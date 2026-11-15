"""Assembling training chunks from the recorded shards. Every bug this code can have is silent: a
one-step misalignment between a frame and the expert's action, a stack that bleeds the previous
episode's frames into the first step, or an invalid lane label counted in the loss would all train
quietly and give a worse network, not an error. So the assembly is a pure function with tests.

Shard layout (torch_rl.expert_record): frames (T+1, H, W, 3) uint8 -- one more than the actions,
because the last frame is the state AFTER the final action -- act_label (T, 2), speed (T,),
pos (T, 2). Lane labels (torch_rl.lane_labels.episode_labels) have length T+1 with a `valid` mask
that is False at frame 0 and wherever heading is undefined (the car was at rest).
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

from torch_rl.control_data import (chunk_from_episode, stack_frames,  # noqa: E402
                                   transition_actions)


def test_behaviour_cloning_takes_the_LABEL_and_a_value_learner_the_EXECUTED_action():
    """The two action columns are not interchangeable and mixing them is silent. act_label is the
    expert's intended action (the BC target); act_exec is what the car did, so it is the action whose
    consequence the next frame records. Conditioning a critic on act_label makes identical (s, a)
    rows map to different s' by unseen noise, and TD then averages the action-dependence out of Q."""
    ep = _episode()
    shard = {"act_label": ep["act_label"],
             "act_exec": ep["act_label"] + 0.3}                 # the executed action differs
    assert np.allclose(transition_actions(shard, 10), ep["act_label"] + 0.3)
    assert np.allclose(chunk_from_episode(ep, 0, 3, 1)["actions"], ep["act_label"][:3])
    # a shard with no act_exec (expert-driven recordings) falls back to the label
    assert np.allclose(transition_actions({"act_label": ep["act_label"]}, 4), ep["act_label"][:4])


def _episode(t=10, px=8):
    """A tiny episode whose frame k is filled with the value k, so alignment is readable."""
    frames = np.stack([np.full((px, px, 3), k, np.uint8) for k in range(t + 1)])
    return {
        "frames": frames,
        "act_label": np.stack([[k * 0.1, -k * 0.1] for k in range(t)]).astype(np.float32),
        "speed": np.arange(t, dtype=np.float32),
        "offset": np.arange(t + 1, dtype=np.float32) * 0.01,
        "heading_err": np.arange(t + 1, dtype=np.float32) * 0.5,
        "valid": np.array([False] + [True] * t),
    }


def test_a_frame_is_paired_with_the_action_taken_from_it():
    ep = _episode()
    c = chunk_from_episode(ep, start=3, length=4, n_frames=1)
    # frame k carries the value k; the action from frame k is [k/10, -k/10]
    assert [int(f[0, 0, 0]) for f in c["frames"]] == [3, 4, 5, 6]
    assert np.allclose(c["actions"][:, 0], [0.3, 0.4, 0.5, 0.6])
    assert np.allclose(c["speed"], [3, 4, 5, 6])


def test_the_stack_holds_the_PRECEDING_frames_newest_last():
    ep = _episode()
    c = chunk_from_episode(ep, start=5, length=2, n_frames=3)
    # step 0 of the chunk is frame 5, so its stack is frames 3, 4, 5 on the channel axis
    assert c["frames"].shape == (2, 9, 8, 8)
    assert [int(c["frames"][0, i * 3, 0, 0]) for i in range(3)] == [3, 4, 5]
    assert [int(c["frames"][1, i * 3, 0, 0]) for i in range(3)] == [4, 5, 6]


def test_the_stack_at_the_episode_start_repeats_frame_zero_and_never_bleeds():
    ep = _episode()
    c = chunk_from_episode(ep, start=0, length=1, n_frames=3)
    # there is no frame -1 or -2: the oldest available frame is repeated, so no other episode's
    # pixels and no wrap-around can enter the stack
    assert [int(c["frames"][0, i * 3, 0, 0]) for i in range(3)] == [0, 0, 0]


def test_lane_labels_come_from_the_same_frame_and_carry_their_mask():
    ep = _episode()
    c = chunk_from_episode(ep, start=0, length=3, n_frames=1)
    assert np.allclose(c["lane"][:, 0], [0.0, 0.01, 0.02])      # offset at frames 0, 1, 2
    assert np.allclose(c["lane"][:, 1], [0.0, 0.5, 1.0])        # heading error at the same frames
    assert c["lane_valid"].tolist() == [False, True, True]      # frame 0 is never a label


def test_a_chunk_cannot_run_past_the_last_action():
    ep = _episode(t=10)
    with pytest.raises(ValueError):
        chunk_from_episode(ep, start=8, length=4, n_frames=1)   # actions exist for 0..9 only
    ok = chunk_from_episode(ep, start=6, length=4, n_frames=1)
    assert ok["actions"].shape == (4, 2)


def test_nstep_rows_reduce_to_single_steps_at_k_one():
    from torch_rl.control_data import nstep_rows
    r = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    a = np.stack([np.arange(4), np.zeros(4)], -1).astype(np.float32)
    out = nstep_rows(a, r, terminal_at_end=False, k=1, gamma=0.99)
    assert out["starts"].tolist() == [0, 1, 2, 3]
    assert np.allclose(out["rewards"], r)
    assert np.allclose(out["actions"], a)
    assert out["masks"].tolist() == [1, 1, 1, 1]
    assert out["nexts"].tolist() == [1, 2, 3, 4]


def test_nstep_rows_discount_inside_the_window_and_average_the_action():
    """The k-step transition is (s_t, mean action over the window, sum gamma^i r_{t+i}, s_{t+k}).
    The mean action is an APPROXIMATION of holding one action for k steps -- the recorded actions
    varied within the window -- and averaging washes variation out, so it biases toward a FLATTER
    Q. A span that grows under it is therefore meaningful; a null result is weak evidence."""
    from torch_rl.control_data import nstep_rows
    r = np.ones(6, np.float32)
    a = np.stack([np.array([1.0, 2, 3, 4, 5, 6]), np.zeros(6)], -1).astype(np.float32)
    out = nstep_rows(a, r, terminal_at_end=False, k=3, gamma=0.5)
    assert out["starts"].tolist() == [0, 3]
    assert np.allclose(out["rewards"], [1 + 0.5 + 0.25, 1 + 0.5 + 0.25])
    assert np.allclose(out["actions"][:, 0], [2.0, 5.0])        # mean of 1,2,3 and of 4,5,6
    assert out["nexts"].tolist() == [3, 6]


def test_nstep_rows_drop_the_bootstrap_only_on_the_window_that_ends_the_episode():
    from torch_rl.control_data import nstep_rows
    r = np.ones(6, np.float32)
    a = np.zeros((6, 2), np.float32)
    out = nstep_rows(a, r, terminal_at_end=True, k=3, gamma=1.0)
    assert out["masks"].tolist() == [1, 0]                      # only the last window terminates
    short = nstep_rows(a, r, terminal_at_end=True, k=4, gamma=1.0)
    assert short["starts"].tolist() == [0]                      # a partial tail window is dropped
    assert short["masks"].tolist() == [1]


def test_non_finite_labels_are_scrubbed_not_merely_masked():
    """THE BUG THIS PINS: episode_labels leaves NaN where the heading is undefined (car at rest) or
    the lane query hit a fork. In a masked loss `nan * 0` is still nan, so the first backward pass
    NaNs every weight -- which is exactly what happened on the first run. The chunk must therefore
    come out finite AND have those steps masked off."""
    ep = _episode()
    ep["heading_err"][4] = np.nan
    ep["offset"][6] = np.inf
    c = chunk_from_episode(ep, start=2, length=6, n_frames=1)       # frames 2..7
    assert np.isfinite(c["lane"]).all()                             # no nan/inf reaches the loss
    assert c["lane_valid"].tolist() == [True, True, False, True, False, True]
    assert c["lane"][2, 1] == 0.0 and c["lane"][4, 0] == 0.0        # replaced, not left as nan


def test_the_stack_stride_matches_the_decision_rate():
    """With ACTION_REPEAT=k the policy sees frames k ticks apart, so a stream pretrained on
    CONSECUTIVE frames would be fed k times the apparent motion at deployment -- the same optical
    flow meaning a different heading rate. stride makes training and deployment agree."""
    ep = _episode(t=20)
    c = chunk_from_episode(ep, start=16, length=1, n_frames=2, stride=8)
    assert [int(c["frames"][0, i * 3, 0, 0]) for i in range(2)] == [8, 16]     # 8 ticks apart
    # at the start of an episode the oldest available frame is repeated, never wrapped
    first = chunk_from_episode(ep, start=2, length=1, n_frames=3, stride=8)
    assert [int(first["frames"][0, i * 3, 0, 0]) for i in range(3)] == [0, 0, 2]


def test_a_chunk_can_step_at_the_decision_rate():
    """At ACTION_REPEAT=k the policy decides once every k ticks: its GRU steps once per decision and
    its frame pair is k ticks apart. A stream pretrained with the GRU stepping every tick would meet
    a different time scale at deployment, so the chunk can advance `step` ticks per GRU step."""
    ep = _episode(t=30)
    c = chunk_from_episode(ep, start=0, length=3, n_frames=2, stride=8, step=8)
    # GRU steps at ticks 0, 8, 16; each stack is (tick - 8, tick), the first repeating frame 0
    assert [[int(c["frames"][j, i * 3, 0, 0]) for i in range(2)] for j in range(3)] == \
        [[0, 0], [0, 8], [8, 16]]
    assert np.allclose(c["actions"][:, 0], [0.0, 0.8, 1.6])        # the label AT each decision tick
    assert np.allclose(c["speed"], [0, 8, 16])
    assert np.allclose(c["lane"][:, 1], [0.0, 4.0, 8.0])           # heading label at those ticks
    with pytest.raises(ValueError):
        chunk_from_episode(ep, start=8, length=4, n_frames=1, step=8)   # tick 32 > last action 29
    assert chunk_from_episode(ep, start=5, length=4, n_frames=1, step=8)["actions"].shape == (4, 2)


def test_stack_frames_is_channels_first_and_uint8():
    frames = _episode()["frames"]
    out = stack_frames(frames, t=4, n_frames=2)
    assert out.shape == (6, 8, 8) and out.dtype == np.uint8
    assert [int(out[i * 3, 0, 0]) for i in range(2)] == [3, 4]


# ---- how a pretrained stream is judged --------------------------------------------------------------

def test_r2_needs_no_fitting_and_is_zero_for_the_mean():
    """The first number to quote for a trained head is its OWN prediction against the label on
    held-out episodes: nothing is fitted, so nothing can be fitted to the test rows."""
    from torch_rl.control_data import r2_score
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(y, y) == 1.0
    assert abs(r2_score(y, np.full(4, y.mean()))) < 1e-9
    assert r2_score(y, -y) < 0                                  # worse than the mean is negative


def _fingerprint_data(n_ep=30, per=40, dim=64, seed=0):
    """Features that identify the EPISODE and know nothing else, and a target that varies mostly
    between episodes -- heading error in the recorded shards: 80% of its variance sat in 3 of the
    validation episodes."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_ep), per)
    level = rng.normal(0.0, 5.0, n_ep)
    y = (level[groups] + rng.normal(0.0, 0.5, n_ep * per))[:, None].astype(np.float32)
    H = (rng.normal(0.0, 1.0, (n_ep, dim))[groups]
         + rng.normal(0.0, 0.05, (n_ep * per, dim))).astype(np.float32)
    return H, y, groups


def test_a_probe_must_hold_out_whole_episodes():
    """THE MISTAKE THIS PINS (2026-11-14). The control stream's "heading R^2 0.81" was a ridge probe
    fitted and scored on the SAME rows, compared with c0's 0.046 from an episode-grouped CV probe.
    Held out by episode the stream scored about 0.1. A probe on rows of the same few episodes can
    memorise the episodes: here the features carry no information about the target beyond WHICH
    episode a row is from, the in-sample fit is near perfect, and the honest one is worthless."""
    from torch_rl.control_data import ridge_r2_episode_cv, ridge_r2_in_sample
    H, y, groups = _fingerprint_data()
    assert ridge_r2_in_sample(H, y)[0] > 0.9
    assert ridge_r2_episode_cv(H, y, groups)[0] < 0.1


def test_an_episode_held_out_probe_still_finds_what_the_features_carry():
    """Holding episodes out does not blind the probe: one feature that really tracks the target,
    among 63 that are noise, is found."""
    from torch_rl.control_data import ridge_r2_episode_cv
    _, y, groups = _fingerprint_data()
    rng = np.random.default_rng(1)
    H = rng.normal(0.0, 1.0, (len(y), 64)).astype(np.float32)
    H[:, 0] = y[:, 0] + rng.normal(0.0, 0.5, len(y))                         # the honest feature
    assert ridge_r2_episode_cv(H, y, groups)[0] > 0.8


def test_an_episode_held_out_probe_never_scores_an_episode_it_was_fitted_on():
    """Every row is predicted exactly once, by a fit that saw none of its episode's rows."""
    from torch_rl.control_data import episode_folds
    groups = np.repeat(np.arange(11), 3)
    folds = episode_folds(groups, 5, seed=0)
    assert folds.shape == groups.shape and set(folds.tolist()) == set(range(5))
    for g in range(11):
        assert len(set(folds[groups == g].tolist())) == 1      # an episode is never split
    with pytest.raises(ValueError):
        episode_folds(np.zeros(6, int), 5)                      # one episode cannot be held out
