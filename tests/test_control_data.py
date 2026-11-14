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


def test_stack_frames_is_channels_first_and_uint8():
    frames = _episode()["frames"]
    out = stack_frames(frames, t=4, n_frames=2)
    assert out.shape == (6, 8, 8) and out.dtype == np.uint8
    assert [int(out[i * 3, 0, 0]) for i in range(2)] == [3, 4]
