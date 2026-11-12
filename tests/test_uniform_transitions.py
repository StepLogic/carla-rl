"""UNIFORM TRANSITIONS (user 2026-11-08: "Fix the window-sampler bias so every
transition is equally likely (plain SAC), and drop the forced 50/50 goal/explore
batch split").

The default sampler draws a trained window's start t0 uniformly in [0, n - 1]
of an episode picked with probability proportional to n: every row STARTS a
window equally often, but a window trains T rows, so row r is trained by the
min(r + 1, T) starts at or before it -- an episode's first T - 1 rows (and the
whole of a short episode) are trained less than its later rows.

uniform_transitions=True draws a VIRTUAL start v uniformly over the n + T - 1
values [-(T - 1), n - 1] (episode probability proportional to n + T - 1) and
trains the rows [v, v + T - 1] that exist: each row is inside exactly T of the
virtual windows, so every stored transition is trained equally often, as plain
SAC's uniform replay would. A window with v < 0 starts at row 0 (hidden state
reset there, as at an env reset) and trains its first v + T rows only.

goal_batch_frac=None drops the per-batch goal/explore split: chunks are drawn
from all episodes together, so each mode's share is its share of the data.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")

from torch_rl.episode_buffer import DeviceEpisodeBuffer  # noqa: E402

K, T = 2, 8          # burn-in, trained window
L = K + T


def _buf(capacity=4000, **kw):
    return DeviceEpisodeBuffer(3, 2, capacity, "cpu", seed=0, **kw)


def _fill(buf, episodes):
    """episodes: [(n, mode)]. Reward = a unique transition id + 1 (exact in
    float32, and 0 marks an untrained slot); obs = (episode, row, 0). Returns
    the first id of each episode and the total number of transitions."""
    firsts, nid = [], 0
    for ep, (n, mode) in enumerate(episodes):
        firsts.append(nid)
        for t in range(n):
            o = np.array([ep, t, 0], np.float32)
            o2 = np.array([ep, t + 1, 0], np.float32)
            buf.add(o, np.zeros(2, np.float32), float(nid + t + 1), t == n - 1, False,
                    o2, mode)
        nid += n
    return firsts, nid


def _counts(buf, n_ids, calls=1000, B=64):
    """How often each transition id was a TRAINED row (valid) in `calls` batches."""
    counts = np.zeros(n_ids + 1)
    for _ in range(calls):
        out = buf.sample_chunks(B, L, K)
        ids = out["rewards"].numpy()[out["valid"].numpy() > 0].astype(np.int64)
        np.add.at(counts, ids, 1)
    return counts[1:]


MIXED = [(1, 1), (3, 0), (10, 1), (57, 0), (120, 1)]


def test_every_transition_is_trained_equally_often():
    b = _buf(uniform_transitions=True, goal_batch_frac=None)
    _, n_ids = _fill(b, MIXED)
    c = _counts(b, n_ids)
    # ~2,270 expected per row: 3.5 sigma is ~7.5 %
    assert np.abs(c / c.mean() - 1.0).max() < 0.12, (c.min(), c.mean(), c.max())


def test_the_default_sampler_is_unchanged_and_undertrains_episode_starts():
    # The documented old behaviour, kept as the default so earlier runs keep
    # their meaning: row r of an episode is trained ~min(r + 1, T) times.
    b = _buf(goal_batch_frac=1.0)
    firsts, n_ids = _fill(b, [(120, 1)])
    c = _counts(b, n_ids, calls=400)
    mid = c[T:].mean()
    assert c[0] / mid == pytest.approx(1.0 / T, abs=0.05)
    assert c[T - 1] / mid == pytest.approx(1.0, abs=0.1)


def test_without_the_split_each_mode_gets_its_share_of_the_data():
    b = _buf(uniform_transitions=True, goal_batch_frac=None)
    _fill(b, [(150, 1), (150, 1), (100, 0)])          # 300 goal rows, 100 explore
    goal = rows = 0.0
    for _ in range(400):
        out = b.sample_chunks(64, L, K)
        per_chunk = out["valid"].numpy().sum(axis=1)
        goal += float((per_chunk * out["mode"].numpy()).sum())
        rows += float(per_chunk.sum())
    assert goal / rows == pytest.approx(0.75, abs=0.03)


def test_the_split_still_applies_when_a_fraction_is_set():
    b = _buf(uniform_transitions=True, goal_batch_frac=0.5)
    _fill(b, [(150, 1), (150, 1), (100, 0)])
    for _ in range(20):
        assert b.sample_chunks(64, L, K)["mode"].numpy().mean() == 0.5


def test_trained_rows_line_up_with_their_observations_and_bootstrap_rows():
    b = _buf(uniform_transitions=True, goal_batch_frac=None)
    firsts, _ = _fill(b, MIXED)
    for _ in range(200):
        out = b.sample_chunks(64, L, K)
        obs, valid = out["obs"].numpy(), out["valid"].numpy()
        rew, keep = out["rewards"].numpy(), out["keep"].numpy()
        for j in range(valid.shape[0]):
            v = valid[j] > 0
            n_v = int(v.sum())
            assert n_v >= 1
            assert v[:n_v].all() and not v[n_v:].any()      # a prefix
            for i in range(n_v):
                ep, row = int(obs[j, K + i, 0]), int(obs[j, K + i, 1])
                assert rew[j, i] == firsts[ep] + row + 1      # acts on its own row
                assert int(obs[j, K + i + 1, 0]) == ep        # bootstraps from the
                assert int(obs[j, K + i + 1, 1]) == row + 1   # next row of it
            if int(obs[j, K, 1]) == 0:                        # window opens at row 0:
                assert keep[j, K] == 0.0                      # hidden reset there


def test_a_window_reaching_back_before_the_episode_trains_only_rows_that_exist():
    # One episode of 3 transitions: virtual starts -7..2. Every chunk's trained
    # rows are a run of real rows starting at max(v, 0), so row 0 is trained by
    # 8 of the 10 starts (v = -7..0), not 1.
    b = _buf(uniform_transitions=True, goal_batch_frac=None)
    _, n_ids = _fill(b, [(3, 1)])
    c = _counts(b, n_ids, calls=300)
    assert np.abs(c / c.mean() - 1.0).max() < 0.05


def test_per_is_not_combined_with_either_setting():
    # PER draws window starts per mode stratum; neither combination is built.
    with pytest.raises(ValueError, match="uniform_transitions"):
        _buf(uniform_transitions=True, per_alpha=0.6)
    with pytest.raises(ValueError, match="goal_batch_frac=None"):
        _buf(goal_batch_frac=None, per_alpha=0.6)


def test_trainer_reads_the_two_settings(monkeypatch):
    import train_carla_sac as T_
    monkeypatch.delenv("UNIFORM_TRANSITIONS", raising=False)
    monkeypatch.delenv("GOAL_BATCH_FRAC", raising=False)
    assert T_.replay_sampling() == {"uniform_transitions": False, "goal_batch_frac": 0.5}
    monkeypatch.setenv("UNIFORM_TRANSITIONS", "1")
    monkeypatch.setenv("GOAL_BATCH_FRAC", "none")
    assert T_.replay_sampling() == {"uniform_transitions": True, "goal_batch_frac": None}
    monkeypatch.setenv("GOAL_BATCH_FRAC", "0.3")
    assert T_.replay_sampling()["goal_batch_frac"] == 0.3


def test_trainer_builds_both_replays_with_the_settings():
    src = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    # the agent's replay and the expert replay of IL_EXPERT_FRAC
    assert src.count("**replay_sampling()") == 2
    assert 'goal_batch_frac=env_float("GOAL_BATCH_FRAC", 0.5)' not in src
