"""Sweep seeds must give DISJOINT episode sets; the same seed the same set.

Measured 2026-09-11: with per-episode seeds seed0 + i, sweep seeds
1234..1237 shared 23-24 of their 25 episodes (identical median routes for
three of four "seeds", sd 0.02 across them). Four seeds were one draw.
"""
import numpy as np


def _episode_seeds(seed0, n=25):
    # the rule used by eval_policy.run_episodes
    return set(int(x) for x in np.random.default_rng(int(seed0)).integers(0, 2 ** 31 - 1, size=n))


def test_consecutive_sweep_seeds_share_no_episodes():
    sets = [_episode_seeds(s) for s in (1234, 1235, 1236, 1237)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not (sets[i] & sets[j]), (i, j)


def test_same_seed_same_set():
    assert _episode_seeds(1234) == _episode_seeds(1234)


def test_the_old_rule_overlapped():
    old = lambda s0, n=25: set(range(s0, s0 + n))
    assert len(old(1235) & old(1236)) == 24      # the trap this file pins
