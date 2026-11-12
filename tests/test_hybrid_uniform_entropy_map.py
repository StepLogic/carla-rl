"""Uniform nodes fill the gaps between entropy seeds, never within the gate of an existing node."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_uniform_entropy_map import place_hybrid  # noqa: E402


def test_uniform_nodes_fill_gaps_and_respect_seeds():
    P = np.array([[x, 0.0] for x in range(0, 31, 1)], np.float64)      # a straight 30 m trace, 1 m steps
    idx, kinds, members = place_hybrid(P, seed_idx=[10], seed_members=[[9, 10, 11]], gate_m=6.0)
    assert 10 in idx and kinds[idx.index(10)] == "entropy"
    xy = P[idx]
    D = np.linalg.norm(xy[:, None] - xy[None], axis=-1); np.fill_diagonal(D, np.inf)
    assert D.min() >= 6.0 - 1e-9                                          # nothing inside the gate of another node
    assert idx == sorted(idx)                                             # trace order
    assert sorted(t for m in members for t in m) == list(range(31))       # every sample assigned exactly once
    assert {9, 10, 11} <= set(members[idx.index(10)])                      # the seed keeps its absorbed samples
    # ...and also takes unclaimed samples nearest to it (5-8 are closer to x=10 than to the uniform node at x=0)
    assert all(members[idx.index(10)].count(t) == 1 for t in (5, 6, 7, 8))


def test_first_trace_sample_becomes_uniform_when_far_from_seeds():
    P = np.array([[0.0, 0.0], [100.0, 0.0]])
    idx, kinds, _ = place_hybrid(P, seed_idx=[1], seed_members=[[1]], gate_m=6.0)
    assert idx == [0, 1] and kinds == ["uniform", "entropy"]
