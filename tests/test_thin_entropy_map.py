"""Spatial thinning after the MAD filter keeps the highest-entropy node of each cluster and records absorption."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thin_entropy_map import thin  # noqa: E402


def test_cluster_keeps_highest_entropy_and_far_nodes_survive():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5], [20.0, 0.0]])
    H = np.array([1.0, 3.0, 2.0, 0.5])
    kept, absorber = thin(xy, H, radius_m=4.0)
    assert sorted(kept) == [1, 3]
    assert absorber.tolist() == [1, 1, 1, 3]


def test_radius_is_strict_less_than():
    xy = np.array([[0.0, 0.0], [4.0, 0.0]])
    kept, _ = thin(xy, np.array([2.0, 1.0]), radius_m=4.0)
    assert sorted(kept) == [0, 1]
