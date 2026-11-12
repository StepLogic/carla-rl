"""SGM map builder: nodes match sgm_faithful, and every sample records the node that absorbed it."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from build_sgm_map import sgm_with_members  # noqa: E402
from vae_nav.sgm_frame import sgm_faithful  # noqa: E402


def test_edgeless_node_absorbs_gated_samples_and_far_sample_is_kept():
    P = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [30.0, 0.0]])
    dist = np.full((4, 4), 100.0)                       # no edges under cutoff 10
    kept, absorber, edges = sgm_with_members(dist, P, consistency=5.0, edge_cutoff=10.0, gate_m=2.0)
    assert kept.tolist() == [0, 3]
    assert absorber.tolist() == [0, 0, 0, 3]
    assert edges == []


def test_inconsistent_sample_becomes_a_node_consistent_one_merges():
    P = np.array([[0.0, 0.0], [10.0, 0.0], [0.5, 0.0], [0.4, 0.0]])
    dist = np.full((4, 4), 100.0)
    dist[0, 1] = dist[1, 0] = 5.0                      # node 0 <-> node 1 edge
    dist[2, 1] = 50.0; dist[1, 2] = 50.0               # sample 2 disagrees with node 0 by 45 steps
    dist[3, 1] = 6.0; dist[1, 3] = 6.0                 # sample 3 disagrees by 1 step
    kept, absorber, _ = sgm_with_members(dist, P, consistency=10.0, edge_cutoff=20.0, gate_m=2.0)
    assert kept.tolist() == [0, 1, 2]
    assert absorber[3] == 0
    ref, _ = sgm_faithful(dist, P, consistency_cutoff=10.0, edge_cutoff=20.0, gate_m=2.0)
    assert kept.tolist() == ref.tolist()
