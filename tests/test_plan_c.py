"""Option C driving logic (user 2026-09-13): re-localise at each node, accept only
nodes near the current plan, replan when the match is not the expected node.

Pure functions only -- the CARLA loop in drive_plan_c.py calls these, so the rules
that decide where the car is sent are pinned here without a simulator.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plan_c import bfs_path, candidate_window, nearest_latent, on_node_event, step_cap  # noqa: E402


# a chain 0-1-2-3-4-5 driven one way, plus a branch 2-6
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6)]


def test_bfs_two_way_reaches_backwards():
    assert bfs_path(7, EDGES, 5, 0, undirected=True) == [5, 4, 3, 2, 1, 0]
    assert bfs_path(7, EDGES, 5, 0, undirected=False) is None


def test_bfs_same_node_is_zero_hops():
    assert bfs_path(7, EDGES, 3, 3, undirected=True) == [3]


def test_window_is_expected_node_plus_next_three_hops():
    path = [0, 1, 2, 3, 4, 5]
    assert candidate_window(path, 1, hops=3) == [1, 2, 3, 4]
    assert candidate_window(path, 4, hops=3) == [4, 5]          # clipped at the plan end


def test_nearest_latent_only_searches_the_window():
    Z = np.eye(4)                                                # node k's latent is e_k
    z = np.array([0.1, 0.2, 0.0, 1.0])                           # most like node 3, then node 1
    assert nearest_latent(z, Z, [0, 1, 2]) == 1                  # node 3 is outside the window
    assert nearest_latent(z, Z, [1, 3]) == 3


def test_match_on_expected_node_advances():
    path, i = [0, 1, 2, 3], 1
    new_path, new_i, replanned = on_node_event(path, i, matched=1, goal_node=3, n=7, edges=EDGES, undirected=True)
    assert (new_path, new_i, replanned) == ([0, 1, 2, 3], 2, False)


def test_match_ahead_on_plan_replans_from_the_match():
    path, i = [0, 1, 2, 3, 4], 1
    new_path, new_i, replanned = on_node_event(path, i, matched=3, goal_node=4, n=7, edges=EDGES, undirected=True)
    assert new_path == [3, 4] and new_i == 1 and replanned


def test_match_on_goal_node_finishes_the_plan():
    path, i = [0, 1, 2, 3], 2
    new_path, new_i, replanned = on_node_event(path, i, matched=3, goal_node=3, n=7, edges=EDGES, undirected=True)
    assert new_path == [3] and new_i == 1 and replanned        # i past the end -> target the true goal


def test_step_cap_formula():
    assert step_cap(240.0, m_per_step=0.24, slack=300) == 2 * 1000 + 300
    assert step_cap(0.0) == 300
