"""ViNT-native topological map (user 2026-11-07: "start with default trace as map and
then drive"; chose "ViNT-native topomap"). The expert-tour trace IS the map, as
ViNT's create_topomap.py records one (an image every --dt 1.0 s); ViNT's own
distance head localises and picks the next node with navigate.py's rule
(--radius 4, --close-threshold 3). What the benchmark adds -- arbitrary start
and goal points -- needs routes, so revisits of a place become graph links when
ViNT predicts the second place is at most about one node ahead (ViNG-style),
proposed by the trace's odometry (allowed for map building, disclosed).
"""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from torch_rl.vint_topomap import (TopoNavigator, build_graph, calibrate_close,  # noqa: E402
                                   context_steps, loop_candidates, node_steps,
                                   sequential_edges, shortest_path)


def test_a_node_every_second_at_ten_hertz():
    assert node_steps(35).tolist() == [0, 10, 20, 30]
    assert node_steps(35, every=20).tolist() == [0, 20]


def test_a_nodes_context_is_vints_stride_padded_at_the_drive_start():
    assert context_steps(40) == [25, 28, 31, 34, 37, 40]
    assert context_steps(7) == [0, 0, 0, 1, 4, 7]


def test_consecutive_nodes_are_linked_in_drive_order_only():
    assert sequential_edges(4) == [(0, 1), (1, 2), (2, 3)]


def test_revisit_candidates_need_nearby_same_heading_and_not_drive_neighbours():
    # a straight east road driven twice (nodes 0-3, then 10-13 again), one node facing west
    xy = np.array([[0, 0], [4, 0], [8, 0], [12, 0], [40, 40], [44, 40],
                   [48, 40], [52, 40], [56, 40], [60, 40], [0.5, 0.3], [4.2, 0], [8, 0.2], [12, 0],
                   [4, 0]], float)
    yaw = np.zeros(len(xy))
    yaw[14] = math.pi                                     # the same spot, driven the other way
    pairs = set(loop_candidates(xy, yaw, radius_m=3.0, min_gap=3))
    assert (1, 11) in pairs and (11, 1) in pairs and (0, 10) in pairs
    assert (1, 2) not in pairs                            # drive neighbours: already linked
    assert not any(14 in p for p in pairs)                # opposite heading never links
    assert (0, 1) not in pairs                            # 4 m apart: outside the radius


def test_the_close_threshold_is_calibrated_on_consecutive_nodes_not_on_success():
    seq = np.arange(1, 101, dtype=float) / 10.0           # 0.1 .. 10.0
    assert calibrate_close(seq, q=0.95) == pytest.approx(np.quantile(seq, 0.95))


def test_the_graph_keeps_drive_links_and_only_close_predicted_revisits():
    edges = build_graph(seq_edges=[(0, 1), (1, 2)], seq_costs=[3.0, 3.5],
                        cand_pairs=[(0, 2), (2, 0)], cand_dists=[2.0, 9.0], tau=4.0)
    assert sorted(edges) == [(0, 1, 3.0), (0, 2, 2.0), (1, 2, 3.5)]


def test_shortest_path_is_dijkstra_and_none_when_unreachable():
    edges = [(0, 1, 3.0), (1, 2, 3.0), (0, 2, 10.0), (2, 3, 1.0)]
    assert shortest_path(4, edges, 0, 3) == [0, 1, 2, 3]
    assert shortest_path(4, edges, 3, 0) is None
    assert shortest_path(4, edges, 2, 2) == [2]


def test_the_navigator_follows_navigate_py():
    """navigate.py (ViNT branch): window closest-r .. min(closest+r+1, goal); argmin;
    a node farther than close_threshold is itself the target, a closer one hands
    over to the next node; reached when the closest node is the goal node."""
    nav = TopoNavigator(list(range(20)), radius=4, close=3.0)
    assert list(nav.window()) == [0, 1, 2, 3, 4, 5]
    d = np.array([5.0, 4.0, 3.5, 6.0, 7.0, 8.0])
    assert nav.step(d) == 2 and nav.closest == 2           # far: drive at the closest node
    assert list(nav.window()) == list(range(0, 8))
    d = np.full(8, 9.0)
    d[3] = 1.0
    assert nav.step(d) == 4 and nav.closest == 4           # close: hand over to the next
    assert not nav.final
    nav.closest = 17
    assert list(nav.window()) == list(range(13, 20))       # clipped at the goal node
    d = np.full(7, 9.0)
    d[5] = 0.5                                             # node 18 close -> 19, the goal
    assert nav.step(d) == 19 and nav.final
