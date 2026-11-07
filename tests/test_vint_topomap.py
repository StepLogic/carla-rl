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

from torch_rl.vint_topomap import (TopoNavigator, build_graph, build_topomap,  # noqa: E402
                                   calibrate_close, context_steps, loop_candidates,
                                   node_steps, scc_sizes, sequential_edges, shortest_path)


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


def test_strongly_connected_component_sizes():
    edges = [(0, 1, 1), (1, 2, 1), (2, 0, 1), (2, 3, 1)]
    assert scc_sizes(4, edges) == [3, 1]


class _PlacePolicy:
    """Stand-in ViNT: a frame's value is its PLACE (metres along the road), so the
    predicted distance from a context to a goal is how far ahead the goal's place
    is, and behind is 'far'."""

    def crop85(self, f):
        return np.asarray(f, np.uint8)

    def pair_dists(self, ctxs, goals):
        here = ctxs[:, -1, 0, 0, 0].astype(float)
        there = goals[:, 0, 0, 0].astype(float)
        ahead = there - here
        return np.where(ahead >= 0, ahead / 4.0, 20.0)   # 4 m per ViNT step, behind = far


def test_build_topomap_links_a_road_driven_twice():
    # 0..39 m east (0.4 m/step, 100 steps), a 100-step detour far away, then 0..39 m again
    road = np.arange(100) * 0.4
    x = np.concatenate([road, np.full(100, 500.0), road])
    y = np.concatenate([np.zeros(100), np.linspace(0, 300, 100), np.zeros(100)])
    place = np.concatenate([road, np.full(100, 200.0), road])
    frames = np.repeat(place.astype(np.uint8)[:, None, None, None], 2, axis=1).repeat(2, axis=2).repeat(3, axis=3)
    xy = np.stack([x, y], 1)
    yaw = np.zeros(len(x))
    m = build_topomap(frames, xy, yaw, _PlacePolicy(), every=10, radius_m=2.0, min_gap=3, q=0.95,
                      log=lambda *a: None)
    n = len(m["node_step"])
    assert n == 30 and m["ctx85"].shape == (30, 6, 2, 2, 3)
    e = {(int(u), int(v)) for u, v, _ in m["edges"]}
    assert all((i, i + 1) in e for i in range(n - 1))               # the drive
    assert (0, 20) in e and (20, 0) in e                              # same place, twice
    assert m["summary"]["tau"] > 0 and m["summary"]["kept_revisits"] >= 2
    assert m["summary"]["scc_largest_frac"] > 0.5
    assert shortest_path(n, m["edges"].tolist(), 25, 5) is not None   # a loop closes back


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
