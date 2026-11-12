"""A closed coverage tour of a town's lanes for the single-trace evaluation maps
(user 2026-11-06: "CAN YOU CREATE MAP FROM A SINGLE TRACE?" -> "I am talking
about evaluations").

The expert drives ONE uninterrupted closed tour that traverses every driving
lane at least once in its legal direction, so the map is built from one
continuous trace instead of v30's many short segments. Choosing that tour is a
directed rural postman problem on the lane graph: the REQUIRED edges are the
lanes outside junctions, any other edge (a junction connector, or a lane again)
may be used to link them. These tests pin the planner on graphs small enough
to check by hand, and the conversion of an edge walk to the waypoint plan the
expert follows.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vae_nav.coverage_tour import (coverage_tour, lane_graph,  # noqa: E402
                                   tour_waypoints, walk_length)


def _assert_closed_walk(walk, edges, start):
    assert walk, "empty tour"
    assert walk[0][0] == start and walk[-1][1] == start
    for (a, b), (c, d) in zip(walk, walk[1:]):
        assert b == c, f"walk breaks between {(a, b)} and {(c, d)}"
    for e in walk:
        assert e in edges, f"{e} is not an edge of the graph"


def test_a_directed_cycle_is_its_own_tour():
    edges = {("a", "b"): 1.0, ("b", "c"): 1.0, ("c", "a"): 1.0}
    walk = coverage_tour(edges, required=edges.keys(), start="a")
    _assert_closed_walk(walk, edges, "a")
    assert walk == [("a", "b"), ("b", "c"), ("c", "a")]


def test_required_lanes_are_linked_through_connectors():
    # two lanes (required) joined by junction connectors (not required)
    edges = {("a", "b"): 10.0, ("c", "d"): 10.0, ("b", "c"): 2.0, ("d", "a"): 2.0}
    walk = coverage_tour(edges, required=[("a", "b"), ("c", "d")], start="a")
    _assert_closed_walk(walk, edges, "a")
    assert walk_length(walk, edges) == pytest.approx(24.0)


def test_an_unbalanced_node_gets_the_cheapest_extra_path():
    # a has two required lanes out and none back in: the tour must come back
    # to a twice, each time by the shortest return (b->a or c->a), never by
    # the long detour b->c->a.
    edges = {("a", "b"): 5.0, ("a", "c"): 5.0, ("b", "a"): 1.0, ("c", "a"): 1.0,
             ("b", "c"): 50.0}
    walk = coverage_tour(edges, required=[("a", "b"), ("a", "c")], start="a")
    _assert_closed_walk(walk, edges, "a")
    assert set(walk) == {("a", "b"), ("a", "c"), ("b", "a"), ("c", "a")}
    assert walk_length(walk, edges) == pytest.approx(12.0)


def test_separate_loops_are_joined_into_one_closed_tour():
    # two balanced required loops that only optional links connect
    edges = {("a", "b"): 1.0, ("b", "a"): 1.0, ("c", "d"): 1.0, ("d", "c"): 1.0,
             ("b", "c"): 7.0, ("c", "b"): 7.0}
    req = [("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")]
    walk = coverage_tour(edges, required=req, start="a")
    _assert_closed_walk(walk, edges, "a")
    assert set(req) <= set(walk)
    assert walk_length(walk, edges) == pytest.approx(18.0)


def test_the_start_may_lie_off_every_required_lane():
    edges = {("s", "a"): 3.0, ("a", "b"): 1.0, ("b", "a"): 1.0, ("a", "s"): 3.0}
    walk = coverage_tour(edges, required=[("a", "b"), ("b", "a")], start="s")
    _assert_closed_walk(walk, edges, "s")
    assert walk_length(walk, edges) == pytest.approx(8.0)


def test_an_unreachable_lane_is_refused():
    edges = {("a", "b"): 1.0, ("b", "a"): 1.0, ("c", "d"): 1.0, ("b", "c"): 1.0}
    with pytest.raises(ValueError, match="reach"):
        coverage_tour(edges, required=[("a", "b"), ("c", "d")], start="a")


def test_a_required_edge_must_exist():
    with pytest.raises(ValueError, match="not an edge"):
        coverage_tour({("a", "b"): 1.0, ("b", "a"): 1.0}, required=[("a", "c")], start="a")


def test_random_strongly_connected_graphs_are_fully_covered():
    rng = np.random.default_rng(0)
    for trial in range(20):
        n = int(rng.integers(5, 25))
        nodes = list(range(n))
        edges = {}
        perm = rng.permutation(n)
        for i in range(n):                       # a Hamiltonian cycle: strongly connected
            edges[(int(perm[i]), int(perm[(i + 1) % n]))] = float(rng.uniform(1, 10))
        for _ in range(2 * n):
            a, b = (int(x) for x in rng.choice(nodes, 2, replace=False))
            edges.setdefault((a, b), float(rng.uniform(1, 10)))
        keys = list(edges)
        req = [keys[i] for i in rng.choice(len(keys), max(1, len(keys) // 2), replace=False)]
        start = int(rng.integers(n))
        walk = coverage_tour(edges, required=req, start=start)
        _assert_closed_walk(walk, edges, start)
        assert set(req) <= set(walk), f"trial {trial}: a required edge was skipped"


# ---- the lane graph from CARLA's GlobalRoutePlanner ------------------------------

class _Loc:
    def __init__(self, x, y):
        self.x, self.y = float(x), float(y)


class _Wp:
    def __init__(self, x, y):
        self.transform = type("T", (), {"location": _Loc(x, y)})()


class _FakeGRP:
    """Just the parts of GlobalRoutePlanner that lane_graph reads: `_graph`
    with GRP's edge attributes (type, intersection, entry/exit waypoints, path)."""

    def __init__(self):
        import networkx as nx
        from types import SimpleNamespace
        self._graph = nx.DiGraph()
        follow, change = SimpleNamespace(name="LANEFOLLOW"), SimpleNamespace(name="CHANGELANELEFT")

        def add(u, v, pts, intersection, kind=follow):
            wps = [_Wp(*p) for p in pts]
            self._graph.add_edge(u, v, type=kind, intersection=intersection,
                                 entry_waypoint=wps[0], exit_waypoint=wps[-1],
                                 path=wps[1:-1], length=len(wps) - 1)
        add(0, 1, [(0, 0), (2, 0), (4, 0), (6, 0)], False)      # a 6 m lane
        add(1, 2, [(6, 0), (8, 1), (9, 3)], True)               # a junction turn
        add(2, 0, [(9, 3), (5, 3), (0, 3), (0, 0)], False)
        add(0, 2, [(0, 0), (9, 3)], False, kind=change)         # a lane change: never used


def test_lane_graph_keeps_lane_following_edges_and_requires_only_lanes():
    edges, required, wps = lane_graph(_FakeGRP(), follow_type="LANEFOLLOW")
    assert set(edges) == {(0, 1), (1, 2), (2, 0)}             # no lane change
    assert set(required) == {(0, 1), (2, 0)}                 # junction turns optional
    assert edges[(0, 1)] == pytest.approx(6.0)                # metres along the lane
    assert len(wps[(0, 1)]) == 4                              # entry + path + exit


def test_tour_waypoints_concatenates_edges_without_duplicate_joints():
    grp = _FakeGRP()
    edges, required, wps = lane_graph(grp, follow_type="LANEFOLLOW")
    walk = coverage_tour(edges, required, start=0)
    plan = tour_waypoints(walk, wps, min_gap_m=0.5)
    xy = np.array([[w.transform.location.x, w.transform.location.y] for w in plan])
    gaps = np.hypot(*np.diff(xy, axis=0).T)
    assert gaps.min() >= 0.5                                  # the shared joints are dropped
    assert xy[0].tolist() == [0.0, 0.0] and xy[-1].tolist() == [0.0, 0.0]
    assert len(plan) == 4 + 2 + 3                             # 11 points, 2 duplicate joints


def test_the_drive_plan_closes_the_loop_past_its_own_start():
    # The car starts a few waypoints into the tour; ending at the tour's last
    # waypoint would stop short of the spawn point, leaving the first and last
    # stretches unjoined. The drive plan runs past the spawn again.
    plan = [_Wp(x, 0.0) for x in range(10)]
    from vae_nav.coverage_tour import closed_drive_plan
    drive, k0 = closed_drive_plan(plan, start_k=2, overlap=3)
    xs = [w.transform.location.x for w in drive]
    assert k0 == 2 and xs[:k0 + 1] == [0.0, 1.0, 2.0]
    assert xs[len(plan):] == [1.0, 2.0, 3.0, 4.0, 5.0]  # past the spawn (2) by `overlap`


# ---- towns whose lanes are not one strongly connected component (2026-11-09) --
# Town05's lane-following graph (no lane changes) is not strongly connected: its
# training-town tour (job 465543) failed "the tour cannot reach and return from
# [108, 109, ...]". EXPERT_TOUR_LARGEST_SCC=1 keeps the required lanes inside the
# strongly connected component holding the most required metres.
from vae_nav.coverage_tour import restrict_to_largest_scc  # noqa: E402


def _two_loops():
    # loop A: 0->1->2->0 (lanes 10 m each); loop B: 5->6->5 (lanes 50 m each);
    # a one-way link 2->5 joins them, so neither reaches the other both ways
    edges = {(0, 1): 10.0, (1, 2): 10.0, (2, 0): 10.0, (5, 6): 50.0, (6, 5): 50.0,
             (2, 5): 3.0}
    required = [(0, 1), (1, 2), (2, 0), (5, 6), (6, 5)]
    return edges, required


def test_the_component_with_the_most_required_metres_is_kept():
    edges, required = _two_loops()
    keep, start, dropped = restrict_to_largest_scc(edges, required)
    assert sorted(keep) == [(5, 6), (6, 5)]                 # 100 m beats 30 m
    assert sorted(dropped) == [(0, 1), (1, 2), (2, 0)]
    assert start in (5, 6)
    walk = coverage_tour(edges, keep, start)                 # and it now plans
    _assert_closed_walk(walk, edges, start)
    assert set(keep) <= set(walk)


def test_a_strongly_connected_town_loses_nothing():
    edges = {(0, 1): 1.0, (1, 2): 1.0, (2, 0): 1.0}
    keep, start, dropped = restrict_to_largest_scc(edges, list(edges))
    assert sorted(keep) == sorted(edges) and dropped == []
    assert start == sorted(keep)[0][0]                       # the old rule's start


def test_the_trace_runner_uses_it_only_when_asked():
    src = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    assert 'EXPERT_TOUR_LARGEST_SCC = os.environ.get("EXPERT_TOUR_LARGEST_SCC", "0") == "1"' in src
    body = src[src.index("def run_expert_trace"):]
    body = body[:body.index("walk = coverage_tour(")]
    assert "if EXPERT_TOUR_LARGEST_SCC:" in body and "restrict_to_largest_scc(edges, req)" in body
    assert "start = sorted(req)[0][0]" in body               # the default path, unchanged
