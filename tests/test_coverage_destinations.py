"""The oracle's destination rule: go to the NEAREST ground not yet driven.

This is the whole difference between the oracle trace and the policy trace. v43's
explore reward has no novelty term, so the policy re-drives one ring road; the
oracle covers the map only if this chooser actually refuses ground already
visited. Each failure mode below was a plausible bug in the first draft:
sending the car somewhere it is already standing, re-picking a visited cell
because the ARRAY index was fresh rather than the CELL, and hanging forever once
the map is covered instead of saying so.

No CARLA here -- waypoints are faked to the one attribute the chooser reads, so
this runs on any machine.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Loc:
    def __init__(self, x, y):
        self.x, self.y = float(x), float(y)


class _Wp:
    """Only `.transform.location.{x,y}` is read by the chooser."""
    def __init__(self, x, y):
        self.transform = type("T", (), {"location": _Loc(x, y)})()


def _chooser(points, cell_m=10.0, seed=0):
    from collect_oracle_trace_v43 import CoverageDestinations
    return CoverageDestinations([_Wp(*p) for p in points], cell_m,
                                np.random.default_rng(seed))


def test_picks_the_nearest_unvisited_not_the_farthest():
    c = _chooser([(0, 0), (50, 0), (500, 0)])
    g = c.next_goal((0.0, 0.0), min_m=10.0)
    assert (g.x, g.y) == (50.0, 0.0)          # not the 500 m one


def test_refuses_ground_already_driven():
    c = _chooser([(0, 0), (50, 0), (500, 0)])
    c.visit((50.0, 0.0))                       # mark that cell driven
    g = c.next_goal((0.0, 0.0), min_m=10.0)
    assert (g.x, g.y) == (500.0, 0.0)          # the only cell left


def test_min_distance_keeps_it_from_routing_under_the_car():
    c = _chooser([(0, 0), (5, 0), (200, 0)])
    g = c.next_goal((0.0, 0.0), min_m=30.0)
    assert (g.x, g.y) == (200.0, 0.0)          # 5 m away is below the floor


def test_visiting_one_point_marks_the_whole_cell():
    """A 10 m cell visited at its corner must not be re-offered 1 m away."""
    c = _chooser([(0, 0), (101, 0), (109, 0), (400, 0)], cell_m=10.0)
    c.visit((101.0, 0.0))                      # cell (10, 0)
    g = c.next_goal((0.0, 0.0), min_m=10.0)
    assert (g.x, g.y) == (400.0, 0.0)          # 109 is the SAME cell as 101


def test_returns_none_only_for_a_map_with_nowhere_else_to_go():
    """None means "no destination exists", which is true only when every
    waypoint shares the car's own cell. A COVERED map is not that case -- see
    test_a_covered_map_starts_a_NEW_SWEEP_instead_of_giving_up."""
    c = _chooser([(0, 0), (1, 1), (2, 2)], cell_m=10.0)   # all one cell
    assert c.next_goal((0.0, 0.0), min_m=1.0) is None


def test_ignores_the_min_distance_rather_than_stalling():
    """Everything far is driven and only near ground is left: go there anyway,
    because returning None would end the run with the map half covered."""
    c = _chooser([(0, 0), (20, 0), (400, 0)], cell_m=10.0)
    c.visit((400.0, 0.0))
    g = c.next_goal((0.0, 0.0), min_m=100.0)   # nothing unvisited is that far
    assert g is not None and (g.x, g.y) == (20.0, 0.0)


def test_coverage_counts_only_reachable_cells():
    c = _chooser([(0, 0), (50, 0), (500, 0)], cell_m=10.0)
    assert c.coverage() == 0.0
    c.visit((0.0, 0.0))
    assert c.coverage() == pytest.approx(1 / 3)
    c.visit((9999.0, 9999.0))                  # off the road graph
    assert c.coverage() == pytest.approx(1 / 3)


def test_ties_are_broken_at_random_so_two_runs_differ():
    pts = [(0, 0)] + [(100, d) for d in (-40, 0, 40)]
    picks = {(_chooser(pts, seed=s).next_goal((0.0, 0.0), 10.0).y) for s in range(12)}
    assert len(picks) > 1


def test_a_covered_map_starts_a_NEW_SWEEP_instead_of_giving_up():
    """THE BUG THAT COST A RUN (2026-10-03). Town02 has only 98 cells of 15 m,
    so the expert covered it in ~1500 samples; next_goal then returned None,
    every episode ended `expert_stuck` after one step, and the collector span
    through 5433 junk episodes without ever filling its sample budget. A covered
    map means START AGAIN, not stop -- a second pass is still good data."""
    c = _chooser([(0, 0), (50, 0), (500, 0)], cell_m=10.0)
    for p in [(0.0, 0.0), (50.0, 0.0), (500.0, 0.0)]:
        c.visit(p)
    assert c.coverage() == pytest.approx(1.0)
    g = c.next_goal((0.0, 0.0), min_m=10.0)
    assert g is not None, "a covered map must re-sweep, not return None"
    assert (g.x, g.y) in {(50.0, 0.0), (500.0, 0.0)}
    assert c.sweeps == 1                      # the re-sweep is COUNTED
    assert c.coverage_total() == pytest.approx(1.0)   # the pass is not forgotten


def test_the_new_sweep_keeps_the_cell_under_the_car_out_of_play():
    c = _chooser([(0, 0), (50, 0)], cell_m=10.0)
    for p in [(0.0, 0.0), (50.0, 0.0)]:
        c.visit(p)
    g = c.next_goal((0.0, 0.0), min_m=5.0)
    assert (g.x, g.y) == (50.0, 0.0)          # never the cell it stands in
