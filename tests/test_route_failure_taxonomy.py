"""A4 (plan 2026-11-09): why does a long route fail? Each failed episode keeps its end
reason (crossing / off_road / collision / clock) and gets three flags from its
per-step trajectory (EVAL_TRAJ=1): WRONG BRANCH (left the planner's road route by
> 5 m for >= 20 steps after a junction), STALL (< 2 km/h for >= 50 steps), and, in
map arms, PASSED A SUBGOAL without switching (came within 3 m of the current
subgoal's node, no switch within 20 steps). Lane faults are split by place: in or
within 15 m after a junction, or elsewhere. Synthetic trajectories, no CARLA.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

import route_failure_taxonomy as rft  # noqa: E402


def test_distance_to_a_polyline():
    L = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    P = np.array([[5.0, 2.0], [12.0, 5.0], [-3.0, 0.0]])
    np.testing.assert_allclose(rft.dist_to_polyline(P, L), [2.0, 2.0, 3.0])


def test_a_stall_needs_a_long_enough_slow_run():
    v = np.r_[np.full(10, 10.0), np.full(60, 1.0), np.full(10, 10.0)]
    assert rft.stalled(v, below_kmh=2.0, min_steps=50)
    assert not rft.stalled(np.r_[np.full(40, 1.0), [10.0], np.full(40, 1.0)], 2.0, 50)


def _route():
    # east along y=0 to x=50, then a left turn north along x=50
    return np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 60.0]])


def test_a_wrong_branch_is_a_lasting_departure_after_a_junction():
    route = _route()
    east = [[x, 0.0] for x in np.arange(0, 50, 1.0)]
    straight_on = [[x, 0.0] for x in np.arange(50, 90, 1.0)]       # did not turn
    P = np.array(east + straight_on)
    junction = np.zeros(len(P), bool)
    junction[45:55] = True
    assert rft.wrong_branch(P, route, junction, off_m=5.0, persist=20)
    turned = [[50.0, y] for y in np.arange(0, 40, 1.0)]
    P2 = np.array(east + turned)
    assert not rft.wrong_branch(P2, route, junction[:len(P2)], off_m=5.0, persist=20)


def test_a_departure_far_from_any_junction_is_not_a_wrong_branch():
    route = np.array([[0.0, 0.0], [200.0, 0.0]])
    P = np.array([[x, 0.0] for x in range(100)] + [[100.0 + x, 8.0] for x in range(30)])
    assert not rft.wrong_branch(P, route, np.zeros(len(P), bool), off_m=5.0, persist=20)


def test_passing_a_subgoal_without_switching():
    node_xy = {7: (10.0, 0.0), 8: (30.0, 0.0)}
    P = np.array([[x, 0.0] for x in np.arange(0, 40, 1.0)])
    cur = np.full(len(P), 7)                      # never switched to node 8
    assert rft.passed_subgoal(P, cur, node_xy, near_m=3.0, window=20)
    cur2 = np.where(P[:, 0] > 11, 8, 7)           # switched right after passing
    assert not rft.passed_subgoal(P, cur2, node_xy, near_m=3.0, window=20)


def test_lane_fault_place_uses_the_last_steps_junction_flags():
    j = np.zeros(100, bool)
    j[80:90] = True
    P = np.array([[float(x), 0.0] for x in range(100)])
    assert rft.fault_place(P, j, after_m=15.0) == "junction"          # ended 10 m after
    j2 = np.zeros(100, bool)
    j2[10:20] = True
    assert rft.fault_place(P, j2, after_m=15.0) == "elsewhere"


def test_classify_reached_episodes_are_not_failures():
    assert rft.classify({"end": "reached"}, None) is None
