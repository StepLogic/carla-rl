"""Explore-mode routes for the expert (imitation-first plan, Task 2).

At a fork the expert takes the exit with the smallest heading change; where no
exit is within 30 degrees of straight (a T-junction) it turns right -- a
positive yaw change in CARLA's left-handed frame (x forward, y right, yaw grows
clockwise seen from above). Random branches would give the clone conflicting
labels at the same view, and a mean-fitting actor averages left and right into
straight ahead, into the kerb.

The CARLA map is replaced by a fake waypoint graph with the interface the
helpers use: wp.next(d), wp.is_junction, wp.transform.location,
wp.transform.rotation.yaw.
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vae_nav.expert_route import explore_route, needs_replan  # noqa: E402

STEP = 2.0


class _Loc:
    def __init__(self, x, y):
        self.x, self.y, self.z = float(x), float(y), 0.0


class _Rot:
    def __init__(self, yaw):
        self.yaw = float(yaw)


class _Tf:
    def __init__(self, x, y, yaw):
        self.location = _Loc(x, y)
        self.rotation = _Rot(yaw)


class _Wp:
    def __init__(self, x, y, yaw, junction=False):
        self.transform = _Tf(x, y, yaw)
        self.is_junction = bool(junction)
        self.succ = []

    def next(self, d):
        assert abs(d - STEP) < 1e-9
        return list(self.succ)


def _road(x0, y0, yaw, n, junction=False):
    """n waypoints, STEP apart, heading `yaw` degrees (direction (cos, sin))."""
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    wps = [_Wp(x0 + i * STEP * c, y0 + i * STEP * s, yaw, junction) for i in range(n)]
    for a, b in zip(wps, wps[1:]):
        a.succ = [b]
    return wps


def _fork(yaw_a, yaw_b, stem_n=10, arm_n=40):
    """A stem heading 0 deg into a junction that splits into two arms. Each
    arm starts with 3 in-junction waypoints, then leaves the junction."""
    stem = _road(0.0, 0.0, 0.0, stem_n)
    end = stem[-1].transform.location
    arms = []
    for yaw in (yaw_a, yaw_b):
        inside = _road(end.x, end.y, yaw, 3, junction=True)
        last = inside[-1].transform.location
        out = _road(last.x, last.y, yaw, arm_n)
        inside[-1].succ = [out[0]]
        arms.append(inside + out)
    stem[-1].succ = [arms[0][0], arms[1][0]]
    return stem, arms


def test_straight_exit_preferred():
    stem, (straight, right) = _fork(0.0, 90.0)
    route = explore_route(stem[0], 60.0)
    assert route[-1] in straight and route[-1] not in right


def test_t_junction_turns_right():
    stem, (left, right) = _fork(-90.0, 90.0)
    route = explore_route(stem[0], 60.0)
    assert route[-1] in right


def test_left_only_fork_takes_the_smallest_turn():
    stem, (sharp, gentle) = _fork(-150.0, -90.0)
    route = explore_route(stem[0], 60.0)
    assert route[-1] in gentle


def test_route_length_is_respected():
    road = _road(0.0, 0.0, 0.0, 200)
    route = explore_route(road[0], 100.0)
    covered = STEP * (len(route) - 1)
    assert 100.0 <= covered <= 102.0
    assert route[0] is road[0]


def test_route_never_ends_inside_a_junction():
    # 20 m of road, then 10 junction waypoints, then road: a 24 m request
    # lands inside the junction and must walk out of it.
    a = _road(0.0, 0.0, 0.0, 11)
    j = _road(22.0, 0.0, 0.0, 10, junction=True)
    b = _road(42.0, 0.0, 0.0, 30)
    a[-1].succ = [j[0]]
    j[-1].succ = [b[0]]
    route = explore_route(a[0], 24.0)
    assert not route[-1].is_junction


def test_dead_end_returns_partial_route():
    road = _road(0.0, 0.0, 0.0, 11)          # 0..20 m, then nothing
    route = explore_route(road[0], 100.0)
    assert len(route) == 11 and route[-1] is road[-1]


def test_needs_replan():
    xy = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    assert not needs_replan(xy, (2.0, 2.9), 3.0)
    assert needs_replan(xy, (2.0, 3.1), 3.0)
    assert needs_replan(np.zeros((0, 2)), (0.0, 0.0), 3.0)
