"""Goal chaining and junction zones (user 2026-11-02: lane following everywhere,
the goal-conditioned reward at junctions, and new goals until 200 m driven).

The CARLA map is replaced by a tiny waypoint graph with the same interface the
helpers use: wp.next(d) -> list of successors, wp.is_junction, wp.transform.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vae_nav import goal_chain as gc  # noqa: E402


class _Loc:
    def __init__(self, x, y):
        self.x, self.y, self.z = float(x), float(y), 0.0


class _Tf:
    def __init__(self, x, y):
        self.location = _Loc(x, y)


class _Wp:
    """A road of 2 m waypoints. `succ` is filled after construction."""

    def __init__(self, x, y, junction=False, name=""):
        self.transform = _Tf(x, y)
        self.is_junction = junction
        self.succ = []
        self.name = name

    def next(self, d):
        # only 2 m steps are used by the helpers
        assert abs(d - gc.STEP_M) < 1e-9
        return list(self.succ)


def _road(xs, y=0.0, junction_at=()):
    wps = [_Wp(x, y, junction=(x in junction_at)) for x in xs]
    for a, b in zip(wps, wps[1:]):
        a.succ = [b]
    return wps


class _Map:
    def __init__(self, wps):
        self.wps = wps

    def get_waypoint(self, loc, **kw):
        return min(self.wps, key=lambda w: np.hypot(w.transform.location.x - loc.x,
                                                     w.transform.location.y - loc.y))


def _fork():
    """Straight road 0..20 m, a junction 22..26 m, then two branches:
    east (y = 0) and north (x = 26, y up)."""
    stem = _road(np.arange(0, 22, 2.0))
    junc = _road([22.0, 24.0, 26.0], junction_at=(22.0, 24.0, 26.0))
    east = _road(np.arange(28, 90, 2.0))
    north = [_Wp(26.0, y) for y in np.arange(2, 64, 2.0)]
    for a, b in zip(north, north[1:]):
        a.succ = [b]
    stem[-1].succ = [junc[0]]
    junc[-1].succ = [east[0], north[0]]
    return _Map(stem + junc + east + north), stem, east, north


def test_junction_ahead_reports_the_distance_along_the_lane():
    m, stem, _, _ = _fork()
    assert gc.junction_ahead_m(m, _Loc(10, 0), lookahead_m=20) == pytest.approx(12.0)
    assert gc.junction_ahead_m(m, _Loc(0, 0), lookahead_m=10) is None


def test_inside_a_junction_the_distance_is_zero():
    m, *_ = _fork()
    assert gc.junction_ahead_m(m, _Loc(24, 0), lookahead_m=10) == 0.0


def test_new_goal_lands_past_the_next_junction_in_the_band():
    m, stem, east, north = _fork()
    rng = np.random.default_rng(0)
    for _ in range(20):
        wp, st = gc.goal_past_next_junction(m, _Loc(4, 0), rng, min_m=10, max_m=30)
        assert st["through_junction"]
        assert not wp.is_junction
        assert 10.0 - 1e-9 <= st["past_junction_m"] <= 30.0 + 2.0
        assert wp in east or wp in north


def test_both_branches_are_drawn():
    m, stem, east, north = _fork()
    rng = np.random.default_rng(1)
    picks = {gc.goal_past_next_junction(m, _Loc(4, 0), rng, 10, 30)[0] in north
             for _ in range(40)}
    assert picks == {True, False}


def test_without_a_junction_ahead_the_goal_is_drawn_along_the_road():
    road = _road(np.arange(0, 400, 2.0))
    m = _Map(road)
    wp, st = gc.goal_past_next_junction(m, _Loc(0, 0), np.random.default_rng(0), 10, 30,
                                        lookahead_m=50)
    assert not st["through_junction"]
    assert 10.0 <= wp.transform.location.x <= 32.0


def test_a_dead_end_returns_none():
    road = _road([0.0, 2.0, 4.0])
    m = _Map(road)
    assert gc.goal_past_next_junction(m, _Loc(0, 0), np.random.default_rng(0), 10, 30)[0] is None


def test_zone_tracker_covers_before_inside_and_after():
    z = gc.JunctionZone(zone_m=10.0)
    assert not z.update(path_m=0.0, in_junction=False, ahead_m=None)
    assert z.update(path_m=5.0, in_junction=False, ahead_m=8.0)       # before
    assert z.update(path_m=15.0, in_junction=True, ahead_m=0.0)       # inside
    assert z.update(path_m=24.0, in_junction=False, ahead_m=None)     # 9 m after
    assert not z.update(path_m=26.0, in_junction=False, ahead_m=None)  # 11 m after
    z.reset()
    assert not z.update(path_m=0.0, in_junction=False, ahead_m=None)
