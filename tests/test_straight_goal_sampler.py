"""Junction-to-junction STRAIGHT-segment origin sampling (user 2026-09-16).

The existing sampler has two branches: with probability JUNCTION_GOAL_FRAC the origin is placed one goal-distance
BEFORE a junction so the route drives through it, otherwise a uniformly random waypoint. Neither asks for a straight
run of road anchored between two junctions, which is the case the user wants represented.

WHY A LENGTH LIMIT COMES BACK WITH THE ORIGIN. GOAL_NOT_IN_JUNCTION only rejects a goal that lands INSIDE a junction;
a goal placed BEYOND the next junction is on open road and is accepted, so the route would drive through the junction
anyway and the straight branch would be decorative. straight_segment_origin therefore returns how much straight road
lies ahead, and the caller filters the goal-distance draw to that.

These are pure functions over anything with previous()/is_junction, so the walk-back is tested without CARLA.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from rlib_integration.carla_core import pick_origin, straight_segment_origin  # noqa: E402


class FakeWP:
    """A waypoint on a linear chain at fixed spacing. previous(d) steps back d/spacing links."""

    def __init__(self, chain, idx, spacing=2.0):
        self.chain, self.idx, self.spacing = chain, idx, spacing

    @property
    def is_junction(self):
        return self.chain.junctions[self.idx]

    def previous(self, d):
        back = self.idx - int(round(d / self.spacing))
        return [FakeWP(self.chain, back, self.spacing)] if back >= 0 else []

    def next(self, d):
        fwd = self.idx + int(round(d / self.spacing))
        return [FakeWP(self.chain, fwd, self.spacing)] if fwd < len(self.chain.junctions) else []

    def __eq__(self, other):
        return isinstance(other, FakeWP) and other.idx == self.idx

    def __repr__(self):
        return f"WP({self.idx}{'J' if self.is_junction else ''})"


class Chain:
    """junctions: list of bools, one per 2 m link."""

    def __init__(self, junctions):
        self.junctions = junctions

    def wp(self, idx):
        return FakeWP(self, idx)


def _road_with_junctions_at(n, idxs):
    return Chain([i in idxs for i in range(n)])


def test_walks_back_to_the_preceding_junction_and_reports_the_gap():
    # junctions at link 10 and link 60; 2 m spacing -> 100 m of straight road between them
    ch = _road_with_junctions_at(100, {10, 60})
    origin, straight_m = straight_segment_origin(ch.wp(60), step_m=2.0, max_m=200.0)
    assert origin is not None
    assert origin.idx == 11                      # the first non-junction after the preceding junction
    assert straight_m == pytest.approx(98.0)     # 49 links x 2 m back to link 11


def test_returns_nothing_when_junctions_are_adjacent():
    ch = _road_with_junctions_at(20, {9, 10})
    origin, straight_m = straight_segment_origin(ch.wp(10), step_m=2.0, max_m=200.0)
    assert origin is None
    assert straight_m == 0.0


def test_stops_at_max_m_on_an_endless_road():
    ch = _road_with_junctions_at(500, {400})
    origin, straight_m = straight_segment_origin(ch.wp(400), step_m=2.0, max_m=40.0)
    assert straight_m <= 40.0
    assert origin is not None


def test_stops_at_the_end_of_the_chain():
    ch = _road_with_junctions_at(30, {25})
    origin, straight_m = straight_segment_origin(ch.wp(25), step_m=2.0, max_m=200.0)
    assert origin is not None
    assert origin.idx == 0                       # ran out of road rather than finding a junction


class _Rng:
    def __init__(self, roll, picks=None):
        self._roll, self._picks, self._i = roll, picks or [], 0

    def random(self):
        return self._roll

    def choice(self, seq):
        if self._picks:
            v = self._picks[self._i % len(self._picks)]
            self._i += 1
            return v
        return seq[0]


def test_straight_branch_returns_a_forward_limit():
    ch = _road_with_junctions_at(100, {10, 60})
    origin, max_fwd, kind = pick_origin(
        waypoints=[ch.wp(0)], junctions=[ch.wp(60)], lenghts=[20.0, 50.0],
        jfrac=0.0, straight_frac=1.0, rng=_Rng(0.0), step_m=2.0)
    assert kind == "straight"
    assert max_fwd == pytest.approx(98.0)
    assert origin.idx == 11


def test_straight_frac_zero_preserves_todays_junction_behaviour():
    ch = _road_with_junctions_at(100, {60})
    origin, max_fwd, kind = pick_origin(
        waypoints=[ch.wp(0)], junctions=[ch.wp(60)], lenghts=[20.0],
        jfrac=1.0, straight_frac=0.0, rng=_Rng(0.5), step_m=2.0)
    assert kind == "junction"
    assert max_fwd is None                        # unconstrained, exactly as before
    assert origin.idx == 50                       # 20 m back from the junction


def test_falls_back_to_uniform_when_there_are_no_junctions():
    ch = _road_with_junctions_at(20, set())
    origin, max_fwd, kind = pick_origin(
        waypoints=[ch.wp(3)], junctions=[], lenghts=[20.0],
        jfrac=0.7, straight_frac=0.2, rng=_Rng(0.0), step_m=2.0)
    assert kind == "uniform"
    assert max_fwd is None
    assert origin.idx == 3


def test_the_two_fractions_partition_the_roll():
    ch = _road_with_junctions_at(100, {10, 60})
    common = dict(waypoints=[ch.wp(0)], junctions=[ch.wp(60)], lenghts=[20.0], step_m=2.0)
    # straight_frac=0.2, jfrac=0.5 -> [0,0.2) straight, [0.2,0.7) junction, [0.7,1) uniform
    assert pick_origin(jfrac=0.5, straight_frac=0.2, rng=_Rng(0.1), **common)[2] == "straight"
    assert pick_origin(jfrac=0.5, straight_frac=0.2, rng=_Rng(0.5), **common)[2] == "junction"
    assert pick_origin(jfrac=0.5, straight_frac=0.2, rng=_Rng(0.9), **common)[2] == "uniform"
