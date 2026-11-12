"""Which events end a budget-mode collection episode.

USER 2026-09-13, Town02 recollection: "reset in nearest waypoint if lane
invasion happens / goes offroad". The earlier rule ended only on collision.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_default_is_collision_only():
    import eval_placement as ep
    assert ep.parse_end_on("") == frozenset({"collision"})
    end = ep.parse_end_on("collision")
    assert ep.collect_should_end({"collision": True}, True, False, end) == "collision"
    assert ep.collect_should_end({"explore_end": "crossing"}, True, False, end) is None
    assert ep.collect_should_end({"off_road": True}, False, False, end) is None


def test_lane_and_off_road_end_when_enabled():
    import eval_placement as ep
    end = ep.parse_end_on("collision,lane,off_road")
    assert ep.collect_should_end({"explore_end": "crossing"}, True, False, end) == "lane"
    # off-road ends even though the v6n reward does not terminate on it
    assert ep.collect_should_end({"off_road": True}, False, False, end) == "off_road"


def test_arrival_and_truncation_are_driven_through():
    import eval_placement as ep
    end = ep.parse_end_on("collision,lane,off_road")
    assert ep.collect_should_end({"explore_end": "reached"}, True, False, end) is None
    assert ep.collect_should_end({}, False, True, end) is None
    # a painted crossing that did NOT terminate (junction-exempt) is not an end
    assert ep.collect_should_end({"lane_invasion": True}, False, False, end) is None


def test_rejects_unknown_cause():
    import eval_placement as ep
    with pytest.raises(ValueError):
        ep.parse_end_on("collision,wrong_dir")
