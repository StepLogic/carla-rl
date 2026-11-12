"""The evaluation fault rule (USER RULE: eval always fault-ends).

A collision, going off-road, or a painted lane crossing on open road ends an
eval episode. A broken line crossed inside a junction (the turn) and an unpainted
road edge do not; solid lines, curbs and grass are faults everywhere:
that is the same reading of a crossing as the training reward, so "crossing"
means one thing in both places.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

tcs = pytest.importorskip("train_carla_sac")


def _info(**kw):
    d = {"collision": False, "off_road": False, "lane_any": False,
         "lane_marking_types": [], "in_junction": False}
    d.update(kw)
    return d


def test_a_clean_step_is_not_a_fault():
    assert tcs._fault(_info()) is None


def test_collision_and_off_road_are_faults():
    assert tcs._fault(_info(collision=True)) == "collision"
    assert tcs._fault(_info(off_road=True)) == "off_road"


def test_a_painted_crossing_on_open_road_is_a_fault():
    assert tcs._fault(_info(lane_any=True, lane_marking_types=["Solid"])) == "crossing"


def test_a_broken_line_crossed_in_a_junction_is_not_a_fault():
    """That is the turn. (The training rule, lane_departure_is_fatal.)"""
    assert tcs._fault(_info(lane_any=True, lane_marking_types=["Broken"],
                            in_junction=True)) is None


def test_a_solid_line_is_a_fault_even_in_a_junction():
    """Solid/SolidSolid/SolidBroken/Curb/Grass are ILLEGAL_MARKS: never forgiven."""
    assert tcs._fault(_info(lane_any=True, lane_marking_types=["Solid"],
                            in_junction=True)) == "crossing"


def test_a_broken_line_on_open_road_is_a_fault():
    assert tcs._fault(_info(lane_any=True, lane_marking_types=["Broken"])) == "crossing"


def test_an_unpainted_edge_is_not_a_fault():
    assert tcs._fault(_info(lane_any=True, lane_marking_types=["NONE"])) is None
