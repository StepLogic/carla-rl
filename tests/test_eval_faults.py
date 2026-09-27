"""eval_faults: per-episode driving-rule accounting for offline evals (2026-09-27)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from vae_nav.eval_faults import EpisodeFaults, aggregate, classify_crossing  # noqa: E402

M = "carla.LaneMarkingType."


@pytest.mark.parametrize("info,kind", [
    ({"lane_any": False}, None),
    ({"lane_any": True, "lane_marking_types": [M + "Broken"], "in_junction": True}, "junction"),
    ({"lane_any": True, "lane_marking_types": [M + "NONE"]}, "edge"),
    ({"lane_any": True, "lane_marking_types": [M + "Broken"]}, "lane_change"),
    ({"lane_any": True, "lane_marking_types": [M + "Solid"]}, "illegal"),
    ({"lane_any": True, "lane_marking_types": [M + "SolidSolid"], "in_junction": True}, "illegal"),
    ({"lane_any": True, "lane_marking_types": []}, "illegal"),     # no data: fails closed
])
def test_classify(info, kind):
    assert classify_crossing(info) == kind


def test_clean_and_dirty_arrivals():
    clean = EpisodeFaults(start_pos=(0, 0))
    for x in range(1, 11):
        clean.step({"hero_pos": (float(x), 0.0)})
    clean.step({"hero_pos": (11.0, 0.0), "lane_any": True,
                "lane_marking_types": [M + "Broken"], "in_junction": True})
    s = clean.summary(arrived=True)
    assert s["clean_arrival"] and s["strict_clean_arrival"]
    assert s["path_m"] == pytest.approx(11.0)

    dirty = EpisodeFaults(start_pos=(0, 0))
    dirty.step({"hero_pos": (1.0, 0.0), "off_road": True})
    d = dirty.summary(arrived=True)
    assert not d["clean_arrival"]                    # arrived, but went off-road

    lc = EpisodeFaults()
    lc.step({"lane_any": True, "lane_marking_types": [M + "Broken"]})
    l = lc.summary(arrived=True)
    assert l["clean_arrival"] and not l["strict_clean_arrival"]


def test_aggregate():
    a = EpisodeFaults(); a.step({"off_road": True})
    b = EpisodeFaults(); b.step({"lane_any": True, "lane_marking_types": [M + "Solid"]})
    c = EpisodeFaults(); c.step({})
    agg = aggregate([a.summary(True), b.summary(True), c.summary(False)])
    assert agg["n"] == 3
    assert agg["clean_arrival_rate"] == pytest.approx(0.0)     # a off-road, b illegal, c no arrival
    assert agg["frac_any_off_road"] == pytest.approx(0.333)
    assert agg["frac_any_illegal_crossing"] == pytest.approx(0.333)
