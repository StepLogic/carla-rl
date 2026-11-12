"""Metres between failures from an explore trace (imitation-first plan, Task 8).

The success bar is >= 300 m between failures on Town02 and Town03, so this is
THE lane-keeping number. A failure is a crossing, off-road or collision end; a
segment that ends any other way (the cap, the budget) still counts its metres.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from trace_failure_stats import failure_stats  # noqa: E402


def test_mean_between_failures():
    s = [{"metres": 30, "end": "crossing"}, {"metres": 50, "end": "cap"},
         {"metres": 20, "end": "off_road"}]
    r = failure_stats(s)
    assert r["faults"] == 2 and r["mean_between"] == pytest.approx(50.0)
    assert r["metres"] == pytest.approx(100.0)


def test_no_faults_is_infinite():
    r = failure_stats([{"metres": 400, "end": "cap"}, {"metres": 250, "end": "budget"}])
    assert r["faults"] == 0 and r["mean_between"] == float("inf")


def test_runs_are_taken_over_fault_ended_segments_only():
    s = [{"metres": 10, "end": "crossing"}, {"metres": 900, "end": "budget"},
         {"metres": 40, "end": "collision"}, {"metres": 70, "end": "off_road"}]
    r = failure_stats(s)
    assert r["median_run"] == pytest.approx(40.0)
    assert r["longest"] == pytest.approx(70.0)


def test_speed_uses_the_env_step_of_0_1_s():
    """The env ticks at 0.1 s (train_env_config_v4_image_goal "timestep": 0.1;
    recorded expert data: 0.38 m/step at 13.8 km/h). The plan's 0.05 s doubled
    every speed: the 15 km/h expert would have read 30 km/h."""
    from trace_failure_stats import mean_speed_kmh
    assert mean_speed_kmh([{"metres": 41.67, "steps": 100}]) == pytest.approx(15.0, abs=0.01)


def test_matches_the_published_v30_town02_figure():
    with open(os.path.join(ROOT, "tests", "fixtures", "trace_v30det_Town02_index.json")) as fh:
        idx = json.load(fh)
    r = failure_stats(idx["segments"])
    assert r["faults"] == 134
    assert round(r["mean_between"], 1) == 31.7
