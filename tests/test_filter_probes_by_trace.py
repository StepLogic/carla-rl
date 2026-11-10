"""Multi-turn validation episodes must start and end on lanes the expert tour drove
(plan 2026-11-09, A2). Town05's tour left out 32 lanes (6 km) that cannot be
reached both ways without a lane change; an episode on them would fall back to
driving without a map for reasons unrelated to the driver.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from filter_probes_by_trace import on_trace  # noqa: E402


def test_start_and_goal_must_both_lie_near_the_trace():
    trace = np.array([[x, 0.0] for x in np.arange(0, 100, 0.5)])
    probes = [{"seed": 1, "start_xy": [10, 1], "goal_xy": [90, -2]},     # both on
              {"seed": 2, "start_xy": [10, 1], "goal_xy": [50, 30]},     # goal off
              {"seed": 3, "start_xy": [150, 0], "goal_xy": [20, 0]},     # start off
              {"seed": 4, "start_xy": None, "goal_xy": [20, 0]}]         # unknown
    kept = on_trace(probes, trace, radius_m=5.0)
    assert [p["seed"] for p in kept] == [1]
