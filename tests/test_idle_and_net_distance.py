"""Two reward changes after v8 (2026-10-19), both off by default (v52).

v8 peaked at goal-arm 0.62 (~400k) then parked: from episode 1920 the car sat
motionless in 92-100% of frames, steer locked at +0.9, because a parked step
cost only -0.03 to -0.15 -- worth about -10 forever at gamma 0.99, the price of
one lane crossing. And explore "reached" counted cumulative path, which a car
looping one junction also accumulates (v8 episode 1360).
"""
import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

FWD = np.array([0.3, 0.0])


def _mod(**env):
    for k, v in env.items():
        os.environ[k] = str(v)
    import vae_nav.gated_reward as g
    importlib.reload(g)
    g.PROGRESS_MODE, g.PROGRESS_SCALE = "heading", 1.0
    return g


@pytest.fixture(autouse=True)
def _restore():
    yield
    for k in ("EXPLORE_IDLE_PENALTY", "EXPLORE_IDLE_SPEED_KMH", "EXPLORE_DIST_MODE"):
        os.environ.pop(k, None)
    importlib.reload(importlib.import_module("vae_nav.gated_reward"))


def _info(speed):
    return {"hero_yaw": 0.0, "speed_factor": 0.5, "action_smoothness_penalty": 0.0,
            "collision": False, "off_road": False, "lane_any": False,
            "lane_marking_types": [], "in_junction": False, "speed_kmh": speed}


def test_both_changes_are_off_by_default():
    g = _mod()
    assert g.EXPLORE_IDLE_PENALTY == 0.0
    assert g.EXPLORE_DIST_MODE == "path"


def test_standing_still_is_charged():
    g = _mod(EXPLORE_IDLE_PENALTY=0.5, EXPLORE_IDLE_SPEED_KMH=2)
    still, _ = g.explore_reward(_info(0.5), FWD * 0)
    off = _mod(EXPLORE_IDLE_PENALTY=0)
    base, _ = off.explore_reward(_info(0.5), FWD * 0)
    assert still == pytest.approx(base - 0.5)


def test_moving_is_not_charged():
    g = _mod(EXPLORE_IDLE_PENALTY=0.5, EXPLORE_IDLE_SPEED_KMH=2)
    moving, _ = g.explore_reward(_info(15.0), FWD)
    off = _mod(EXPLORE_IDLE_PENALTY=0)
    base, _ = off.explore_reward(_info(15.0), FWD)
    assert moving == pytest.approx(base)


def test_the_idle_threshold_sits_below_the_arrival_speed():
    """Slowing to arrive must never be charged as idling."""
    os.environ["GOAL_ARRIVAL_MAX_SPEED"] = "10"
    try:
        g = _mod(EXPLORE_IDLE_PENALTY=0.5, EXPLORE_IDLE_SPEED_KMH=2)
        assert g.EXPLORE_IDLE_SPEED_KMH < g.GOAL_ARRIVAL_MAX_SPEED
    finally:
        os.environ.pop("GOAL_ARRIVAL_MAX_SPEED", None)


def test_parking_forever_now_costs_more_than_several_crossings():
    """The arithmetic the change exists for, at gamma 0.99 and a fine of 3."""
    g = _mod(EXPLORE_IDLE_PENALTY=0.5)
    parked_step, _ = g.explore_reward(_info(0.0), FWD * 0)
    assert parked_step / (1 - 0.99) < -10 * 3, parked_step


class _W:
    """Just enough of GatedReward to exercise _explore_dist."""


def _dist(g, path_m, start, now):
    w = _W()
    w._path_m = path_m
    w._start = np.asarray(start, float)
    w._prev = np.asarray(now, float)
    return g.GatedReward._explore_dist(w)


def test_net_distance_ignores_looping():
    g = _mod(EXPLORE_DIST_MODE="net")
    # 150 m of loops ending 3 m from where it began
    assert _dist(g, 150.0, (0, 0), (3, 0)) == pytest.approx(3.0)


def test_path_distance_is_unchanged_by_default():
    g = _mod()
    assert _dist(g, 150.0, (0, 0), (3, 0)) == pytest.approx(150.0)


def test_bad_mode_is_refused():
    with pytest.raises(ValueError):
        _mod(EXPLORE_DIST_MODE="crow")
