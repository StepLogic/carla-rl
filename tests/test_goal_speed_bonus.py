"""GOAL_SPEED_BONUS: the v10 target-velocity term in goal_reward_sorb.

On a CLEAN step (on-road, no lane crossing, no collision this step) the
counting branch pays ``+GOAL_SPEED_BONUS * max(0, 1 - |v - v*| / v*)``
on top of the ordinary step cost and fines; a fault step pays its fines
and NO bonus; a missing or non-positive target disables the term; the
default weight 0.0 leaves every pre-v10 run byte-identical.

Constants are monkeypatched on the module (single source of truth), as in
test_episode_her_recost.py -- environment writes here would be silently
dead once gated_reward has been imported by another test module.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from vae_nav import gated_reward as gr  # noqa: E402

STEP_COST = 1.0
BONUS = 10.0
W = 0.5


@pytest.fixture
def w05(monkeypatch):
    """Counting regime with the velocity bonus at weight 0.5."""
    monkeypatch.setattr(gr, "GOAL_TRUNCATE_COUNTS", True)
    monkeypatch.setattr(gr, "STEP_COST", STEP_COST)
    monkeypatch.setattr(gr, "GOAL_ARRIVAL_BONUS", BONUS)
    monkeypatch.setattr(gr, "GOAL_OFF_ROAD_PENALTY", 100.0)
    monkeypatch.setattr(gr, "GOAL_LANE_PENALTY", 100.0)
    monkeypatch.setattr(gr, "GOAL_COLLISION_PENALTY", 100.0)
    monkeypatch.setattr(gr, "GOAL_OFFROAD_TRUNCATE_STEPS", 20)
    monkeypatch.setattr(gr, "GOAL_COLLISION_TRUNCATE_STEPS", 20)
    monkeypatch.setattr(gr, "GOAL_CROSSING_TRUNCATE_STEPS", 20)
    monkeypatch.setattr(gr, "GOAL_MAX_STEPS", 1000)
    monkeypatch.setattr(gr, "GOAL_SPEED_BONUS", W)


def call(info, step=5, max_steps=1000):
    return gr.goal_reward_sorb(
        info, step, max_steps=max_steps, step_cost=STEP_COST,
        offroad_steps=0, collision_steps=0, crossing_steps=0)


def _clean(v, vt=20.0):
    return {"off_road": False, "lane_any": False, "collision": False,
            "speed_kmh": v, "target_speed_kmh": vt}


def test_bonus_peaks_at_target(w05):
    r, terminated, truncated, _ = call(_clean(20.0))
    assert r == pytest.approx(-STEP_COST + W)
    assert not terminated and not truncated


def test_bonus_tapers_linearly(w05):
    r, *_ = call(_clean(10.0))          # |10-20|/20 = 0.5 -> half the peak
    assert r == pytest.approx(-STEP_COST + W / 2)
    r, *_ = call(_clean(5.0))           # 0.25 of the peak
    assert r == pytest.approx(-STEP_COST + W / 4)


def test_bonus_floors_at_zero(w05):
    """Standing still and absurd overspeed both pay NO bonus, never negative."""
    r, *_ = call(_clean(0.0))           # |0-20|/20 = 1.0 -> taper reaches 0
    assert r == pytest.approx(-STEP_COST)
    r, *_ = call(_clean(80.0))          # 3x target -> clamped at 0
    assert r == pytest.approx(-STEP_COST)
    r, *_ = call(_clean(200.0))
    assert r == pytest.approx(-STEP_COST)


def test_bonus_blocked_on_fault_steps(w05):
    """A fault step pays its fine and nothing else, whatever the speed."""
    for key in ("off_road", "lane_any", "collision"):
        info = _clean(20.0)
        info[key] = True
        r, *_ = call(info)
        assert r == pytest.approx(-STEP_COST - 100.0), key


def test_bonus_absent_without_a_target(w05):
    """A missing or non-positive target disables the term."""
    info = _clean(20.0)
    del info["target_speed_kmh"]
    r, *_ = call(info)
    assert r == pytest.approx(-STEP_COST)
    r, *_ = call(_clean(20.0, vt=0.0))
    assert r == pytest.approx(-STEP_COST)


def test_bonus_absent_without_a_speed(w05):
    """A missing speed observation counts as v=0: no bonus, still on-road."""
    info = _clean(0.0)
    del info["speed_kmh"]
    r, *_ = call(info)
    assert r == pytest.approx(-STEP_COST)


def test_weight_zero_is_byte_identical(monkeypatch):
    """GOAL_SPEED_BONUS=0 (every pre-v10 run) pays exactly the old reward."""
    monkeypatch.setattr(gr, "GOAL_TRUNCATE_COUNTS", True)
    monkeypatch.setattr(gr, "STEP_COST", STEP_COST)
    monkeypatch.setattr(gr, "GOAL_ARRIVAL_BONUS", BONUS)
    monkeypatch.setattr(gr, "GOAL_OFF_ROAD_PENALTY", 100.0)
    monkeypatch.setattr(gr, "GOAL_LANE_PENALTY", 100.0)
    monkeypatch.setattr(gr, "GOAL_COLLISION_PENALTY", 100.0)
    monkeypatch.setattr(gr, "GOAL_OFFROAD_TRUNCATE_STEPS", 20)
    monkeypatch.setattr(gr, "GOAL_COLLISION_TRUNCATE_STEPS", 20)
    monkeypatch.setattr(gr, "GOAL_CROSSING_TRUNCATE_STEPS", 20)
    monkeypatch.setattr(gr, "GOAL_MAX_STEPS", 1000)
    monkeypatch.setattr(gr, "GOAL_SPEED_BONUS", 0.0)
    r, *_ = call(_clean(20.0))
    assert r == pytest.approx(-STEP_COST)
    # Arrival pays only the arrival bonus, with or without the term.
    info = _clean(20.0)
    info["arrived"] = True
    r, terminated, truncated, _ = call(info)
    assert r == pytest.approx(BONUS)
    assert terminated and not truncated