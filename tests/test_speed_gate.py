"""The progress term's speed gate, and turning it off.

User 2026-10-14: "remove speed gating". The progress term was paid only below
the target speed, so a car at or above it earned NOTHING for driving forward.
That is also why emptying the state block hurt so much: speed_factor decides
the reward and the agent could no longer observe it -- v52's state block carried
clip(speed_kmh / TARGET_SPEED, 0, 1) precisely because, as run_gated.sh puts it,
"BOTH gated rewards are speed-gated".

The default stays ON so v52 (SR 0.48) remains reproducible; the launcher turns
it off, which puts the deviation in the run's environment rather than in a
default nobody reads.
"""
import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

FWD = np.array([0.5, 0.0])          # half a metre straight ahead


def _mod(gate):
    """Re-import the reward with the flag set: it is read at import time."""
    os.environ["PROGRESS_SPEED_GATE"] = "1" if gate else "0"
    import vae_nav.gated_reward as g
    importlib.reload(g)
    g.PROGRESS_MODE, g.PROGRESS_SCALE = "heading", 1.0
    return g


def _info(speed_factor):
    return {"hero_yaw": 0.0, "speed_factor": speed_factor,
            "action_smoothness_penalty": 0.0, "collision": False,
            "off_road": False, "lane_any": False, "lane_marking_types": [],
            "in_junction": False}


@pytest.fixture(autouse=True)
def _restore():
    yield
    os.environ.pop("PROGRESS_SPEED_GATE", None)
    importlib.reload(importlib.import_module("vae_nav.gated_reward"))


def test_with_the_gate_ON_speed_above_target_pays_nothing():
    """v52's behaviour, and the trap: at or above target speed, driving forward
    earns nothing at all."""
    g = _mod(True)
    slow, _ = g.explore_reward(_info(0.5), FWD)
    fast, _ = g.explore_reward(_info(1.0), FWD)
    assert slow > 0.4, slow
    assert fast < 0.0, f"expected only the base penalty, got {fast}"


def test_with_the_gate_OFF_progress_pays_at_any_speed():
    g = _mod(False)
    slow, _ = g.explore_reward(_info(0.5), FWD)
    fast, _ = g.explore_reward(_info(1.5), FWD)
    assert slow == pytest.approx(fast), (slow, fast)
    assert fast > 0.4


def test_the_gate_defaults_to_v52s_behaviour():
    """v52 is the recorded run that reaches SR 0.48; its reward must stay
    reproducible without anyone setting anything."""
    os.environ.pop("PROGRESS_SPEED_GATE", None)
    import vae_nav.gated_reward as g
    importlib.reload(g)
    assert g.PROGRESS_SPEED_GATE is True
    assert g._pays_progress(0.5) and not g._pays_progress(1.0)


def test_turning_it_off_changes_nothing_but_the_progress_term():
    """One knob, not a reward rewrite: with no displacement there is no progress
    either way, so the two must agree exactly."""
    on = _mod(True)
    hit = dict(_info(1.0), collision=True)
    r_on, fatal_on = on.explore_reward(hit, FWD * 0)
    off = _mod(False)
    r_off, fatal_off = off.explore_reward(hit, FWD * 0)
    assert r_on == pytest.approx(r_off) and fatal_on == fatal_off


def test_reversing_still_pays_negatively_with_the_gate_off():
    """Progress is measured along the heading, so the reward still prices
    reverse -- which matters now that negative throttle IS reverse."""
    g = _mod(False)
    fwd, _ = g.explore_reward(_info(0.8), FWD)
    rev, _ = g.explore_reward(_info(0.8), -FWD)
    assert fwd > 0 > rev
