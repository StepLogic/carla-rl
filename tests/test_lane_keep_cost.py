"""LANE-CENTRE COST (user 2026-10-25, "Both modes"): a small per-step cost for
distance from the lane centre and for heading error, from CARLA's map, charged
in BOTH modes through the shared explore_reward. Crossings stay fined, not
terminal; this term only adds a gradient towards the centre that the boundary
fine alone never gave.
"""
import importlib
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vae_nav.lane_priv import PRIV_DIM, lane_keep_cost, lane_priv  # noqa: E402


def _wp(x=0.0, y=0.0, yaw_deg=0.0, width=3.5, junction=False):
    mark = SimpleNamespace(type="Broken")
    return SimpleNamespace(
        transform=SimpleNamespace(location=SimpleNamespace(x=x, y=y),
                                  rotation=SimpleNamespace(yaw=yaw_deg)),
        lane_width=width, is_junction=junction,
        left_lane_marking=mark, right_lane_marking=mark)


def _priv(lat=0.0, herr_deg=0.0, width=3.5, junction=False):
    # Lane runs along +x (yaw 0). CARLA is left-handed: LEFT is -y, so a
    # LEFT offset of `lat` metres puts the car at y = -lat.
    return lane_priv((0.0, -lat), math.radians(herr_deg), _wp(width=width,
                                                              junction=junction))


HALF_CAR = 0.74          # microlino, the hero blueprint


def test_centred_and_aligned_costs_nothing():
    assert lane_keep_cost(_priv(), 0.2, 0.2, HALF_CAR) == 0.0


def test_offset_cost_reaches_k_where_the_body_meets_the_line():
    # 3.5 m lane: half 1.75, room before the body touches the marking 1.01 m.
    room = 1.75 - HALF_CAR
    assert lane_keep_cost(_priv(lat=room), 0.2, 0.0, HALF_CAR) == pytest.approx(0.2)
    assert lane_keep_cost(_priv(lat=-room), 0.2, 0.0, HALF_CAR) == pytest.approx(0.2)


def test_offset_cost_is_quadratic_so_the_centre_is_nearly_free():
    room = 1.75 - HALF_CAR
    half = lane_keep_cost(_priv(lat=room / 2), 0.2, 0.0, HALF_CAR)
    assert half == pytest.approx(0.05)


def test_offset_cost_is_capped():
    far = lane_keep_cost(_priv(lat=9.0), 0.2, 0.0, HALF_CAR)
    assert far == pytest.approx(0.2 * 1.5 ** 2)


def test_heading_cost_reaches_k_at_45_degrees_and_caps_there():
    assert lane_keep_cost(_priv(herr_deg=45), 0.0, 0.2, HALF_CAR) == pytest.approx(0.2)
    assert lane_keep_cost(_priv(herr_deg=-45), 0.0, 0.2, HALF_CAR) == pytest.approx(0.2)
    assert lane_keep_cost(_priv(herr_deg=170), 0.0, 0.2, HALF_CAR) == pytest.approx(0.2)


def test_junctions_are_forgiven_like_the_crossing_rule():
    assert lane_keep_cost(_priv(lat=1.5, herr_deg=60, junction=True),
                          0.2, 0.2, HALF_CAR) == 0.0


def test_no_waypoint_or_no_block_costs_nothing():
    assert lane_keep_cost(np.zeros(PRIV_DIM, np.float32), 0.2, 0.2, HALF_CAR) == 0.0
    assert lane_keep_cost(None, 0.2, 0.2, HALF_CAR) == 0.0


def test_narrow_lane_keeps_a_minimum_room():
    # A 2 m lane leaves 0.26 m of room for a 0.74 m half-car; the floor (0.3 m)
    # stops the cost exploding to k at 26 cm off centre.
    c = lane_keep_cost(_priv(lat=0.3, width=2.0), 0.2, 0.0, HALF_CAR)
    assert c == pytest.approx(0.2)


def _reward(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    import vae_nav.gated_reward as gr
    return importlib.reload(gr)


def _info(lat=0.0, herr_deg=0.0, junction=False):
    return {"lane_priv": _priv(lat=lat, herr_deg=herr_deg, junction=junction),
            "hero_half_width": HALF_CAR, "wp_forward": np.array([1.0, 0.0]),
            "speed_kmh": 20.0, "speed_factor": 0.5,
            "action_smoothness_penalty": 0.0}


def test_explore_reward_charges_the_cost_when_on(monkeypatch):
    gr = _reward(monkeypatch, LANE_CENTRE_COST=0.2, LANE_HEADING_COST=0.2)
    disp = np.array([0.25, 0.0])
    r_mid, _ = gr.explore_reward(_info(), disp)
    r_off, _ = gr.explore_reward(_info(lat=1.75 - HALF_CAR), disp)
    assert r_mid - r_off == pytest.approx(0.2)


def test_explore_reward_is_unchanged_when_off(monkeypatch):
    gr = _reward(monkeypatch, LANE_CENTRE_COST=0.0, LANE_HEADING_COST=0.0)
    disp = np.array([0.25, 0.0])
    r_mid, _ = gr.explore_reward(_info(), disp)
    r_off, _ = gr.explore_reward(_info(lat=1.0, herr_deg=40), disp)
    assert r_mid == pytest.approx(r_off)


def test_cost_on_without_lane_block_is_a_loud_error(monkeypatch):
    gr = _reward(monkeypatch, LANE_CENTRE_COST=0.2, LANE_HEADING_COST=0.0)
    info = _info()
    del info["lane_priv"]
    with pytest.raises(KeyError):
        gr.explore_reward(info, np.array([0.25, 0.0]))


def test_core_state_publishes_the_block_when_the_cost_is_on(monkeypatch):
    monkeypatch.setenv("OBS_PRIV", "0")
    monkeypatch.setenv("LANE_CENTRE_COST", "0.2")
    import vae_nav.core_state as cs
    cs = importlib.reload(cs)
    assert cs.LANE_GEOM is True
    monkeypatch.setenv("LANE_CENTRE_COST", "0")
    monkeypatch.setenv("LANE_HEADING_COST", "0")
    cs = importlib.reload(cs)
    assert cs.LANE_GEOM is False


@pytest.fixture(autouse=True)
def _restore_modules(monkeypatch):
    yield
    for k in ("LANE_CENTRE_COST", "LANE_HEADING_COST", "OBS_PRIV"):
        monkeypatch.delenv(k, raising=False)
    import vae_nav.gated_reward as gr
    import vae_nav.core_state as cs
    importlib.reload(gr)
    importlib.reload(cs)
