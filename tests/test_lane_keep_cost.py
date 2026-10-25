"""LANE-CENTRE SHAPING (user 2026-10-25, "Both modes", then "Potential
shaping"): c(s) from the lane block (distance from the lane centre and heading
error, CARLA's map) is a POTENTIAL; every step in both modes pays
c(s) - gamma c(s'), c = 0 at a terminal state. That points at the centre but
telescopes, so no early end can ever be made cheaper -- the cheap-crash risk a
per-step cost carried. Crossings stay fined, not terminal.
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


# ---- the shaping in GatedReward ------------------------------------------
pytest.importorskip("gymnasium")
import gymnasium as gym                       # noqa: E402
from gymnasium import spaces                  # noqa: E402


class _LaneEnv(gym.Env):
    """Scripted lane offsets: step t reports offset lats[t] (reset: lats0)."""
    metadata = {}

    def __init__(self, lat0, lats, crash_at=None):
        self.observation_space = spaces.Dict({})
        self.action_space = spaces.Box(np.array([-1.0, 0.0], np.float32),
                                       np.array([1.0, 1.0], np.float32))
        self.lat0, self.lats, self.crash_at = lat0, list(lats), crash_at
        self.t = 0

    def _info(self, lat, x):
        return {"lane_priv": _priv(lat=lat), "hero_half_width": HALF_CAR,
                "hero_pos": (x, 0.0), "hero_yaw": 0.0,
                "wp_forward": np.array([1.0, 0.0]), "speed_kmh": 20.0,
                "speed_factor": 0.5, "action_smoothness_penalty": 0.0}

    def reset(self, **kw):
        self.t = 0
        return np.zeros(4, np.float32), self._info(self.lat0, 0.0)

    def step(self, action):
        lat = self.lats[self.t]
        info = self._info(lat, 0.25 * (self.t + 1))
        if self.crash_at is not None and self.t == self.crash_at:
            info["collision"] = True
        self.t += 1
        return np.zeros(4, np.float32), 0.0, False, False, info


def _gr(monkeypatch, k_off=2.0, k_head=1.0, gamma=0.99):
    monkeypatch.setenv("LANE_CENTRE_K", str(k_off))
    monkeypatch.setenv("LANE_HEADING_K", str(k_head))
    monkeypatch.setenv("DISCOUNT", str(gamma))
    monkeypatch.setenv("EXPLORE_DISCOUNT", str(gamma))
    monkeypatch.setenv("GATED_GOAL_REWARD", "explore")
    import vae_nav.gated_reward as gr
    return importlib.reload(gr)


def _run(gr, lat0, lats, crash_at=None, mode=None, max_steps=50):
    w = gr.GatedReward(_LaneEnv(lat0, lats, crash_at),
                       force_mode=gr.EXPLORE if mode is None else mode,
                       max_steps=max_steps, seed=0)
    w.reset()
    out = []
    for _ in lats:
        _o, r, term, trunc, info = w.step(np.zeros(2))
        out.append((r, term, trunc, info))
        if term or trunc:
            break
    return out


ROOM = 1.75 - HALF_CAR


def test_drifting_out_is_charged_and_coming_back_is_paid(monkeypatch):
    gr = _gr(monkeypatch)
    out = _run(gr, 0.0, [ROOM, 0.0, 0.0])
    f = [i["lane_shaping"] for _r, _t, _tr, i in out]
    assert f[0] == pytest.approx(0.0 - 0.99 * 2.0)       # drift to the line
    assert f[1] == pytest.approx(2.0 - 0.0)              # back to the centre
    assert f[2] == pytest.approx(0.0)


def test_discounted_total_is_fixed_by_the_start_however_it_ends(monkeypatch):
    """The whole point: an early crash and a long drive from the same start
    collect the same discounted shaping, c(s0), so no lane term can make
    ending early cheaper."""
    gr = _gr(monkeypatch)
    g = 0.99
    lat0 = ROOM / 2
    c0 = 2.0 * 0.25

    def total(lats, crash_at):
        out = _run(gr, lat0, lats, crash_at=crash_at)
        assert out[-1][1]                                # ended TERMINAL
        return sum(g ** t * i["lane_shaping"] for t, (_r, _t, _tr, i) in enumerate(out))

    early = total([ROOM, ROOM], crash_at=1)
    late = total([ROOM, 0.2, -0.3, 0.9, 0.0, 0.0, 0.4, 0.0], crash_at=7)
    assert early == pytest.approx(c0)
    assert late == pytest.approx(c0)


def test_terminal_step_has_no_tail(monkeypatch):
    gr = _gr(monkeypatch)
    out = _run(gr, 0.0, [ROOM, ROOM], crash_at=1)
    _r, term, _tr, info = out[-1]
    assert term
    assert info["lane_shaping_tail"] == 0.0
    assert info["lane_shaping"] == pytest.approx(2.0)   # c(s) - 0


def test_truncation_keeps_the_tail(monkeypatch):
    gr = _gr(monkeypatch)
    out = _run(gr, 0.0, [ROOM, ROOM], max_steps=2)
    _r, term, trunc, info = out[-1]
    assert trunc and not term
    assert info["lane_shaping_tail"] == pytest.approx(0.99 * 2.0)


def test_goal_mode_is_shaped_too(monkeypatch):
    gr = _gr(monkeypatch)
    out = _run(gr, 0.0, [ROOM, 0.0], mode=gr.GOAL)
    f = [i["lane_shaping"] for _r, _t, _tr, i in out]
    assert f[0] == pytest.approx(-0.99 * 2.0)
    assert f[1] == pytest.approx(2.0)


def test_shaping_reaches_the_reward(monkeypatch):
    gr = _gr(monkeypatch)
    on = _run(gr, 0.0, [ROOM])
    gr = _gr(monkeypatch, k_off=0.0, k_head=0.0)
    off = _run(gr, 0.0, [ROOM])
    assert "lane_shaping" not in off[0][3]
    assert on[0][0] - off[0][0] == pytest.approx(on[0][3]["lane_shaping"])


def test_shaping_on_without_lane_block_is_a_loud_error(monkeypatch):
    gr = _gr(monkeypatch)
    env = _LaneEnv(0.0, [0.0])
    w = gr.GatedReward(env, force_mode=gr.EXPLORE, max_steps=5, seed=0)
    orig = env.reset

    def reset_without_block(**kw):
        o, i = orig(**kw)
        del i["lane_priv"]
        return o, i
    env.reset = reset_without_block
    with pytest.raises(KeyError):
        w.reset()


def test_core_state_publishes_the_block_when_shaping_is_on(monkeypatch):
    monkeypatch.setenv("OBS_PRIV", "0")
    monkeypatch.setenv("LANE_CENTRE_K", "2.0")
    import vae_nav.core_state as cs
    cs = importlib.reload(cs)
    assert cs.LANE_GEOM is True
    monkeypatch.setenv("LANE_CENTRE_K", "0")
    monkeypatch.setenv("LANE_HEADING_K", "0")
    cs = importlib.reload(cs)
    assert cs.LANE_GEOM is False


@pytest.fixture(autouse=True)
def _restore_modules(monkeypatch):
    yield
    for k in ("LANE_CENTRE_K", "LANE_HEADING_K", "OBS_PRIV", "DISCOUNT",
              "EXPLORE_DISCOUNT", "GATED_GOAL_REWARD"):
        monkeypatch.delenv(k, raising=False)
    import vae_nav.gated_reward as gr
    import vae_nav.core_state as cs
    importlib.reload(gr)
    importlib.reload(cs)
