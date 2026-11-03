"""Junction-zone goal reward and goal chaining in GatedReward (user 2026-11-02).

GOAL_ZONE_M: goal mode is paid lane following everywhere; inside the junction
zone its forward progress is REPLACED by route progress toward the goal, with
retreat charged at ROUTE_RETREAT_SCALE. GOAL_CHAIN_M: an arrival short of the
chain distance makes the next reset soft (same car, new goal).
"""
import importlib
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
pytest.importorskip("gymnasium")
from tests.test_route_and_junction_reward import Road, STEP_M  # noqa: E402

GOAL = 1


@pytest.fixture(autouse=True)
def _clean_modules():
    yield
    import vae_nav.lane_explore_reward as ler
    import vae_nav.gated_reward as gr
    for k in ("GOAL_ZONE_M", "GOAL_CHAIN_M", "ROUTE_RETREAT_SCALE", "GOAL_ROUTE_ONLY",
              "GOAL_ARRIVAL_MAX_SPEED", "EXPLORE_LANE_MODE",
              "EXPLORE_COLLISION_FATAL", "GOAL_STEP_PENALTY",
              "STALL_TRUNCATE_STEPS", "STALL_SPEED_KMH"):
        os.environ.pop(k, None)
    importlib.reload(ler)
    importlib.reload(gr)


def _gr(monkeypatch, **env):
    base = dict(GATED_GOAL_REWARD="explore", LANE_PROGRESS_MODE="projected",
                LANE_PROGRESS_SCALE=1, POTENTIAL_K=1, GOAL_ROUTE_ONLY=0,
                EXPLORE_JUNCTION_BONUS=0, EXPLORE_GOAL_DIST_M=0,
                EXPLORE_IDLE_PENALTY=0, LANE_CENTRE_K=0, LANE_HEADING_K=0,
                STOP_ACTION=0, PROGRESS_SPEED_GATE=0, SHAPING_DISCOUNTED=0,
                GOAL_ZONE_M=1.0, ROUTE_RETREAT_SCALE=0.3, EXPLORE_LANE_MODE="terminal",
                GOAL_ARRIVAL_MAX_SPEED=10)
    base.update(env)
    for k, v in base.items():
        monkeypatch.setenv(k, str(v))
    import vae_nav.lane_explore_reward as ler
    importlib.reload(ler)
    import vae_nav.gated_reward as gr
    return importlib.reload(gr)


def _run(gr, env, n, chain=False):
    w = gr.GatedReward(env, force_mode=GOAL, max_steps=500, seed=0, chain=chain)
    w.reset()
    out = []
    for _ in range(n):
        _o, r, term, trunc, info = w.step(np.zeros(2, np.float32))
        out.append((float(r), term, trunc, dict(info)))
        if term or trunc:
            break
    return w, out


def test_outside_the_zone_goal_mode_is_paid_lane_following_only(monkeypatch):
    gr = _gr(monkeypatch)
    # route falls 1 m a step, the car drives 0.3 m: lane pay is 0.3, route pay 1.0
    env = Road(junction=[False] * 10, route=[100.0 - t for t in range(12)])
    _w, out = _run(gr, env, 6)
    for r, _te, _tr, info in out[1:]:
        assert not info["goal_zone"]
        assert r == pytest.approx(STEP_M, abs=0.01)
        assert info["gated_shaping"] == 0.0


def test_inside_the_zone_route_progress_replaces_lane_progress(monkeypatch):
    gr = _gr(monkeypatch)
    env = Road(junction=[False, False, True, True, True, False] + [False] * 10,
               route=[100.0 - t for t in range(20)])
    _w, out = _run(gr, env, 5)
    # steps 3-5 end inside the junction
    for r, _te, _tr, info in out[2:5]:
        assert info["goal_zone"]
        assert r == pytest.approx(1.0, abs=0.01)
        assert info["gated_shaping"] == pytest.approx(1.0)


def test_driving_away_in_the_zone_is_charged_at_the_retreat_scale(monkeypatch):
    gr = _gr(monkeypatch)
    env = Road(junction=[True] * 6, route=[50.0 + t for t in range(10)])
    _w, out = _run(gr, env, 4)
    for r, _te, _tr, info in out[1:]:
        assert info["goal_zone"]
        assert r == pytest.approx(-0.3, abs=0.01)


def test_the_zone_extends_past_the_junction_exit(monkeypatch):
    gr = _gr(monkeypatch, GOAL_ZONE_M=1.0)    # 1 m = three 0.3 m steps after exit
    env = Road(junction=[True, True, False, False, False, False, False, False],
               route=[100.0 - t for t in range(12)])
    _w, out = _run(gr, env, 8)
    zones = [o[3]["goal_zone"] for o in out]
    assert zones[:5] == [True, True, True, True, True]
    assert zones[-1] is False


class _Arrive(Road):
    def __init__(self, arrive_at, **kw):
        super().__init__(**kw)
        self.arrive_at = arrive_at

    def _info(self):
        inf = super()._info()
        inf["arrived"] = self.t == self.arrive_at
        return inf


def test_an_arrival_short_of_the_chain_makes_the_next_reset_soft(monkeypatch):
    gr = _gr(monkeypatch, GOAL_CHAIN_M=200)
    w, out = _run(gr, _Arrive(4, junction=[False] * 10, route=[10.0] * 12), 10, chain=True)
    assert out[-1][1] and out[-1][3]["is_success"]
    assert out[-1][3]["chain_continue"]
    calls = []

    def fake_soft():
        calls.append(1)
        return np.zeros(4, np.float32), dict(w.env._info(), chain_soft_reset=True)
    w._soft_reset = fake_soft
    _o, info = w.reset()
    assert calls and info["chain_goals"] == 1
    assert info["chain_m"] == pytest.approx(4 * STEP_M)


def test_the_chain_ends_once_the_distance_is_covered(monkeypatch):
    gr = _gr(monkeypatch, GOAL_CHAIN_M=1.0)       # 4 steps x 0.3 m = 1.2 m >= 1
    w, out = _run(gr, _Arrive(4, junction=[False] * 10, route=[10.0] * 12), 10, chain=True)
    assert out[-1][3]["is_success"] and not out[-1][3]["chain_continue"]
    w._soft_reset = lambda: pytest.fail("soft reset after the chain was complete")
    _o, info = w.reset()
    assert info["chain_goals"] == 0 and info["chain_m"] == 0.0


def test_eval_envs_never_chain(monkeypatch):
    gr = _gr(monkeypatch, GOAL_CHAIN_M=200)
    w, out = _run(gr, _Arrive(4, junction=[False] * 10, route=[10.0] * 12), 10, chain=False)
    assert not out[-1][3]["chain_continue"]


def test_a_fault_resets_the_chain(monkeypatch):
    gr = _gr(monkeypatch, GOAL_CHAIN_M=200, EXPLORE_COLLISION_FATAL=1)
    w, out = _run(gr, Road(junction=[False] * 10, route=[10.0] * 12, collide_at=3), 10,
                  chain=True)
    assert out[-1][1] and not out[-1][3]["chain_continue"]


def test_the_step_penalty_is_charged_on_every_goal_step(monkeypatch):
    gr0 = _gr(monkeypatch, GOAL_ZONE_M=0)
    _w, base = _run(gr0, Road(junction=[False] * 10, route=[10.0] * 12), 5)
    gr1 = _gr(monkeypatch, GOAL_ZONE_M=0, GOAL_STEP_PENALTY=0.02)
    _w, pen = _run(gr1, Road(junction=[False] * 10, route=[10.0] * 12), 5)
    assert [round(a[0] - b[0], 6) for a, b in zip(base, pen)] == [0.02] * len(base)


# ---- stall truncation (STALL_TRUNCATE_STEPS, 2026-11-03) -------------------

def test_a_stationary_car_truncates_instead_of_running_the_clock(monkeypatch):
    gr = _gr(monkeypatch, GOAL_ZONE_M=0, STALL_TRUNCATE_STEPS=3, STALL_SPEED_KMH=0.5)
    _w, out = _run(gr, Road(junction=[False] * 10, route=[10.0] * 12, speed=0.0), 10)
    assert len(out) == 3
    r, term, trunc, info = out[-1]
    assert trunc and not term
    assert info["goal_end"] == "stalled" and info["stalled"]


def test_a_moving_car_is_never_stalled_and_the_counter_resets(monkeypatch):
    gr = _gr(monkeypatch, GOAL_ZONE_M=0, STALL_TRUNCATE_STEPS=3, STALL_SPEED_KMH=0.5)
    _w, out = _run(gr, Road(junction=[False] * 10, route=[10.0] * 12, speed=10.0), 6)
    assert len(out) == 6 and not any(o[2] for o in out)
    assert all(o[3]["stall_steps"] == 0 for o in out)


def test_stall_truncation_is_off_by_default(monkeypatch):
    gr = _gr(monkeypatch, GOAL_ZONE_M=0)
    _w, out = _run(gr, Road(junction=[False] * 10, route=[10.0] * 12, speed=0.0), 6)
    assert len(out) == 6 and not any(o[2] for o in out)


def test_a_fault_still_wins_over_the_stall_clock(monkeypatch):
    gr = _gr(monkeypatch, GOAL_ZONE_M=0, STALL_TRUNCATE_STEPS=2, STALL_SPEED_KMH=0.5,
             EXPLORE_COLLISION_FATAL=1)
    _w, out = _run(gr, Road(junction=[False] * 10, route=[10.0] * 12, speed=0.0,
                            collide_at=2), 10)
    assert out[-1][1] and out[-1][3].get("goal_end") != "stalled"


def test_a_chain_continuation_stays_a_goal_episode(monkeypatch):
    """With explore episodes in the mix, redrawing the mode on a soft reset
    would hand continuations to the explore arm, which has no goal."""
    gr = _gr(monkeypatch, GOAL_ZONE_M=0, GOAL_CHAIN_M=200)
    env = Road(junction=[False] * 10, route=[10.0] * 12)
    w = gr.GatedReward(env, explore_prob=1.0, max_steps=500, seed=0, chain=True)
    w.reset()
    w._chain_continue = True
    w._soft_reset = lambda: (np.zeros(4, np.float32), dict(env._info()))
    w.reset()
    assert w.mode == gr.GOAL


def test_an_ordinary_reset_still_draws_the_mode(monkeypatch):
    gr = _gr(monkeypatch, GOAL_ZONE_M=0, GOAL_CHAIN_M=200)
    w = gr.GatedReward(Road(junction=[False] * 10, route=[10.0] * 12),
                       explore_prob=1.0, max_steps=500, seed=0, chain=True)
    w.reset()
    assert w.mode == gr.EXPLORE
