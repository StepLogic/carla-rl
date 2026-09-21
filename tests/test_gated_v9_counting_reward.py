"""The v9 goal-mode COUNTING REGIME (user 2026-09-21).

GOAL_TRUNCATE_COUNTS=1 replaces goal_reward_sorb's terminal bills: arrival
(+GOAL_ARRIVAL_BONUS) is the ONLY terminus; every other end is a truncation
(mask=1.0, the critic bootstraps through the cut) paying its normal per-step
reward; thresholds are cumulative STEPS -- 20 off-road, 20 collision,
20 crossing. The legacy bill regime must stay byte-identical when the flag is
off, so these tests reload the module with the env set and restore it after --
the module reads its constants at import time, and the sibling file
(test_gated_sorb_reward.py) imports it under the defaults.
"""
import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

pytest.importorskip("gymnasium")

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402

ACT = np.zeros(2, np.float32)

V9_ENV = {
    "GOAL_TRUNCATE_COUNTS": "1",
    "GOAL_ARRIVAL_BONUS": "10.0",
    "GOAL_OFFROAD_TRUNCATE_STEPS": "20",
    "GOAL_COLLISION_TRUNCATE_STEPS": "20",
    "GOAL_CROSSING_TRUNCATE_STEPS": "20",
    "GOAL_LANE_PENALTY": "10.0",
    "GOAL_OFF_ROAD_PENALTY": "10.0",
    "GOAL_COLLISION_PENALTY": "10.0",
}


@pytest.fixture()
def v9():
    import vae_nav.gated_reward as gr
    saved = {k: os.environ.get(k) for k in V9_ENV}
    os.environ.update(V9_ENV)
    try:
        importlib.reload(gr)
        yield gr
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(gr)


class _StubEnv(gym.Env):
    metadata = {}

    def __init__(self, info=None):
        self.observation_space = spaces.Dict({})
        self.action_space = spaces.Box(
            np.array([-0.5, 0.0], np.float32), np.array([0.5, 0.6], np.float32))
        self.info = dict(info or {})
        self.info.setdefault("hero_pos", (0.0, 0.0))
        self.info.setdefault("speed_factor", 0.0)

    def reset(self, **kw):
        return np.zeros(4, np.float32), dict(self.info)

    def step(self, action):
        return (np.zeros(4, np.float32), 0.0, False, False, dict(self.info))


def _wrap(gr, info=None):
    env = _StubEnv(info)
    w = gr.GatedReward(env, force_mode=gr.GOAL, max_steps=500, seed=0,
                       goal_kind="sorb", goal_max_steps=1000, step_cost=1.0)
    w.reset()
    return w


# ---- arrival: the only terminus, and it pays the bonus ----------------------

def test_arrival_pays_the_bonus_and_terminates(v9):
    w = _wrap(v9, {"arrived": True})
    _, reward, terminated, truncated, info = w.step(ACT)
    assert (reward, terminated, truncated) == (10.0, True, False)
    assert info["is_success"] is True


def test_arrival_wins_over_a_fault_on_the_same_step(v9):
    # reached is checked first: an episode that arrives on a step it also
    # crosses a line still arrives.
    w = _wrap(v9, {"arrived": True, "lane_any": True})
    _, reward, terminated, truncated, _ = w.step(ACT)
    assert (reward, terminated, truncated) == (10.0, True, False)


def test_default_bonus_stays_zero_for_older_configs(v9):
    # GOAL_TRUNCATE_COUNTS is on but the bonus is its 0.0 default: arrival
    # behaves exactly as v8's (0.0, terminated).
    import os as _os
    _os.environ["GOAL_ARRIVAL_BONUS"] = "0.0"
    try:
        importlib.reload(v9)
        w = _wrap(v9, {"arrived": True})
        _, reward, terminated, truncated, _ = w.step(ACT)
        assert (reward, terminated, truncated) == (0.0, True, False)
    finally:
        _os = V9_ENV["GOAL_ARRIVAL_BONUS"]
        os.environ["GOAL_ARRIVAL_BONUS"] = _os
        importlib.reload(v9)


# ---- per-step fines, and NO bill on the truncating step ---------------------

def test_collision_steps_fine_at_ten_and_do_not_end_early(v9):
    w = _wrap(v9, {"collision": True})
    rewards = [w.step(ACT)[1] for _ in range(15)]
    assert rewards == [-11.0] * 15          # -1 step cost -10 collision fine
    assert w.step(ACT)[2:4] == (False, False)   # 16th step: still running


def test_crossing_steps_fine_at_ten(v9):
    w = _wrap(v9, {"lane_any": True})
    assert [w.step(ACT)[1] for _ in range(3)] == [-11.0] * 3


def test_off_road_steps_fine_at_ten(v9):
    w = _wrap(v9, {"off_road": True})
    assert [w.step(ACT)[1] for _ in range(3)] == [-11.0] * 3


# ---- the thresholds: 20 steps, cumulative, truncated (never terminated) -----

def _run_to_truncation(gr, info):
    w = _wrap(gr, info)
    for _ in range(19):
        assert w.step(ACT)[3] is False       # not truncated yet
    obs, reward, terminated, truncated, _ = w.step(ACT)
    return w, reward, terminated, truncated


def test_twenty_collision_steps_truncate(v9):
    w, reward, terminated, truncated = _run_to_truncation(v9, {"collision": True})
    assert (terminated, truncated) == (False, True)
    assert reward == -11.0                   # normal step reward, NO bill


def test_twenty_crossing_steps_truncate(v9):
    w, reward, terminated, truncated = _run_to_truncation(v9, {"lane_any": True})
    assert (terminated, truncated) == (False, True)
    assert reward == -11.0


def test_twenty_off_road_steps_truncate(v9):
    w, reward, terminated, truncated = _run_to_truncation(v9, {"off_road": True})
    assert (terminated, truncated) == (False, True)
    assert reward == -11.0


def test_nineteen_fault_steps_plus_a_clean_step_do_not_truncate(v9):
    # 19 fault steps, then one CLEAN step: no counter reaches 20, so the
    # episode keeps running (the counters are cumulative, not consecutive --
    # a clean step does not reset them, but 19 < 20 either way).
    w = _wrap(v9, {"collision": True})
    for _ in range(19):
        assert w.step(ACT)[3] is False
    env_info = w.env.info
    env_info.pop("collision")
    _, reward, terminated, truncated, _ = w.step(ACT)
    assert (terminated, truncated) == (False, False)
    assert reward == -1.0                    # clean step pays only the meter


def test_counters_are_cumulative_across_fault_types_and_reset(v9):
    # 10 collisions + 10 crossings does NOT trip (different counters -- each
    # needs 20 of its own kind), and a reset zeroes every counter.
    w = _wrap(v9, {"collision": True})
    for _ in range(10):
        assert w.step(ACT)[3] is False
    w.env.info.clear()
    w.env.info.update({"lane_any": True, "hero_pos": (0.0, 0.0),
                       "speed_factor": 0.0})
    for _ in range(10):
        _, _, terminated, truncated, _ = w.step(ACT)
        assert (terminated, truncated) == (False, False)   # 10+10, no trip
    w.reset()
    assert w._collision_steps == 0 and w._crossing_steps == 0
    assert w.step(ACT)[3] is False


def test_the_clock_cap_still_truncates_under_counting(v9):
    w = _wrap(v9)
    w.goal_max_steps = 3
    for _ in range(3):
        _, _, terminated, truncated, _ = w.step(ACT)
    assert (terminated, truncated) == (False, True)


def test_legacy_bill_regime_is_untouched_when_flag_off():
    # Without GOAL_TRUNCATE_COUNTS the v4-v8 behaviour is byte-identical: a
    # collision step under COLLISION_TERMINAL=1 bills the remaining clock and
    # TERMINATES (the v8 semantics; v9's truncate-everything rule lives only
    # in the counting branch). Called directly -- the wrapper passes only
    # wedge_truncate, and def-time defaults bake COLLISION_TERMINAL at import.
    import vae_nav.gated_reward as gr
    assert gr.GOAL_TRUNCATE_COUNTS is False
    reward, terminated, truncated, arrived = gr.goal_reward_sorb(
        {"collision": True}, 3, 1000, 1.0, False, True)
    assert (terminated, truncated, arrived) == (True, False, False)
    assert reward == pytest.approx(-(1 + 1000 - 3))     # exact remaining clock