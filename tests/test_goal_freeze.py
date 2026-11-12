"""GOAL_FREEZE_STEPS: a fined crossing in goal mode freezes the car for K sim
steps and comes back as ONE lumped transition (user 2026-09-29).

Pinned: the car is held (the env config's goal_frozen flag is up for exactly
the frozen steps), the reward is the crossing step plus K discounted -step_cost,
the bootstrap mask is gamma^K, the clock advances by K, the fault counters do
not, the crossing's fault flags survive into the returned info, and a legal
(junction / unpainted) crossing, explore mode and freeze_steps=0 never freeze.
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
G = 0.995
K = 50
CLEAN = {"lane_any": False, "in_junction": False, "off_road": False,
         "collision": False, "lane_marking_types": [], "speed_kmh": 0.0,
         "target_speed_kmh": 0.0}
BROKEN = dict(CLEAN, lane_any=True, lane_marking_types=["Broken"])


@pytest.fixture
def m(monkeypatch):
    for k, v in {"DISCOUNT": str(G), "GOAL_FREEZE_STEPS": str(K),
                 "GOAL_TRUNCATE_COUNTS": "1", "GOAL_LANE_GRADED": "1",
                 "GOAL_LANE_PENALTY": "10", "GOAL_LANE_PENALTY_BROKEN": "10",
                 "GOAL_OFF_ROAD_PENALTY": "10", "GOAL_COLLISION_PENALTY": "10",
                 "GOAL_ARRIVAL_BONUS": "100", "GOAL_SPEED_BONUS": "0",
                 "POTENTIAL_K": "0", "GATED_GOAL_LANE_TERMINAL": "0",
                 "EPISODE_HER": "0"}.items():
        monkeypatch.setenv(k, v)
    import vae_nav.gated_reward as mod
    yield importlib.reload(mod)
    for k in ("GOAL_FREEZE_STEPS", "DISCOUNT"):
        monkeypatch.delenv(k, raising=False)
    importlib.reload(mod)


class _Exp:
    goal_frozen = False


class _ScriptEnv(gym.Env):
    """Steps through a list of infos (the last repeats); records whether the
    experiment's hold was up on each step."""
    metadata = {}

    def __init__(self, infos):
        self.observation_space = spaces.Dict({})
        self.action_space = spaces.Box(np.array([-1, -1], np.float32),
                                       np.array([1, 1], np.float32))
        self.infos = infos
        self.experiment = _Exp()
        self.n = 0
        self.held = []

    def reset(self, **kw):
        self.n = 0
        return np.zeros(4, np.float32), dict(CLEAN, hero_pos=(0.0, 0.0))

    def step(self, a):
        self.held.append(self.experiment.goal_frozen)
        info = dict(self.infos[min(self.n, len(self.infos) - 1)])
        info.setdefault("hero_pos", (float(self.n), 0.0))
        self.n += 1
        return np.full(4, self.n, np.float32), 0.0, False, False, info


def _goal(m, infos, max_steps=1000, **kw):
    env = _ScriptEnv(infos)
    w = m.GatedReward(env, force_mode=m.GOAL, goal_kind="sorb",
                      goal_max_steps=max_steps, step_cost=1.0, seed=0, **kw)
    w.reset()
    return env, w


def test_broken_crossing_freezes_for_k_steps_as_one_transition(m):
    env, w = _goal(m, [BROKEN] + [CLEAN] * 200)
    obs, r, term, trunc, info = w.step(ACT)
    freeze_cost = sum(G ** k for k in range(1, K + 1))
    assert r == pytest.approx(-1.0 - 10.0 - freeze_cost)      # ~ -55.2
    assert info["freeze_mask"] == pytest.approx(G ** K)
    assert info["goal_freeze_steps"] == K
    assert not term and not trunc
    assert env.n == 1 + K and w._t == 1 + K
    assert env.held == [False] + [True] * K
    assert env.experiment.goal_frozen is False                # released
    assert obs[0] == 1 + K                                    # post-freeze obs
    assert info["lane_any"] and info["lane_marking_types"] == ["Broken"]
    assert w._crossing_steps == 1                             # frozen steps not counted
    # the next step is ordinary again
    _, r2, *_ , info2 = w.step(ACT)
    assert r2 == pytest.approx(-1.0) and "freeze_mask" not in info2


@pytest.mark.parametrize("info", [
    dict(BROKEN, in_junction=True),                           # the turn
    dict(CLEAN, lane_any=True, lane_marking_types=["NONE"]),  # unpainted edge
    dict(CLEAN, off_road=True),                               # not a crossing
])
def test_unfined_or_non_crossing_faults_never_freeze(m, info):
    env, w = _goal(m, [info] + [CLEAN] * 5)
    _, r, *_ , out = w.step(ACT)
    assert env.n == 1 and "freeze_mask" not in out


def test_clock_cap_during_freeze_truncates_and_bootstraps_at_gamma_k(m):
    env, w = _goal(m, [BROKEN] + [CLEAN] * 200, max_steps=11)
    _, r, term, trunc, info = w.step(ACT)
    assert trunc and not term
    assert info["goal_freeze_steps"] == 10 and env.n == 11
    assert info["freeze_mask"] == pytest.approx(G ** 10)


def test_arrival_while_braking_pays_the_discounted_bonus_and_terminates(m):
    env, w = _goal(m, [BROKEN, CLEAN, CLEAN, dict(CLEAN, arrived=True)])
    _, r, term, trunc, info = w.step(ACT)
    assert term and not trunc and info["is_success"]
    assert r == pytest.approx(-11.0 - G - G ** 2 + 100.0 * G ** 3)


def test_freeze_off_and_explore_mode_are_untouched(m):
    env, w = _goal(m, [BROKEN] + [CLEAN] * 5, freeze_steps=0)
    _, r, *_ = w.step(ACT)
    assert env.n == 1 and r == pytest.approx(-11.0)
    env = _ScriptEnv([BROKEN] + [CLEAN] * 5)
    w = m.GatedReward(env, force_mode=m.EXPLORE, max_steps=50, seed=0)
    w.reset()
    w.step(ACT)
    assert env.n == 1


def test_config_refuses_freeze_with_episode_her_or_nstep():
    from vae_nav.gated_config import check_gated_run
    base = {"GATED": "1", "OBS_WITH_MODE": "1", "GOAL_FREEZE_STEPS": "50"}
    for extra in ({"EPISODE_HER": "1"}, {"N_STEP": "3"}):
        with pytest.raises(ValueError, match="GOAL_FREEZE_STEPS"):
            check_gated_run({**base, **extra})
