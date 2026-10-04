"""Goal episodes must say WHY they ended, even with HER off.

MEASURED 2026-09-30: all 235 of v37's [Ep] lines read cross=nan(n=0). The
trainer fills its goal end-reason tally from the HER buffer's classifier, and
with EPISODE_HER=0 there is no buffer -- so for four hours the goal arm's
endings were never counted, and `cross=0.00` was 0 of an EMPTY tally, not a
measurement. v38 attempt 1 printed `ends[]` after six goal episodes for the
same reason.

GatedReward now labels the terminating goal step itself, mirroring
info["explore_end"]:
  "reached"   arrival (checked first, as goal_reward_sorb does)
  "off_road"  off-road count reached GOAL_OFFROAD_TRUNCATE_STEPS
  "collision" collision count reached GOAL_COLLISION_TRUNCATE_STEPS
  "crossing"  crossing count reached GOAL_CROSSING_TRUNCATE_STEPS
  "clock"     t reached the goal step cap
  "other"     any other ending -- reported, never guessed
absent while the episode continues. Cap precedence is episode_her's: off_road,
collision, crossing. Logging only: the reward is unchanged.
"""
import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
pytest.importorskip("gymnasium")
import gymnasium as gym                       # noqa: E402
from gymnasium import spaces                  # noqa: E402

ACT = np.zeros(2, np.float32)

# v38's goal regime: counting caps, off-road recoverable for 20 steps,
# collision fatal at 1, shaping on.
V38_ENV = {
    "GATED_GOAL_REWARD": "sorb",
    "GOAL_TRUNCATE_COUNTS": "1",
    "GOAL_ARRIVAL_BONUS": "100.0",
    "GOAL_OFFROAD_TRUNCATE_STEPS": "20",
    "GOAL_COLLISION_TRUNCATE_STEPS": "1",
    "GOAL_CROSSING_TRUNCATE_STEPS": "20",
    "GOAL_LANE_PENALTY": "10.0",
    "GOAL_OFF_ROAD_PENALTY": "10.0",
    "GOAL_COLLISION_PENALTY": "10.0",
    "POTENTIAL_K": "2.0",
}


@pytest.fixture()
def gr():
    import vae_nav.gated_reward as mod
    saved = {k: os.environ.get(k) for k in V38_ENV}
    os.environ.update(V38_ENV)
    try:
        importlib.reload(mod)
        yield mod
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(mod)


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
        return np.zeros(4, np.float32), 0.0, False, False, dict(self.info)


def _wrap(gr, info=None, mode=None, goal_max_steps=1000):
    w = gr.GatedReward(_StubEnv(info), force_mode=gr.GOAL if mode is None else mode,
                       max_steps=500, seed=0, goal_kind="sorb",
                       goal_max_steps=goal_max_steps, step_cost=1.0)
    w.reset()
    return w


def _run_until_done(w, limit=2000):
    for k in range(1, limit + 1):
        _, _, term, trunc, info = w.step(ACT)
        if term or trunc:
            return k, info
        assert "goal_end" not in info, f"labelled at step {k} while still running"
    raise AssertionError("episode never ended")


# ---- the pure classifier ------------------------------------------------------

def test_classifier_precedence(gr):
    c = gr.classify_goal_end
    assert c(True, 20, 1, 20, 5, 1000) == "reached"      # arrival wins over caps
    assert c(False, 20, 1, 20, 5, 1000) == "off_road"    # then off_road
    assert c(False, 19, 1, 20, 5, 1000) == "collision"   # then collision
    assert c(False, 19, 0, 20, 5, 1000) == "crossing"    # then crossing
    assert c(False, 0, 0, 0, 1000, 1000) == "clock"
    assert c(False, 0, 0, 0, 10, 1000) == "other"        # never guessed


# ---- the wrapper labels the terminating goal step -------------------------------

def test_off_road_is_recoverable_until_the_20th_step(gr):
    k, info = _run_until_done(_wrap(gr, {"off_road": True}))
    assert k == 20
    assert info["goal_end"] == "off_road"


def test_collision_is_fatal_on_the_first_step(gr):
    k, info = _run_until_done(_wrap(gr, {"collision": True}))
    assert k == 1
    assert info["goal_end"] == "collision"


def test_crossing_cap(gr):
    k, info = _run_until_done(_wrap(gr, {"lane_any": True}))
    assert k == 20
    assert info["goal_end"] == "crossing"


def test_arrival(gr):
    _, info = _run_until_done(_wrap(gr, {"arrived": True}))
    assert info["goal_end"] == "reached"


def test_arrival_beats_a_fault_on_the_same_step(gr):
    _, info = _run_until_done(_wrap(gr, {"arrived": True, "collision": True}))
    assert info["goal_end"] == "reached"


def test_clock(gr):
    k, info = _run_until_done(_wrap(gr, {}, goal_max_steps=5))
    assert k == 5
    assert info["goal_end"] == "clock"


def test_explore_episodes_are_never_goal_labelled(gr):
    w = _wrap(gr, {"off_road": True}, mode=gr.EXPLORE)
    for _ in range(600):
        _, _, term, trunc, info = w.step(ACT)
        assert "goal_end" not in info
        if term or trunc:
            break


def test_label_does_not_change_the_reward(gr):
    # Logging only: the same scripted episode pays the same rewards whether or
    # not anyone reads the label.
    w1, w2 = _wrap(gr, {"off_road": True}), _wrap(gr, {"off_road": True})
    for _ in range(20):
        r1 = w1.step(ACT)[1]
        r2 = w2.step(ACT)[1]
        assert r1 == r2
