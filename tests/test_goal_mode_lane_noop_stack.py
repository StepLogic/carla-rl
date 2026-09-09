"""Goal mode ignores lane invasion THROUGH THE REAL WRAPPER STACK.

WHY THIS EXISTS SEPARATELY from test_lane_invasion_noop.py. That file drives
GatedReward over a bare stub, which proves the reward function ignores lane
state. It does NOT prove the assembled arm does, because in the real trainer
GatedReward sits on top of LaneFollowReward:

    CarlaEnv -> CoreStateWrapper -> LaneFollowReward -> GatedReward -> ...

LaneFollowReward computes its own reward AND its own termination, and a lane
crossing is terminal for it. If GatedReward's SORB branch let that `terminated`
through, a lane crossing would end a goal episode no matter what the reward
function said -- and every stub test would still pass.

So this file asserts the claim at the level it is actually made: with the real
LaneFollowReward underneath, signalling a lane crossing and a termination.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

pytest.importorskip("gymnasium")

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402

from vae_nav.gated_reward import EXPLORE, GOAL, GatedReward  # noqa: E402
from vae_nav.lane_follow_reward import LaneFollowReward  # noqa: E402

ACT = np.zeros(2, np.float32)


class _CarlaLike(gym.Env):
    """Stands in for CoreStateWrapper's output, including a terminal `done`.

    done_from_env=True models the env ending the episode on a lane crossing,
    which is what CarlaEnv actually does.
    """
    metadata = {}

    def __init__(self, info=None, done_from_env=True):
        self.observation_space = spaces.Dict({})
        self.action_space = spaces.Box(
            np.array([-0.5, 0.0], np.float32), np.array([0.5, 0.6], np.float32))
        self.info = dict(info or {})
        self.info.setdefault("hero_pos", (0.0, 0.0))
        self.info.setdefault("speed_factor", 0.0)
        self.info.setdefault("delta_distance", 0.0)
        self.info.setdefault("projected_distance", 0.0)
        self.done_from_env = done_from_env

    def reset(self, **kw):
        return np.zeros(4, np.float32), dict(self.info)

    def step(self, a):
        return (np.zeros(4, np.float32), 0.0,
                bool(self.done_from_env), False, dict(self.info))


def _stack(info, mode=GOAL, goal_kind="sorb", goal_max_steps=50):
    """The real order: LaneFollowReward BELOW GatedReward."""
    env = _CarlaLike(info)
    env = LaneFollowReward(env)
    return GatedReward(env, force_mode=mode, goal_kind=goal_kind,
                       goal_max_steps=goal_max_steps, step_cost=1.0,
                       max_steps=50, seed=0)


CROSSING = {"lane_any": True, "lane_invasion": True, "in_junction": False,
            "off_road": False, "lane_marking_types": ["Solid"]}


def test_a_lane_crossing_does_not_end_a_goal_episode():
    # THE LOAD-BEARING ONE. LaneFollowReward below is signalling terminal AND
    # the env's own done is True. Goal mode must discard both.
    w = _stack(CROSSING)
    w.reset()
    assert w.step(ACT)[2] is False


def test_a_lane_crossing_costs_a_goal_episode_exactly_one_step():
    # No penalty leaks through from the wrapper below: the step costs the
    # step cost and nothing more.
    w = _stack(CROSSING)
    w.reset()
    assert w.step(ACT)[1] == pytest.approx(-1.0)


def test_a_clean_step_and_a_crossing_step_cost_the_same():
    crossing = _stack(CROSSING)
    clean = _stack({})
    crossing.reset()
    clean.reset()
    assert crossing.step(ACT)[1] == pytest.approx(clean.step(ACT)[1])


def test_off_road_does_not_end_a_goal_episode_either():
    # Under minimum time the penalty for any failure IS the time it costs, so
    # nothing except arrival and the wedge stops the clock.
    w = _stack({"lane_any": True, "off_road": True, "in_junction": False})
    w.reset()
    assert w.step(ACT)[2] is False


def test_a_collision_does_not_end_a_goal_episode_either():
    w = _stack({"collision": True})
    w.reset()
    _, reward, terminated, _, _ = w.step(ACT)
    assert terminated is False
    assert reward == pytest.approx(-1.0)


def test_arrival_still_ends_a_goal_episode_through_the_stack():
    # The one thing that MUST still terminate. LaneFollowReward publishes
    # is_success, and goal_reward_sorb reads it.
    w = _stack({"arrived": True})
    w.reset()
    _, reward, terminated, _, info = w.step(ACT)
    assert terminated is True
    assert reward == 0.0
    assert info["is_success"] is True


def test_the_goal_return_is_still_minus_the_step_count_through_the_stack():
    w = _stack({}, goal_max_steps=100)
    w.reset()
    assert sum(w.step(ACT)[1] for _ in range(20)) == pytest.approx(-20.0)


def test_repeated_crossings_never_shorten_a_goal_episode():
    # A crossing on every single step still runs the full clock.
    w = _stack(CROSSING, goal_max_steps=10)
    w.reset()
    truncs = [w.step(ACT)[3] for _ in range(10)]
    assert truncs == [False] * 9 + [True]


def test_explore_mode_through_the_stack_still_ignores_crossings_under_the_flag():
    # Sanity that the stack test agrees with the flag's behaviour in the other
    # mode; depends on LANE_INVASION_NOOP, so it only asserts non-termination
    # when the flag is on.
    from vae_nav import gated_reward as m
    if not m.LANE_INVASION_NOOP:
        pytest.skip("LANE_INVASION_NOOP is off in this environment")
    w = _stack(CROSSING, mode=EXPLORE)
    w.reset()
    assert w.step(ACT)[2] is False
