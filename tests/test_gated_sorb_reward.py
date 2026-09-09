"""The SORB minimum-time goal reward: -1 per step until arrival.

WHY THIS REWARD. The critic becomes the cost-to-go -- Q = -(steps to goal) --
which is the reading a topological map builder needs and which no shaped reward
provides. Matches the Habitat peer's gated point-goal arm.

THE TRAP IT HAS TO AVOID, and the reason several tests here look paranoid: with
a per-step cost and a TERMINAL failure, crashing STOPS THE METER, so crashing on
step one is optimal. Every escape route has to be closed:

  collision      episode continues, the bill keeps running
  lane departure episode continues, the bill keeps running
  wedge          ends the episode but is BILLED EXACTLY for the time it would
                 have burned, so ending early is worth precisely nothing

Only arrival stops the clock, and it pays zero, which is the best score
available. sparse_reward.assert_no_suicide_incentive exists because an earlier
formulation in this project got this wrong.
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

from vae_nav.gated_reward import GOAL, GatedReward  # noqa: E402

ACT = np.zeros(2, np.float32)


class _StubEnv(gym.Env):
    metadata = {}

    def __init__(self, info=None, done_from_env=False):
        self.observation_space = spaces.Dict({})
        self.action_space = spaces.Box(
            np.array([-0.5, 0.0], np.float32), np.array([0.5, 0.6], np.float32))
        self.info = dict(info or {})
        self.info.setdefault("hero_pos", (0.0, 0.0))
        self.info.setdefault("speed_factor", 0.0)
        self.done_from_env = done_from_env

    def reset(self, **kw):
        return np.zeros(4, np.float32), dict(self.info)

    def step(self, action):
        return (np.zeros(4, np.float32), 0.0,
                bool(self.done_from_env), False, dict(self.info))


def _wrap(info=None, step_cost=1.0, goal_max_steps=10, done_from_env=False):
    env = _StubEnv(info, done_from_env)
    w = GatedReward(env, force_mode=GOAL, max_steps=500, seed=0,
                    goal_kind="sorb", goal_max_steps=goal_max_steps,
                    step_cost=step_cost)
    return w, env


# ------------------------------------------------------------ the meter ----

def test_it_pays_exactly_minus_one_per_step():
    w, _ = _wrap()
    w.reset()
    assert [w.step(ACT)[1] for _ in range(5)] == [-1.0] * 5


def test_the_step_cost_is_configurable():
    w, _ = _wrap(step_cost=0.1)
    w.reset()
    assert w.step(ACT)[1] == pytest.approx(-0.1)


def test_the_return_equals_minus_the_step_count():
    # THE WHOLE POINT: the return IS the elapsed time, so a critic fitted to it
    # reads as -(steps to goal) rather than as an arbitrary score.
    w, _ = _wrap(goal_max_steps=100)
    w.reset()
    assert sum(w.step(ACT)[1] for _ in range(37)) == pytest.approx(-37.0)


def test_it_truncates_at_the_goal_cap():
    w, _ = _wrap(goal_max_steps=3)
    w.reset()
    assert [w.step(ACT)[3] for _ in range(3)] == [False, False, True]


def test_the_clock_resets_between_episodes():
    w, _ = _wrap(goal_max_steps=3)
    w.reset()
    w.step(ACT)
    w.step(ACT)
    w.reset()
    assert w.step(ACT)[3] is False


# ------------------------------------------------------------- arrival ----

def test_arrival_pays_zero_and_terminates():
    w, _ = _wrap({"arrived": True})
    w.reset()
    _, reward, terminated, truncated, info = w.step(ACT)
    assert (reward, terminated, truncated) == (0.0, True, False)
    assert info["is_success"] is True


def test_the_is_success_flag_is_accepted_too():
    # The env publishes `arrived`; LaneFollowReward publishes `is_success`.
    # Both must count, because LaneFollowReward sits below this wrapper.
    w, _ = _wrap({"is_success": True})
    w.reset()
    assert w.step(ACT)[2] is True


def test_arrival_is_the_best_outcome_available():
    arrive, _ = _wrap({"arrived": True})
    drive, _ = _wrap()
    wedge, _ = _wrap({"wedged": True})
    for w in (arrive, drive, wedge):
        w.reset()
    assert arrive.step(ACT)[1] > drive.step(ACT)[1] > wedge.step(ACT)[1]


# ------------------------------------------------- no escape from the bill ----

def test_a_collision_does_not_stop_the_meter():
    w, _ = _wrap({"collision": True})
    w.reset()
    _, reward, terminated, _, _ = w.step(ACT)
    assert terminated is False
    assert reward == -1.0


def test_a_lane_departure_does_not_stop_the_meter():
    w, _ = _wrap({"lane_any": True, "off_road": True})
    w.reset()
    _, reward, terminated, _, _ = w.step(ACT)
    assert terminated is False
    assert reward == -1.0


def test_the_envs_own_termination_is_discarded():
    w, _ = _wrap(done_from_env=True)
    w.reset()
    assert w.step(ACT)[2] is False


def test_crashing_immediately_is_not_cheaper_than_driving():
    # The suicide check, stated as economics rather than as a flag.
    crash, _ = _wrap({"collision": True}, goal_max_steps=5)
    clean, _ = _wrap(goal_max_steps=5)
    crash.reset()
    clean.reset()
    crash_total = sum(crash.step(ACT)[1] for _ in range(5))
    clean_total = sum(clean.step(ACT)[1] for _ in range(5))
    assert crash_total == pytest.approx(clean_total)


# ---------------------------------------------------------- the wedge bill ----

def test_a_wedge_is_billed_for_this_step_plus_the_time_remaining():
    w, _ = _wrap({"wedged": True}, goal_max_steps=10)
    w.reset()
    _, reward, terminated, truncated, _ = w.step(ACT)
    assert (terminated, truncated) == (True, False)
    assert reward == pytest.approx(-10.0)      # 1 step + 9 remaining


def test_wedging_late_costs_only_the_time_left():
    w, env = _wrap(goal_max_steps=10)
    w.reset()
    for _ in range(6):
        w.step(ACT)
    env.info["wedged"] = True
    assert w.step(ACT)[1] == pytest.approx(-4.0)   # step 7, plus 3 remaining


@pytest.mark.parametrize("wedge_at", [1, 3, 9, 10])
def test_a_wedged_episode_returns_exactly_the_full_bill(wedge_at):
    # WHENEVER the wedge happens, the episode total is -step_cost * max_steps,
    # byte-for-byte what simulating the frozen frames would have cost. That
    # exactness is what stops the wedge becoming an escape.
    w, env = _wrap(goal_max_steps=10)
    w.reset()
    total = 0.0
    for t in range(1, 11):
        if t == wedge_at:
            env.info["wedged"] = True
        _, reward, terminated, _, _ = w.step(ACT)
        total += reward
        if terminated:
            break
    assert total == pytest.approx(-10.0)


def test_the_wedge_terminates_rather_than_truncates():
    # TERMINATED means masks=0, so the critic does NOT bootstrap. Bootstrapping
    # a frozen state would let its value be whatever the critic guessed --
    # optimistic early in training -- making a wedge look cheap, and would add
    # an unbounded gamma=1 chain from a state the agent never leaves.
    w, _ = _wrap({"wedged": True})
    w.reset()
    _, _, terminated, truncated, _ = w.step(ACT)
    assert terminated is True and truncated is False


def test_wedge_billing_can_be_disabled():
    w, env = _wrap(goal_max_steps=10)
    w.wedge_truncate = False
    env.info["wedged"] = True
    w.reset()
    _, reward, terminated, _, _ = w.step(ACT)
    assert terminated is False
    assert reward == -1.0
