"""Ending a wedged episode early must not change what the episode is worth.

THE PROPERTY UNDER TEST. A wedged vehicle is frozen for the rest of the episode
and provably cannot arrive, so its return is -STEP_COST * MAX_STEPS whether the
frozen steps are simulated or billed. If billing and simulating ever disagree,
the wedge becomes either an escape (cheaper -> the agent seeks it, the
suicide-incentive trap) or an extra punishment (dearer -> the reward is no
longer pure time). Both are silent in the success rate.
"""
import gymnasium as gym
import numpy as np
import pytest

from vae_nav.time_to_goal_reward import TimeToGoalReward

STEP_COST = 0.1
MAX_STEPS = 50


class FakeEnv(gym.Env):
    """Drives `arrive_at` steps to the goal, or wedges at `wedge_at` forever."""

    def __init__(self, arrive_at=None, wedge_at=None):
        self.arrive_at = arrive_at
        self.wedge_at = wedge_at
        self.t = 0
        box = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = box
        self.action_space = box

    def reset(self, **kw):
        self.t = 0
        return np.zeros(2, np.float32), {}

    def step(self, action):
        self.t += 1
        wedged = self.wedge_at is not None and self.t >= self.wedge_at
        arrived = (self.arrive_at is not None and self.t == self.arrive_at
                   and not wedged)
        info = {"arrived": arrived, "wedged": wedged,
                "collision": False, "lane_invasion": False}
        # The env's own `done` is deliberately noisy: the wrapper must ignore it.
        return np.zeros(2, np.float32), 999.0, True, False, info


def rollout(env, wedge_truncate=True):
    w = TimeToGoalReward(env, step_cost=STEP_COST, max_steps=MAX_STEPS,
                         wedge_truncate=wedge_truncate)
    w.reset()
    total, steps = 0.0, 0
    while True:
        _o, r, term, trunc, info = w.step(np.zeros(2, np.float32))
        total += r
        steps += 1
        if term or trunc:
            return total, steps, term, trunc, info


@pytest.mark.parametrize("wedge_at", [1, 2, 7, 25, MAX_STEPS - 1, MAX_STEPS])
def test_billing_equals_simulating(wedge_at):
    """THE CENTRAL PROPERTY, at every point the wedge can fire."""
    billed, billed_steps, term, trunc, _ = rollout(
        FakeEnv(wedge_at=wedge_at), wedge_truncate=True)
    simulated, sim_steps, _, _, _ = rollout(
        FakeEnv(wedge_at=wedge_at), wedge_truncate=False)

    assert billed == pytest.approx(simulated), (
        f"wedge at {wedge_at}: billing gave {billed}, simulating gave "
        f"{simulated}. Ending early changed what the episode is worth.")
    assert billed == pytest.approx(-STEP_COST * MAX_STEPS)
    assert billed_steps < sim_steps or wedge_at == MAX_STEPS, (
        "billing saved no wall-clock, which is the entire point")
    assert term and not trunc, (
        "a wedge must TERMINATE (masks=0), not truncate -- bootstrapping "
        "through a frozen state lets the critic guess it cheap")


def test_wedging_is_never_better_than_running_out_the_clock():
    """No escape: the worst honest outcome costs exactly what a wedge costs."""
    wedged, _, _, _, _ = rollout(FakeEnv(wedge_at=3))
    timeout, _, _, _, _ = rollout(FakeEnv())
    assert wedged == pytest.approx(timeout), (
        f"wedging returned {wedged} against {timeout} for a plain timeout; a "
        "gap either way is an incentive the reward should not contain")


def test_arrival_is_unaffected():
    total, steps, term, trunc, info = rollout(FakeEnv(arrive_at=8))
    assert total == pytest.approx(-STEP_COST * 7)  # the arriving step pays 0
    assert steps == 8 and term and not trunc and info["is_success"]


def test_arrival_beats_wedging_at_every_wedge_point():
    """The ordering the policy must learn, stated as an assertion."""
    arrive, _, _, _, _ = rollout(FakeEnv(arrive_at=20))
    for wedge_at in (1, 10, 19, 40):
        wedged, _, _, _, _ = rollout(FakeEnv(wedge_at=wedge_at))
        assert arrive > wedged


def test_return_stays_in_range_whatever_happens():
    for env in (FakeEnv(), FakeEnv(arrive_at=1), FakeEnv(arrive_at=MAX_STEPS),
                FakeEnv(wedge_at=1), FakeEnv(wedge_at=MAX_STEPS)):
        total, _, _, _, _ = rollout(env)
        assert -STEP_COST * MAX_STEPS - 1e-6 <= total <= 0.0


def test_flag_off_restores_the_old_behaviour():
    total, steps, term, trunc, _ = rollout(FakeEnv(wedge_at=5),
                                           wedge_truncate=False)
    assert steps == MAX_STEPS and trunc and not term
    assert total == pytest.approx(-STEP_COST * MAX_STEPS)
