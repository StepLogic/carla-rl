"""A terminal collision must not become an escape, and shaping must not pay one.

THE TRAP. With a per-step cost and a collision that simply ENDS the episode,
crashing stops the meter: crash on step one returns -c, drive honestly to a
goal 80 steps away and return -80. Crashing immediately would be the optimal
policy. time_to_goal_reward's docstring records this as the reason collisions
were made non-terminal in the first place, and sparse_reward has a dedicated
assert_no_suicide_incentive for the same failure.

Billing every remaining step removes it: a crash costs exactly what running out
of time costs, so ending early is worth precisely nothing.

THE SECOND WAY IN IS THE SHAPING. Potential-based shaping paid ON a terminal
step would hand a crash near the goal a bonus over a crash far from it, which
turns "how far did I get before dying" into something worth optimising and
makes the billed terminal inexact again. The wrapper skips shaping on terminal
steps; these tests pin that.
"""
import gymnasium as gym
import numpy as np
import pytest

from vae_nav.time_to_goal_reward import TimeToGoalReward

STEP_COST = 1.0
MAX_STEPS = 50
ROUTE_M = 100.0          # a 100 m route, closing 2 m per step
CLOSE_PER_STEP = 2.0


class FakeEnv(gym.Env):
    """Closes CLOSE_PER_STEP metres of route per step; may arrive or crash."""

    def __init__(self, arrive_at=None, crash_at=None, route=True):
        self.arrive_at, self.crash_at, self.route = arrive_at, crash_at, route
        self.t = 0
        box = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = box
        self.action_space = box

    def reset(self, **kw):
        self.t = 0
        return np.zeros(2, np.float32), {}

    def step(self, action):
        self.t += 1
        crashed = self.crash_at is not None and self.t >= self.crash_at
        arrived = (self.arrive_at is not None and self.t == self.arrive_at
                   and not crashed)
        info = {"arrived": arrived, "wedged": False, "collision": crashed}
        if self.route:
            info["route_remaining_m"] = max(
                0.0, ROUTE_M - CLOSE_PER_STEP * self.t)
        return np.zeros(2, np.float32), 999.0, True, False, info


def rollout(env, **kw):
    kw.setdefault("step_cost", STEP_COST)
    kw.setdefault("max_steps", MAX_STEPS)
    w = TimeToGoalReward(env, **kw)
    w.reset()
    total, shaping, steps = 0.0, 0.0, 0
    while True:
        _o, r, term, trunc, info = w.step(np.zeros(2, np.float32))
        total += r
        shaping += float(info.get("ttg_shaping", 0.0))
        steps += 1
        if term or trunc:
            return total, shaping, steps, term, info


@pytest.mark.parametrize("crash_at", [1, 2, 9, 25, MAX_STEPS - 1, MAX_STEPS])
def test_crashing_costs_exactly_a_timeout(crash_at):
    """The whole anti-suicide property, at every step a crash can happen."""
    total, _sh, _n, term, _i = rollout(
        FakeEnv(crash_at=crash_at, route=False), collision_terminal=True)
    assert term, "a billed collision must TERMINATE (masks=0), not truncate"
    assert total == pytest.approx(-STEP_COST * MAX_STEPS), (
        f"crash at {crash_at} returned {total}, but running out of time returns "
        f"{-STEP_COST * MAX_STEPS}. Any gap is an incentive to seek or avoid a "
        "crash for reasons other than the time it costs.")


def test_arriving_always_beats_crashing():
    arrive, _sh, _n, _t, _i = rollout(
        FakeEnv(arrive_at=8), collision_terminal=True)
    crash, _sh2, _n2, _t2, _i2 = rollout(
        FakeEnv(crash_at=8, route=False), collision_terminal=True)
    assert arrive > crash, f"arrive={arrive} crash={crash}"


def test_collision_is_not_terminal_by_default():
    """The flag is opt-in; without it a collision costs only the time it loses."""
    total, _sh, steps, term, _i = rollout(FakeEnv(crash_at=3, route=False))
    assert not term and steps == MAX_STEPS
    assert total == pytest.approx(-STEP_COST * MAX_STEPS)


def test_shaping_is_skipped_on_the_first_step():
    """A difference needs two samples; paying on step 1 would gift the route."""
    env = FakeEnv(arrive_at=MAX_STEPS)
    w = TimeToGoalReward(env, step_cost=STEP_COST, max_steps=MAX_STEPS,
                         potential_k=1.0)
    w.reset()
    _o, r, _te, _tr, info = w.step(np.zeros(2, np.float32))
    assert info["ttg_shaping"] == 0.0
    assert r == pytest.approx(-STEP_COST)


def test_shaping_telescopes_to_the_progress_made():
    """Sum of the shaping = k * (start remaining - remaining before terminal)."""
    k = 0.5
    total, shaping, steps, _t, _i = rollout(
        FakeEnv(arrive_at=10), potential_k=k)
    # Step 1 pays nothing; steps 2..9 each close CLOSE_PER_STEP; step 10 is
    # terminal and pays no shaping.
    expected = k * CLOSE_PER_STEP * (steps - 2)
    assert shaping == pytest.approx(expected), (
        f"shaping {shaping} != {expected} over {steps} steps")
    assert total == pytest.approx(-STEP_COST * (steps - 1) + expected)


def test_shaping_does_not_pay_a_crash():
    """A crash near the goal must not out-earn a crash at the start line."""
    early, _s1, _n1, _t1, _i1 = rollout(
        FakeEnv(crash_at=2), collision_terminal=True, potential_k=1.0)
    late, _s2, _n2, _t2, _i2 = rollout(
        FakeEnv(crash_at=20), collision_terminal=True, potential_k=1.0)
    # Both are billed the same base return; the LATE one legitimately collected
    # shaping for the metres it actually covered before crashing. What must not
    # happen is the crash STEP ITSELF paying out the remaining potential.
    assert _i1["ttg_shaping"] == 0.0 and _i2["ttg_shaping"] == 0.0
    assert late > early, "progress made before a crash should still count"
    base = -STEP_COST * MAX_STEPS
    assert late == pytest.approx(base + 1.0 * CLOSE_PER_STEP * (20 - 2))
