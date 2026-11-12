"""Minimum-time reward: pay for every step until you arrive.

    r = -STEP_COST              every step
    r =  0    and TERMINATE     on arrival
              and TRUNCATE      at MAX_STEPS

    collision      -> no special reward, the episode CONTINUES
    lane departure -> no special reward, the episode CONTINUES

    WEDGED (LANE_WEDGE only) -> TERMINATE, billed -STEP_COST * (1 + remaining)

A wedged vehicle is frozen under full brake for the rest of the episode, so it
cannot reach the goal and will certainly run to MAX_STEPS. Ending it early and
charging the time it would have burned leaves the episode return at exactly
-STEP_COST * MAX_STEPS -- identical to simulating the frozen steps -- while
saving ~950 dead frames of wall-clock per wedged episode. The objective does
not change; only the simulation cost does. WEDGE_TRUNCATE=0 restores the old
behaviour of sitting out the clock.

Return is -STEP_COST * steps_to_goal, or -STEP_COST * MAX_STEPS if the goal is
never reached. The whole range is [-100, 0] at the defaults, and every unit of
it is time. There is no arrival bonus to trade against, no shaping term to
farm, and no coefficient ratio to tune.

THE PENALTY FOR CRASHING IS THE TIME IT COSTS. A car that hits a wall and
cannot recover stops making progress and pays the full -100. One that clips a
kerb and drives on pays only the seconds it lost. That is the right economics
for a minimum-time task and it needs no coefficient.

WHY THERE IS NO FAILURE TERM AT ALL. An earlier version billed every remaining
step on collision, which existed to solve a problem this formulation does not
have: with a per-step cost and a TERMINAL collision, crashing stops the meter,
so crashing on step one is optimal -- the failure sparse_reward's
assert_no_suicide_incentive guards against. Once collisions do not end the
episode there is nothing to escape to, and the bill, the terminal flag and the
bootstrap question all disappear with it.

THE ENV'S OWN TERMINATIONS ARE DISCARDED HERE. CarlaEnv.step returns
(obs, reward, done, False, info) where `done` covers idle, falling, distance,
lane invasion, collision, speed and wrong-direction. This wrapper ignores that
flag entirely and publishes its own, so ONLY arrival and the step limit end an
episode. The env's flags still reach `info`, where the trainer's [Why]
breakdown reports them as RATES rather than as termination causes.
"""
import os

import numpy as np

try:
    from gymnasium import Wrapper
except ImportError:
    from gym import Wrapper

STEP_COST = float(os.environ.get("STEP_COST", 0.1))
TTG_MAX_STEPS = int(os.environ.get("TTG_MAX_STEPS", 1000))
# End a wedged episode immediately and bill the time it would have burned.
WEDGE_TRUNCATE = os.environ.get("WEDGE_TRUNCATE", "1") == "1"


class TimeToGoalReward(Wrapper):
    """Every step costs. Arrival stops the clock. Nothing else ends an episode."""

    def __init__(self, env, step_cost=STEP_COST, max_steps=TTG_MAX_STEPS,
                 wedge_truncate=WEDGE_TRUNCATE):
        super().__init__(env)
        self.step_cost = float(step_cost)
        self.max_steps = int(max_steps)
        self.wedge_truncate = bool(wedge_truncate)
        self._t = 0
        _w = ("wedge ends the episode and is billed for the remaining time"
              if self.wedge_truncate else "only the step limit truncates")
        print(f"[TimeToGoal] step_cost={self.step_cost} "
              f"max_steps={self.max_steps} "
              f"-> return in [{-self.step_cost * self.max_steps:.1f}, 0.0]  "
              f"| only arrival terminates, {_w}")

    def reset(self, **kwargs):
        self._t = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        # The env's `done` (third element) is deliberately dropped -- see the
        # module docstring. Its `truncated` is always False.
        obs, _env_reward, _env_done, _env_trunc, info = self.env.step(action)
        self._t += 1

        # The env's OWN arrival flag, not a recomputed distance: the env
        # resamples the goal the moment it is reached, so recomputing the
        # distance here reports zero successes against a goal that has already
        # moved. That mistake cost hundreds of episodes earlier in this project.
        reached = bool(info.get("arrived", False) or info.get("is_success", False))

        # WEDGED -> END NOW, BILLED FOR EVERY STEP IT WOULD HAVE BURNED.
        #
        # A wedged vehicle is frozen under full brake and the flag is never
        # cleared, so it cannot reach the goal and WILL run to the step limit.
        # Simulating those steps costs wall-clock and teaches nothing: ~950
        # identical frozen frames per wedged episode.
        #
        # THE BILL IS EXACT, NOT A PENALTY. It pays for this step plus the
        # (max_steps - t) steps left on the clock, so the episode's total return
        # is -step_cost * max_steps -- byte-for-byte what it would have been had
        # the frozen steps actually been simulated. The objective is unchanged;
        # only the wall-clock is. That exactness is what stops this becoming an
        # escape: ending early is worth precisely nothing, so there is no
        # incentive to seek a wedge, which is the suicide-incentive trap that
        # sparse_reward.assert_no_suicide_incentive exists to catch.
        #
        # TERMINATED, NOT TRUNCATED, so masks=0 and the critic does NOT
        # bootstrap. Bootstrapping here would be the bug: the wedged state's
        # value would be whatever the critic happened to guess -- optimistic
        # early in training -- and wedging would look cheap. mask=0 pins it to
        # the billed value, which is the true one. It also removes the
        # unbounded gamma=1 bootstrap chain that a frozen episode would
        # otherwise contribute, since V(s) = -c + V(s') has no fixed point on a
        # state the agent never escapes.
        wedged = self.wedge_truncate and bool(info.get("wedged", False))

        if reached:
            reward, terminated, truncated = 0.0, True, False
        elif wedged:
            remaining = max(0, self.max_steps - self._t)
            reward = -self.step_cost * (1 + remaining)
            terminated, truncated = True, False
        else:
            reward = -self.step_cost
            terminated = False
            truncated = self._t >= self.max_steps

        info["is_success"] = reached
        info["ttg_wedged"] = wedged
        info["ttg_step"] = self._t
        # Collision and lane state stay in info for LOGGING. They no longer end
        # the episode, so the trainer's [Why] breakdown reports them as rates.
        info["ttg_collision"] = bool(info.get("collision", False))
        info["ttg_lane"] = bool(info.get("lane_invasion", False))
        return obs, float(reward), terminated, truncated, info
