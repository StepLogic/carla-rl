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

# COLLISION ENDS THE EPISODE, AND IS BILLED EXACTLY LIKE A WEDGE. Off by
# default, so the existing "collision costs only the time it loses" behaviour
# is unchanged everywhere it is not asked for.
#
# THE BILL IS NOT A PENALTY, IT IS THE TRAP THIS FEATURE WOULD OTHERWISE WALK
# INTO. With a per-step cost and a TERMINAL collision that simply stops the
# episode, crashing STOPS THE METER: a crash on step one returns -c, while
# driving honestly to the goal returns -c * 80. Crashing immediately would be
# the optimal policy. That is the suicide incentive
# sparse_reward.assert_no_suicide_incentive exists to catch, and this module's
# docstring records it as the reason collisions were made non-terminal in the
# first place.
#
# Billing every remaining step makes a crash cost exactly what running out of
# time costs -- return -c * max_steps either way -- so ending early is worth
# precisely nothing and there is no incentive to seek a crash. Identical
# reasoning, and identical arithmetic, to the wedge above.
COLLISION_TERMINAL = os.environ.get("COLLISION_TERMINAL", "0") == "1"

# POTENTIAL-BASED SHAPING ON ALONG-ROAD DISTANCE. 0.0 disables it.
#
#   Phi(s) = -POTENTIAL_K * route_remaining_m(s)
#   F      = Phi(s') - Phi(s) = POTENTIAL_K * (remaining(s) - remaining(s'))
#
# so the agent is paid POTENTIAL_K per metre of route progress and charged the
# same for backtracking. It telescopes to a bounded constant over an episode,
# so it cannot be farmed by driving in circles, and by Ng et al. (1999) it
# leaves the optimal policy unchanged -- it only moves credit earlier in time,
# which is the whole point when the alternative is one terminal bit per episode.
#
# THE VALUE FUNCTION IS NO LONGER RAW NEGATIVE TIME:
#
#   V(s) = -c * E[steps] + POTENTIAL_K * remaining(s) - E[Phi-at-terminal]
#
# DECIDED 2026-09-10: FEED -V TO THE SORB GRAPH UNMODIFIED. Do not subtract the
# potential. At the measured 0.283 m/step, ideal steps ~= 3.53 * remaining, so
#
#   -V ~= 3.53*r - K*r = 2.53*r   at K=1, c=1
#
# which is (a) POSITIVE, so Dijkstra is safe -- that needs K/c < 3.53, the same
# ceiling this file's K already respects; (b) a LINEAR RESCALE of true steps by
# 1 - K/(c*3.53) = 0.717, and Dijkstra sums edge costs, so path RANKING is
# exactly preserved; and (c) departing in the useful direction where the policy
# is unreliable -- E[steps] inflates on a hard edge because failures are billed
# the full clock, while `remaining` does not move, so hard edges cost more.
#
# THE PRICE IS THE CUTOFF, NOT THE WEIGHTS. H_STEPS=106 was derived for unshaped
# V-units; on a 0.717 rescale the equivalent is ~76, and edge_reject_mask will
# otherwise reject edges it should keep. Recalibrate per checkpoint, never carry
# a learned-distance cutoff across one. Two further consequences: no absolute
# step count can be quoted from this critic, and a POTENTIAL_K=0 critic is on a
# different scale and must never share a table with it.
#
# The alternative -- subtracting K * remaining for an exact recovery -- was
# rejected because the exact form needs GlobalRoutePlanner, i.e. the road graph
# the mapping is supposed to PRODUCE.
POTENTIAL_K = float(os.environ.get("POTENTIAL_K", 0.0))

# CHAIN OF GOALS. 1 = stop at the first arrival, which is every existing run.
#
# The ENV already chains by itself: on arrival within 2 m it sets passed_chckp,
# increments checkpoint_count and calls core.resample_goal(), which draws the
# next goal from the SAME curriculum band (GOAL_DIST_MIN/MAX). What stops the
# chain is this wrapper -- `reached` terminates the episode on the first goal,
# so the env's freshly resampled goal is never driven to.
#
# CHAIN_GOALS > 1 makes the wrapper keep going until that many legs are done.
# Arrival still pays 0 for the step, so the return remains -c * total steps over
# the whole chain and V stays a cost-to-go over the remaining legs.
#
# TWO COUPLED LIMITS MUST BE RAISED WITH IT, or the chain ends for reasons that
# have nothing to do with the policy:
#   TTG_MAX_STEPS   the whole chain shares one clock. N goals at up to
#                   GOAL_DIST_MAX metres need N * MAX/0.283 steps of perfect
#                   driving.
#   MAX_DISTANCE_M  the env's own cumulative-distance terminator, default 200.
#                   The config says it "must exceed chain_max_goals *
#                   max_goal_distance"; at 5 x 50 m the default would end the
#                   episode mid-chain and report it as done_dist.
CHAIN_GOALS = int(os.environ.get("CHAIN_GOALS", 1))


class TimeToGoalReward(Wrapper):
    """Every step costs. Arrival stops the clock. Nothing else ends an episode."""

    def __init__(self, env, step_cost=STEP_COST, max_steps=TTG_MAX_STEPS,
                 wedge_truncate=WEDGE_TRUNCATE,
                 collision_terminal=COLLISION_TERMINAL,
                 potential_k=POTENTIAL_K, chain_goals=CHAIN_GOALS):
        super().__init__(env)
        self.step_cost = float(step_cost)
        self.max_steps = int(max_steps)
        self.wedge_truncate = bool(wedge_truncate)
        self.collision_terminal = bool(collision_terminal)
        self.potential_k = float(potential_k)
        self.chain_goals = max(1, int(chain_goals))
        self._t = 0
        self._goals = 0
        self._prev_remaining = None
        _w = ("wedge ends the episode and is billed for the remaining time"
              if self.wedge_truncate else "only the step limit truncates")
        _c = (", collision ends it and is billed the same"
              if self.collision_terminal else "")
        _p = (f", shaping {self.potential_k:g}/m of route progress"
              if self.potential_k else "")
        _g = (f", CHAIN of {self.chain_goals} goals"
              if self.chain_goals > 1 else "")
        print(f"[TimeToGoal] step_cost={self.step_cost} "
              f"max_steps={self.max_steps} "
              f"-> return in [{-self.step_cost * self.max_steps:.1f}, 0.0]  "
              f"| only arrival terminates, {_w}{_c}{_p}{_g}")

    def reset(self, **kwargs):
        self._t = 0
        self._goals = 0
        # The potential is a DIFFERENCE, so the first step of an episode has
        # nothing to difference against and must pay no shaping. Leaving a
        # stale value here would pay the whole route length as a one-off reward
        # on step 1 of every episode.
        self._prev_remaining = None
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
        # SAME BILLING AS THE WEDGE, and for the same reason -- see
        # COLLISION_TERMINAL above. A terminal collision that merely stopped the
        # episode would make crashing on step one optimal.
        crashed = self.collision_terminal and bool(info.get("collision", False))

        if reached:
            # A leg is done. Only the LAST one ends the episode; the env has
            # already resampled the next goal, so the chain continues from here.
            self._goals += 1
            reward = 0.0
            terminated = self._goals >= self.chain_goals
            truncated = (not terminated) and self._t >= self.max_steps
            # THE ROUTE IS REPLANNED ON ARRIVAL, so `remaining` jumps from ~0 to
            # the whole of the NEXT leg. Differencing across that boundary would
            # charge the agent K * (next leg length) for having arrived -- a
            # large negative exactly when it did the right thing. Dropping the
            # anchor makes the first step of each leg pay no shaping, the same
            # rule as the first step of an episode.
            #
            # Only reachable with CHAIN_GOALS > 1: at 1 an arrival terminates
            # and the shaping is skipped anyway, which is why this survived.
        elif wedged or crashed:
            remaining = max(0, self.max_steps - self._t)
            reward = -self.step_cost * (1 + remaining)
            terminated, truncated = True, False
        else:
            reward = -self.step_cost
            terminated = False
            truncated = self._t >= self.max_steps

        # POTENTIAL-BASED SHAPING, ADDED TO WHATEVER THE BASE REWARD WAS.
        #
        # Phi = -k * route_remaining_m, so F = k * (prev_remaining - remaining):
        # paid per metre of route progress, charged for backtracking.
        #
        # NO SHAPING ON A TERMINAL STEP. Ng et al.'s invariance needs
        # Phi(terminal) = 0, and neither an arrival (remaining ~ 0 but snapped)
        # nor a crash (remaining large) satisfies it. Paying the difference into
        # a terminal step would hand a crash near the goal a bonus over a crash
        # far from it, which is a change to the objective rather than a
        # redistribution of credit within it. Skipping it keeps the billed
        # terminal returns EXACT, which is what stops the wedge and the crash
        # becoming escapes.
        remaining = info.get("route_remaining_m", None)
        shaped = 0.0
        if reached:
            self._prev_remaining = None          # see the note above
        if self.potential_k and not (terminated or truncated or reached):
            if remaining is not None and self._prev_remaining is not None:
                shaped = self.potential_k * (self._prev_remaining - float(remaining))
                reward += shaped
        self._prev_remaining = (None if remaining is None else float(remaining))

        # SUCCESS IS THE WHOLE CHAIN, not one leg -- otherwise a 5-goal
        # episode that completed 1 would report the same as one that
        # completed 5. The per-leg count is published alongside it.
        info["is_success"] = bool(self._goals >= self.chain_goals)
        info["ttg_goals"] = int(self._goals)
        info["ttg_wedged"] = wedged
        info["ttg_crashed"] = crashed
        info["ttg_shaping"] = float(shaped)
        info["ttg_step"] = self._t
        # Collision and lane state stay in info for LOGGING. They no longer end
        # the episode, so the trainer's [Why] breakdown reports them as rates.
        info["ttg_collision"] = bool(info.get("collision", False))
        info["ttg_lane"] = bool(info.get("lane_invasion", False))
        return obs, float(reward), terminated, truncated, info
