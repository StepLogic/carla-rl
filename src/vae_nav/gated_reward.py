"""One env serving TWO modes: goal-reaching and goal-free driving.

The mode is COMMANDED, not learned. It is drawn once per episode at reset (or
forced by an evaluator) and published in info["mode"]: 1 = goal, 0 = explore.
This mirrors the peer's gated point-goal policy, where the same bit is an actor
input, selects which critic supplies the value, and stratifies the replay.

WHY BOTH REWARDS ARE COMPUTED IN ONE WRAPPER rather than switching between two
wrapper objects. A gym wrapper is fixed at construction; swapping it per episode
would mean rebuilding the env on every reset, which in CARLA costs a world
reload. Both rewards are pure functions of the same `info`, so computing both
and selecting is cheap and keeps the two definitions side by side where they can
be compared.

THE TWO REWARDS ARE NOT ON THE SAME SCALE, AND THAT IS THE POINT OF TWO CRITICS.

    goal mode, "sorb" (DEFAULT)  minimum time: -STEP_COST every step, 0 on
                  arrival. Return is -(steps to goal), in [-step_cost*cap, 0].
    goal mode, "shaped"          LaneFollowReward's delta_distance +
                  projected_distance, speed-gated. Returns of order +100.
    explore mode  the goal-free half: lane projection, base penalty,
                  steer-change, terminal penalties. Returns of order +10.

A single critic fitted to any of these pairings would be dominated by one side.

WHY TWO CRITICS -- THE ANSWER DEPENDS ON WHICH GOAL REWARD IS SELECTED, and an
earlier version of this docstring asserted one case as though it were the only
one. Both are recorded here because the file supports both.

  under "sorb"    The goal arm runs gamma=1.0 (its Q must read as time) and the
                  explore arm 0.99. The discounts differ AND the returns differ
                  in SIGN, so one critic could not serve both even at equal
                  scale. This is the Habitat peer's rationale, and under this
                  reward it DOES transfer.

  under "shaped"  Both arms run gamma=0.99 -- run_lanefollow.sh never sets
                  DISCOUNT and inherits the trainer's 0.99, run_lanefollow_nogoal.sh
                  sets it explicitly. The horizons are then identical and the
                  Habitat rationale does NOT apply; the split is justified by
                  return SCALE alone, order 100 against order 10, and by TQC's
                  kappa being derived from that scale.

WHAT IS DELIBERATELY SHARED: the observation shape. In explore mode the goal
block is ZEROED rather than removed, because one actor cannot take two different
input shapes. That is the dead-input cost of gating, and it is the thing the
separate goal-free policy (OBS_WITH_GOAL=0, 132-d) exists to be compared against.
"""
import os

import numpy as np

try:
    from gymnasium import Wrapper
except ImportError:
    from gym import Wrapper

from vae_nav.lane_explore_reward import (BASE_PENALTY, COLLISION_PENALTY,
                                         LANE_DEPARTURE_PENALTY, PROGRESS_MODE,
                                         lane_departure_is_fatal)

GOAL, EXPLORE = 1, 0
EXPLORE_PROB = float(os.environ.get("EXPLORE_PROB", 0.5))

# ---- THE GOAL ARM'S REWARD --------------------------------------------------
#
# "sorb" (default): MINIMUM TIME. -STEP_COST every step, 0 on arrival. The
# critic then IS the cost-to-go -- Q = -(steps to goal) -- which is the reading
# a topological map builder needs, and which no shaped reward provides. Matches
# the Habitat peer's gated point-goal arm.
#
# "shaped": the original LaneFollowReward shape, delta_distance +
# projected_distance, kept so the two can be compared without a code change.
#
# THREE THINGS FOLLOW FROM CHOOSING sorb, AND ALL THREE ARE LOAD-BEARING.
#
# 1. THE GOAL CRITIC MUST RUN AT gamma = 1.0. At 0.99 the horizon is
#    1/(1-gamma) = 100 steps, so Q saturates near -100 and cannot separate a
#    400-step goal from a 1000-step one -- measured in this repo at 0.18 apart.
#    The whole point of a minimum-time reward is that Q reads as time, and 0.99
#    destroys exactly that. The explore critic stays at 0.99.
#
# 2. THIS REVIVES THE HABITAT TWO-CRITIC RATIONALE, which the shaped reward did
#    not. Under "shaped" both arms ran at 0.99 and the split was justified by
#    return SCALE alone. Under "sorb" the discounts genuinely differ, 1.0
#    against 0.99, and one critic could not serve both even at equal scale.
#
# 3. gamma = 1.0 HAS NO BOUNDED FIXED POINT on a state the agent never escapes,
#    so the goal critic needs value bounds. They are applied PER CRITIC in the
#    gated learner, never through the global TQC_TARGET_MIN/MAX: those are
#    module-level in critic_updater and would clip the explore critic's
#    POSITIVE targets to zero, silently destroying it.
#
# WHY COLLISIONS DO NOT END A GOAL EPISODE HERE. With a per-step cost and a
# TERMINAL collision, crashing stops the meter, so crashing on step one is
# optimal. Letting the episode continue removes the escape entirely: a car that
# hits a wall and cannot recover simply pays the full bill. The penalty for
# crashing IS the time it costs, and it needs no coefficient. This mirrors
# time_to_goal_reward.py, which learned it the hard way.
GOAL_REWARD_KIND = os.environ.get("GATED_GOAL_REWARD", "sorb").lower()
if GOAL_REWARD_KIND not in ("sorb", "shaped"):
    raise ValueError(f"GATED_GOAL_REWARD={GOAL_REWARD_KIND!r} is not sorb|shaped")
STEP_COST = float(os.environ.get("STEP_COST", 1.0))
GOAL_MAX_STEPS = int(os.environ.get("GATED_GOAL_MAX_STEPS", 1000))
WEDGE_TRUNCATE = os.environ.get("WEDGE_TRUNCATE", "1") == "1"

# LANE_INVASION_NOOP=1 MAKES CROSSING A LANE MARKING COST NOTHING AND END
# NOTHING. Off-road is NOT covered by it and stays fatal -- see below.
#
# WHERE IT APPLIES. Goal mode under "sorb" already ignores lane state entirely:
# the minimum-time reward looks only at arrival and the wedge, so a crossing is
# already a no-op there and this flag changes nothing. It bites on the EXPLORE
# half, where a crossing currently costs LANE_DEPARTURE_PENALTY (5.0) and
# terminates the episode, and on the "shaped" goal reward's lane term.
#
# WHAT IT REMOVES, STATED PLAINLY. lane_explore_reward.py's own docstring says
# the lane-departure penalty is "the ONLY thing discouraging" a car that drives
# tight circles, because the progress term pays the same for circling as for
# driving down the road. Turning it off removes that discouragement. OFF_ROAD IS
# DELIBERATELY LEFT FATAL as the remaining backstop: leaving the drivable
# surface is not "crossing a lane", and without at least one such term the
# explore arm has no incentive to stay on the road at all.
#
# COMPARABILITY WARNING FOR THE ABLATION. The goal-free baseline
# (carla_lanefollow_nogoal, checkpoint_350000) was trained WITH the lane penalty
# active. A gated arm trained without it differs from that baseline by more than
# the gate, which is the one thing the three-way comparison is supposed to
# isolate. Either the baseline is retrained with the same setting, or the
# explore halves are not directly comparable and the writeup has to say so.
LANE_INVASION_NOOP = os.environ.get("LANE_INVASION_NOOP", "0") == "1"
# Terminal penalties for the GOAL arm, matching LaneFollowReward's defaults.
LF_K_LANE = float(os.environ.get("LF_K_LANE", 1.0))
LF_K_COLLISION = float(os.environ.get("LF_K_COLLISION", 1.0))
LF_K_IDLE = float(os.environ.get("LF_K_IDLE", 1.0))
LF_K_STEER = float(os.environ.get("LF_K_STEER_CHANGE", 1.0))
LF_K_GOAL_BONUS = float(os.environ.get("LF_K_GOAL_BONUS", 1.0))


def goal_reward(info, delta_mode="positive"):
    """LaneFollowReward's shape, as a pure function of one step's info."""
    d = float(info.get("delta_distance", 0.0) or 0.0)
    if delta_mode == "positive":
        d = max(d, 0.0)
    elif delta_mode == "abs":
        d = abs(d)
    proj = float(info.get("projected_distance", 0.0) or 0.0)
    steer_change = float(info.get("action_smoothness_penalty", 0.0) or 0.0)
    speed_factor = float(info.get("speed_factor", 0.0) or 0.0)

    r = -LF_K_STEER * steer_change
    if speed_factor < 1.0:
        r += d + proj
    if info.get("collision"):
        r -= LF_K_COLLISION
    # Under LANE_INVASION_NOOP a marking crossing is free here too; off-road
    # still bills, for the same reason it does in explore mode.
    if info.get("off_road") or (info.get("lane_any") and not LANE_INVASION_NOOP):
        r -= LF_K_LANE
    if info.get("is_success"):
        r += LF_K_GOAL_BONUS
    return float(np.nan_to_num(r, nan=0.0))


def goal_reward_sorb(info, t, max_steps=GOAL_MAX_STEPS, step_cost=STEP_COST,
                     wedge_truncate=WEDGE_TRUNCATE):
    """Minimum time. Returns (reward, terminated, truncated).

    Mirrors time_to_goal_reward.TimeToGoalReward exactly, as a pure function of
    one step's info plus the step index, so the gated wrapper can select it per
    episode without swapping wrapper objects.

    ONLY ARRIVAL TERMINATES (and a wedge, billed exactly). The env's own `done`
    -- idle, falling, distance, lane invasion, collision, speed, wrong direction
    -- is deliberately discarded, because with a per-step cost any terminal
    failure stops the meter and becomes an escape.
    """
    # THE ENV'S OWN ARRIVAL FLAG, not a recomputed distance: the env resamples
    # the goal the moment it is reached, so recomputing here reports zero
    # successes against a goal that has already moved.
    reached = bool(info.get("arrived", False) or info.get("is_success", False))
    wedged = wedge_truncate and bool(info.get("wedged", False))

    if reached:
        return 0.0, True, False, True
    if wedged:
        # THE BILL IS EXACT, NOT A PENALTY: this step plus every step left on
        # the clock, so the episode's return is exactly -step_cost * max_steps,
        # identical to simulating the frozen frames. Ending early is worth
        # precisely nothing, so a wedge is not an escape. TERMINATED, so the
        # critic does not bootstrap an unbounded gamma=1 chain from a state the
        # agent never leaves.
        remaining = max(0, int(max_steps) - int(t))
        return -step_cost * (1 + remaining), True, False, False
    return -step_cost, False, int(t) >= int(max_steps), False


def explore_reward(info, disp):
    """The goal-free half. Same terms as LaneExploreReward, novelty disabled."""
    if PROGRESS_MODE == "projected":
        fwd = np.asarray(info.get("wp_forward", (0.0, 0.0)), np.float64)
        progress = float(np.dot(disp, fwd))
    elif PROGRESS_MODE == "heading":
        yaw = float(info.get("hero_yaw", 0.0) or 0.0)
        progress = float(disp[0] * np.cos(yaw) + disp[1] * np.sin(yaw))
    elif PROGRESS_MODE == "raw":
        progress = float(np.linalg.norm(disp))
    else:
        progress = 0.0

    steer_change = float(info.get("action_smoothness_penalty", 0.0) or 0.0)
    speed_factor = float(info.get("speed_factor", 0.0) or 0.0)
    r = -BASE_PENALTY - steer_change
    if speed_factor < 1.0:
        r += progress
    if info.get("collision"):
        r -= COLLISION_PENALTY
    if LANE_INVASION_NOOP:
        # Crossing a marking costs nothing and ends nothing. OFF-ROAD STILL
        # DOES: it is not a lane crossing, and it is the only term left that
        # discourages leaving the road, since the progress term pays the same
        # for circling as for driving down it.
        fatal = bool(info.get("off_road", False))
    else:
        fatal = lane_departure_is_fatal(
            bool(info.get("lane_any", False)), info.get("lane_marking_types"),
            bool(info.get("in_junction", False)),
            bool(info.get("off_road", False)))
    if fatal:
        r -= LANE_DEPARTURE_PENALTY
    return float(np.nan_to_num(r, nan=0.0)), fatal


class GatedReward(Wrapper):
    """Draw a mode per episode; return that mode's reward and termination."""

    def __init__(self, env, explore_prob=EXPLORE_PROB, force_mode=None,
                 max_steps=500, delta_mode="positive", seed=0,
                 goal_kind=GOAL_REWARD_KIND, goal_max_steps=GOAL_MAX_STEPS,
                 step_cost=STEP_COST, wedge_truncate=WEDGE_TRUNCATE):
        super().__init__(env)
        self.explore_prob = float(explore_prob)
        self.force_mode = force_mode
        self.max_steps = int(max_steps)
        self.delta_mode = delta_mode
        self.goal_kind = str(goal_kind).lower()
        self.goal_max_steps = int(goal_max_steps)
        self.step_cost = float(step_cost)
        self.wedge_truncate = bool(wedge_truncate)
        self._rng = np.random.default_rng(seed)
        self.mode = GOAL
        self._t = 0
        self._prev = np.zeros(2)
        if self.goal_kind == "sorb":
            lo = -self.step_cost * self.goal_max_steps
            print(f"[Gated] goal=SORB minimum-time step_cost={self.step_cost} "
                  f"goal_max_steps={self.goal_max_steps} "
                  f"-> goal return in [{lo:.1f}, 0.0]; only arrival terminates. "
                  f"REQUIRES DISCOUNT=1.0 for the goal critic.")
        else:
            print(f"[Gated] goal=SHAPED (delta+projected), returns ~order 100")
        print(f"[Gated] explore_prob={self.explore_prob} "
              f"force_mode={self.force_mode} explore_max_steps={self.max_steps} "
              f"progress={PROGRESS_MODE}")

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        obs, info = out if isinstance(out, tuple) else (out, {})
        # DRAWN ONCE PER EPISODE, never mid-episode: the critics are per-mode, so
        # a transition's mode must be constant for its whole return.
        self.mode = (int(self.force_mode) if self.force_mode is not None
                     else (EXPLORE if self._rng.random() < self.explore_prob else GOAL))
        self._t = 0
        self._prev = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
        info["mode"] = self.mode
        return obs, info

    def step(self, action):
        obs, _r, term, trunc, info = self.env.step(action)
        self._t += 1
        pos = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
        disp = pos - self._prev
        self._prev = pos

        if self.mode == GOAL and self.goal_kind == "sorb":
            # MINIMUM TIME. The env's own `done` is discarded on purpose: with a
            # per-step cost, any terminal failure stops the meter and crashing
            # immediately becomes optimal. Only arrival ends the episode, so a
            # car that hits a wall simply pays the remaining bill.
            reward, terminated, truncated, arrived = goal_reward_sorb(
                info, self._t, self.goal_max_steps, self.step_cost,
                self.wedge_truncate)
            info["is_success"] = arrived
        elif self.mode == GOAL:
            reward = goal_reward(info, self.delta_mode)
            terminated = bool(term)          # the env's own goal-arm rules
            truncated = bool(trunc)
        else:
            reward, fatal = explore_reward(info, disp)
            # The env's `done` fires on any lane crossing, which would end an
            # episode every time the car turned through a junction. Explore mode
            # re-derives it with the junction exemption, exactly as
            # LaneExploreReward does.
            terminated = bool(info.get("collision", False) or fatal)
            truncated = (not terminated) and self._t >= self.max_steps

        info["mode"] = self.mode
        info["gated_reward"] = reward
        return obs, reward, terminated, truncated, info
