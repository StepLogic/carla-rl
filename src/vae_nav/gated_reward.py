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
                                         LANE_DEPARTURE_PENALTY, PROGRESS_MODE, PROGRESS_SCALE,
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
# PORTED FROM time_to_goal_reward.py 2026-09-10 so the two arms share ONE goal
# reward. This module is a pure-function MIRROR of that wrapper, and the mirror
# had silently drifted: collision-terminal and the route-progress shaping were
# added there and not here, so the gated arm would have trained goal mode on the
# older reward while every log said the two matched.
#
# COLLISION ENDS THE EPISODE AND IS BILLED THE REMAINING CLOCK, exactly like the
# wedge. The bill is not a penalty -- with a per-step cost and a terminal that
# merely STOPS the episode, crashing on step one returns -c against -c*N for
# driving honestly, so crashing immediately would be optimal.
COLLISION_TERMINAL = os.environ.get("COLLISION_TERMINAL", "0") == "1"
# GOAL MODE ONLY. A lane departure ends the goal episode and is billed the
# remaining clock. Off by default so the existing gated arms are unchanged.
# Explore mode is untouched: its lane handling stays in explore_reward, and
# LANE_INVASION_NOOP still governs it.
LANE_TERMINAL = os.environ.get("GATED_GOAL_LANE_TERMINAL", "0") == "1"
# GOAL MODE ("sorb" arm) ARRIVAL BONUS (v9, user 2026-09-21: "add +10 for
# goal"). Paid on the arrival step INSTEAD of the old 0.0; default 0.0 keeps
# every earlier arm identical. The explore arm's EXPLORE_ARRIVAL_BONUS is a
# separate constant and is untouched.
GOAL_ARRIVAL_BONUS = float(os.environ.get("GOAL_ARRIVAL_BONUS", 0.0))
# v9 COUNTING REGIME (user 2026-09-21: "only truncate after N steps offroad,
# 2 collisions or 2 lane crossings", later "make it 20" -- ALL THREE
# thresholds are 20 STEPS, counted per step, cumulative within one goal
# episode; event-dedup was offered and declined). When GOAL_TRUNCATE_COUNTS=1
# the goal arm's terminal bills are REPLACED by counters the wrapper keeps:
#   nothing terminates except arrival (+GOAL_ARRIVAL_BONUS, terminated=True);
#   the episode truncates (truncated=True, mask=1.0, the critic bootstraps
#   through the cut) when ANY of
#     off-road steps  >= GOAL_OFFROAD_TRUNCATE_STEPS   (20)
#     collision steps >= GOAL_COLLISION_TRUNCATE_STEPS (20)
#     crossing steps  >= GOAL_CROSSING_TRUNCATE_STEPS  (20)
#   and the truncating step pays its NORMAL per-step reward -- no
#   remaining-clock bill, the bootstrap carries the cost (user's pick).
# GOAL_COLLISION_PENALTY (default 10.0) fines each collision step, consistent
# with the 10.0 lane/off-road fines; under the old bill regime collision was
# billed terminally, so no per-step constant existed for it before.
GOAL_TRUNCATE_COUNTS = os.environ.get("GOAL_TRUNCATE_COUNTS", "0") == "1"
GOAL_OFFROAD_TRUNCATE_STEPS = int(os.environ.get("GOAL_OFFROAD_TRUNCATE_STEPS", 20))
GOAL_COLLISION_TRUNCATE_STEPS = int(os.environ.get("GOAL_COLLISION_TRUNCATE_STEPS", 20))
GOAL_CROSSING_TRUNCATE_STEPS = int(os.environ.get("GOAL_CROSSING_TRUNCATE_STEPS", 20))
GOAL_COLLISION_PENALTY = float(os.environ.get("GOAL_COLLISION_PENALTY", 10.0))
# GOAL-MODE TARGET-VELOCITY BONUS (user 2026-09-22: "add a term to the goal
# reward that favors moving at target velocity in the lane", "make it
# positive", "make it max out at 0.5"). On a CLEAN step -- on-road and no
# lane crossing THIS step -- a bonus of
#     GOAL_SPEED_BONUS * max(0, 1 - |speed_kmh - target_speed_kmh|
#                               / target_speed_kmh)
# is ADDED to the step reward: it peaks at GOAL_SPEED_BONUS (0.5) when the
# hero moves exactly at the env's target speed, tapers linearly to 0 at half
# and double the target, and is floored at 0 so an absurd speed can never
# make the term negative. Fault steps pay their fines and NO bonus (the
# bonus is a preference about how well you drive, not a discount on driving
# badly). Default 0.0 keeps every earlier run byte-identical. The speeds are
# the env's own published per-step values (core_state.py: speed_kmh,
# target_speed_kmh); a missing or non-positive target disables the term.
GOAL_SPEED_BONUS = float(os.environ.get("GOAL_SPEED_BONUS", 0.0))
# POTENTIAL-BASED SHAPING on ALONG-ROAD progress. Phi = -K * route_remaining_m,
# so the agent is paid K per metre of route progress and charged for
# backtracking. Telescopes to a bounded constant, so it cannot be farmed, and by
# Ng et al. (1999) the optimum is unchanged -- it only moves credit earlier.
#
# ROAD distance, not euclidean: on the CORL Town02 set one trajectory's
# endpoints are 39.8 m apart and 621.6 m by road, so a straight-line potential
# points the wrong way in a fifth of episodes.
#
# -V stays usable as an SGM edge weight WITHOUT subtracting the potential: it
# remains positive while K/c < 3.53, and is a linear rescale of true steps by
# 1 - K/(c*3.53), so Dijkstra's path ranking is exact. The CUTOFF must be
# recalibrated instead. See time_to_goal_reward.POTENTIAL_K for the derivation.
POTENTIAL_K = float(os.environ.get("POTENTIAL_K", 0.0))

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
    if info.get("off_road") or (info.get("lane_any",
            info.get("lane_invasion_any", False)) and not LANE_INVASION_NOOP):
        r -= LF_K_LANE
    if info.get("is_success"):
        r += LF_K_GOAL_BONUS
    return float(np.nan_to_num(r, nan=0.0))


def goal_lane_departed(info):
    """Is this step a FATAL lane departure for the goal arm? Pure; one source.

    THE ONE DEFINITION both reward paths use: goal_reward_sorb below, and the
    trainer's HER record call, which stages it per step so hindsight rows are
    recosted by the same rule (episode_her). Two copies of this test would be
    two reward functions over the same states.

    lane_departure_is_fatal with LANE_STRICT off -- the junction-forgiving rule
    the explore arm and the env wedge already share:
      Broken line on open road          -> fatal (the car left its lane)
      Broken line inside a junction     -> forgiven (that is the turn)
      every mark unpainted (NONE)       -> forgiven (the road edge; making it
                                           fatal took the scripted oracle to
                                           0.000 success at 100 m goals)
      illegal mark, or no marking data  -> fatal (fails closed)
    off_road is passed as False ON PURPOSE: this is the LANE rule, and off-road
    keeps its own per-step fine and step cap.
    """
    return bool(lane_departure_is_fatal(
        bool(info.get("lane_any", info.get("lane_invasion_any", False))),
        info.get("lane_marking_types"),
        bool(info.get("in_junction", False)),
        False))


def goal_reward_sorb(info, t, max_steps=GOAL_MAX_STEPS, step_cost=STEP_COST,
                     wedge_truncate=WEDGE_TRUNCATE,
                     collision_terminal=COLLISION_TERMINAL,
                     offroad_steps=0, collision_steps=0, crossing_steps=0):
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
    # Billed identically to the wedge -- see COLLISION_TERMINAL above.
    crashed = collision_terminal and bool(info.get("collision", False))

    # LANE DEPARTURE IS TERMINAL IN GOAL MODE, AND IT IS DECIDED HERE, NOT IN
    # THE ENV.
    #
    # The time-to-goal arm gets this from LANE_WEDGE, which lives in
    # get_done_status and freezes the vehicle. That flag is MODE-BLIND: the env
    # runs the same code for both modes, so enabling it would wedge the car
    # during EXPLORE episodes too, where a crossing is meant to cost nothing and
    # end nothing. Deciding it in this function instead keeps every mode-specific
    # rule in the gated wrapper, which is the whole point of the wrapper.
    #
    # Behaviourally identical for goal mode: WEDGE_TRUNCATE ends the episode on
    # the same step, so the freeze never gets to matter -- it is the BILL that
    # does the work, and it is charged below.
    #
    # Same junction-forgiving filter the explore half and the env wedge use, so
    # the three cannot disagree about what counts as a departure.
    departed = LANE_TERMINAL and lane_departure_is_fatal(
        bool(info.get("lane_any",
                                 info.get("lane_invasion_any", False))), info.get("lane_marking_types"),
        bool(info.get("in_junction", False)),
        bool(info.get("off_road", False)))

    if reached:
        # v9 (user 2026-09-21): arrival pays GOAL_ARRIVAL_BONUS (default 0.0
        # keeps every earlier run byte-identical). The ONLY termination: every
        # other end is a truncation, so the critic bootstraps through it.
        return GOAL_ARRIVAL_BONUS, True, False, True
    if GOAL_TRUNCATE_COUNTS:
        # LANE INVASION IS TERMINAL AND BILLED IN THE COUNTING REGIME TOO
        # (user 2026-09-26: "make lane invasion terminal and bill remaining
        # steps in goal mode"). GATED_GOAL_LANE_TERMINAL existed, but only the
        # LEGACY branch below ever read `departed`; this branch returned first,
        # so the flag was silently inert for every counting-regime run.
        #
        # THE BILL IS THE LEGACY ONE, EXACT: this step plus every step left on
        # the clock at the step cost, so the episode's return is what it would
        # be had the car sat there until time ran out. That makes a crossing
        # precisely as bad as never arriving -- never an escape from a costly
        # episode, never a punishment larger than the clock. The lane FINE is
        # deliberately NOT multiplied in: 11 x 1000 would sit far below the
        # goal critic's TQC_VALUE_FLOOR and be clipped, which breaks exactness.
        #
        # TERMINATED (mask 0): the bill carries the whole future, so the critic
        # must not also bootstrap one. Checked AFTER arrival (arrival wins) and
        # BEFORE the fault caps, so the first fatal crossing ends the episode.
        # HER_RECOST mirrors this through goal_lane_departed and cuts the
        # relabelled leg at the same step (episode_her._counting_end).
        if LANE_TERMINAL and goal_lane_departed(info):
            remaining = max(0, int(max_steps) - int(t))
            return -step_cost * (1 + remaining), True, False, False
        # v9 counting regime (see the constants block): per-step fines, no
        # terminal bills, truncate when any cumulative count trips. The
        # truncating step pays its normal reward -- the wrapper's counters
        # (offroad/collision/crossing steps) arrive as kwargs.
        _r = -step_cost
        if GOAL_OFF_ROAD_PENALTY and info.get("off_road"):
            _r -= GOAL_OFF_ROAD_PENALTY
        if GOAL_LANE_PENALTY and info.get("lane_any",
                                          info.get("lane_invasion_any", False)):
            _r -= GOAL_LANE_PENALTY
        if GOAL_COLLISION_PENALTY and info.get("collision", False):
            _r -= GOAL_COLLISION_PENALTY
        # TARGET-VELOCITY BONUS (see GOAL_SPEED_BONUS above): only on a CLEAN
        # step -- on-road, no crossing and no collision THIS step; a fault
        # step pays its fines and no bonus. Floored at 0 so an absurd speed
        # cannot make the term negative; a missing or non-positive target
        # disables it.
        if GOAL_SPEED_BONUS:
            _clean = (not info.get("off_road")
                      and not info.get("lane_any",
                                       info.get("lane_invasion_any", False))
                      and not info.get("collision", False))
            if _clean:
                _v = float(info.get("speed_kmh", 0.0) or 0.0)
                _vt = float(info.get("target_speed_kmh", 0.0) or 0.0)
                if _vt > 0.0:
                    _r += GOAL_SPEED_BONUS * max(
                        0.0, 1.0 - abs(_v - _vt) / _vt)
        truncated = (int(t) >= int(max_steps)
                     or offroad_steps >= GOAL_OFFROAD_TRUNCATE_STEPS
                     or collision_steps >= GOAL_COLLISION_TRUNCATE_STEPS
                     or crossing_steps >= GOAL_CROSSING_TRUNCATE_STEPS)
        return _r, False, truncated, False
    if wedged or crashed or departed:
        # THE BILL IS EXACT, NOT A PENALTY: this step plus every step left on
        # the clock, so the episode's return is exactly -step_cost * max_steps,
        # identical to simulating the frozen frames. Ending early is worth
        # precisely nothing, so a wedge is not an escape. TERMINATED, so the
        # critic does not bootstrap an unbounded gamma=1 chain from a state the
        # agent never leaves. LEGACY REGIME (GOAL_TRUNCATE_COUNTS=0): kept
        # byte-identical to v4-v8 so old run_env snapshots replay exactly;
        # v9's truncate-everything rule lives in the counting branch above.
        remaining = max(0, int(max_steps) - int(t))
        return -step_cost * (1 + remaining), True, False, False
    # PER-STEP DRIVING PENALTIES (default 0.0). NOT applied to the terminal
    # bills above: those are exact by construction -- the wedge and crash bills
    # pay the remaining clock precisely so that ending early is never an escape
    # -- and adding to them would make the escape cheaper or dearer than the
    # clock it replaces, which is the one thing those bills exist to prevent.
    _r = -step_cost
    if GOAL_OFF_ROAD_PENALTY and info.get("off_road"):
        _r -= GOAL_OFF_ROAD_PENALTY
    if GOAL_LANE_PENALTY and info.get("lane_any",
                                      info.get("lane_invasion_any", False)):
        _r -= GOAL_LANE_PENALTY
    return _r, False, int(t) >= int(max_steps), False


# EXPLORE-MODE LANE RULE (user decision 2026-09-11): OFF-ROAD IS FATAL; a lane
# crossing is fatal unless EVERY marking crossed is in EXPLORE_EXEMPT_MARKS
# (default "NONE" -- the unpainted road edge, which is what a kerb-hop or a
# lane-edge brush reports). Painted markings (Broken, Solid...) stay fatal on
# open road; inside a junction a painted crossing is the turn and is forgiven
# (the repo's shared junction clause in lane_departure_is_fatal).
# Goal mode is untouched: it keeps GATED_GOAL_LANE_TERMINAL / LANE_STRICT.
#
# WHY. v4 trained with EVERY crossing fatal in explore mode and its explore
# half converged to "cross a line at step ~37": return rose from -55 to -10 as
# episode length fell from 184 to 40 steps, and at evaluation it drove a
# median 1.7 m. The 5.0 departure penalty was a cheap exit next to accumulated
# jitter cost. Exempting NONE removes roughly half of those exits (measured
# NONE@road 0.40-0.60 of explore deaths) while keeping the painted-line and
# off-road fences.
EXPLORE_EXEMPT_MARKS = frozenset(
    m.strip() for m in os.environ.get("EXPLORE_EXEMPT_MARKS", "NONE").split(",")
    if m.strip())
EXPLORE_OFF_ROAD_FATAL = os.environ.get("EXPLORE_OFF_ROAD_FATAL", "1") == "1"
# EXPLORE_LANE_MODE (user decision 2026-09-11, "lane invasion no-op is not a
# good idea"):
#   penalise  a crossing (per explore_lane_fatal: painted, on open road) costs
#             LANE_DEPARTURE_PENALTY and the episode CONTINUES; off-road ends
#             it. No cheap exit exists, so the only way to raise the return is
#             progress, and one crossing costs ~23 steps of progress -- a
#             lane-keeping explorer strictly out-earns a weaving one.
#   terminal  the crossing is also fatal (v4's rule; produced the 40-step trap).
# LANE_INVASION_NOOP=1 still overrides both (crossings free); kept for the
# earlier runs that used it, not for new ones.
# EXPLORE_GOAL_DIST_M: explore mode ARRIVES once the car has driven this much
# CUMULATIVE PATH (not net displacement), ending the episode with
# EXPLORE_ARRIVAL_BONUS: charged on top of that step's reward when the car has
# driven EXPLORE_GOAL_DIST_M of cumulative path, ending the episode. User
# decision 2026-09-11: "terminate exploration episode at 100m and given +2.0
# reward"; raised to 10.0 by user decision 2026-09-18 -- at 2.0 the bonus was
# SMALLER THAN ONE STEP'S PROGRESS (~2.2 at 8 km/h under scale 10), so arrival
# was mildly discouraged and the bonus was invisible at gamma=0.99 against
# returns of ~1100.
#
# WHY 100 m IS THE RIGHT TARGET AND NOT A CAP CHANGE. Measured on v6c@75k over
# five episodes: explore drives at 2.0-2.35 m/s, so a surviving 500-step (50 s)
# episode covers 100-117 m. 100 m is therefore reachable INSIDE the existing
# cap -- the binding constraint was survival, not the step budget, so the cap
# stays at GATED_EXPLORE_MAX_STEPS and this becomes a positive terminal.
# Set EXPLORE_GOAL_DIST_M=0 to disable (pre-2026-09-11 behaviour).
# EXPLORE_OFF_ROAD_PENALTY: charged EVERY STEP the car is off the drivable
# surface, on top of whatever else that step earns. DEFAULT 0.0 (off), so every
# existing run is unchanged.
#
# READ THIS BEFORE SETTING IT. With EXPLORE_OFF_ROAD_FATAL=1 -- the current
# default -- the episode ENDS on the first off-road step, so a "per-step"
# penalty can only ever be charged ONCE, and it simply adds to the terminal
# LANE_DEPARTURE_PENALTY. For it to behave as a graded, per-step cost the
# off-road terminal must be switched off (EXPLORE_OFF_ROAD_FATAL=0), which
# turns leaving the road from a fatal mistake into an expensive one.
# GOAL-MODE DRIVING PENALTIES, charged PER STEP on top of the -1/step clock.
# Both default to 0.0, so the minimum-time contract is untouched unless set.
#
# WHAT THEY COST YOU. Under pure minimum time the goal return is exactly
# -(steps) and lies in [-GOAL_MAX_STEPS, 0], which is what lets the goal
# critic's Q be read as -(steps to goal) -- the property the SGM graph is
# built on. Any per-step penalty breaks that: the return is no longer a step
# count and no longer has a known range, so -Q becomes an ORDERING over
# "badness" rather than a distance in steps. Calibrated cutoffs expressed in
# steps stop meaning steps.
GOAL_OFF_ROAD_PENALTY = float(os.environ.get("GOAL_OFF_ROAD_PENALTY", 0.0))
GOAL_LANE_PENALTY = float(os.environ.get("GOAL_LANE_PENALTY", 0.0))
EXPLORE_OFF_ROAD_PENALTY = float(os.environ.get("EXPLORE_OFF_ROAD_PENALTY", 0.0))
EXPLORE_GOAL_DIST_M = float(os.environ.get("EXPLORE_GOAL_DIST_M", 100.0))
EXPLORE_ARRIVAL_BONUS = float(os.environ.get("EXPLORE_ARRIVAL_BONUS", 10.0))
EXPLORE_LANE_MODE = os.environ.get("EXPLORE_LANE_MODE", "terminal")
if EXPLORE_LANE_MODE not in ("penalise", "terminal"):
    raise ValueError(f"EXPLORE_LANE_MODE={EXPLORE_LANE_MODE!r}: penalise|terminal")


def explore_lane_fatal(lane_any, marking_types, off_road, in_junction=False,
                       off_road_fatal=EXPLORE_OFF_ROAD_FATAL):
    """Explore mode's own rule. Pure, so it is unit-tested without CARLA.

    off_road                          -> fatal (EXPLORE_OFF_ROAD_FATAL, default on;
                                         independent of WEDGE_ON_OFF_ROAD, which
                                         run_gated.sh sets to 0 for the goal arm)
    no crossing                       -> not fatal
    crossing, every mark unpainted    -> NOT fatal (NONE = the road edge)
    crossing, painted, IN A JUNCTION  -> NOT fatal: the turn, not a departure
                                         (the repo's shared junction clause)
    crossing, painted, on open road   -> fatal
    crossing with NO marking data     -> fatal: absent data is not a legal crossing
    Everything after the off-road line is lane_departure_is_fatal with
    strict=False, so the explorer and the wedge share one reading of a
    crossing and only the goal arm's strictness is switched off here.
    """
    if off_road_fatal and bool(off_road):
        return True
    if not bool(lane_any):
        return False
    return bool(lane_departure_is_fatal(True, marking_types, bool(in_junction),
                                        False, strict=False))


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
        r += PROGRESS_SCALE * progress          # LANE_PROGRESS_SCALE (10 for v5)
    if info.get("collision"):
        r -= COLLISION_PENALTY
    # Per-step cost of being off the road. Independent of whether off-road is
    # also terminal: see EXPLORE_OFF_ROAD_PENALTY for how the two interact.
    if EXPLORE_OFF_ROAD_PENALTY and info.get("off_road"):
        r -= EXPLORE_OFF_ROAD_PENALTY
    if LANE_INVASION_NOOP:
        # Crossing a marking costs nothing and ends nothing. OFF-ROAD STILL
        # DOES: it is not a lane crossing, and it is the only term left that
        # discourages leaving the road, since the progress term pays the same
        # for circling as for driving down it.
        fatal = bool(info.get("off_road", False))
    else:
        crossing = explore_lane_fatal(
            bool(info.get("lane_any", info.get("lane_invasion_any", False))),
            info.get("lane_marking_types"), False,
            in_junction=bool(info.get("in_junction", False)))
        off = EXPLORE_OFF_ROAD_FATAL and bool(info.get("off_road", False))
        if EXPLORE_LANE_MODE == "penalise":
            if crossing:
                r -= LANE_DEPARTURE_PENALTY      # charged, not ended
            fatal = off
        else:                                    # terminal
            fatal = crossing or off
            if fatal:
                r -= LANE_DEPARTURE_PENALTY
        return float(np.nan_to_num(r, nan=0.0)), fatal
    if fatal:
        r -= LANE_DEPARTURE_PENALTY
    return float(np.nan_to_num(r, nan=0.0)), fatal


class GatedReward(Wrapper):
    """Draw a mode per episode; return that mode's reward and termination."""

    def __init__(self, env, explore_prob=EXPLORE_PROB, force_mode=None,
                 max_steps=1000, delta_mode="positive", seed=0,
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
        self._prev_remaining = None
        self._t = 0
        # v9 counting regime: per-episode cumulative fault counters (steps,
        # not events -- user "make it 20"). Zeroed per episode in reset().
        self._offroad_steps = 0
        self._collision_steps = 0
        self._crossing_steps = 0
        self._prev = np.zeros(2)
        if self.goal_kind == "sorb":
            lo = -self.step_cost * self.goal_max_steps
            print(f"[Gated] goal=SORB minimum-time step_cost={self.step_cost} "
                  f"goal_max_steps={self.goal_max_steps} "
                  f"-> goal return in [{lo:.1f}, 0.0]; only arrival terminates. "
                  f"REQUIRES DISCOUNT=1.0 for the goal critic.")
        if self.goal_kind == "sorb" and GOAL_TRUNCATE_COUNTS:
            print(f"[Gated] v9 COUNTING regime: no terminal bills; truncate "
                  f"at offroad>={GOAL_OFFROAD_TRUNCATE_STEPS} "
                  f"collision>={GOAL_COLLISION_TRUNCATE_STEPS} "
                  f"crossing>={GOAL_CROSSING_TRUNCATE_STEPS} steps; "
                  f"fines lane/off/coll="
                  f"{GOAL_LANE_PENALTY}/{GOAL_OFF_ROAD_PENALTY}/"
                  f"{GOAL_COLLISION_PENALTY}; "
                  f"arrival bonus={GOAL_ARRIVAL_BONUS} is the only terminus"
                  + (f"; velocity bonus {GOAL_SPEED_BONUS} "
                     f"(peak at target speed, clean steps only)"
                     if GOAL_SPEED_BONUS else ""),
                  flush=True)
        else:
            print(f"[Gated] goal=SHAPED (delta+projected), returns ~order 100")
        if (self.goal_kind == "sorb" and GOAL_TRUNCATE_COUNTS
                and LANE_TERMINAL):
            # PRINTED SO A RUN'S OWN LOG PROVES THE LANE TERMINAL WAS ON.
            # train_v10.sh hardcodes GATED_GOAL_LANE_TERMINAL=0 with an
            # unconditional export, so a launch override is silently
            # discarded unless the launcher is changed -- this line is the
            # only way to see which one actually happened.
            print("[Gated] goal lane invasion TERMINAL: first fatal crossing "
                  "(junction-forgiving, off-road excluded) ends the episode "
                  "with the exact bill -step_cost*(1+remaining), mask=0",
                  flush=True)
        print(f"[Gated] explore lane rule: mode={EXPLORE_LANE_MODE} "
              f"off_road_fatal={EXPLORE_OFF_ROAD_FATAL} "
              f"exempt_marks={sorted(EXPLORE_EXEMPT_MARKS)} junction=forgiven "
              f"penalty={LANE_DEPARTURE_PENALTY} "
              f"(LANE_INVASION_NOOP={LANE_INVASION_NOOP})", flush=True)
        print(f"[Gated] explore_prob={self.explore_prob} "
              f"force_mode={self.force_mode} explore_max_steps={self.max_steps} "
              f"progress={PROGRESS_MODE} x{PROGRESS_SCALE:g}")
        # A SETTING THAT CHANGES THE REWARD MUST SAY SO ON SCREEN. Without this
        # line the only way to confirm it on a live run is to read
        # /proc/<pid>/environ, which is exactly the gap that made the
        # observation banner misleading earlier.
        if LANE_INVASION_NOOP:
            print("[Gated] LANE INVASION IS A NO-OP: crossing a marking costs "
                  "nothing and ends nothing. OFF-ROAD IS STILL FATAL and is "
                  "now the only term discouraging the explore arm from leaving "
                  "the road, since its progress term pays the same for "
                  "circling. NOTE: carla_lanefollow_nogoal was trained WITH "
                  "the lane penalty, so the explore halves are not directly "
                  "comparable unless that baseline is retrained.")
        else:
            print(f"[Gated] lane invasion: penalty {LANE_DEPARTURE_PENALTY} "
                  "and terminal, with the junction exemption")

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        obs, info = out if isinstance(out, tuple) else (out, {})
        # DRAWN ONCE PER EPISODE, never mid-episode: the critics are per-mode, so
        # a transition's mode must be constant for its whole return.
        self.mode = (int(self.force_mode) if self.force_mode is not None
                     else (EXPLORE if self._rng.random() < self.explore_prob else GOAL))
        self._t = 0
        self._offroad_steps = 0
        self._collision_steps = 0
        self._crossing_steps = 0
        self._path_m = 0.0
        self._prev = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
        # The potential is a DIFFERENCE, so step 1 has nothing to difference
        # against and must pay no shaping. A stale value here would pay the
        # whole route length as a one-off reward on the first step of every
        # episode.
        self._prev_remaining = None
        info["mode"] = self.mode
        return obs, info

    def step(self, action):
        obs, _r, term, trunc, info = self.env.step(action)
        self._t += 1
        pos = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
        disp = pos - self._prev
        self._prev = pos
        # CUMULATIVE PATH, not net displacement: "how far did it drive", which
        # is what the explore arm is paid for and what the distance target and
        # the logs both mean. A car that loops back to its start has still
        # driven the distance.
        self._path_m = getattr(self, "_path_m", 0.0) + float(np.linalg.norm(disp))
        info["explore_dist_m"] = float(self._path_m)

        if self.mode == GOAL and self.goal_kind == "sorb":
            # MINIMUM TIME. The env's own `done` is discarded on purpose: with a
            # per-step cost, any terminal failure stops the meter and crashing
            # immediately becomes optimal. Only arrival ends the episode, so a
            # car that hits a wall simply pays the remaining bill.
            # v9 counting regime: the fault counters are the WRAPPER's
            # per-episode state (goal_reward_sorb stays a pure function), so
            # they are updated here, from this step's info, before the call.
            self._offroad_steps += int(bool(info.get("off_road", False)))
            self._collision_steps += int(bool(info.get("collision", False)))
            self._crossing_steps += int(bool(
                info.get("lane_any", info.get("lane_invasion_any", False))))
            reward, terminated, truncated, arrived = goal_reward_sorb(
                info, self._t, self.goal_max_steps, self.step_cost,
                self.wedge_truncate,
                offroad_steps=self._offroad_steps,
                collision_steps=self._collision_steps,
                crossing_steps=self._crossing_steps)
            info["is_success"] = arrived
            # POTENTIAL-BASED SHAPING, GOAL MODE ONLY. Explore mode has no goal,
            # so route_remaining_m is meaningless there and the explore critic
            # must not see a goal-shaped signal.
            #
            # NO SHAPING ON A TERMINAL STEP. Ng et al.'s invariance wants
            # Phi(terminal) = 0, and neither an arrival nor a billed crash
            # satisfies it. Paying into a terminal step would also make the
            # wedge and crash bills inexact, which is what stops them becoming
            # escapes.
            _rem = info.get("route_remaining_m", None)
            _shaped = 0.0
            if POTENTIAL_K and not (terminated or truncated):
                if _rem is not None and self._prev_remaining is not None:
                    _shaped = POTENTIAL_K * (self._prev_remaining - float(_rem))
                    reward += _shaped
            self._prev_remaining = None if _rem is None else float(_rem)
            info["gated_shaping"] = float(_shaped)
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
            _collided = bool(info.get("collision", False))
            terminated = bool(_collided or fatal)
            truncated = (not terminated) and self._t >= self.max_steps
            # ARRIVAL BEATS THE CAP BUT NOT A FAILURE: a car that crosses a
            # line on the step it passes 100 m still crossed a line.
            if (not terminated and EXPLORE_GOAL_DIST_M > 0.0
                    and self._path_m >= EXPLORE_GOAL_DIST_M):
                reward += EXPLORE_ARRIVAL_BONUS
                terminated, truncated = True, False
                info["explore_end"] = "reached"
                info["is_success"] = True
            elif terminated:
                # off_road is only a CAUSE when it is actually fatal. With
                # EXPLORE_OFF_ROAD_FATAL=0 the car can be off the surface and
                # still be ended by a painted crossing, and blaming the
                # off-road would name something that cannot end an episode.
                info["explore_end"] = (
                    "collision" if _collided
                    else "off_road" if (EXPLORE_OFF_ROAD_FATAL
                                        and bool(info.get("off_road", False)))
                    else "crossing")
            elif truncated:
                info["explore_end"] = "cap"

        info["mode"] = self.mode
        info["gated_reward"] = reward
        return obs, reward, terminated, truncated, info
