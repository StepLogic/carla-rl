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

from vae_nav.lane_priv import lane_keep_cost
from vae_nav.lane_explore_reward import (BASE_PENALTY, COLLISION_PENALTY,
                                         LANE_DEPARTURE_PENALTY, PROGRESS_MODE, PROGRESS_SCALE,
                                         STEER_CHANGE_SCALE, lane_departure_is_fatal)

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
# LANE-TERMINAL BILL (user 2026-09-29, option 2 "stronger crossing penalty").
#   clock       -step_cost * (1 + remaining)   -- v15's bill, UNDISCOUNTED: at
#               DISCOUNT 0.99 a crossing at step 100 cost ~-900 while parking to
#               the cap is worth only ~-100, so stopping looked 10x safer.
#   discounted  -step_cost * (1 - g^(1+remaining)) / (1 - g): exactly the value
#               of idling out the rest of the clock -- a crossing costs what a
#               failed episode costs, no more (at most ~100 at g=0.99).
LANE_TERMINAL_BILL = os.environ.get("GOAL_LANE_TERMINAL_BILL", "clock").lower()
if LANE_TERMINAL_BILL not in ("clock", "discounted"):
    raise ValueError(f"GOAL_LANE_TERMINAL_BILL={LANE_TERMINAL_BILL!r} is not clock|discounted")
if LANE_TERMINAL_BILL == "discounted" and os.environ.get("EPISODE_HER", "0") == "1":
    raise ValueError("GOAL_LANE_TERMINAL_BILL=discounted is not mirrored in the HER "
                     "recost; run it with EPISODE_HER=0")
_BILL_GAMMA = float(os.environ.get("DISCOUNT", 0.99))
# GAMMA-AWARE POTENTIAL SHAPING (v40, 2026-10-01). The wrapper pays
# K*(d_prev - d) = Phi(s') - Phi(s) for Phi = -K*d, which is Ng et al.'s
# policy-invariant gamma*Phi(s') - Phi(s) ONLY at gamma = 1. At gamma < 1 the
# gap (1-gamma)*K*d is an extra per-step penalty proportional to distance to
# the goal -- -4/step at 200 m with K=2, g=0.99, against a 1.5 step cost. v40
# moves DISCOUNT 1.0 -> 0.99 to test whether the undiscounted goal critic is
# what diverged in v37-v39; this keeps the shaping's meaning fixed across that
# change. Off by default (every recipe byte-identical); equal at gamma = 1.
SHAPING_DISCOUNTED = os.environ.get("GOAL_SHAPING_DISCOUNTED", "0") == "1"
if SHAPING_DISCOUNTED and os.environ.get("EPISODE_HER", "0") == "1":
    raise ValueError("GOAL_SHAPING_DISCOUNTED=1 is not mirrored in the HER "
                     "recost; run it with EPISODE_HER=0")


def lane_terminal_bill(remaining, step_cost=1.0):
    """The terminal lane-departure cost for `remaining` steps left on the clock."""
    if LANE_TERMINAL_BILL == "discounted" and _BILL_GAMMA < 1.0:
        g = _BILL_GAMMA
        return step_cost * (1.0 - g ** (1 + remaining)) / (1.0 - g)
    return step_cost * (1 + remaining)
# GRADED LANE FINES (v18, user 2026-09-27: "any suggestions to make agent transfer
# to other towns and respect driving rules?" -> "proceed"). Until now the goal arm
# fined RAW lane_any at GOAL_LANE_PENALTY per step: a legal turn through a junction,
# a legal lane change over a broken line and crossing a solid line all cost the
# same -- while JUNCTION_GOAL_FRAC=0.5 means half the goals REQUIRE a junction. The
# explore arm already distinguished them; the goal arm did not. v15 showed that
# making the undifferentiated rule harsher pushed the agent toward idling (0.09).
# With GOAL_LANE_GRADED=1 the fine follows the marking crossed:
#   broken line inside a junction     0                        (the turn)
#   unpainted edge only (NONE)        GOAL_LANE_PENALTY_NONE   (see below)
#   broken line on open road          GOAL_LANE_PENALTY_BROKEN (a lane change)
#   solid / double / kerb / grass     GOAL_LANE_PENALTY        (illegal)
#   no marking data                   GOAL_LANE_PENALTY        (fails closed)
# The SAME verdict drives the 20-step crossing counter and the velocity bonus's
# clean gate, so a legal turn neither accrues toward truncation nor forfeits the
# bonus. Off by default: every earlier recipe is byte-identical.
LANE_GRADED = os.environ.get("GOAL_LANE_GRADED", "0") == "1"
GOAL_LANE_PENALTY_BROKEN = float(os.environ.get("GOAL_LANE_PENALTY_BROKEN", 2.0))
# NONE IS NO LONGER FREE (user 2026-09-30: "fine NONE 5").
#
# The comment above said off_road fines the unpainted edge. MEASURED, and it does
# not: over 120 goal-mode eval episodes (38,855 steps) lane_any fired 79 times and
# off_road 18, and they NEVER coincided -- 0 steps carried both. They are different
# events. off_road is get_waypoint(project_to_road=False) is None, which needs the
# vehicle REFERENCE POINT off every lane; the lane sensor traces the four BOUNDING
# BOX CORNERS (CARLA LaneInvasionSensor.cpp: CalculateCrossedLanes over each
# corner's previous->current segment). A wheel or overhang on a kerb is seen by the
# sensor as NONE and is invisible to off_road, so under the old rule mounting a
# kerb cost nothing at all. lane_explore_reward records the same: Solid/Curb/Grass
# have never once fired in this repo, so "mounting a kerb reports as NONE".
#
# WHY A FINE AND NOT A FATAL. The same corner geometry means a competent driver
# clips an unpainted boundary on bends -- the bounding box is ~2 m in a ~3.5 m lane,
# so the outer corner sweeps wider than the centre. Making NONE fatal at 100 m goals
# took the scripted PID oracle to 0.000 success; a fine charges the kerb mount
# without declaring correct cornering a failure. Set BELOW the painted fines (5.0 vs
# 10.0) for the same reason: it is a weaker claim about a weaker boundary.
# Default 0.0 keeps every earlier run byte-identical.
GOAL_LANE_PENALTY_NONE = float(os.environ.get("GOAL_LANE_PENALTY_NONE", 0.0))
# GOAL MODE ("sorb" arm) ARRIVAL BONUS (v9, user 2026-09-21: "add +10 for
# goal"). Paid on the arrival step INSTEAD of the old 0.0; default 0.0 keeps
# every earlier arm identical. The explore arm's EXPLORE_ARRIVAL_BONUS is a
# separate constant and is untouched.
GOAL_ARRIVAL_BONUS = float(os.environ.get("GOAL_ARRIVAL_BONUS", 0.0))
# v9 COUNTING REGIME (user 2026-09-21: "only truncate after N steps offroad,
# 2 collisions or 2 lane crossings", later "make it 20" -- thresholds are
# CUMULATIVE STEPS within one goal episode; event-dedup was offered and
# declined). When GOAL_TRUNCATE_COUNTS=1 the wrapper keeps per-episode fault
# counters and goal_reward_sorb applies, IN THIS ORDER:
#   arrival (+GOAL_ARRIVAL_BONUS, terminated=True) is checked FIRST and wins;
#   a tripping fault cap TERMINATES the episode with the EXACT remaining-clock
#   bill, mask=0 (user 2026-09-24: "terminate collision, lane invasion and
#   offroad and bill 101 per step", "bill remaining steps but set mask=0"):
#     off-road steps  >= GOAL_OFFROAD_TRUNCATE_STEPS   (20)
#     collision steps >= GOAL_COLLISION_TRUNCATE_STEPS (20)
#     crossing steps  >= GOAL_CROSSING_TRUNCATE_STEPS  (20; the counter is
#       every lane_any step, so a lane invasion IS the counted crossing)
#   and the bill is -(step_cost + fine) * (1 + remaining) under the
#   stay-in-terminal assumption (user: "assume the agent stays in a terminal
#   state till the episode elapsed") -- every remaining step would keep
#   faulting, each paying its step cost plus its fine. TERMINATED, not
#   truncated: the bill carries the cost, so the critic must not bootstrap an
#   unbounded gamma=1 chain on top of it (the legacy wedge/crash bill's rule).
#   fault STEPS BEFORE the cap pay their per-step fines (the truncating step
#   does NOT -- the bill replaces its normal reward);
#   the CLOCK cap alone (max_steps with no fault) still truncates
#   (truncated=True, mask=1.0, the critic bootstraps through the cut).
# GOAL_COLLISION_PENALTY (default 10.0) fines each collision step, consistent
# with the 10.0 lane/off-road fines; under the old bill regime collision was
# billed terminally, so no per-step constant existed for it before.
GOAL_TRUNCATE_COUNTS = os.environ.get("GOAL_TRUNCATE_COUNTS", "0") == "1"
# GOAL FREEZE (user 2026-09-29: "a freeze penalty -- like lane crossings --
# stops the car for 50 steps with 0.995 with current reward setup"; frozen
# steps cost the normal -step_cost, not a larger fine, because at 0.995 a
# -10/step freeze (~-452) would cost more than parking out the episode (~-200)).
# GOAL MODE ONLY, COUNTING REGIME ONLY. On a FINED crossing (goal_lane_fault:
# graded, so junction turns and the unpainted edge never freeze) the step pays
# its usual reward, then the wrapper holds the car under full brake for
# GOAL_FREEZE_STEPS sim steps, ignoring the policy, and returns ONE lumped
# transition (a semi-MDP option):
#   reward  r_cross + sum_{k=1..K} g^k * (-step_cost)   (fines/counters off)
#   s'      the state after the freeze
#   info["freeze_mask"] = g^K, which the trainer multiplies into the replay
#   mask, so the critic bootstraps gamma * g^K * Q(s') = g^(K+1) * Q(s').
# The frozen steps never reach the buffer: the policy's actions do nothing
# there, and a stationary car that is frozen and one that is not would look
# identical to the critic. The clock (and so the goal cap) advances by K.
# 0 = off, byte-identical to every earlier run. g is DISCOUNT, the goal
# critic's own discount (same read as the lane-terminal bill above).
GOAL_FREEZE_STEPS = int(os.environ.get("GOAL_FREEZE_STEPS", 0))
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
# GATED_GOAL_REWARD=progress (user 2026-09-26: "use the reward from
# https://arxiv.org/pdf/2309.16634 as the reward for the goal arm"). Bono et al.
# 2023, "End-to-End (Instance)-Image Goal Navigation through Correspondence as an
# Emergent Phenomenon", Sec. 3.2 "Training for navigation", verbatim:
#     r_t = K * 1_success - Delta_t^Geo - lambda,  K=10,  lambda=0.01,
#     "Delta_t^Geo is the increase in geodesic distance to the goal".
# In CARLA the geodesic is the ALONG-ROAD route distance the env already
# publishes as route_remaining_m (ROUTE_POTENTIAL=1). The paper has NO collision,
# lane or off-road term, so none is paid here; GOAL_PROGRESS_FINES=1 adds the
# GOAL_*_PENALTY fines back as an explicit, off-by-default deviation.
PROGRESS_K = float(os.environ.get("GOAL_PROGRESS_SUCCESS", 10.0))
PROGRESS_SLACK = float(os.environ.get("GOAL_PROGRESS_SLACK", 0.01))
PROGRESS_FINES = os.environ.get("GOAL_PROGRESS_FINES", "0") == "1"
# A route distance that moves more than this in ONE step is not driving: the car
# covers ~0.3 m per 0.05 s step (72 km/h would be 1 m). A larger jump is a
# replan, a goal switch or a planner snapping to another carriageway, and paying
# it would inject a fake reward of tens of metres in a single step.
PROGRESS_MAX_STEP_M = float(os.environ.get("GOAL_PROGRESS_MAX_STEP_M", 5.0))
# GOAL_PROGRESS_POSITIVE (user 2026-10-04: "max((d_prev - d),0)"). Pay progress
# toward the goal and charge nothing for moving away.
#
# READ THIS BEFORE SETTING IT. The signed delta is a POTENTIAL and telescopes to
# d_start - d_end, which is what makes shaping unable to change the optimal
# policy. Clipping at zero gives that up on purpose: an approach-retreat cycle
# returns to the same state having been paid for the approach half, every time
# round, so the reward now contains a loop the agent can farm. Bono et al. pay
# the signed term; this is a deliberate deviation from the paper, not the paper.
# The step cost (-0.01) is the only thing pricing the wasted time, and at ~0.2 m
# per step a 2 m approach pays 2.0 against a cycle cost of about 0.2.
PROGRESS_POSITIVE = os.environ.get("GOAL_PROGRESS_POSITIVE", "0") == "1"
# Read here only to warn: the env publishes route_remaining_m only when this is 1.
ROUTE_POTENTIAL_ON = os.environ.get("ROUTE_POTENTIAL", "0") == "1"
# "progress" added 2026-09-26. This import-time check was missed when the mode
# was added -- the unit tests import with the default and then monkeypatch the
# constant, so they never reach it -- and v16 died on it at startup.
# tests/test_progress_startup.py now imports under the real v16 snapshot.
if GOAL_REWARD_KIND not in ("sorb", "shaped", "progress", "explore"):
    raise ValueError(f"GATED_GOAL_REWARD={GOAL_REWARD_KIND!r} is not "
                     f"sorb|shaped|progress|explore")
# GATED_GOAL_REWARD=explore (user 2026-10-05): goal mode and explore mode share
# one setup -- the same per-step reward, the same fault rules and the same episode
# cap -- and differ in two things only (user: "keep the bonus different"):
#   * what counts as arriving: the goal, versus EXPLORE_GOAL_DIST_M of path;
#   * the arrival bonus: this one for goal mode, EXPLORE_ARRIVAL_BONUS for explore.
# It must exceed what driving on is worth, progress_per_step / (1 - DISCOUNT) --
# about 30 at 0.99 -- or a goal on one branch of a junction would be worth less
# than taking the other branch and continuing.
GOAL_EXPLORE_ARRIVAL_BONUS = float(os.environ.get("GOAL_EXPLORE_ARRIVAL_BONUS", 50.0))
# GOAL_ROUTE_ONLY (v26, user 2026-10-30: both modes pay "metres along the road
# towards the target", the reward of nisheeth-golakiya/carla-rl adapted to a
# goal). Goal episodes were paid TWICE for driving: explore's forward progress,
# which pays any direction the lane goes -- including down the wrong branch,
# away from the goal -- plus the route potential towards the goal
# (POTENTIAL_K). With this on, goal mode keeps only the route term.
GOAL_ROUTE_ONLY = os.environ.get("GOAL_ROUTE_ONLY", "0") == "1"
if GOAL_ROUTE_ONLY and not POTENTIAL_K:
    raise ValueError("GOAL_ROUTE_ONLY=1 with POTENTIAL_K=0 leaves goal mode no "
                     "positive dense term at all; set POTENTIAL_K (1 = one per "
                     "metre of route progress)")
# EXPLORE_JUNCTION_BONUS (v26). Explore episodes have no goal image, so a
# specific target junction would be invisible to the policy: at a fork it would
# be paid for one branch and charged for the other with nothing in its view to
# tell them apart. The target is instead THE NEXT JUNCTION AHEAD, ANY EXIT:
# forward road progress already pays every metre towards it, clearing it pays
# this bonus, and the following junction becomes the target -- the episode goes
# on. Cleared = leaving a junction after at least MIN_STEPS steps inside it and
# at least MIN_GAP_M of driving after the last paid one, so a waypoint
# projection flickering at a junction edge cannot be farmed. 0 = off.
EXPLORE_JUNCTION_BONUS = float(os.environ.get("EXPLORE_JUNCTION_BONUS", 0.0))
EXPLORE_JUNCTION_MIN_STEPS = int(os.environ.get("EXPLORE_JUNCTION_MIN_STEPS", 5))
EXPLORE_JUNCTION_MIN_GAP_M = float(os.environ.get("EXPLORE_JUNCTION_MIN_GAP_M", 10.0))
_NO_DISP = np.zeros(2, np.float64)
STEP_COST = float(os.environ.get("STEP_COST", 1.0))

# PROGRESS_SPEED_GATE (user 2026-10-14: "remove speed gating"). The progress
# term is paid only below the target speed, so a car at or above it earns
# NOTHING for driving forward. That gate is also why removing the state block
# hurt so much: speed_factor decides the reward and the agent could no longer
# see it. With the gate off, forward progress always pays.
#
# DEFAULT 1 KEEPS v52's BEHAVIOUR, because v52 is the recorded run that reaches
# SR 0.48 and must stay reproducible; the launcher sets it to 0 explicitly, so
# the deviation is in the run's environment rather than hidden in a default.
PROGRESS_SPEED_GATE = os.environ.get("PROGRESS_SPEED_GATE", "1") == "1"

# GOAL_ARRIVAL_MAX_SPEED (user 2026-10-14: "the agent should brake near goal"),
# in km/h. Above it, being inside the goal radius does NOT count as arriving:
# the bonus is not paid and the episode continues, so the car has to slow down
# to collect it. Braking becomes instrumentally necessary rather than merely
# encouraged.
#
# IT HAS TO BE THE ARRIVAL CONDITION, not a penalty on speed. Negative throttle
# is REVERSE now and the progress term is ungated, so slowing is already paid
# for negatively -- the reward as it stands wants maximum forward speed forever,
# and nothing short of withholding the bonus gives a reason to stop.
#
# 0 DISABLES IT, which is v52's behaviour and stays the default.
GOAL_ARRIVAL_MAX_SPEED = float(os.environ.get("GOAL_ARRIVAL_MAX_SPEED", 0.0))


def _stopped_enough(info) -> bool:
    """Is the car slow enough for being at the goal to count as arriving?"""
    if GOAL_ARRIVAL_MAX_SPEED <= 0.0:
        return True
    return float(info.get("speed_kmh", 0.0) or 0.0) <= GOAL_ARRIVAL_MAX_SPEED


# THE CAR DECLARES ARRIVAL (user 2026-10-28, "Yes, add a stop action"). With
# automatic arrival the wrong-goal eval of v18@300k scored 0.600 against 0.725
# with the true goal image: driving the lane and halting collected most
# successes without reading the goal. Habitat's imagenav (DEBiT's own task)
# scores success only when the agent calls STOP inside the radius. Here
# declaring also requires being slow, so a random a[2] cannot end episodes at
# speed. 0 = off: every earlier run keeps automatic arrival.
STOP_ACTION = os.environ.get("STOP_ACTION", "0") == "1"
# FALSE_STOP_PENALTY (user 2026-10-28, after v20 scored 0.000 on 120 eval
# episodes): what declaring away from the goal costs. 0 (v20) ENDS the episode,
# which forfeits every reward the car would still have earned -- so never
# declaring was safest, the eval's mean action never declared, and the score
# was exactly zero however well it drove. A positive value charges that much
# and DRIVES ON, so declaring is something the policy can try.
FALSE_STOP_PENALTY = float(os.environ.get("FALSE_STOP_PENALTY", 0.0) or 0.0)


def declared_arrival(info) -> bool:
    """Did the car claim it has arrived on this step? (STOP_ACTION only.)"""
    return (float(info.get("stop_action", 0.0) or 0.0) > 0.0
            and _stopped_enough(info))


def _pays_progress(speed_factor: float) -> bool:
    """Whether the progress term is earned at this speed."""
    return (not PROGRESS_SPEED_GATE) or speed_factor < 1.0
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
    if _pays_progress(speed_factor):
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


def progress_term(d_prev, d_now):
    """-Delta geodesic: metres of along-road progress this step, THE one
    definition the goal wrapper and the hindsight relabeller both use.

    0.0 when either distance is missing (no route) or the step moved it by more
    than PROGRESS_MAX_STEP_M, which no car does in one step (see above).

    Signed by default, so it telescopes and cannot change the optimum. Under
    GOAL_PROGRESS_POSITIVE it is clipped at zero and no longer telescopes -- see
    that constant for what is being traded away.
    """
    if d_prev is None or d_now is None:
        return 0.0
    dp = float(d_prev) - float(d_now)
    if abs(dp) > PROGRESS_MAX_STEP_M:
        return 0.0
    # The jump guard runs FIRST so a planner artefact cannot become a payout.
    return max(dp, 0.0) if PROGRESS_POSITIVE else dp


def goal_reward_progress(info, t, prev_remaining, remaining,
                         max_steps=GOAL_MAX_STEPS,
                         offroad_steps=0, collision_steps=0, crossing_steps=0):
    """Bono et al. 2023's goal reward. Returns (reward, terminated, truncated,
    arrived), the same contract as goal_reward_sorb. Pure.

        arrival:  K - lambda                 terminated (mask 0)
        else:     progress_term - lambda     (+ fines only if PROGRESS_FINES)

    THE FINAL STEP'S PROGRESS IS NOT PAID ON ARRIVAL. The env resamples the goal
    the moment it is reached, so route_remaining_m on that step may already
    describe the NEXT goal; differencing it would pay a jump. Dropping it costs
    at most the goal radius (2 m) once per goal. The relabeller does the same.

    TERMINATIONS ARE THE COUNTING REGIME'S, UNCHANGED, AND CARRY NO BILL. A fatal
    lane crossing (GATED_GOAL_LANE_TERMINAL) ends the episode paying only this
    step; the fault caps and the clock truncate (the critic bootstraps). No bill
    is needed: under a progress reward ending early forfeits the progress still
    to be made and the arrival bonus, so it is never an escape.
    """
    reached = bool(info.get("arrived", False) or info.get("is_success", False))
    if reached:
        return PROGRESS_K - PROGRESS_SLACK, True, False, True
    r = progress_term(prev_remaining, remaining) - PROGRESS_SLACK
    if PROGRESS_FINES:
        if GOAL_OFF_ROAD_PENALTY and info.get("off_road"):
            r -= GOAL_OFF_ROAD_PENALTY
        r -= goal_lane_fine(info) if GOAL_LANE_PENALTY else 0.0
        if GOAL_COLLISION_PENALTY and info.get("collision", False):
            r -= GOAL_COLLISION_PENALTY
    if LANE_TERMINAL and goal_lane_departed(info):
        return r, True, False, False
    truncated = (int(t) >= int(max_steps)
                 or (GOAL_TRUNCATE_COUNTS and (
                     offroad_steps >= GOAL_OFFROAD_TRUNCATE_STEPS
                     or collision_steps >= GOAL_COLLISION_TRUNCATE_STEPS
                     or crossing_steps >= GOAL_CROSSING_TRUNCATE_STEPS)))
    return r, False, truncated, False


def _lane_any(info):
    return bool(info.get("lane_any", info.get("lane_invasion_any", False)))


def goal_lane_fine(info):
    """This step's goal-arm lane fine. THE one definition, read by the reward, by
    goal_lane_fault (counter + clean gate) and, through the trainer's staging, by
    the hindsight relabeller. Ungraded it is exactly the old rule."""
    if not _lane_any(info):
        return 0.0
    if not LANE_GRADED:
        return float(GOAL_LANE_PENALTY)
    if not GOAL_LANE_PENALTY:
        return 0.0           # fines off: graded or not, nothing is paid
    from vae_nav.lane_explore_reward import ILLEGAL_MARKS, UNPAINTED_MARKS
    marks = [str(m).split(".")[-1] for m in (info.get("lane_marking_types") or [])]
    if not marks or any(m in ILLEGAL_MARKS for m in marks):
        return float(GOAL_LANE_PENALTY)
    if all(m in UNPAINTED_MARKS for m in marks):
        return float(GOAL_LANE_PENALTY_NONE)
    if bool(info.get("in_junction", False)):
        return 0.0
    return float(GOAL_LANE_PENALTY_BROKEN)


def goal_lane_fault(info):
    """Does this step count as a lane FAULT -- toward the 20-step crossing cap and
    against the velocity bonus's clean gate? Ungraded: any crossing (old rule).
    Graded: only a crossing that is fined, so legal turns are not faults."""
    if not LANE_GRADED:
        return _lane_any(info)
    return goal_lane_fine(info) > 0.0


def classify_goal_end(arrived, offroad_steps, collision_steps, crossing_steps,
                      t, max_steps):
    """Why a goal episode ended, in episode_her's vocabulary and precedence.

    Logging only (v38, 2026-09-30). The trainer's goal end-reason tally was
    filled solely from the HER buffer's classifier, so with EPISODE_HER=0 it
    stayed EMPTY: all 235 of v37's [Ep] lines read cross=nan(n=0). The wrapper
    holds the same counters, so it labels the terminating step itself, the way
    explore mode writes info["explore_end"]. Arrival is checked first, as
    goal_reward_sorb does; an ending none of the rules explain is "other",
    reported rather than guessed."""
    if arrived:
        return "reached"
    if offroad_steps >= GOAL_OFFROAD_TRUNCATE_STEPS:
        return "off_road"
    if collision_steps >= GOAL_COLLISION_TRUNCATE_STEPS:
        return "collision"
    if crossing_steps >= GOAL_CROSSING_TRUNCATE_STEPS:
        return "crossing"
    if int(t) >= int(max_steps):
        return "clock"
    return "other"


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
            # Same rule as the fault caps below: this step's fine once, plus
            # the wasted clock.
            remaining = max(0, int(max_steps) - int(t))
            return (-goal_lane_fine(info) - lane_terminal_bill(remaining, step_cost),
                    True, False, False)
        # v9 counting regime (see the constants block): per-step fines while
        # the episode runs; the fault caps END THE EPISODE WITH A BILL.
        # User 2026-09-24 ("terminate collision, lane invasion and offroad and
        # bill 101 per step", then "bill remaining steps but set mask=0"):
        # when a cumulative count trips on this step the episode TERMINATES
        # (terminated=True -> mask=0, the critic does NOT bootstrap through)
        # and the step pays the exact remaining-clock bill.
        # ARRIVAL IS CHECKED BEFORE THIS BRANCH and still wins on a step that
        # both arrives and faults. The CLOCK cap (below) stays a truncation:
        # a time limit is not a fault and still bootstraps (mask=1.0).
        #
        # ONE RULE FOR EVERY NON-ARRIVAL ENDING (user 2026-09-30: "terminal
        # unification", then "first fine + wasted time"):
        #
        #     terminal = -(this step's fine) - (the clock left on the episode)
        #
        # The fine is paid ONCE, for the fault that ended the episode, and the
        # wasted time is the same bill every other ending pays. This branch
        # used to charge -(step_cost + fine) for EVERY remaining step, on the
        # assumption that a capped episode would keep faulting to the horizon.
        # That made the exits disagree by an order of magnitude -- at step 100
        # of a 1000 cap the wedge/crash bill is -901 and this one was -9911, a
        # flat 11x -- and TQC_VALUE_FLOOR=-3500 CLIPPED it for any fault before
        # step ~682, so the reward asserted a return the critic could not
        # represent. Charging the fine once keeps every ending inside the floor
        # while still pricing collision, off-road and crossing differently.
        # lane_terminal_bill rather than a bare (1 + remaining) so the clock
        # stays correct under a discount -- v30's lesson, where the
        # undiscounted bill over-penalised 10x at gamma=0.99 and the car parked.
        if (offroad_steps >= GOAL_OFFROAD_TRUNCATE_STEPS
                or collision_steps >= GOAL_COLLISION_TRUNCATE_STEPS
                or crossing_steps >= GOAL_CROSSING_TRUNCATE_STEPS):
            _fine = 0.0
            if (GOAL_OFF_ROAD_PENALTY and offroad_steps
                    >= GOAL_OFFROAD_TRUNCATE_STEPS):
                _fine = max(_fine, GOAL_OFF_ROAD_PENALTY)
            if (GOAL_COLLISION_PENALTY and collision_steps
                    >= GOAL_COLLISION_TRUNCATE_STEPS):
                _fine = max(_fine, GOAL_COLLISION_PENALTY)
            if (GOAL_LANE_PENALTY and crossing_steps
                    >= GOAL_CROSSING_TRUNCATE_STEPS):
                _fine = max(_fine, GOAL_LANE_PENALTY)
            remaining = max(0, int(max_steps) - int(t))
            return (-_fine - lane_terminal_bill(remaining, step_cost),
                    True, False, False)
        _r = -step_cost
        if GOAL_OFF_ROAD_PENALTY and info.get("off_road"):
            _r -= GOAL_OFF_ROAD_PENALTY
        _r -= goal_lane_fine(info) if GOAL_LANE_PENALTY else 0.0
        if GOAL_COLLISION_PENALTY and info.get("collision", False):
            _r -= GOAL_COLLISION_PENALTY
        # TARGET-VELOCITY BONUS (see GOAL_SPEED_BONUS above): only on a CLEAN
        # step -- on-road, no crossing and no collision THIS step; a fault
        # step pays its fines and no bonus. Floored at 0 so an absurd speed
        # cannot make the term negative; a missing or non-positive target
        # disables it.
        if GOAL_SPEED_BONUS:
            _clean = (not info.get("off_road")
                      and not goal_lane_fault(info)
                      and not info.get("collision", False))
            if _clean:
                _v = float(info.get("speed_kmh", 0.0) or 0.0)
                _vt = float(info.get("target_speed_kmh", 0.0) or 0.0)
                if _vt > 0.0:
                    _r += GOAL_SPEED_BONUS * max(
                        0.0, 1.0 - abs(_v - _vt) / _vt)
        truncated = int(t) >= int(max_steps)
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
        #
        # Under the counting regime this is lane_terminal_bill, the SAME bill
        # the fault caps and the lane-departure terminal pay, so every
        # non-arrival ending costs exactly the clock (user 2026-09-30,
        # "terminal unification"). The legacy arm keeps the bare undiscounted
        # form so pre-v9 snapshots replay byte-identically.
        remaining = max(0, int(max_steps) - int(t))
        if GOAL_TRUNCATE_COUNTS:
            return -lane_terminal_bill(remaining, step_cost), True, False, False
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
# NONE IS FINED IN EXPLORE TOO (user 2026-09-30: "fine NONE in both modes").
#
# The explore arm exempts NONE via EXPLORE_EXEMPT_MARKS so that clipping an
# unpainted boundary is not a fatal departure -- the bounding-box corners sweep
# wider than the wheels on a bend (CARLA traces all four corners), so a competent
# driver touches unpainted edges while cornering correctly. That exemption is
# still right about FATALITY and was wrong about COST: a wheel on a kerb reports
# as NONE and was free. This charges it without ending the episode, mirroring
# GOAL_LANE_PENALTY_NONE so the same event costs the same in both modes.
# Default 0.0 keeps every earlier run byte-identical.
EXPLORE_LANE_PENALTY_NONE = float(
    os.environ.get("EXPLORE_LANE_PENALTY_NONE", 0.0))
# COLLISION ENDS AN EXPLORE EPISODE (user 2026-09-30: "collision should be fatal
# and fined"). It was fined COLLISION_PENALTY (3.0) and NOT terminal, so with
# PROGRESS_SCALE=10 a single step of progress (~+2.4) very nearly paid for a
# crash and the explorer could grind along a wall at a small net cost. The fine
# is unchanged; this only makes the episode end. Default off: earlier runs are
# byte-identical.
EXPLORE_COLLISION_FATAL = os.environ.get("EXPLORE_COLLISION_FATAL", "0") == "1"
EXPLORE_GOAL_DIST_M = float(os.environ.get("EXPLORE_GOAL_DIST_M", 100.0))
# HOW EXPLORE "ARRIVAL" MEASURES DISTANCE (2026-10-19, after v8). "path" (the
# default, v52) counts CUMULATIVE path, which a car driving tight loops also
# accumulates -- v8 episode 1360 was logged "reached" while circling a single
# junction. "net" counts straight-line displacement from where the episode
# started, which looping cannot build up.
EXPLORE_DIST_MODE = os.environ.get("EXPLORE_DIST_MODE", "path")
if EXPLORE_DIST_MODE not in ("path", "net"):
    raise ValueError(f"EXPLORE_DIST_MODE={EXPLORE_DIST_MODE!r}: use path or net")
# A COST FOR STANDING STILL, both modes (2026-10-19, after v8 parked). v8's
# last ~120 episodes sat motionless (92-100% still frames) with steer locked at
# +0.9 and throttle 0.03-0.19: a parked step cost only -0.03 to -0.15, so at
# gamma 0.99 parking forever was worth about -10 -- the price of ONE lane
# crossing. Charging EXPLORE_IDLE_PENALTY a step below EXPLORE_IDLE_SPEED_KMH
# makes parking forever worth about -50 at 0.5, clearly worse than driving.
# The threshold sits below GOAL_ARRIVAL_MAX_SPEED so slowing to arrive is free.
# 0 = off (v52).
EXPLORE_IDLE_PENALTY = float(os.environ.get("EXPLORE_IDLE_PENALTY", 0.0))
EXPLORE_IDLE_SPEED_KMH = float(os.environ.get("EXPLORE_IDLE_SPEED_KMH", 2.0))
EXPLORE_ARRIVAL_BONUS = float(os.environ.get("EXPLORE_ARRIVAL_BONUS", 10.0))
# LANE-CENTRE SHAPING (user 2026-10-25: "Both modes", then "Potential
# shaping" to remove the cheap-crash risk of a per-step cost). The potential is
#     c(s) = vae_nav.lane_priv.lane_keep_cost(lane block of s)
#          = LANE_CENTRE_K * min(|offset| / room, 1.5)^2       (room: slack
#            before the BODY meets the marking; = K at the line, ~0 centred)
#          + LANE_HEADING_K * min(|heading err| / 45 deg, 1)^2
# and every step, in BOTH modes, pays F = c(s) - gamma * c(s'), with c = 0 at a
# terminal state and gamma the LEARNER's discount (DISCOUNT / EXPLORE_DISCOUNT,
# the trainer's own variables). Over any episode the discounted F telescopes to
# c(start): crashing at step 5 and arriving at step 500 get the same total, so
# the term points at the centre without ever making an early end cheaper (Ng,
# Harada & Russell 1999). It cannot by itself make lane keeping MORE valuable --
# the crossing fine (LANE_DEPARTURE_PENALTY) still does that. 0 = off.
LANE_CENTRE_K = float(os.environ.get("LANE_CENTRE_K", 0.0) or 0.0)
LANE_HEADING_K = float(os.environ.get("LANE_HEADING_K", 0.0) or 0.0)
LANE_SHAPING = bool(LANE_CENTRE_K or LANE_HEADING_K)
LANE_GAMMA_GOAL = float(os.environ.get("DISCOUNT", 0.99))
LANE_GAMMA_EXPLORE = float(os.environ.get("EXPLORE_DISCOUNT", 0.99))
# Fallback when info carries no hero_half_width: vehicle.micro.microlino.
LANE_CAR_HALF_WIDTH = float(os.environ.get("LANE_CAR_HALF_WIDTH", 0.74))
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


def goal_reward_explore(info, disp, t, max_steps=GOAL_MAX_STEPS):
    """Goal mode paid EXPLORE mode's reward, plus a bonus on arrival. Pure.
    Returns (reward, terminated, truncated, arrived), goal_reward_sorb's contract.

    THE DENSE TERM CARRIES NO GOAL. Same v43 policy, same town: explore drives a
    median 189 m before an episode ends, goal mode with a far goal 9 m, ending
    off-road -- and 3 m with 0 successes when the goal is behind the car. Goal
    mode's old dense reward paid closing the STRAIGHT-LINE gap, so it steered at
    the goal through whatever was in the way. explore_reward pays forward progress
    along the road only; the goal enters through the arrival bonus alone.

    ARRIVAL ENDS THE EPISODE; a fatal step does not count as arriving, matching
    explore's own "arrival beats the cap but not a failure". The clock TRUNCATES
    so the critic bootstraps rather than treating time-out as absorbing.
    """
    # GOAL_ROUTE_ONLY: everything explore pays EXCEPT its forward progress; the
    # route potential added by the wrapper is then the only metre-pay.
    reward, fatal = explore_reward(info, _NO_DISP if GOAL_ROUTE_ONLY else disp)
    collided = bool(info.get("collision", False))
    failed = bool(collided and EXPLORE_COLLISION_FATAL) or bool(fatal)
    # ARRIVAL REQUIRES BEING SLOW ENOUGH (GOAL_ARRIVAL_MAX_SPEED). Blasting
    # through the goal radius at speed is not arriving: the bonus is withheld
    # and the episode continues, so the car must brake to collect it.
    reached = bool(info.get("arrived", False) or info.get("is_success", False))
    reached = reached and _stopped_enough(info)
    if failed:
        return float(reward), True, False, False
    if STOP_ACTION:
        # Being there is not arriving: the car must say so, and saying so
        # anywhere else ends the episode unpaid (Habitat's STOP rule).
        info["declared"] = declared_arrival(info)
        if not info["declared"]:
            return float(reward), False, bool(t >= max_steps), False
        if not reached:
            info["false_stop"] = True
            if FALSE_STOP_PENALTY > 0.0:
                return (float(reward) - FALSE_STOP_PENALTY, False,
                        bool(t >= max_steps), False)
            return float(reward), True, False, False
    if reached:
        return float(reward) + GOAL_EXPLORE_ARRIVAL_BONUS, True, False, True
    return float(reward), False, bool(t >= max_steps), False


def goal_explore_end(info, arrived, truncated):
    """Why a GATED_GOAL_REWARD=explore episode ended, from this step's info.
    Same order as explore mode's explore_end: arrival, then the clock, then the
    fatal cause. Pure."""
    if arrived:
        return "reached"
    if truncated:
        return "clock"
    if info.get("false_stop"):
        return "false_stop"
    if EXPLORE_COLLISION_FATAL and bool(info.get("collision", False)):
        return "collision"
    if EXPLORE_OFF_ROAD_FATAL and bool(info.get("off_road", False)):
        return "off_road"
    return "crossing"


def lane_potential(info):
    """c(s) for the lane-centre shaping, from this step's info. A missing lane
    block is a wiring error (CoreStateWrapper publishes it whenever the shaping
    is on), and is raised rather than read as a centred car."""
    if "lane_priv" not in info:
        raise KeyError("LANE_CENTRE_K/LANE_HEADING_K is on but info has no "
                       "'lane_priv': CoreStateWrapper must run below the reward "
                       "with the same environment.")
    half_car = info.get("hero_half_width")
    return lane_keep_cost(info["lane_priv"], LANE_CENTRE_K, LANE_HEADING_K,
                          LANE_CAR_HALF_WIDTH if half_car is None else half_car)


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
    r = -BASE_PENALTY - STEER_CHANGE_SCALE * steer_change
    if _pays_progress(speed_factor):
        r += PROGRESS_SCALE * progress          # LANE_PROGRESS_SCALE (10 for v5)
    if (EXPLORE_IDLE_PENALTY
            and float(info.get("speed_kmh", 0.0) or 0.0) < EXPLORE_IDLE_SPEED_KMH):
        r -= EXPLORE_IDLE_PENALTY
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
        info["lane_fined"] = False
        fatal = bool(info.get("off_road", False))
    else:
        crossing = explore_lane_fatal(
            bool(info.get("lane_any", info.get("lane_invasion_any", False))),
            info.get("lane_marking_types"), False,
            in_junction=bool(info.get("in_junction", False)))
        off = EXPLORE_OFF_ROAD_FATAL and bool(info.get("off_road", False))
        # An exempt (NONE) crossing is not fatal but is no longer free: see
        # EXPLORE_LANE_PENALTY_NONE. Charged only when EVERY mark is exempt, so
        # a painted crossing is priced by LANE_DEPARTURE_PENALTY below and never
        # pays both.
        _marks = [str(m).split(".")[-1]
                  for m in (info.get("lane_marking_types") or [])]
        if (EXPLORE_LANE_PENALTY_NONE
                and bool(info.get("lane_any", info.get("lane_invasion_any", False)))
                and _marks and all(m in EXPLORE_EXEMPT_MARKS for m in _marks)):
            r -= EXPLORE_LANE_PENALTY_NONE
        # A crash ends the episode (EXPLORE_COLLISION_FATAL); COLLISION_PENALTY
        # was already charged above, so this adds termination, not a second fine.
        _crash = EXPLORE_COLLISION_FATAL and bool(info.get("collision", False))
        info["lane_fined"] = bool(crossing)
        if EXPLORE_LANE_MODE == "penalise":
            if crossing:
                r -= LANE_DEPARTURE_PENALTY      # charged, not ended
            fatal = off or _crash
        else:                                    # terminal
            fatal = crossing or off or _crash
            if crossing or off:
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
                 step_cost=STEP_COST, wedge_truncate=WEDGE_TRUNCATE,
                 freeze_steps=None):
        super().__init__(env)
        # None = GOAL_FREEZE_STEPS; the trainer passes 0 for eval envs, whose
        # episodes end at the first fault (EVAL_FAULT_ENDS) anyway.
        self.freeze_steps = int(GOAL_FREEZE_STEPS if freeze_steps is None
                                else freeze_steps)
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
        if self.goal_kind == "sorb":
            # PRINTED SO A RUN'S OWN LOG PROVES WHICH SHAPING FORM AND WHICH
            # FAULT BILL WERE LIVE (v40). Neither is visible anywhere else: the
            # bill's own banner prints only when the lane terminal is on.
            print(f"[Gated] goal shaping K={POTENTIAL_K:g} form="
                  + (f"K*(d_prev - {_BILL_GAMMA:g}*d) [gamma-aware]"
                     if SHAPING_DISCOUNTED
                     else "K*(d_prev - d) [undiscounted difference]")
                  + "; fault/terminal bill="
                  + (f"DISCOUNTED -step_cost*(1-g^(1+remaining))/(1-g) g={_BILL_GAMMA:g}"
                     if (LANE_TERMINAL_BILL == "discounted" and _BILL_GAMMA < 1.0)
                     else "CLOCK -step_cost*(1+remaining)"), flush=True)
        if self.goal_kind == "sorb" and GOAL_TRUNCATE_COUNTS:
            print(f"[Gated] v9 COUNTING regime: fault caps TERMINATE with the "
                  f"exact remaining-clock bill, mask=0 (stays-faulting bill "
                  f"-(step_cost+fine)*(1+remaining)); caps offroad>="
                  f"{GOAL_OFFROAD_TRUNCATE_STEPS} "
                  f"collision>={GOAL_COLLISION_TRUNCATE_STEPS} "
                  f"crossing>={GOAL_CROSSING_TRUNCATE_STEPS} steps; "
                  f"fines lane/off/coll="
                  f"{GOAL_LANE_PENALTY}/{GOAL_OFF_ROAD_PENALTY}/"
                  f"{GOAL_COLLISION_PENALTY}; "
                  f"arrival bonus={GOAL_ARRIVAL_BONUS} is the only other "
                  f"terminus; the clock cap alone truncates (mask=1)"
                  + (f"; velocity bonus {GOAL_SPEED_BONUS} "
                     f"(peak at target speed, clean steps only)"
                     if GOAL_SPEED_BONUS else ""),
                  flush=True)
        elif self.goal_kind == "shaped":
            print(f"[Gated] goal=SHAPED (delta+projected), returns ~order 100")
        if self.freeze_steps and self.goal_kind == "sorb":
            print(f"[Gated] GOAL FREEZE: a fined crossing holds the car under "
                  f"full brake for {self.freeze_steps} steps at "
                  f"-{self.step_cost:g}/step (discounted at {_BILL_GAMMA:g}), "
                  f"one lumped transition, bootstrap mask "
                  f"{_BILL_GAMMA:g}^{self.freeze_steps}="
                  f"{_BILL_GAMMA ** self.freeze_steps:.4f}; goal mode only"
                  + ("" if GOAL_TRUNCATE_COUNTS else
                     " -- INERT: needs GOAL_TRUNCATE_COUNTS=1"), flush=True)
        if (self.goal_kind == "sorb" and GOAL_TRUNCATE_COUNTS
                and LANE_TERMINAL):
            # PRINTED SO A RUN'S OWN LOG PROVES THE LANE TERMINAL WAS ON.
            # train_v10.sh hardcodes GATED_GOAL_LANE_TERMINAL=0 with an
            # unconditional export, so a launch override is silently
            # discarded unless the launcher is changed -- this line is the
            # only way to see which one actually happened.
            print("[Gated] goal lane invasion TERMINAL: first fatal crossing "
                  "(junction-forgiving, off-road excluded) ends the episode with "
                  + ("the DISCOUNTED bill -step_cost*(1-g^(1+remaining))/(1-g)"
                     if LANE_TERMINAL_BILL == "discounted"
                     else "the clock bill -step_cost*(1+remaining)")
                  + ", mask=0", flush=True)
        if self.goal_kind == "explore":
            # Say which reward ran: without this line the log never states that
            # the goal arm was paid explore's reward rather than a goal reward.
            _mode = PROGRESS_MODE
            print(f"[Gated] goal=EXPLORE reward + arrival: r = explore_reward "
                  f"(progress={_mode} x{PROGRESS_SCALE:g}, lane rule "
                  f"{EXPLORE_LANE_MODE}, off_road_fatal={EXPLORE_OFF_ROAD_FATAL}, "
                  f"collision_fatal={EXPLORE_COLLISION_FATAL}) + "
                  f"{GOAL_EXPLORE_ARRIVAL_BONUS:g}*1[arrived] (explore pays "
                  f"{EXPLORE_ARRIVAL_BONUS:g} at {EXPLORE_GOAL_DIST_M:g} m); cap "
                  f"{self.max_steps} (shared with explore); "
                  + (f"GOAL_ROUTE_ONLY: goal mode drops forward progress and is "
                     f"paid {POTENTIAL_K:g}/m of ROUTE progress only; "
                     if GOAL_ROUTE_ONLY else
                     f"route potential K={POTENTIAL_K:g}; " if POTENTIAL_K else
                     "NO term pulls towards the goal; ")
                  + (f"explore +{EXPLORE_JUNCTION_BONUS:g} per junction cleared "
                     f"(>= {EXPLORE_JUNCTION_MIN_STEPS} steps inside, "
                     f">= {EXPLORE_JUNCTION_MIN_GAP_M:g} m apart); "
                     if EXPLORE_JUNCTION_BONUS else "")
                  + f"steer-change cost x{STEER_CHANGE_SCALE:g}; "
                  + "arrival ends the episode; clock truncates",
                  flush=True)
        if self.goal_kind == "progress":
            # Precomputed: a nested same-quote expression inside an
            # f-string is a syntax error before Python 3.12.
            _delta = ("max(Delta(route m), 0) [CLIPPED: not a potential, "
                      "retreat is free]" if PROGRESS_POSITIVE
                      else "Delta(route m)")
            print(f"[Gated] goal=PROGRESS (Bono et al. 2023, arXiv 2309.16634): "
                  f"r = {PROGRESS_K:g}*1[arrived] - {_delta} - "
                  f"{PROGRESS_SLACK:g}; fines "
                  f"{'ON (deviation from the paper)' if PROGRESS_FINES else 'off'}; "
                  f"per-step jump guard {PROGRESS_MAX_STEP_M:g} m; lane terminal "
                  f"{'ON (no bill)' if LANE_TERMINAL else 'off'}; caps "
                  f"{'truncate' if GOAL_TRUNCATE_COUNTS else 'off'}"
                  + (f"; POTENTIAL_K={POTENTIAL_K:g} IGNORED (progress is the "
                     f"reward itself)" if POTENTIAL_K else ""), flush=True)
            if not ROUTE_POTENTIAL_ON:
                print("[Gated] WARNING: GATED_GOAL_REWARD=progress but "
                      "ROUTE_POTENTIAL is not 1 -- route_remaining_m will be None "
                      "and the reward degenerates to -lambda per step",
                      flush=True)
        if LANE_GRADED and self.goal_kind in ("sorb", "progress"):
            print(f"[Gated] goal lane fines GRADED: junction turn 0, unpainted "
                  f"edge 0, broken on open road {GOAL_LANE_PENALTY_BROKEN:g}, "
                  f"solid/illegal/unknown {GOAL_LANE_PENALTY:g}; counter and "
                  f"clean gate count only fined crossings", flush=True)
        print(f"[Gated] explore lane rule: mode={EXPLORE_LANE_MODE} "
              f"off_road_fatal={EXPLORE_OFF_ROAD_FATAL} "
              f"exempt_marks={sorted(EXPLORE_EXEMPT_MARKS)} junction=forgiven "
              f"penalty={LANE_DEPARTURE_PENALTY} "
              f"(LANE_INVASION_NOOP={LANE_INVASION_NOOP})", flush=True)
        print(f"[Gated] idle: -{EXPLORE_IDLE_PENALTY:g}/step below "
              f"{EXPLORE_IDLE_SPEED_KMH:g} km/h (both modes; 0 = off) | explore "
              f"arrival: {EXPLORE_GOAL_DIST_M:g} m of {EXPLORE_DIST_MODE} distance",
              flush=True)
        if STOP_ACTION:
            print(f"[Gated] DECLARED ARRIVAL: action dim 2 > 0 and speed <= "
                  f"{GOAL_ARRIVAL_MAX_SPEED:g} km/h means 'I am at the goal'. "
                  f"At the goal: +{GOAL_EXPLORE_ARRIVAL_BONUS:g} and the episode ends. "
                  + (f"Elsewhere: -{FALSE_STOP_PENALTY:g} and it DRIVES ON."
                     if FALSE_STOP_PENALTY > 0 else
                     "Elsewhere: the episode ENDS unpaid."), flush=True)
        print(f"[Gated] lane-centre SHAPING: F = c(s) - gamma c(s'), c = "
              f"{LANE_CENTRE_K:g} where the body meets the line (quadratic, cap "
              f"x2.25) + {LANE_HEADING_K:g} at 45 deg heading error (cap there); "
              f"gamma goal {LANE_GAMMA_GOAL:g} / explore {LANE_GAMMA_EXPLORE:g}; "
              f"c=0 at terminal; both modes, junctions forgiven"
              + ("" if LANE_SHAPING else " -- OFF"), flush=True)
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
            print(f"[Gated] lane invasion: penalty {LANE_DEPARTURE_PENALTY}, "
                  f"{'terminal' if EXPLORE_LANE_MODE == 'terminal' else 'NOT terminal (fined)'}"
                  ", with the junction exemption")

    def _explore_dist(self) -> float:
        """Distance that counts towards explore arrival (EXPLORE_DIST_MODE)."""
        if EXPLORE_DIST_MODE == "net":
            start = getattr(self, "_start", self._prev)
            return float(np.linalg.norm(self._prev - start))
        return float(self._path_m)

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
        # EXPLORE_JUNCTION_BONUS state: consecutive steps inside the current
        # junction, the path metre of the last paid clearing (-inf: the first
        # junction of an episode always qualifies), and the episode's count.
        self._j_steps = 0
        self._j_last_m = -np.inf
        self._j_count = 0
        self._prev = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
        self._start = self._prev.copy()
        # The potential is a DIFFERENCE, so step 1 has nothing to difference
        # against and must pay no shaping. A stale value here would pay the
        # whole route length as a one-off reward on the first step of every
        # episode.
        self._prev_remaining = None
        if self.goal_kind == "progress":
            # The progress reward is a difference too, but its FIRST step is a
            # real step of driving, so seed it from the reset frame instead of
            # dropping a step of progress per episode. None if no route.
            _r0 = info.get("route_remaining_m", None)
            self._prev_remaining = None if _r0 is None else float(_r0)
        # The shaping differences c(s) - gamma c(s'), so it needs c(s0).
        self._lane_c = lane_potential(info) if LANE_SHAPING else 0.0
        info["mode"] = self.mode
        return obs, info

    def _experiment(self):
        base = getattr(self.env, "unwrapped", self.env)
        return getattr(base, "experiment", None)

    # The crossing step's fault flags survive into the returned info, so the
    # training logs and the eval fault rule still see the crossing; the rest
    # (pose, frame, route) is the post-freeze step's.
    _FREEZE_KEEP = ("lane_any", "lane_invasion_any", "lane_marking_types",
                    "in_junction", "off_road", "collision")

    def _goal_freeze(self, action, reward, info):
        """Hold the car for self.freeze_steps sim steps; return the lumped
        (obs, reward, terminated, truncated, arrived, info). See
        GOAL_FREEZE_STEPS."""
        exp = self._experiment()
        if exp is not None:
            exp.goal_frozen = True
        acc, g, k = float(reward), 1.0, 0
        terminated = truncated = arrived = False
        obs, last = None, info
        _p0 = self._path_m
        try:
            while k < self.freeze_steps:
                obs, _r, _te, _tr, last = self.env.step(action)
                k += 1
                self._t += 1
                g *= _BILL_GAMMA
                pos = np.asarray(last.get("hero_pos", (0.0, 0.0)), np.float64)
                self._path_m += float(np.linalg.norm(pos - self._prev))
                self._prev = pos
                if bool(last.get("arrived", False)
                        or last.get("is_success", False)):
                    # A braking car can still roll onto the goal.
                    acc += g * GOAL_ARRIVAL_BONUS
                    terminated = arrived = True
                    break
                acc += g * (-self.step_cost)
                if self._t >= self.goal_max_steps:
                    truncated = True
                    break
        finally:
            if exp is not None:
                exp.goal_frozen = False
        # One line per freeze: proves on a live run that the car stops.
        print(f"[Freeze] t={self._t} held {k} steps: "
              f"{float(info.get('speed_kmh', 0) or 0):.1f} -> "
              f"{float(last.get('speed_kmh', 0) or 0):.1f} km/h, rolled "
              f"{self._path_m - _p0:.2f} m, r={acc:.1f} mask={g:.3f}"
              + (" ARRIVED" if arrived else " CAP" if truncated else ""),
              flush=True)
        out = dict(last)
        for key in self._FREEZE_KEEP:
            if key in info:
                out[key] = info[key]
        out["explore_dist_m"] = float(self._path_m)
        out["goal_freeze_steps"] = int(k)
        out["freeze_mask"] = float(g)
        _rem = out.get("route_remaining_m", None)
        self._prev_remaining = None if _rem is None else float(_rem)
        return obs, acc, terminated, truncated, arrived, out

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

        if self.mode == GOAL and self.goal_kind == "progress":
            # BONO ET AL. 2023 (see PROGRESS_K). The same fault counters as the
            # counting regime drive the caps; the reward itself is progress.
            self._offroad_steps += int(bool(info.get("off_road", False)))
            self._collision_steps += int(bool(info.get("collision", False)))
            self._crossing_steps += int(goal_lane_fault(info))
            _rem = info.get("route_remaining_m", None)
            reward, terminated, truncated, arrived = goal_reward_progress(
                info, self._t, self._prev_remaining, _rem, self.goal_max_steps,
                offroad_steps=self._offroad_steps,
                collision_steps=self._collision_steps,
                crossing_steps=self._crossing_steps)
            info["is_success"] = arrived
            # After an arrival the next distance belongs to a new goal, so there
            # is nothing to difference against for one step.
            self._prev_remaining = (None if (arrived or _rem is None)
                                    else float(_rem))
            # The progress term IS the reward here; POTENTIAL_K is not added on
            # top (it would pay the same metres twice). Logged as shaping=0.
            info["gated_shaping"] = 0.0
        elif self.mode == GOAL and self.goal_kind == "explore":
            # EXPLORE'S REWARD + A GOAL BONUS (goal_reward_explore). The fault
            # counters are kept so goal_end and the [GatedEp] ends read the same
            # way as the other kinds.
            self._offroad_steps += int(bool(info.get("off_road", False)))
            self._collision_steps += int(bool(info.get("collision", False)))
            self._crossing_steps += int(goal_lane_fault(info))
            # THE EXPLORE CAP, not goal_max_steps: the two modes share one
            # setup and differ only in what counts as arriving.
            reward, terminated, truncated, arrived = goal_reward_explore(
                info, disp, self._t, self.max_steps)
            info["is_success"] = arrived
            # A DENSE GOAL TERM, GOAL MODE ONLY (POTENTIAL_K; 0 = off, which is
            # v52 and the default). My decision 2026-10-14 under the user's
            # reward delegation, after v6: the explore reward is identical in
            # both modes, so goal mode had nothing to make it behave DIFFERENTLY
            # -- its only distinction was a rare arrival bonus. v6's goal arm
            # duly decayed 0.100 -> 0.010 while the explore arm improved to
            # 0.20: the policy optimised the objective that pays every step.
            # The peer's Habitat arm, which works, has exactly this term
            # (d_prev - d on every step); ours had nothing until arrival.
            #
            # POTENTIAL-BASED, so the optimal policy is unchanged (Ng et al.):
            # Phi(s) = -K * route_remaining, and the step pays
            # gamma*Phi(s') - Phi(s) = K*(prev - gamma*rem). NOT paid on a
            # terminal step, where Phi must be 0 for the invariance to hold --
            # paying into an arrival or a crash would also make those bills
            # inexact.
            _rem = info.get("route_remaining_m", None)
            _shaped = 0.0
            if POTENTIAL_K and not (terminated or truncated):
                if _rem is not None and self._prev_remaining is not None:
                    _g = _BILL_GAMMA if SHAPING_DISCOUNTED else 1.0
                    _shaped = POTENTIAL_K * (self._prev_remaining - _g * float(_rem))
                    # THE SPEED GATE (v27). Under GOAL_ROUTE_ONLY this term IS
                    # goal mode's progress, so PROGRESS_SPEED_GATE gates it like
                    # explore's -- but only what it PAYS. Moving away is charged
                    # at any speed, or "approach slowly, retreat fast" would be
                    # a loop that earns.
                    if (GOAL_ROUTE_ONLY and _shaped > 0.0 and not _pays_progress(
                            float(info.get("speed_factor", 0.0) or 0.0))):
                        _shaped = 0.0
                    reward += _shaped
            self._prev_remaining = None if _rem is None else float(_rem)
            info["gated_shaping"] = float(_shaped)
            if terminated or truncated:
                # NAME THE ACTUAL CAUSE, in explore mode's own order. NOT
                # classify_goal_end: that reads the goal arm's fault COUNTERS
                # and caps, and has no notion of explore's fatal-crossing rule,
                # so every episode this reward ended on a painted line came out
                # "other" -- v51's first launch logged other=0.83, cross=0.00,
                # hiding exactly the lane invasions this run is about.
                info["goal_end"] = goal_explore_end(info, arrived, truncated)
        elif self.mode == GOAL and self.goal_kind == "sorb":
            # MINIMUM TIME. The env's own `done` is discarded on purpose: with a
            # per-step cost, any terminal failure stops the meter and crashing
            # immediately becomes optimal. Only arrival ends the episode, so a
            # car that hits a wall simply pays the remaining bill.
            # v9 counting regime: the fault counters are the WRAPPER's
            # per-episode state (goal_reward_sorb stays a pure function), so
            # they are updated here, from this step's info, before the call.
            self._offroad_steps += int(bool(info.get("off_road", False)))
            self._collision_steps += int(bool(info.get("collision", False)))
            self._crossing_steps += int(goal_lane_fault(info))
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
                    _g = _BILL_GAMMA if SHAPING_DISCOUNTED else 1.0
                    _shaped = POTENTIAL_K * (self._prev_remaining - _g * float(_rem))
                    reward += _shaped
            self._prev_remaining = None if _rem is None else float(_rem)
            info["gated_shaping"] = float(_shaped)
            if (self.freeze_steps and GOAL_TRUNCATE_COUNTS
                    and not (terminated or truncated)
                    and goal_lane_fault(info)):
                obs, reward, terminated, truncated, arrived, info = \
                    self._goal_freeze(action, reward, info)
                info["is_success"] = arrived
            if terminated or truncated:
                info["goal_end"] = classify_goal_end(
                    arrived, self._offroad_steps, self._collision_steps,
                    self._crossing_steps, self._t, self.goal_max_steps)
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
            # What only explore pays, for HER: a relabelled explore episode
            # becomes GOAL rows, which must not carry it. Forward progress is
            # explore-only when goal mode is route-only; the junction bonus
            # always is.
            _explore_only = (reward - explore_reward(info, _NO_DISP)[0]
                             if GOAL_ROUTE_ONLY else 0.0)
            cleared = False
            if EXPLORE_JUNCTION_BONUS:
                if bool(info.get("in_junction", False)):
                    self._j_steps += 1
                else:
                    # Leaving a junction. A crash on the exit step is not a
                    # clearing; the clock running out on it is.
                    cleared = (not terminated
                               and self._j_steps >= EXPLORE_JUNCTION_MIN_STEPS
                               and self._path_m - self._j_last_m
                               >= EXPLORE_JUNCTION_MIN_GAP_M)
                    self._j_steps = 0
                if cleared:
                    reward += EXPLORE_JUNCTION_BONUS
                    _explore_only += EXPLORE_JUNCTION_BONUS
                    self._j_last_m = self._path_m
                    self._j_count += 1
            info["junction_cleared"] = cleared
            info["junctions_cleared"] = self._j_count
            info["explore_only"] = float(_explore_only)
            # ARRIVAL BEATS THE CAP BUT NOT A FAILURE: a car that crosses a
            # line on the step it passes 100 m still crossed a line.
            if (not terminated and EXPLORE_GOAL_DIST_M > 0.0
                    and self._explore_dist() >= EXPLORE_GOAL_DIST_M):
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

        if LANE_SHAPING:
            gamma = LANE_GAMMA_GOAL if self.mode == GOAL else LANE_GAMMA_EXPLORE
            c_next = 0.0 if terminated else lane_potential(info)
            tail = gamma * c_next
            shaping = self._lane_c - tail
            reward = float(reward) + shaping
            # The hindsight relabeller makes a non-terminal row TERMINAL at the
            # leg's arrival; there c(s') must be 0, so it adds `tail` back.
            info["lane_shaping"] = shaping
            info["lane_shaping_tail"] = tail
            info["lane_potential"] = c_next
            self._lane_c = c_next
        info["mode"] = self.mode
        info["gated_reward"] = reward
        return obs, reward, terminated, truncated, info
