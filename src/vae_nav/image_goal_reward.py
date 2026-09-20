"""The image-goal navigation reward: ReViND-style, all-negative until success.

THE TASK. Reach a physical target viewpoint -- the viewpoint the goal image was
taken from -- and STOP there, facing the goal's road heading. Success is judged
on what the deployed policy will have onboard: current proprioception and the
goal. The reward itself may read privileged CARLA labels; the claim it supports
does not depend on them at execution time.

THE SHAPE (user spec 2026-09-18; 2026-09-20 removed the flat in-lane bonus
and the evening batch made the speed term a POSITIVE SATURATING BONUS in
both legs and made GOAL-mode failures TERMINAL):

    GOAL leg (mode 1):
    r_t = -1.0                              the base cost, EVERY step
        + K min(v, v*)/v*  if L_t           the saturating speed bonus,
                                            K = IG_SPEED_PREF_K, v* =
                                            TARGET_SPEED (20 km/h); +0.2/step
                                            at v >= v*, exactly 0 parked
        - 1.5    if off the drivable surface (each step it persists)
        - 1.5    if a collision has EVER happened this episode
        [- 5.0    on a FAILING step -- the TWO-MODE goal branch terminates
                  on the first off-road / crossing / collision step and pays
                  this flat on top of the ordinary reward (wrapper-level)]

    EXPLORE leg (mode 0):
    r_t = + K min(v, v*)/v*  if L_t         the SAME saturating bonus, alone
                                            (no base cost, no flat penalties)

    L_t = on the drivable lane, moving in the route direction, no lane
          crossing this step, and no collision yet.

The |v - v*| PENALTY form is GONE in both modes (user 2026-09-20 evening):
both legs now carry the same positive saturating bonus and differ only in
base cost, flat penalties and terminal design. The old +0.75 flat in-lane
bonus stays gone; the constant IN_LANE_BONUS remains an env-overridable
ablation knob (IG_IN_LANE_BONUS) defaulting to 0.0. The speed term stays
gated on L_t exactly as it was.

THE TWO USER DECISIONS THAT CHANGED IT FROM THE WRITTEN SPEC:

  1. NO SUCCESS BONUS. The original spec paid +10 on success; the user removed
     the term. Success now only TERMINATES the episode, and its step pays the
     ordinary reward. The incentive to finish is the per-step cost ending --
     which is the strictest form of the ReViND property: r_t < 0 at EVERY step,
     including the last one. A critic fitted to this return reads "less
     negative = closer to done", and loitering is never better than arriving.

  2. NO IMAGE-SIMILARITY GATE. The original spec conditioned success on
     goal_image_similarity >= 0.85. The user removed it: success = sustained
     low speed, within GOAL_DISTANCE_THRESHOLD_M of the goal, and facing within
     GOAL_YAW_THRESHOLD_DEG of the goal waypoint's road heading.

THE SPEED TERM (user spec 2026-09-20 evening). While L_t holds, each step
also earns +IG_SPEED_PREF_K * min(v, v*)/v* with v* = TARGET_SPEED/3.6: a
bonus that saturates at IG_SPEED_PREF_K (0.2) once v reaches v* and is
exactly 0 when parked. It is a bonus times the in-lane indicator. In the
goal leg the base cost dominates it (best row -0.8, every row < 0 -- the
ReViND property survives), and in the explore leg it IS the whole reward
(+0.2/step max; positive explore rewards are legal by the same decision).
A car off the lane, crossing, off-road or post-collision pays its ordinary
rows and NOT this term (its indicator is false); an unreadable speed reads
as stopped, not as a free pass.

WHY COLLISION IS A STATE, NOT AN EVENT. One impact is one sensor reading;
the DAMAGE persists. A one-time event penalty is paid once and then forgotten,
so a car that crashes early and drives on afterwards looks identical to one
that never crashed. The persistent flag keeps the post-collision trajectory in
a lower-reward mode until reset, and it must be visible to the value function
(Phase 2: critic conditioning z = has_collided) or the same apparent image has
two different values depending on unseen history -- a Markov violation.

WHAT TERMINATES (user decisions 2026-09-20, evening batch). Success, invalid
and the clock terminate as before. This paragraph describes the GOAL leg of
the pure function (termination-wise; the single-mode arm keeps exactly these
terminals). The TWO-MODE arm's GOAL branch diverges further by user decision
2026-09-20 evening: the FIRST off-road step, lane crossing or collision
TERMINATES the goal episode (terminated=True, truncated=False -- a
termination, not a truncation, so info["ig_truncated"] stays None), paying
the ordinary reward plus a flat -IG_TERMINAL_PENALTY on that step. The old
|v-v*| penalty and Task 16's OFFROAD_TRUNC_STEPS debounce counter are GONE:
the first off-road step ends the episode, so there is nothing left to
debounce. The STUCK truncation (speed under IG_STUCK_SPEED_MPS=0.3 m/s for
IG_STUCK_STEPS=100 consecutive steps outside the success spatial/yaw gate ->
truncated=True, ordinary reward, no penalty) is KEPT exactly as reviewed --
truncation stops the negative stream without buying the terminal escapes,
and a gate step never counts toward the stuck counter, so parking at the
goal still succeeds. The counters live in TwoModeImageGoalReward (where the
stop window lives); explore mode runs none of them.
The two legs are combined as a MASKED SUM -- each leg is zeroed
outside its own mode -- not a formula swap; see image_goal_reward and
TwoModeImageGoalReward for the explore contract.

DEVIATIONS FROM THE WRITTEN SPEC, all deliberate:
  - success pays no bonus (decision above);
  - the yaw reference is the GOAL WAYPOINT's road heading, which is the yaw the
    goal camera was spawned at, so "matches the intended orientation" means
    facing along the road as in the goal image;
  - "driving_with_route_direction" is measured, not asserted: displacement
    projected on the lane's forward vector, and a parked or creeping car
    (speed under STOP_SPEED_MPS) is NOT counted as driving the wrong way --
    its L_t stays True, exactly as before (user 2026-09-20 keeps the
    carve-out). Since the 2026-09-20 evening batch the speed term is the
    saturating bonus, so a parked car earns none of it and pays the bare
    base cost (-1.0) while it waits out the stop window -- ending the
    negative stream is still the best a parked step can do;
  - "simulator_invalid" is the hero being destroyed or not alive, which is all
    the health signal CARLA exposes here;
  - the timeout uses gym's `truncated`, and whether the offline TD target treats
    a truncation as done is a Phase 2 decision, not this module's.

WHAT IS DELIBERATELY ABSENT: any comparison between the current camera frame
and the goal image. The env encodes both to MobileNet features and the user
decided none of that gates success in this phase.
"""
import math
import os

import numpy as np

try:
    from gymnasium import Wrapper
except ImportError:
    from gym import Wrapper

# ---- THE CONSTANTS (user spec; every one configurable) ----------------------
SUCCESS_REWARD = float(os.environ.get("IG_SUCCESS_REWARD", 0.0))
# NOTE: the spec's SUCCESS_REWARD was 10.0; the user removed the success term,
# so the constant is kept (configurable) but defaults to 0.0 -- success
# terminates and pays the ordinary step reward. Setting it above 0 restores the
# spec's original bonus.
BASE_STEP_COST = float(os.environ.get("IG_BASE_STEP_COST", 1.0))
# The flat in-lane bonus was REMOVED from both modes (user decision
# 2026-09-20): the default is 0.0, so in-lane driving pays the base cost and
# the speed preference, nothing more. The constant stays env-overridable via
# IG_IN_LANE_BONUS for ablations; setting it above 0 restores the old term.
IN_LANE_BONUS = float(os.environ.get("IG_IN_LANE_BONUS", 0.0))
OFFROAD_PENALTY = float(os.environ.get("IG_OFFROAD_PENALTY", 1.5))
# Both flat penalties raised 0.75 -> 1.5 (user 2026-09-20) to restore the
# in-lane/off-road contrast that the in-lane-bonus removal halved: the best
# goal-mode row is -1.0 (in lane at v*), off-road and post-collision rows are
# -2.5 and the worst stacks both at -4.0, so the contrast is back to 1.5/step
# (the pre-removal ratio). Env-overridable as before for the old-row ablation.
PERSISTENT_COLLISION_PENALTY = float(
    os.environ.get("IG_PERSISTENT_COLLISION_PENALTY", 1.5))
GOAL_DISTANCE_THRESHOLD_M = float(
    os.environ.get("IG_GOAL_DISTANCE_THRESHOLD_M", 2.0))
GOAL_YAW_THRESHOLD_DEG = float(
    os.environ.get("IG_GOAL_YAW_THRESHOLD_DEG", 15.0))
# PLACEHOLDER until calibrated from real collision-sensor readings: it must
# ignore negligible contacts and sensor noise while catching real impacts.
COLLISION_IMPULSE_THRESHOLD = float(
    os.environ.get("IG_COLLISION_IMPULSE_THRESHOLD", 1.0))
MAX_STEPS = int(os.environ.get("IG_MAX_STEPS", 1000))
STOP_SPEED_MPS = float(os.environ.get("IG_STOP_SPEED_MPS", 0.2))
STOP_WINDOW_STEPS = int(os.environ.get("IG_STOP_WINDOW_STEPS", 10))
# THE SPEED TERM (user spec 2026-09-20 evening): while in lane, every step
# earns +SPEED_PREF_K * min(v, v*)/v* with v* = TARGET_SPEED -- a SATURATING
# BONUS, +K/step at or above v*, exactly 0 parked, in BOTH legs (the
# |v - v*| penalty form is gone). K's default was raised 0.1 -> 0.2 (user
# decision 2026-09-20, both modes); the bonus form is the same evening's
# follow-up decision.
SPEED_PREF_K = float(os.environ.get("IG_SPEED_PREF_K", 0.2))
TARGET_SPEED_KMH = float(os.environ.get("TARGET_SPEED", 20.0))
# The failing step's flat penalty (user decisions 2026-09-20): a failing
# step (off-road / lane-crossing / collision) TERMINATES its episode and
# pays this flat on top of the ordinary reward. In EXPLORE the leg pays it
# directly; in GOAL mode the two-mode wrapper pays it when it terminates on
# the first failure (the single-mode arm does not terminate on failures and
# never reads it). The failing step's ordinary component is exactly 0.0 in
# explore (in_lane is False on every one of them), so the explore failing
# step pays exactly this.
TERMINAL_PENALTY = float(os.environ.get("IG_TERMINAL_PENALTY", 5.0))
# GOAL-MODE STUCK TRUNCATION (user decision 2026-09-20): a goal episode
# TRUNCATES (truncated=True, NO extra penalty -- the truncating step pays its
# ordinary reward) on sustained low speed instead of running to the cap:
# speed under STUCK_SPEED_MPS for STUCK_STEPS consecutive steps, where a
# step inside the success spatial/yaw gate (route distance under
# GOAL_DISTANCE_THRESHOLD_M and |yaw error| under GOAL_YAW_THRESHOLD_DEG)
# is NEVER a stuck step: parking at the goal still reaches the sustained
# stop in STOP_WINDOW_STEPS=10 steps and can never be stuck-truncated
# first. The counter resets on wrapper reset and whenever the condition
# fails that step (consecutive, non-consecutive-safe). Truncation
# is not termination: terminated stays False, success still wins the step
# it fires on. The OFFROAD_TRUNC_STEPS off-road counter this block once
# carried is REMOVED (user 2026-09-20 evening): the first off-road step now
# TERMINATES the goal episode, so there is nothing left to debounce.
STUCK_SPEED_MPS = float(os.environ.get("IG_STUCK_SPEED_MPS", 0.3))
STUCK_STEPS = int(os.environ.get("IG_STUCK_STEPS", 100))

IMAGE_GOAL_TASK = os.environ.get("IMAGE_GOAL_TASK", "0") == "1"

# The two modes of the gated image-goal arm, matching vae_nav.gated_reward.
GOAL, EXPLORE = 1, 0

# The explore arm's arrival distance. IMPORTED, not redefined: the 2-mode arm
# shares the gated explore arm's semantics (cumulative path, not net
# displacement) and must never drift from the frozen constant's meaning.
from vae_nav.gated_reward import EXPLORE_GOAL_DIST_M  # noqa: E402


def image_goal_coherence():
    """Refuse image-goal configurations that cannot learn, BEFORE CARLA starts.

    Mirrors gated_config.check_gated_run: a run that starts and then dies at
    step 1 has still cost a server start, a world load and a GPU share.

    TWO BRANCHES, chosen on GATED:

      GATED=1 (the 2-mode pairing, TwoModeImageGoalReward): the pairing is
      REQUIRED, not refused -- it is the arm this module now serves -- but a
      2-mode run must state every flag its learner depends on, and must NOT
      carry any flag from the single-mode arm it replaces. The refusals are
      exact pairings, not style: OBS_WITH_COLLISION=1 would put the single
      arm's `coll` key beside the 2-mode arm's `state` key, whose first two
      slots already carry has_collided; a stacked reward would be dead code
      below a wrapper that reads none of its info keys; NO_TIMELIMIT=1 is a
      trace-collection setting that would remove the TimeLimit wrapper and
      leave only the reward's own cap.

      GATED=0 (the single-mode arm, ImageGoalReward): the original checks,
      unchanged.
    """
    if not IMAGE_GOAL_TASK:
        return
    if os.environ.get("GATED", "0") == "1":
        if os.environ.get("OBS_WITH_STATE", "0") != "1":
            raise SystemExit(
                "IMAGE_GOAL_TASK=1 with GATED=1 requires OBS_WITH_STATE=1. "
                "Both modes publish has_collided and speed through the "
                "mode-independent `state` block; without it neither critic "
                "sees its own Markov state and the persistent collision "
                "penalty makes the reward depend on history the critic "
                "cannot see.")
        if os.environ.get("OBS_WITH_MODE", "0") != "1":
            raise SystemExit(
                "IMAGE_GOAL_TASK=1 with GATED=1 requires OBS_WITH_MODE=1: "
                "the mode is commanded per episode, so the actor must see "
                "the bit or the two modes are indistinguishable to it.")
        if os.environ.get("PERCEPTION", "vae").lower() not in ("vc1",
                                                               "vc1_vitb"):
            raise SystemExit(
                "IMAGE_GOAL_TASK=1 expects PERCEPTION=vc1. The arm's "
                "checkpoint identity, latent width (768) and edge-weight "
                "calibration are tied to the VC-1 backend; running it on "
                "another encoder is an experiment change, not a run option.")
        if os.environ.get("VC1_READOUT", "cls").lower() not in ("cls",
                                                                "grid2"):
            raise SystemExit(
                "IMAGE_GOAL_TASK=1 with GATED=1 requires VC1_READOUT=cls or "
                "grid2. The readout is part of the encoder's output shape; "
                "anything else changes the observation under a pairing the "
                "plan's checkpoints were sized for.")
        if os.environ.get("GATED_ONE_CRITIC", "0") != "1":
            raise SystemExit(
                "IMAGE_GOAL_TASK=1 with GATED=1 requires GATED_ONE_CRITIC=1. "
                "The unified reward puts both modes on ONE scale (every step "
                "< 0), so the two-critic split has nothing left to justify "
                "and the single bounded critic is the arm's design; a "
                "two-critic learner under one reward would train a second "
                "critic on the same signal and double the replay cost.")
        if os.environ.get("EPISODE_HER", "0") != "1":
            raise SystemExit(
                "IMAGE_GOAL_TASK=1 with GATED=1 requires EPISODE_HER=1: goal "
                "mode's returns are sparse against the 1000-step cap, and "
                "without hindsight relabelling the goal critic fits almost "
                "nothing but failures.")
        for _bound in ("TQC_TARGET_MIN", "TQC_TARGET_MAX"):
            if not os.environ.get(_bound):
                raise SystemExit(
                    f"IMAGE_GOAL_TASK=1 with GATED=1 requires {_bound} set. "
                    "The goal leg is all-negative but unbounded below except "
                    "by the step cap, and the one bounded critic "
                    "the arm runs clips its targets explicitly; running "
                    "without both bounds is the configuration the bounded "
                    "critic exists to replace, not a variant of it.")
        if os.environ.get("OBS_WITH_COLLISION", "0") == "1":
            raise SystemExit(
                "IMAGE_GOAL_TASK=1 with GATED=1 refuses OBS_WITH_COLLISION=1: "
                "the single-mode arm's 1-d `coll` key shares a run with the "
                "2-mode arm's 10-d `state` block, whose first two slots "
                "ALREADY carry has_collided and the speed ratio. Two keys "
                "for one flag means the critic's Markov repair is applied "
                "twice under different widths; the state block is the one "
                "the 2-mode arm trains on.")
        if (os.environ.get("LANE_FOLLOW", "0") == "1"
                or os.environ.get("LANE_EXPLORE", "0") == "1"
                or os.environ.get("TIME_TO_GOAL", "0") == "1"):
            raise SystemExit(
                "IMAGE_GOAL_TASK=1 must not stack another reward wrapper: "
                "its own reward is TwoModeImageGoalReward, and a second "
                "reward below it would be dead code the run banner would "
                "not reflect.")
        if os.environ.get("NO_TIMELIMIT", "0") == "1":
            raise SystemExit(
                "IMAGE_GOAL_TASK=1 with GATED=1 refuses NO_TIMELIMIT=1: it "
                "is a training run, and the trainer's TimeLimit must agree "
                "with the reward's own max_steps cap. NO_TIMELIMIT is a "
                "trace-collection setting; on a training run it silently "
                "changes what a 'done' step means.")
        return
    if os.environ.get("OBS_WITH_COLLISION", "0") != "1":
        raise SystemExit(
            "IMAGE_GOAL_TASK=1 requires OBS_WITH_COLLISION=1. The persistent "
            "collision penalty makes the reward depend on history; the critic "
            "must see z = has_collided or the MDP is not Markov and the "
            "critic fits noise.")
    if os.environ.get("PERCEPTION", "vae").lower() not in ("vc1", "vc1_vitb"):
        raise SystemExit(
            "IMAGE_GOAL_TASK=1 expects PERCEPTION=vc1. The arm's checkpoint "
            "identity, latent width (768) and edge-weight calibration are "
            "tied to the VC-1 backend; running it on another encoder is an "
            "experiment change, not a run option.")
    if (os.environ.get("LANE_FOLLOW", "0") == "1"
            or os.environ.get("LANE_EXPLORE", "0") == "1"
            or os.environ.get("TIME_TO_GOAL", "0") == "1"):
        raise SystemExit(
            "IMAGE_GOAL_TASK=1 must not stack another reward wrapper: its "
            "own reward is ImageGoalReward, and a second reward below it "
            "would be dead code the run banner would not reflect.")


def wrap_deg(angle_deg):
    """Wrap an angle difference in degrees to (-180, 180]."""
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped <= -180.0 else wrapped


def _speed_mps(info):
    """Speed in m/s. The env publishes km/h; absent data reads as stopped."""
    return float(info.get("speed_kmh", 0.0) or 0.0) / 3.6


def _goal_distance_m(info):
    """Along-road metres to the goal, with an euclidean fallback.

    The env publishes route_remaining_m when ROUTE_POTENTIAL is on and None
    otherwise; route_distance_to_goal_m carries the fallback already. If both
    are missing the wrapper recomputes the straight-line distance from the
    poses in info -- for a 2 m threshold the two coincide at the target.
    """
    for key in ("route_distance_to_goal_m", "route_remaining_m"):
        val = info.get(key, None)
        if val is not None:
            return float(val)
    hero = info.get("hero_pos")
    goal = info.get("goal_pos")
    if hero is not None and goal is not None:
        return float(np.linalg.norm(
            np.asarray(hero, np.float64)[:2] - np.asarray(goal, np.float64)[:2]))
    return float("inf")


def _driving_with_route_direction(info, disp):
    """The route-compatibility predicate, measured not asserted.

    Displacement projected on the lane's own forward vector, positive when the
    car moves the way the road points. A parked or creeping car moves nothing
    and is NOT counted as driving the wrong way: the carve-out keeps L_t True
    at standstill so the goal-mode stop condition does not pay -1.00 while it
    waits out the stop window. Since the 2026-09-20 removal of the flat
    in-lane bonus, L_t True at standstill also means a parked car still pays
    the speed term -SPEED_PREF_K * |v - v*| (it reads v = 0 against the
    20 km/h target) in both modes -- stopping remains the cheapest way to end
    the negative stream, so the agent is never paid to roll through the goal.
    """
    if _speed_mps(info) < STOP_SPEED_MPS:
        return True
    fwd = np.asarray(info.get("wp_forward", (0.0, 0.0)), np.float64)
    if not np.any(fwd):
        return False
    return bool(float(np.dot(np.asarray(disp, np.float64), fwd)) > 0.0)


def _speed_bonus(info, in_lane):
    """The saturating speed bonus (user spec 2026-09-20 evening), shared by
    BOTH legs: +SPEED_PREF_K * min(v, v*)/v* while in lane -- +K/step at or
    above v* = TARGET_SPEED/3.6, pro-rated below it, exactly 0 when parked.
    Not in lane -> 0.0 (the term is gated on L_t like the penalty it
    replaced)."""
    if not in_lane:
        return 0.0
    v = _speed_mps(info)
    if math.isnan(v):
        v = 0.0    # an unreadable speed reads as stopped, not as a free pass
    v_star = TARGET_SPEED_KMH / 3.6
    return SPEED_PREF_K * min(v, v_star) / v_star


def image_goal_reward(info, disp, has_collided, mode=GOAL):
    """One step's reward, as a pure function. Returns (reward, components).

    ONE REWARD, A MASKED SUM OF TWO LEGS (user amendment 2026-09-20). The
    mode is exclusive per episode, and each leg is zeroed outside its own
    mode:

        r_t = (mode == GOAL    ? goal_reward    : 0)
            + (mode == EXPLORE ? explore_reward : 0)

    so the reward is a SUM, never a branch that swaps formulas wholesale;
    components carry both masked terms ("goal_reward_term",
    "explore_reward_term") so a log line can show the composition.

    THE GOAL LEG (mode 1, the default): the ReViND invariant -- reward < 0
    at every step, success included -- is a property of the constants: the
    best case is -0.8 (in lane at v*, the saturating bonus at full tilt) and
    the worst ordinary row stacks all three penalties (-4.0 under the 1.5
    flats). No terminal here is ever positive. The flat penalties were
    raised 0.75 -> 1.5 by user revision (2026-09-20) to restore the
    in-lane/off-road contrast; the speed term became the saturating bonus
    and the two-mode branch made failures terminal the same evening.

    THE EXPLORE LEG (mode 0, user decisions 2026-09-20): failures are
    TERMINAL and there is NO TIME PRESSURE. The leg's step reward is the
    same saturating speed bonus the goal leg carries, alone (no base
    cost, no off-road penalty, no persistent-collision penalty), so a clean
    in-lane step at v* pays exactly +SPEED_PREF_K and the explore invariant
    is -TERMINAL_PENALTY <= r <= SPEED_PREF_K, with r > 0 possible. The
    FIRST off-road step, lane crossing or collision terminates
    the episode and pays a flat -TERMINAL_PENALTY: on each of those steps
    in_lane is False by construction, the ordinary component is exactly 0.0,
    and the failing step pays exactly -TERMINAL_PENALTY. A simulator-invalid
    step terminates and pays the same flat. The failure cause travels out in
    components["explore_end"] ("off_road" / "crossed" / "collision" /
    "invalid"); arrival at EXPLORE_GOAL_DIST_M and the step cap stay
    wrapper-level, and neither pays a penalty.
    """
    goal_r, goal_comp = _image_goal_reward_goal(info, disp, has_collided)
    explore_r, explore_comp = _image_goal_reward_explore(info, disp,
                                                         has_collided)
    # THE MASK. Each leg is zeroed outside its own mode; since the mode is
    # exclusive per episode exactly one term is nonzero and the sum equals
    # the leg the episode is in.
    goal_term = goal_r if mode == GOAL else 0.0
    explore_term = explore_r if mode == EXPLORE else 0.0
    r = float(np.nan_to_num(goal_term + explore_term, nan=0.0))

    if mode == GOAL:
        comp = dict(goal_comp)
    else:
        comp = dict(explore_comp)
    comp["goal_reward_term"] = goal_term
    comp["explore_reward_term"] = explore_term
    return r, comp


def _image_goal_reward_goal(info, disp, has_collided):
    """The goal leg of image_goal_reward: the CURRENT formula (2026-09-20
    evening).

    r = -BASE_STEP_COST + IN_LANE_BONUS + SPEED_PREF_K min(v, v*)/v* (if in
        lane) - OFFROAD_PENALTY (if off-road) - PERSISTENT_COLLISION_PENALTY
        (if the collision latch holds). The speed term is the SATURATING
        BONUS (user 2026-09-20 evening); the |v - v*| penalty form is gone.
        The two flat penalties are 1.5 (user revision 2026-09-20;
        env-overridable for the 0.75-row ablation). The best row is -0.8
        (in lane at v*), parked in lane is exactly the base cost, and every
        row stays < 0 -- the ReViND invariant survives the bonus.

    The failing-step flat (-IG_TERMINAL_PENALTY) is NOT added here: the
    pure function cannot know whether the caller terminates on failures
    (the single-mode arm does not), so the two-mode wrapper pays it when it
    terminates the episode (mirroring explore, whose leg owns the flat
    because explore ALWAYS terminates on one).
    """
    off_road = bool(info.get("off_road", False))
    crossed = bool(info.get("lane_invasion_any",
                            info.get("lane_any", False)))
    in_lane = (
        not off_road
        and _driving_with_route_direction(info, disp)
        and not crossed
        and not has_collided)

    speed_bonus = _speed_bonus(info, in_lane)

    r = -BASE_STEP_COST
    if in_lane:
        r += IN_LANE_BONUS + speed_bonus
    if off_road:
        r -= OFFROAD_PENALTY
    if has_collided:
        r -= PERSISTENT_COLLISION_PENALTY
    r = float(np.nan_to_num(r, nan=0.0))
    return r, {
        "base": -BASE_STEP_COST,
        "lane_bonus": IN_LANE_BONUS if in_lane else 0.0,
        "speed_penalty": speed_bonus,
        "offroad_penalty": -OFFROAD_PENALTY if off_road else 0.0,
        "collision_penalty": (-PERSISTENT_COLLISION_PENALTY
                              if has_collided else 0.0),
        "in_lane": in_lane,
        "off_road": off_road,
    }


def _image_goal_reward_explore(info, disp, has_collided):
    """The explore leg of image_goal_reward (user decisions 2026-09-20).

    No base cost, no flat penalties: the ordinary step is the SAME
    saturating speed bonus the goal leg carries (user 2026-09-20 evening --
    the |v - v*| penalty form is gone, so explore rows are >= 0 and the
    leg's range is [-TERMINAL_PENALTY, +SPEED_PREF_K]). The FIRST failure
    terminates and pays a flat -TERMINAL_PENALTY, exactly, because in_lane
    is False on every failing step (off-road, crossing and collision each
    zero the ordinary component by construction, and the invalid step's
    proprioception is garbage -- the destroyed hero's speed is not read as
    an ordinary reward). The cause order is the spec's: invalid > off-road >
    crossing > collision.
    """
    off_road = bool(info.get("off_road", False))
    crossed = bool(info.get("lane_invasion_any",
                            info.get("lane_any", False)))
    in_lane = (
        not off_road
        and _driving_with_route_direction(info, disp)
        and not crossed
        and not has_collided)

    speed_bonus = _speed_bonus(info, in_lane)

    invalid = bool(info.get("simulator_invalid", False))
    if invalid:
        cause = "invalid"
    elif off_road:
        cause = "off_road"
    elif crossed:
        cause = "crossed"
    elif has_collided:
        cause = "collision"
    else:
        cause = None

    if cause is not None:
        r = -TERMINAL_PENALTY
        speed_bonus = 0.0
    else:
        r = speed_bonus
    r = float(np.nan_to_num(r, nan=0.0))
    return r, {
        "speed_penalty": speed_bonus,
        "in_lane": in_lane,
        "off_road": off_road,
        "terminal_penalty": (-TERMINAL_PENALTY if cause is not None else 0.0),
        "explore_end": cause,
    }


def goal_at_target(info):
    """Position + orientation half of the success condition (not the stop)."""
    distance = _goal_distance_m(info)
    yaw_err = info.get("yaw_error_to_goal_deg", None)
    on_position = distance < GOAL_DISTANCE_THRESHOLD_M
    on_heading = (yaw_err is not None
                  and abs(float(yaw_err)) < GOAL_YAW_THRESHOLD_DEG)
    return on_position, on_heading


class ImageGoalReward(Wrapper):
    """Serves the image-goal reward; owns the persistent collision latch and
    the sustained-stop window, which are per-episode state the pure function
    must not carry."""

    def __init__(self, env, max_steps=MAX_STEPS):
        super().__init__(env)
        self.max_steps = int(max_steps)
        self._t = 0
        self._has_collided = False
        self._stop_window = 0
        self._prev = np.zeros(2)
        print(f"[ImageGoal] reward: -{BASE_STEP_COST:g} base "
              f"(flat in-lane bonus removed 2026-09-20; "
              f"IG_IN_LANE_BONUS={IN_LANE_BONUS:g})  "
              f"-{OFFROAD_PENALTY:g} off-road  "
              f"-{PERSISTENT_COLLISION_PENALTY:g} persistent-collision  "
              f"speed bonus +{SPEED_PREF_K:g} x min(v, "
              f"{TARGET_SPEED_KMH:g} km/h)/v* in-lane "
              f"(saturating, user 2026-09-20 evening)  "
              f"success_bonus={SUCCESS_REWARD:g}; success = speed<{STOP_SPEED_MPS:g}"
              f" m/s x{STOP_WINDOW_STEPS} steps while dist<"
              f"{GOAL_DISTANCE_THRESHOLD_M:g} m and |yaw|<"
              f"{GOAL_YAW_THRESHOLD_DEG:g} deg; collision impulse>"
              f"{COLLISION_IMPULSE_THRESHOLD:g} latches until reset; "
              f"MAX_STEPS={self.max_steps}. NOTHING ELSE TERMINATES.", flush=True)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        obs, info = out if isinstance(out, tuple) else (out, {})
        self._t = 0
        self._has_collided = False
        self._stop_window = 0
        self._prev = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
        return obs, info

    def step(self, action):
        obs, _env_r, _env_term, _env_trunc, info = self.env.step(action)
        self._t += 1
        pos = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
        disp = pos - self._prev
        self._prev = pos

        # THE PERSISTENT COLLISION FLAG. Latched on impulse magnitude, not on
        # the boolean event flag: the spec charges meaningful impacts, not
        # sensor noise, and the flag avoids making the reward depend on how
        # often the sensor re-reports one physical impact.
        impulse = float(info.get("collision_impulse", 0.0) or 0.0)
        if impulse > COLLISION_IMPULSE_THRESHOLD:
            self._has_collided = True

        reward, components = image_goal_reward(info, disp, self._has_collided)

        # THE SUSTAINED STOP. Success needs speed under the gate for the whole
        # window WHILE on position and heading; any step that breaks a check
        # restarts the window, so a drive-through can never collect it.
        on_position, on_heading = goal_at_target(info)
        stopped = _speed_mps(info) < STOP_SPEED_MPS
        if stopped and on_position and on_heading:
            self._stop_window += 1
        else:
            self._stop_window = 0
        success = self._stop_window >= STOP_WINDOW_STEPS

        invalid = bool(info.get("simulator_invalid", False))
        terminated = bool(success or invalid)
        truncated = (not terminated) and self._t >= self.max_steps
        # ARRIVAL BEATS THE CLOCK, NOT A FAILURE: there are no other
        # terminals, so the ordering concern the gated explore arm has does
        # not arise here.
        if success and not invalid:
            reward += SUCCESS_REWARD
            components["success_bonus"] = SUCCESS_REWARD

        info["is_success"] = bool(success)       # ours, not the env's
        info["ig_success"] = bool(success)
        info["has_collided"] = self._has_collided
        info["collision_impulse"] = impulse
        info["ig_reward_components"] = components
        info["mode"] = 1                          # goal-conditioned task
        info["gated_reward"] = reward
        return obs, reward, terminated, truncated, info


class TwoModeImageGoalReward(Wrapper):
    """ONE reward, TWO modes, drawn per reset -- the 2-mode image-goal arm.

    STRUCTURE MIRRORS GatedReward (mode draw at reset, `_t`, `_prev`,
    `_path_m`) but the reward is the SAME pure function for both modes:
    image_goal_reward(info, disp, has_collided, mode), a MASKED SUM of a
    goal leg and an explore leg (each zeroed outside its own mode), with the
    persistent collision latch and sustained-stop window carried HERE as
    per-episode state, exactly as ImageGoalReward carries them. The mode is
    passed through, so the two legs do the rest.

    GOAL MODE (mode 1): the image-goal task, as revised by the user on
    2026-09-20 (the flat penalties are 1.5) and the evening batch (the speed
    term is the SATURATING BONUS; failures are TERMINAL). The step reward is
    base cost, flat penalties, and the saturating speed bonus gated on L_t;
    success = speed under
    STOP_SPEED_MPS for STOP_WINDOW_STEPS consecutive steps while within
    GOAL_DISTANCE_THRESHOLD_M and within GOAL_YAW_THRESHOLD_DEG of the goal
    heading. The success step pays the ordinary reward -- no bonus, in either
    mode; termination is the only thing success buys. THE FIRST off-road
    step, lane crossing or collision (impulse over the latch threshold)
    TERMINATES the goal episode, paying the ordinary reward plus a flat
    -IG_TERMINAL_PENALTY on that step -- off-road -7.5, crossed -6.0,
    collision -7.5 (the latch sets has_collided -> L_t false -> bonus 0),
    all with terminated=True and NO ig_truncated cause (a termination, not
    a truncation).

    GOAL-MODE STUCK TRUNCATION (user decision 2026-09-20): sustained low
    speed no longer runs to the cap. Speed under IG_STUCK_SPEED_MPS=0.3 m/s
    for IG_STUCK_STEPS=100 CONSECUTIVE steps sets truncated=True (terminated
    stays False) with NO extra penalty -- the truncating step pays its
    ordinary reward. The cause travels in info["ig_truncated"] ("stuck",
    None otherwise). The stuck counter NEVER counts a step inside the success
    spatial/yaw gate (route distance under GOAL_DISTANCE_THRESHOLD_M and
    |yaw error| under GOAL_YAW_THRESHOLD_DEG -- position and heading, NOT the
    stop window's speed check), so parking at the goal reaches the sustained
    stop in STOP_WINDOW_STEPS=10 steps and can never be stuck-truncated
    first. The counter resets on wrapper reset and whenever the condition
    fails that step. The OFFROAD_TRUNC_STEPS counter this branch once
    carried is REMOVED (2026-09-20 evening): the first off-road step
    terminates, so there is nothing to debounce. ORDERING: success still
    wins -- it is evaluated first (terminated=True) and neither a
    failure-termination nor a truncation fires on a terminated step; a
    failure beats the stuck truncation (it terminates). Explore mode runs
    NO counter.

    EXPLORE MODE (mode 0): drive EXPLORE_GOAL_DIST_M of CUMULATIVE PATH, with
    NO TIME PRESSURE and TERMINAL FAILURES (user decisions 2026-09-20). The
    step reward is the SAME saturating speed bonus the goal leg carries --
    no base cost, no flat penalties -- so a clean in-lane step at v* pays
    exactly +IG_SPEED_PREF_K, a parked step exactly 0.0, and the explore
    invariant is -IG_TERMINAL_PENALTY <= r <= +IG_SPEED_PREF_K (positive
    explore rewards are legal, user 2026-09-20 evening). The FIRST off-road
    step
    (info["explore_end"]="off_road"), lane crossing ("crossed") or collision
    ("collision", the latch just set) TERMINATES the episode, paying a flat
    -IG_TERMINAL_PENALTY on that step; a simulator-invalid step terminates
    and pays the same flat ("invalid"). Arrival at EXPLORE_GOAL_DIST_M still
    terminates with NO bonus ("reached") and pays that step's ordinary
    reward (+IG_SPEED_PREF_K at v*). max_steps still TRUNCATES ("cap"): with
    no base cost a truncated clean episode returns ~+0.2/step, and the cap
    stays as a safety
    bound. ORDERING: invalid > failure > reached > cap -- a failure on the
    step the path reaches 100 m is still a failure; arrival beats the cap.

    The persistent-collision LATCH stays for info["has_collided"] statistics
    in both modes; in explore it now coincides with termination (the first
    latching step ends the episode), in goal mode the latching step is the
    terminating one (the persistent penalty and the terminal flat stack on
    it, -7.5). info["is_success"] is the goal mode's success only --
    explore arrival is not task success and must not feed the HER/eval
    success counters as one.
    """

    def __init__(self, env, explore_prob=0.5, force_mode=None, max_steps=1000,
                 seed=0):
        super().__init__(env)
        self.explore_prob = float(explore_prob)
        self.force_mode = force_mode
        self.max_steps = int(max_steps)
        self._rng = np.random.default_rng(seed)
        self.mode = GOAL
        self._t = 0
        self._prev = np.zeros(2)
        self._path_m = 0.0
        self._has_collided = False
        self._stop_window = 0
        self._stuck_steps = 0
        # A SETTING THAT CHANGES THE REWARD MUST SAY SO ON SCREEN (the repo's
        # own rule after the observation-banner drift): the two knobs an
        # evaluator will ask about and the two constants the reward is
        # computed from. BOTH BRANCHES are named: goal mode carries the 1.5
        # flats and the 2026-09-20-evening terminal failures, explore mode
        # is the terminal-failure design of 2026-09-20 with the bonus.
        print(f"[TwoModeIG] unified image-goal reward, TWO modes: "
              f"explore_prob={self.explore_prob} force_mode={self.force_mode} "
              f"EXPLORE_GOAL_DIST_M={EXPLORE_GOAL_DIST_M:g} "
              f"IG_SPEED_PREF_K={SPEED_PREF_K:g} (v*={TARGET_SPEED_KMH:g} km/h,"
              f" saturating bonus, user 2026-09-20 evening);"
              f" goal mode: base cost, flat penalties "
              f"off={OFFROAD_PENALTY:g} coll={PERSISTENT_COLLISION_PENALTY:g}"
              f" (raised 0.75 -> 1.5, user 2026-09-20), speed bonus;"
              f" sustained-stop success, no bonus; off-road / crossing / "
              f"collision TERMINATE GOAL mode too, paying "
              f"-IG_TERMINAL_PENALTY={TERMINAL_PENALTY:g} on top (user "
              f"2026-09-20 evening); explore mode: no base cost, no flat "
              f"penalties, speed bonus only -- off-road / crossing / "
              f"collision TERMINATE "
              f"paying -IG_TERMINAL_PENALTY={TERMINAL_PENALTY:g} (invalid "
              f"too); explore arrival terminates with NO bonus; "
              f"goal-mode stuck truncation (user 2026-09-20): "
              f"speed<"
              f"{STUCK_SPEED_MPS:g} m/s x IG_STUCK_STEPS={STUCK_STEPS} "
              f"consecutive steps outside the success gate truncate "
              f"(truncated=True, no extra penalty; the off-road debounce "
              f"counter is gone -- first off-road step terminates); "
              f"max_steps={self.max_steps}.", flush=True)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        obs, info = out if isinstance(out, tuple) else (out, {})
        # DRAWN ONCE PER EPISODE, never mid-episode: the mode must be constant
        # for a transition's whole return, exactly as in GatedReward.
        self.mode = (int(self.force_mode) if self.force_mode is not None
                     else (EXPLORE if self._rng.random() < self.explore_prob
                           else GOAL))
        self._t = 0
        self._path_m = 0.0
        self._prev = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
        # THE PER-EPISODE STATE. The collision latch and the stop window are
        # the same objects ImageGoalReward owns; under one wrapper serving
        # both modes they reset with the episode, not with the mode. The
        # goal-mode truncation counters (2026-09-20) join them.
        self._has_collided = False
        self._stop_window = 0
        self._stuck_steps = 0
        info["mode"] = self.mode
        return obs, info

    def step(self, action):
        obs, _env_r, _env_term, _env_trunc, info = self.env.step(action)
        self._t += 1
        pos = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
        disp = pos - self._prev
        self._prev = pos
        self._path_m += float(np.linalg.norm(disp))
        info["explore_dist_m"] = float(self._path_m)

        # THE PERSISTENT COLLISION FLAG, one latch for both modes (ONE
        # reward): latched on impulse magnitude, not on the event flag, so
        # the sensor's re-reports cannot re-charge the penalty.
        impulse = float(info.get("collision_impulse", 0.0) or 0.0)
        if impulse > COLLISION_IMPULSE_THRESHOLD:
            self._has_collided = True

        # THE MASKED SUM. One pure function, two legs: the mode zeroes the
        # other leg (never a formula swap), so the step reward is
        # (goal leg in goal) + (explore leg in explore), and what terminates
        # changes with the mode.
        reward, components = image_goal_reward(info, disp, self._has_collided,
                                               mode=self.mode)

        invalid = bool(info.get("simulator_invalid", False))
        if self.mode == GOAL:
            # THE SUSTAINED STOP, goal mode only. A car that reaches the
            # target at speed never collects the window; any step that
            # breaks a check restarts it.
            on_position, on_heading = goal_at_target(info)
            # THE SUCCESS SPATIAL/YAW GATE: position and heading only -- the
            # stuck counter's exemption (below) uses it WITHOUT the stop
            # window's speed check.
            in_goal_gate = on_position and on_heading
            stopped = _speed_mps(info) < STOP_SPEED_MPS
            if stopped and in_goal_gate:
                self._stop_window += 1
            else:
                self._stop_window = 0
            success = self._stop_window >= STOP_WINDOW_STEPS

            # GOAL-MODE STUCK TRUNCATION (user decision 2026-09-20):
            # sustained low speed truncates instead of running to the cap. A
            # step inside the success spatial/yaw gate NEVER counts toward
            # stuck (parking at the goal must reach the sustained stop, not
            # the truncation), so the gate is the same position/heading pair
            # WITHOUT the speed check. The off-road counter this block once
            # carried is GONE (2026-09-20 evening): the first off-road step
            # TERMINATES below, so there is nothing to debounce.
            if _speed_mps(info) < STUCK_SPEED_MPS and not in_goal_gate:
                self._stuck_steps += 1
            else:
                self._stuck_steps = 0

            info["is_success"] = bool(success)
            info["arrived_this_step"] = bool(success)
            # SUCCESS/INVALID STILL WIN THE STEP (user ruling): terminated is
            # evaluated first, and neither a failure-termination nor a
            # truncation ever fires on a terminated step.
            terminated = bool(success or invalid)
            truncated = (not terminated) and self._t >= self.max_steps
            # GOAL-MODE TERMINAL FAILURES (user decision 2026-09-20 evening):
            # the FIRST off-road step, lane crossing or collision terminates
            # (truncated stays False, ig_truncated stays None -- a
            # termination, not a truncation) and pays the ordinary reward
            # plus the flat -IG_TERMINAL_PENALTY, mirroring explore. The
            # flat lands in components["terminal_penalty"] so the sum
            # identity stays exact (the key is always present, 0.0 on clean
            # steps, exactly like the explore leg's); success and invalid
            # keep their step and pay no flat. ORDINARY ROWS: off-road -2.5
            # (+ -5 = -7.5), crossed -1.0 (+ -5 = -6.0), collision -2.5
            # (+ -5 = -7.5, the latch sets has_collided -> L_t false ->
            # bonus 0).
            components.setdefault("terminal_penalty", 0.0)
            if not terminated:
                crossed = bool(info.get("lane_invasion_any",
                                        info.get("lane_any", False)))
                off_road = bool(info.get("off_road", False))
                if off_road or crossed or self._has_collided:
                    terminated, truncated = True, False
                    reward -= TERMINAL_PENALTY
                    components["terminal_penalty"] = -TERMINAL_PENALTY
            trunc_cause = None
            if not terminated and self._stuck_steps >= STUCK_STEPS:
                trunc_cause = "stuck"
                truncated = True
            info["ig_truncated"] = trunc_cause
        else:
            # EXPLORE MODE (user decision 2026-09-20): failures are TERMINAL
            # and there is no time pressure. The pure function already paid
            # the flat -IG_TERMINAL_PENALTY and stamped the cause into
            # components["explore_end"] when this step failed (invalid >
            # off-road > crossing > collision); the wrapper only decides
            # WHICH terminal won: invalid/failure beats arrival beats cap.
            # ARRIVAL BEATS THE CAP BUT NOT A FAILURE: a car that crosses a
            # line on the step it passes the distance still crossed it, and
            # the failure's terminal penalty is what it pays.
            info["is_success"] = False
            info["arrived_this_step"] = False
            terminated = False
            truncated = self._t >= self.max_steps
            cause = components.get("explore_end")
            if cause is not None:
                terminated, truncated = True, False
                info["explore_end"] = cause
            elif (EXPLORE_GOAL_DIST_M > 0.0
                    and self._path_m >= EXPLORE_GOAL_DIST_M):
                terminated, truncated = True, False
                info["explore_end"] = "reached"
            elif truncated:
                info["explore_end"] = "cap"
            else:
                terminated = False

        info["mode"] = self.mode
        info["has_collided"] = self._has_collided
        info["collision_impulse"] = impulse
        info["ig_reward_components"] = components
        info["gated_reward"] = reward
        return obs, reward, terminated, truncated, info