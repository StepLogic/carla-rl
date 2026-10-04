"""The image-goal navigation reward: ReViND-style, all-negative until success.

THE TASK. Reach a physical target viewpoint -- the viewpoint the goal image was
taken from -- and STOP there, facing the goal's road heading. Success is judged
on what the deployed policy will have onboard: current proprioception and the
goal. The reward itself may read privileged CARLA labels; the claim it supports
does not depend on them at execution time.

THE SHAPE (user spec 2026-09-18, revised twice in review):

    r_t = -1.0                              the base cost, EVERY step
        + 0.75   if L_t                     the in-lane preference
        - 0.75   if off the drivable surface (each step it persists)
        - 0.75   if a collision has EVER happened this episode

    L_t = on the drivable lane, moving in the route direction, no lane
          crossing this step, and no collision yet.

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

WHY COLLISION IS A STATE, NOT AN EVENT. One impact is one sensor reading;
the DAMAGE persists. A one-time event penalty is paid once and then forgotten,
so a car that crashes early and drives on afterwards looks identical to one
that never crashed. The persistent flag keeps the post-collision trajectory in
a lower-reward mode until reset, and it must be visible to the value function
(Phase 2: critic conditioning z = has_collided) or the same apparent image has
two different values depending on unseen history -- a Markov violation.

WHY NOTHING HERE TERMINATES except success, invalid, and the clock. The spec
removes every terminal escape: a crossing costs the bonus it loses, off-road
costs per step, a crash costs per step from then on. Ending the episode early
buys nothing -- termination STOPS a negative stream, and it is granted only for
the thing the task is (stopped at the goal) or the thing the sim cannot
recover from.

DEVIATIONS FROM THE WRITTEN SPEC, all deliberate:
  - success pays no bonus (decision above);
  - the yaw reference is the GOAL WAYPOINT's road heading, which is the yaw the
    goal camera was spawned at, so "matches the intended orientation" means
    facing along the road as in the goal image;
  - "driving_with_route_direction" is measured, not asserted: displacement
    projected on the lane's forward vector, and a parked or creeping car
    (speed under STOP_SPEED_MPS) is NOT punished for not moving -- otherwise
    the sustained stop the success condition demands would cost the lane bonus
    for its whole window and bias the agent against stopping;
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
IN_LANE_BONUS = float(os.environ.get("IG_IN_LANE_BONUS", 0.75))
OFFROAD_PENALTY = float(os.environ.get("IG_OFFROAD_PENALTY", 0.75))
PERSISTENT_COLLISION_PENALTY = float(
    os.environ.get("IG_PERSISTENT_COLLISION_PENALTY", 0.75))
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

IMAGE_GOAL_TASK = os.environ.get("IMAGE_GOAL_TASK", "0") == "1"


def image_goal_coherence():
    """Refuse image-goal configurations that cannot learn, BEFORE CARLA starts.

    Mirrors gated_config.check_gated_run: a run that starts and then dies at
    step 1 has still cost a server start, a world load and a GPU share.
    """
    if not IMAGE_GOAL_TASK:
        return
    if os.environ.get("GATED", "0") == "1":
        raise SystemExit(
            "IMAGE_GOAL_TASK=1 with GATED=1 is not a supported pairing: the "
            "image-goal arm is a single-mode run. GatedReward draws a mode "
            "per episode and two critics; ImageGoalReward defines its own "
            "single reward and termination. Run one or the other.")
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
    and is NOT counted against the lane bonus: the sustained stop the success
    condition demands would otherwise pay -1.00 while it waits, and the agent
    would be paid to roll through the goal instead of stopping on it.
    """
    if _speed_mps(info) < STOP_SPEED_MPS:
        return True
    fwd = np.asarray(info.get("wp_forward", (0.0, 0.0)), np.float64)
    if not np.any(fwd):
        return False
    return bool(float(np.dot(np.asarray(disp, np.float64), fwd)) > 0.0)


def image_goal_reward(info, disp, has_collided):
    """One step's reward, as a pure function. Returns (reward, components).

    The ReViND invariant -- reward < 0 at every step, success included -- is a
    property of the constants: the best case is -1.0 + IN_LANE_BONUS < 0 and
    the worst stacks all three penalties. No terminal here is ever positive.
    """
    off_road = bool(info.get("off_road", False))
    crossed = bool(info.get("lane_invasion_any",
                            info.get("lane_any", False)))
    in_lane = (
        not off_road
        and _driving_with_route_direction(info, disp)
        and not crossed
        and not has_collided)

    r = -BASE_STEP_COST
    if in_lane:
        r += IN_LANE_BONUS
    if off_road:
        r -= OFFROAD_PENALTY
    if has_collided:
        r -= PERSISTENT_COLLISION_PENALTY
    r = float(np.nan_to_num(r, nan=0.0))
    return r, {
        "base": -BASE_STEP_COST,
        "lane_bonus": IN_LANE_BONUS if in_lane else 0.0,
        "offroad_penalty": -OFFROAD_PENALTY if off_road else 0.0,
        "collision_penalty": (-PERSISTENT_COLLISION_PENALTY
                              if has_collided else 0.0),
        "in_lane": in_lane,
        "off_road": off_road,
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
              f"+{IN_LANE_BONUS:g} in-lane  -{OFFROAD_PENALTY:g} off-road  "
              f"-{PERSISTENT_COLLISION_PENALTY:g} persistent-collision  "
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