"""Per-step distance reward, gated on speed. Lane following with a goal.

WHY THIS EXISTS RATHER THAN SparseGoalReward. That wrapper discards the env's
reward entirely and substitutes +600 on arrival, -5 collision, -5 lane. Nothing
in it is a function of the ACTIONS, so bang-bang control is not a failure mode
under it -- it is optimal. MEASURED on the policy trained that way: p95
|acceleration| 196 m/s^2, 79.5% of steps above the 4 m/s^2 harsh threshold,
steering reversing on 32.6% of steps, 123 km/h in a town. It reached goals
(0.808 in-distribution, 0.776 held-out, zero collisions) despite how it drove.

This wrapper instead uses the reward the env config ALREADY computes, which is
the shape used by the upstream nisheeth-golakiya/carla-rl repo this config was
forked from (train_env_config_v4_image_goal.py:639-641):

    if speed_factor < 1.0:                  # speed <= target_speed
        reward += delta_distance + projected_distance

so most of the work here is not discarding it.

THE DISTANCE TERM AND ITS RESIDUAL EXPOSURE. delta_distance is the per-step
REDUCTION in distance to the goal (:595); last_distance is updated every step
(:596), so it is genuinely per-step and not an accumulator. The default is
clip(delta, 0, None): retreating pays nothing, approaching pays the reduction.
That removes the worst case -- an abs() form pays identically for retreating, so
a car oscillating about a point earns forever -- but does NOT fully close it,
because the approach half of a loop still pays:

                        toward   away    closed loop (there and back)
    abs(delta)           +0.3    +0.3          0.6
    clip(delta, 0, None) +0.3     0.0          0.3    <- default, as specified
    delta (signed)       +0.3    -0.3          0.0    <- only loop-neutral form

Only the signed form telescopes to d_start - d_end. DELTA_MODE switches between
them at no cost. The tripwire is behavioural: if mean distance-to-goal stops
falling while episode reward keeps rising, the term is being farmed.

ORDERING GOTCHA. carla_env.py:82-84 runs get_done_status BEFORE compute_reward,
and distance_completed is only incremented inside compute_reward (:622-623), so
the env's done_dist lags one step. Everything here reads the CURRENT step's
info, never an accumulator.
"""
import os

import numpy as np

try:
    from gymnasium import Wrapper
except ImportError:
    from gym import Wrapper

# How the per-step goal-progress term is taken. See the module docstring.
DELTA_MODE = os.environ.get("DELTA_MODE", "positive")   # positive | signed | abs

# Terminal-event penalties. Small relative to the accumulated distance reward,
# because the real deterrent is the episode ENDING and forfeiting the rest of
# the run, not the magnitude of the penalty itself.
K_LANE = float(os.environ.get("LF_K_LANE", 1.0))
K_COLLISION = float(os.environ.get("LF_K_COLLISION", 1.0))
K_IDLE = float(os.environ.get("LF_K_IDLE", 1.0))

# Charged every step on |change in steer|. The env already computes this as
# `action_smoothness_penalty` (:719); it is applied here rather than recomputed
# so the two cannot drift.
K_STEER_CHANGE = float(os.environ.get("LF_K_STEER_CHANGE", 1.0))

# Paid once when the env's own distance target (MAX_DISTANCE_M, default 200 m)
# is met -- the upstream repo's "+1 on travelling 200 m".
K_GOAL_BONUS = float(os.environ.get("LF_K_GOAL_BONUS", 1.0))
K_CHECKPOINT = float(os.environ.get("LF_K_CHECKPOINT", 1.0))


def delta_term(delta, mode=None):
    """Apply DELTA_MODE to a per-step goal-distance reduction."""
    mode = DELTA_MODE if mode is None else mode
    if mode == "signed":
        return float(delta)
    if mode == "abs":
        return abs(float(delta))
    return float(max(delta, 0.0))          # "positive": clip at 0


class LaneFollowReward(Wrapper):
    """Reward = per-step progress while under target speed, minus failures."""

    def __init__(self, env, delta_mode=None, k_lane=K_LANE,
                 k_collision=K_COLLISION, k_idle=K_IDLE,
                 k_steer_change=K_STEER_CHANGE, k_goal_bonus=K_GOAL_BONUS,
                 k_checkpoint=K_CHECKPOINT, terminal=True):
        super().__init__(env)
        self.delta_mode = delta_mode or DELTA_MODE
        self.k_lane = float(k_lane)
        self.k_collision = float(k_collision)
        self.k_idle = float(k_idle)
        self.k_steer_change = float(k_steer_change)
        self.k_goal_bonus = float(k_goal_bonus)
        self.k_checkpoint = float(k_checkpoint)
        self.terminal = bool(terminal)
        if self.delta_mode not in ("positive", "signed", "abs"):
            raise SystemExit(f"DELTA_MODE={self.delta_mode!r} is not one of "
                             "positive | signed | abs")
        if self.delta_mode == "abs":
            print("[LaneFollow] WARNING: DELTA_MODE=abs pays the same for "
                  "retreating as approaching, so oscillating about a point "
                  "earns indefinitely without arriving.")
        print(f"[LaneFollow] delta_mode={self.delta_mode} "
              f"k_lane={self.k_lane} k_collision={self.k_collision} "
              f"k_idle={self.k_idle} k_steer_change={self.k_steer_change} "
              f"terminal={self.terminal}")

    def step(self, action):
        obs, _env_reward, terminated, truncated, info = self.env.step(action)

        # SPEED GATE. The env's own gate is `speed_factor < 1.0`, i.e. speed
        # below target_speed; speed_factor is already published (:717) and is
        # clipped to [0, 1] there, so it is read from the raw speeds instead to
        # avoid the saturation making the gate always-true.
        speed = float(info.get("speed_kmh", 0.0) or 0.0)
        target = float(info.get("target_speed_kmh", 0.0) or 0.0)
        under_target = (speed <= target) if target > 0 else True

        delta = float(info.get("delta_distance", 0.0) or 0.0)
        progress = delta_term(delta, self.delta_mode) if under_target else 0.0

        hit = bool(info.get("collision", False))
        lane = bool(info.get("lane_invasion", False))
        reasons = info.get("done_reasons") or {}
        idle = bool(reasons.get("idle", False))
        arrived = bool(info.get("is_success", False))
        checkpoint = bool(info.get("arrived", False))

        # Steering smoothness, charged unconditionally, exactly as the env does
        # at :635. This is the only term that is a function of the ACTIONS, and
        # its absence is why the previous reward permitted 196 m/s^2.
        steer_pen = self.k_steer_change * float(
            info.get("action_smoothness_penalty", 0.0) or 0.0)

        reward = (progress
                  + (self.k_goal_bonus if arrived else 0.0)
                  + (self.k_checkpoint if checkpoint else 0.0)
                  - (self.k_collision if hit else 0.0)
                  - (self.k_lane if lane else 0.0)
                  - (self.k_idle if idle else 0.0)
                  - steer_pen)

        task_terminal = bool(self.terminal and (hit or lane or idle or arrived))

        info["lf_progress"] = float(progress)
        info["lf_steer_penalty"] = float(steer_pen)
        info["lf_under_target_speed"] = bool(under_target)
        info["lf_delta_distance"] = delta
        info["is_success"] = arrived

        return (obs, float(reward), task_terminal,
                bool(truncated or (terminated and not task_terminal)), info)
