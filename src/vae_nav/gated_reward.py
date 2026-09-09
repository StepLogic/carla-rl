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

    goal mode     LaneFollowReward: delta_distance + projected_distance,
                  speed-gated, plus terminal penalties. Returns of order 100.
    explore mode  the goal-free half: |displacement| or lane projection, base
                  penalty, steer-change, terminal penalties. Returns of order 10.

A single critic fitted to both would be dominated by the larger. The gated
learner therefore keeps one critic per mode, and this wrapper's only job is to
say which mode a transition belongs to.

THE REASON IS RETURN SCALE, NOT DISCOUNT -- correcting a rationale imported from
the Habitat peer that does NOT hold here. There, goal mode runs gamma=1 against
explore's 0.99, so the two critics fit different horizons. In CARLA both lane
arms run gamma=0.99: run_lanefollow.sh never sets DISCOUNT and inherits
train_carla_tqc.py's 0.99 default, and run_lanefollow_nogoal.sh sets 0.99
explicitly. gamma=1.0 belongs to the TIME_TO_GOAL arm alone (run_timetogoal.sh).
So two critics are still right, but because the returns differ by an order of
magnitude and TQC's kappa is derived from that scale -- not because the horizons
differ.

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
    if info.get("lane_any") or info.get("off_road"):
        r -= LF_K_LANE
    if info.get("is_success"):
        r += LF_K_GOAL_BONUS
    return float(np.nan_to_num(r, nan=0.0))


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
    fatal = lane_departure_is_fatal(
        bool(info.get("lane_any", False)), info.get("lane_marking_types"),
        bool(info.get("in_junction", False)), bool(info.get("off_road", False)))
    if fatal:
        r -= LANE_DEPARTURE_PENALTY
    return float(np.nan_to_num(r, nan=0.0)), fatal


class GatedReward(Wrapper):
    """Draw a mode per episode; return that mode's reward and termination."""

    def __init__(self, env, explore_prob=EXPLORE_PROB, force_mode=None,
                 max_steps=500, delta_mode="positive", seed=0):
        super().__init__(env)
        self.explore_prob = float(explore_prob)
        self.force_mode = force_mode
        self.max_steps = int(max_steps)
        self.delta_mode = delta_mode
        self._rng = np.random.default_rng(seed)
        self.mode = GOAL
        self._t = 0
        self._prev = np.zeros(2)
        print(f"[Gated] explore_prob={self.explore_prob} "
              f"force_mode={self.force_mode} max_steps={self.max_steps} "
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

        if self.mode == GOAL:
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
