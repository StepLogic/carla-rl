"""The image-goal navigation reward: ReViND-style, all-negative until done.

THE INVARIANT THE WHOLE FILE GUARDS, stated once here: with the success bonus
removed (user decision 2026-09-18), EVERY step pays a negative reward --
including the success step. The best score available to a step is -0.25
(in-lane), the worst is -2.50 (post-collision, off-road). Termination buys
nothing except the end of the negative stream, and it is granted only for the
thing the task IS (sustained stop at the target viewpoint and heading), a
destroyed hero, or the clock.

The reward table under test (user spec 2026-09-18):

    in-lane, no collision              -0.25
    one-step lane crossing             -1.00
    off-road                           -1.75
    post-collision, back on the road   -1.75
    post-collision AND off-road        -2.50
    success (sustained stop on target) ordinary reward, terminated, NO bonus
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

pytest.importorskip("gymnasium")

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402

from vae_nav.image_goal_reward import (ImageGoalReward,  # noqa: E402
                                       STOP_WINDOW_STEPS, wrap_deg)

ACT = np.zeros(2, np.float32)
# One metre of forward motion per step along the lane's own forward vector,
# at 1 m/s: the "driving the route, in lane" state.
DRIVING = {
    "wp_forward": (1.0, 0.0),
    "speed_kmh": 3.6,                       # 1 m/s
    "route_distance_to_goal_m": 50.0,
    "yaw_error_to_goal_deg": 0.0,
    "off_road": False,
    "lane_invasion_any": False,
    "collision_impulse": 0.0,
    "simulator_invalid": False,
}


class _StubEnv(gym.Env):
    """Steps `advance` (metres) along +x each step, so disp is real."""
    metadata = {}

    def __init__(self, info=None, done_from_env=False, advance=(1.0, 0.0)):
        self.observation_space = spaces.Dict({})
        self.action_space = spaces.Box(
            np.array([-0.5, 0.0], np.float32), np.array([0.5, 0.6], np.float32))
        self.info = dict(DRIVING)
        self.info.update(info or {})
        self.done_from_env = done_from_env
        self.advance = np.asarray(advance, np.float64)
        self._n = 0

    def reset(self, **kw):
        self._n = 0
        return np.zeros(4, np.float32), dict(self.info)

    def step(self, action):
        self._n += 1
        pos = self.advance * self._n
        info = dict(self.info)
        info["hero_pos"] = (float(pos[0]), float(pos[1]))
        return (np.zeros(4, np.float32), 0.0,
                bool(self.done_from_env), False, info)


def _wrap(info=None, max_steps=1000, done_from_env=False, advance=(1.0, 0.0)):
    env = _StubEnv(info, done_from_env, advance)
    w = ImageGoalReward(env, max_steps=max_steps)
    w.reset()
    return w, env


# ------------------------------------------------------------ the reward ----

def test_in_lane_driving_pays_minus_a_quarter():
    w, _ = _wrap()
    assert w.step(ACT)[1] == pytest.approx(-0.25)


def test_a_lane_crossing_step_pays_minus_one_and_continues():
    w, _ = _wrap({"lane_invasion_any": True})
    _, reward, terminated, truncated, _ = w.step(ACT)
    assert (reward, terminated, truncated) == (pytest.approx(-1.0), False, False)


def test_off_road_pays_minus_175():
    w, _ = _wrap({"off_road": True})
    _, reward, terminated, *_ = w.step(ACT)
    assert reward == pytest.approx(-1.75)
    assert terminated is False


def test_a_collision_latches_and_costs_every_later_step():
    w, env = _wrap()
    w.step(ACT)
    env.info["collision_impulse"] = 5.0
    _, reward, terminated, *_ = w.step(ACT)
    assert terminated is False
    assert reward == pytest.approx(-1.75)     # the impact step itself
    env.info["collision_impulse"] = 0.0       # recovered to in-lane driving
    assert w.step(ACT)[1] == pytest.approx(-1.75)   # STILL, on the road


def test_sub_threshold_impulse_does_not_latch():
    w, env = _wrap()
    env.info["collision_impulse"] = 0.5       # sensor noise, not an impact
    w.step(ACT)
    env.info["collision_impulse"] = 0.0
    assert w._has_collided is False
    assert w.step(ACT)[1] == pytest.approx(-0.25)


def test_post_collision_off_road_pays_minus_250():
    w, env = _wrap()
    env.info["collision_impulse"] = 5.0
    w.step(ACT)
    env.info["collision_impulse"] = 0.0
    env.info["off_road"] = True
    assert w.step(ACT)[1] == pytest.approx(-2.50)


def test_the_flag_resets_between_episodes():
    w, env = _wrap()
    env.info["collision_impulse"] = 5.0
    w.step(ACT)
    env.info["collision_impulse"] = 0.0
    w.reset()
    assert w.step(ACT)[1] == pytest.approx(-0.25)


# ------------------------------------------------- the ReViND invariant ----

def test_every_step_is_negative_including_success():
    # THE STRENGTHENED INVARIANT. The success bonus was removed, so no
    # transition anywhere pays >= 0. Walk every state in the reward table.
    w, env = _wrap()
    w.reset()
    states = [DRIVING, {**DRIVING, "lane_invasion_any": True},
              {**DRIVING, "off_road": True},
              {**DRIVING, "speed_kmh": 0.0, "advance": None}]
    for state in states:
        env.info.update({k: v for k, v in state.items() if k != "advance"})
        assert w.step(ACT)[1] < 0.0


# ---------------------------------------------------------------- success ----

def _stopped(**extra):
    info = {"hero_pos": (0.0, 0.0), "speed_kmh": 0.0,
            "route_distance_to_goal_m": 1.0, "yaw_error_to_goal_deg": 0.0}
    info.update(extra)
    return _wrap(info, advance=(0.0, 0.0))


def test_a_single_slow_step_is_not_success():
    w, _ = _stopped()
    _, _, terminated, *_ = w.step(ACT)
    assert terminated is False


def test_success_fires_on_the_last_window_step_and_pays_no_bonus():
    w, _ = _stopped()
    for i in range(STOP_WINDOW_STEPS - 1):
        _, _, terminated, *_ = w.step(ACT)
        assert terminated is False, f"fired early at step {i + 1}"
    _, reward, terminated, truncated, info = w.step(ACT)
    assert terminated is True and truncated is False
    assert reward == pytest.approx(-0.25)     # the ordinary reward, NOT +10
    assert info["is_success"] is True


def test_moving_breaks_the_stop_window():
    w, env = _stopped()
    for _ in range(STOP_WINDOW_STEPS - 1):
        w.step(ACT)
    env.info["speed_kmh"] = 10.8              # 3 m/s: rolls through
    assert w.step(ACT)[2] is False
    env.info["speed_kmh"] = 0.0
    assert w.step(ACT)[2] is False            # window restarted, step 1 of 10


def test_being_off_target_breaks_the_stop_window():
    w, env = _stopped()
    for _ in range(STOP_WINDOW_STEPS - 1):
        w.step(ACT)
    env.info["route_distance_to_goal_m"] = 50.0
    assert w.step(ACT)[2] is False


def test_success_outside_the_yaw_gate_never_fires():
    w, _ = _stopped(yaw_error_to_goal_deg=90.0)
    for _ in range(STOP_WINDOW_STEPS + 3):
        assert w.step(ACT)[2] is False


def test_a_drive_through_at_speed_is_not_success():
    # Arriving at the goal point AT SPEED is exactly what the stop window
    # exists to reject: the car never holds the gate for the full window.
    w, _ = _wrap({"route_distance_to_goal_m": 0.5, "speed_kmh": 21.6})  # 6 m/s
    for _ in range(STOP_WINDOW_STEPS + 3):
        assert w.step(ACT)[2] is False


# ------------------------------------------------------------ termination ----

def test_timeout_truncates_with_the_ordinary_reward():
    w, _ = _wrap(max_steps=5)
    rewards = [w.step(ACT)[1] for _ in range(5)]
    assert all(r < 0 for r in rewards)
    assert w.step(ACT)[3] is True


def test_invalid_terminates():
    w, _ = _wrap({"simulator_invalid": True})
    assert w.step(ACT)[2] is True


def test_the_envs_own_done_is_discarded():
    # The env ORs lane/collision/distance into done; this reward owns
    # termination and none of those are terminal under the spec.
    w, _ = _wrap(done_from_env=True)
    assert w.step(ACT)[2] is False


# ------------------------------------------------------- route direction ----

def test_a_parked_car_keeps_the_lane_bonus():
    # The sustained stop the success condition demands must not pay -1.00
    # while it waits: a stationary car is not driving the wrong way.
    w, _ = _stopped()
    assert w.step(ACT)[1] == pytest.approx(-0.25)


def test_driving_against_the_lane_forward_loses_the_bonus():
    w, _ = _wrap(advance=(-1.0, 0.0))         # moves -x while the road points +x
    assert w.step(ACT)[1] == pytest.approx(-1.0)


def test_no_forward_vector_means_no_direction_credit_while_moving():
    w, _ = _wrap({"wp_forward": (0.0, 0.0)})
    assert w.step(ACT)[1] == pytest.approx(-1.0)


# ---------------------------------------------------------------- helpers ----

def test_wrap_deg_handles_the_seam():
    assert wrap_deg(-180) == pytest.approx(180.0)
    assert wrap_deg(180) == pytest.approx(180.0)
    assert wrap_deg(350) == pytest.approx(-10.0)
    assert wrap_deg(-370) == pytest.approx(-10.0)
    assert wrap_deg(725) == pytest.approx(5.0)


def test_euclidean_fallback_when_no_route_distance_is_published():
    # No route keys at all: the straight-line distance decides the gate.
    near, _ = _wrap({"route_distance_to_goal_m": None,
                     "route_remaining_m": None, "goal_pos": (1.5, 0.0),
                     "speed_kmh": 0.0},
                    advance=(0.0, 0.0))
    for _ in range(STOP_WINDOW_STEPS):
        _, _, terminated, *_ = near.step(ACT)
    assert terminated is True                 # 1.5 m < the 2 m gate

    far, _ = _wrap({"route_distance_to_goal_m": None,
                    "route_remaining_m": None, "goal_pos": (50.0, 0.0),
                    "speed_kmh": 0.0},
                   advance=(0.0, 0.0))
    for _ in range(STOP_WINDOW_STEPS):
        assert far.step(ACT)[2] is False      # 50 m away: no success