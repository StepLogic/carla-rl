"""The image-goal navigation reward: ReViND-style, all-negative until done.

THE INVARIANT THE WHOLE FILE GUARDS, stated once here: with the success bonus
removed (user decision 2026-09-18), EVERY step pays a negative reward --
including the success step. The best score available to a step is -0.8
(in lane at the target speed, under the SATURATING speed bonus of user
decision 2026-09-20 evening), the worst is -4.0 (post-collision, off-road,
under the 1.5 flats of user decision 2026-09-20). Termination buys nothing
except the end of the negative stream, and it is granted only for the thing
the task IS (sustained stop at the target viewpoint and heading), a
destroyed hero, or the clock -- the SINGLE-MODE arm keeps exactly these
terminals; the two-mode arm's goal branch pays the failing-step flat on
every failing step NON-terminally and adds the ungated steering smoothness
(test_two_mode_image_goal.py's business).

The reward table under test (user spec 2026-09-18; 2026-09-20 REMOVES the
+0.75 flat in-lane bonus, RAISES the two flat penalties 0.75 -> 1.5, and the
evening batch turns the speed term into the positive SATURATING bonus
+IG_SPEED_PREF_K * min(v, v*)/v* paid by every IN-LANE step, so the in-lane
rows below are exact at v = TARGET_SPEED and lower by the pro-rated bonus
when the car crawls; the 2026-09-20 night batch adds the UNGATED steering
smoothness -IG_STEER_SMOOTH_W * |steer_t - steer_(t-1)|, which the
SINGLE-MODE arm never pays -- its wrapper passes no steer_delta, so every
row below is exact at delta 0):

    in-lane, at v* = 20 km/h, no collision    -0.80  = -1.0 + K
    parked in lane (L_t via the carve-out)    -1.00  (bonus 0 at v = 0)
    one-step lane crossing                    -1.00   (crossing gates only
                                               L_t, not a flat: unchanged)
    off-road                                  -2.50
    post-collision, back on the road          -2.50
    post-collision AND off-road               -4.00
    success (sustained stop on target) ordinary reward, terminated, NO bonus
"""
import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

pytest.importorskip("gymnasium")

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402

import vae_nav.image_goal_reward as igm  # noqa: E402
from vae_nav.image_goal_reward import (IN_LANE_BONUS,  # noqa: E402
                                       SPEED_PREF_K, STOP_WINDOW_STEPS,
                                       TARGET_SPEED_KMH, ImageGoalReward,
                                       _speed_mps, image_goal_reward,
                                       wrap_deg)

ACT = np.zeros(2, np.float32)
# One metre of forward motion per step along the lane's own forward vector,
# at TARGET_SPEED (20 km/h): the "driving the route, in lane" state. The
# fixture drives AT the target so the velocity preference term is zero here
# and the reward table below stays exact; slow in-lane driving is the
# velocity tests' business, not this table's.
DRIVING = {
    "wp_forward": (1.0, 0.0),
    "speed_kmh": 20.0,                      # 5.556 m/s = TARGET_SPEED
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

def _kmh_to_mps(kmh):
    """km/h -> m/s, borrowed from the module's own convention."""
    return _speed_mps({"speed_kmh": kmh})


def disp_forward_1m():
    """One metre of forward motion along the lane's own forward vector."""
    return (1.0, 0.0)


def test_in_lane_driving_at_target_speed_pays_minus_point_eight():
    # The saturating speed bonus (user 2026-09-20 evening) is at full tilt
    # at v*: the step pays the base cost MINUS the bonus, -0.8.
    w, _ = _wrap()
    assert w.step(ACT)[1] == pytest.approx(-1.0 + SPEED_PREF_K)
    assert w.step(ACT)[1] == pytest.approx(-0.8)


def test_a_lane_crossing_step_pays_minus_one_and_continues():
    # UNCHANGED by the 2026-09-20 flat-penalty raise: crossed gates only
    # L_t (which zeroes the speed bonus) -- neither flat penalty is gated on
    # a crossing, so the step still pays exactly the base cost. The
    # SINGLE-MODE arm keeps per-step crossings; only the two-mode goal
    # branch terminates on them.
    w, _ = _wrap({"lane_invasion_any": True})
    _, reward, terminated, truncated, _ = w.step(ACT)
    assert (reward, terminated, truncated) == (pytest.approx(-1.0), False, False)


def test_off_road_pays_minus_250():
    # The flat off-road penalty is 1.5 (user 2026-09-20): -1.0 - 1.5.
    w, _ = _wrap({"off_road": True})
    _, reward, terminated, *_ = w.step(ACT)
    assert reward == pytest.approx(-2.5)
    assert terminated is False


def test_a_collision_latches_and_costs_every_later_step():
    w, env = _wrap()
    w.step(ACT)
    env.info["collision_impulse"] = 5.0
    _, reward, terminated, *_ = w.step(ACT)
    assert terminated is False
    assert reward == pytest.approx(-2.5)      # the impact step itself
    env.info["collision_impulse"] = 0.0       # recovered to in-lane driving
    assert w.step(ACT)[1] == pytest.approx(-2.5)    # STILL, on the road


def test_sub_threshold_impulse_does_not_latch():
    w, env = _wrap()
    env.info["collision_impulse"] = 0.5       # sensor noise, not an impact
    w.step(ACT)
    env.info["collision_impulse"] = 0.0
    assert w._has_collided is False
    assert w.step(ACT)[1] == pytest.approx(-1.0 + SPEED_PREF_K)


def test_post_collision_off_road_pays_minus_400():
    # The WORST ordinary row: both flats stack (-1.0 - 1.5 - 1.5 = -4.0).
    # The 2-mode arm's TQC_TARGET_MIN floor (-1050) is sized against its
    # worst SUSTAINED row (-10.0 = off-road + latch + failing + max
    # smoothness); this stacked row is the single-mode arm's worst.
    w, env = _wrap()
    env.info["collision_impulse"] = 5.0
    w.step(ACT)
    env.info["collision_impulse"] = 0.0
    env.info["off_road"] = True
    assert w.step(ACT)[1] == pytest.approx(-4.0)


def test_the_flag_resets_between_episodes():
    w, env = _wrap()
    env.info["collision_impulse"] = 5.0
    w.step(ACT)
    env.info["collision_impulse"] = 0.0
    w.reset()
    assert w.step(ACT)[1] == pytest.approx(-1.0 + SPEED_PREF_K)


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


# ------------------------------------------------- the speed term ----

def test_speed_bonus_is_pro_rated_below_v_star():
    info = dict(DRIVING); info["speed_kmh"] = 10.0        # half of 20 km/h
    _, comp = image_goal_reward(info, disp_forward_1m(), has_collided=False)
    assert comp["speed_penalty"] == pytest.approx(
        SPEED_PREF_K * _kmh_to_mps(10.0) / _kmh_to_mps(20.0))
    assert comp["speed_penalty"] == pytest.approx(0.1)
    r, _ = image_goal_reward(info, disp_forward_1m(), False)
    assert r == pytest.approx(-1.0 + comp["speed_penalty"])
    assert r == pytest.approx(-0.9)                        # the crawl row


def test_speed_bonus_saturates_above_v_star():
    # 2x v* (40 km/h): min(v, v*) clips, so the bonus never exceeds K.
    info = dict(DRIVING); info["speed_kmh"] = 40.0
    r, comp = image_goal_reward(info, disp_forward_1m(), False)
    assert comp["speed_penalty"] == pytest.approx(SPEED_PREF_K)
    assert r == pytest.approx(-1.0 + SPEED_PREF_K)


def test_speed_bonus_zero_off_lane():
    info = dict(DRIVING); info["speed_kmh"] = 10.0
    off = dict(info); off["off_road"] = True
    assert image_goal_reward(off, disp_forward_1m(), False)[1]["speed_penalty"] == 0.0
    crossed = dict(info); crossed["lane_invasion_any"] = True
    assert image_goal_reward(crossed, disp_forward_1m(), False)[1]["speed_penalty"] == 0.0
    collided = dict(info)
    assert image_goal_reward(collided, disp_forward_1m(), True)[1]["speed_penalty"] == 0.0


def test_invariant_with_velocity_term():
    # Every cell of the reward table, bonus included, stays < 0: the base
    # cost dominates the +K bonus.
    for speed in (0.0, 10.0, 20.0, 40.0):
        for off in (False, True):
            for coll in (False, True):
                info = dict(DRIVING); info["speed_kmh"] = speed
                info["off_road"] = off
                r, _ = image_goal_reward(info, disp_forward_1m(), coll)
                assert r < 0.0


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
    comp = info["ig_reward_components"]
    assert comp["speed_penalty"] == pytest.approx(0.0)   # parked: no bonus
    assert reward == pytest.approx(-1.0 + comp["speed_penalty"])
    assert reward == pytest.approx(-1.0)
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

def test_parked_in_lane_keeps_L_t_and_pays_exactly_the_base_cost():
    # The sustained stop the success condition demands still runs through the
    # carve-out in _driving_with_route_direction: a stationary car is not
    # driving the wrong way, so L_t stays True. But the saturating bonus at
    # v = 0 is exactly 0, so what L_t now grants the parked car is freedom
    # from the direction penalty -- the parked-in-lane row is exactly the
    # base cost (user 2026-09-20 evening; the old -2.1111 row died with the
    # |v - v*| penalty).
    w, _ = _stopped()
    _, reward, _, _, info = w.step(ACT)
    comp = info["ig_reward_components"]
    assert comp["lane_bonus"] == pytest.approx(IN_LANE_BONUS)   # 0.0 now
    assert comp["speed_penalty"] == pytest.approx(0.0)
    assert reward == pytest.approx(-1.0 + comp["speed_penalty"])
    assert reward == pytest.approx(-1.0)


# ------------------------------------------------- the ablation override ----

def test_goal_truncation_constants_carry_the_ruled_defaults():
    # The stuck-truncation constants (user decision 2026-09-20): the module
    # carries them env-overridable, and the shipped defaults ARE the
    # ruling. (IG_OFFROAD_TRUNC_STEPS is gone -- the failing-step flat is
    # paid per failing step with no debounce, so there is no off-road
    # budget left to rule on.)
    assert igm.STUCK_SPEED_MPS == pytest.approx(0.3)
    assert igm.STUCK_STEPS == 100
    assert not hasattr(igm, "OFFROAD_TRUNC_STEPS")
    # The steering smoothness weight (user 2026-09-20 night) ships at 0.5,
    # env-overridable as IG_STEER_SMOOTH_W.
    assert igm.STEER_SMOOTH_W == pytest.approx(0.5)


def test_smoothness_term_rides_in_the_goal_leg_only_when_asked():
    # The smoothness term is an ARGUMENT, not a state: a caller that passes
    # a steer_delta pays -W*|delta| in the goal leg (every step, ungated);
    # the default call form (the single-mode arm's) pays none and keeps the
    # frozen rows exactly.
    r, comp = image_goal_reward(DRIVING, (1.0, 0.0), False,
                                steer_delta=2.0)
    assert comp["steer_smooth"] == pytest.approx(-1.0)
    assert r == pytest.approx(-1.0 + SPEED_PREF_K - 1.0)     # -1.8, still < 0
    r0, comp0 = image_goal_reward(DRIVING, (1.0, 0.0), False)
    assert comp0["steer_smooth"] == pytest.approx(0.0)
    assert r0 == pytest.approx(-1.0 + SPEED_PREF_K)          # -0.8, unchanged


def test_goal_truncation_constants_env_override(monkeypatch):
    # The stuck-truncation constants are env-overridable like the other
    # IG_* knobs: reload-with-env, and the finally restores the ruled
    # defaults for the later tests.
    monkeypatch.setenv("IG_STUCK_SPEED_MPS", "0.5")
    monkeypatch.setenv("IG_STUCK_STEPS", "42")
    importlib.reload(igm)
    try:
        assert igm.STUCK_SPEED_MPS == pytest.approx(0.5)
        assert igm.STUCK_STEPS == 42
        # the constants live in the module the wrapper reads at call time,
        # so a wrapper built from the pre-reload class still truncates on
        # the shortened stuck budget (its methods resolve the module's
        # globals dict, which reload reuses). The 20 km/h fixture is above
        # even the raised threshold, so drive the PARKED row 50 m out.
        w = igm.TwoModeImageGoalReward(
            _StubEnv({"speed_kmh": 0.0, "route_distance_to_goal_m": 50.0},
                     advance=(0.0, 0.0)),
            explore_prob=0.0, force_mode=1)
        w.reset()
        for _ in range(41):
            _, _, terminated, truncated, info = w.step(ACT)
            assert (terminated, truncated, info["ig_truncated"]) \
                == (False, False, None)
        _, _, terminated, truncated, info = w.step(ACT)
        assert truncated is True and terminated is False
        assert info["ig_truncated"] == "stuck"
    finally:
        monkeypatch.delenv("IG_STUCK_SPEED_MPS", raising=False)
        monkeypatch.delenv("IG_STUCK_STEPS", raising=False)
        importlib.reload(igm)
    assert igm.STUCK_SPEED_MPS == pytest.approx(0.3)
    assert igm.STUCK_STEPS == 100


def test_ig_in_lane_bonus_env_override_restores_the_old_row(monkeypatch):
    # IG_IN_LANE_BONUS stays env-overridable for ablations: with 0.75 in the
    # env the removed term comes back, and the in-lane-at-v* row is
    # -1.0 + 0.75 + 0.2 = -0.05 under the saturating bonus. Reload-with-env,
    # matching the coherence tests' pattern; the reload in the finally puts
    # the shipped default (0.0) back for the later tests.
    monkeypatch.setenv("IG_IN_LANE_BONUS", "0.75")
    importlib.reload(igm)
    try:
        assert igm.IN_LANE_BONUS == pytest.approx(0.75)
        r, comp = igm.image_goal_reward(dict(DRIVING), disp_forward_1m(),
                                        False)
        assert r == pytest.approx(-1.0 + 0.75 + SPEED_PREF_K)
        assert comp["lane_bonus"] == pytest.approx(0.75)
    finally:
        monkeypatch.delenv("IG_IN_LANE_BONUS", raising=False)
        importlib.reload(igm)
    assert igm.IN_LANE_BONUS == pytest.approx(0.0)


def test_ig_flat_penalty_env_overrides_restore_the_old_rows(monkeypatch):
    # The two flat penalties stay env-overridable for the ablation: with
    # 0.75 in the env the pre-2026-09-20 rows come back exactly (off-road
    # -1.75, post-collision on-road -1.75, off-road+collided -2.5). Same
    # reload-with-env pattern; the finally restores the shipped 1.5.
    monkeypatch.setenv("IG_OFFROAD_PENALTY", "0.75")
    monkeypatch.setenv("IG_PERSISTENT_COLLISION_PENALTY", "0.75")
    importlib.reload(igm)
    try:
        assert igm.OFFROAD_PENALTY == pytest.approx(0.75)
        assert igm.PERSISTENT_COLLISION_PENALTY == pytest.approx(0.75)
        r, comp = igm.image_goal_reward(
            dict(DRIVING, off_road=True), disp_forward_1m(), False)
        assert r == pytest.approx(-1.75)
        assert comp["offroad_penalty"] == pytest.approx(-0.75)
        r, comp = igm.image_goal_reward(
            dict(DRIVING), disp_forward_1m(), True)
        assert r == pytest.approx(-1.75)
        assert comp["collision_penalty"] == pytest.approx(-0.75)
        r, _ = igm.image_goal_reward(
            dict(DRIVING, off_road=True), disp_forward_1m(), True)
        assert r == pytest.approx(-2.5)
    finally:
        monkeypatch.delenv("IG_OFFROAD_PENALTY", raising=False)
        monkeypatch.delenv("IG_PERSISTENT_COLLISION_PENALTY", raising=False)
        importlib.reload(igm)
    assert igm.OFFROAD_PENALTY == pytest.approx(1.5)
    assert igm.PERSISTENT_COLLISION_PENALTY == pytest.approx(1.5)


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