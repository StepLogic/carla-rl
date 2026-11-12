"""The 2-mode image-goal arm: ONE pure function, TWO modes, TWO contracts.

THE INVARIANTS THE FILE GUARDS (user decisions 2026-09-20, evening and
night batches):
  - GOAL mode (mode 1): the ReViND reward with the 1.5 flats, the
    SATURATING speed bonus (user 2026-09-20 evening) and the UNGATED
    steering smoothness (user 2026-09-20 night, -IG_STEER_SMOOTH_W *
    |steer_t - steer_(t-1)| on EVERY step, parked and failing included)
    -- every step < 0, the sustained-stop success terminates and pays no
    bonus. FAILURES ARE NON-TERMINAL (user 2026-09-20 night: Task 18
    section A's first-failure termination is REVERTED): every FAILING
    step (off-road / crossing / collision impulse) pays the flat
    -IG_TERMINAL_PENALTY on top of its ordinary row and the episode
    CONTINUES -- off-road -7.5/step, crossed -6.0, the latching collision
    step -7.5 then -2.5/step latched. ig_truncated stays None on failing
    steps; sustained low speed outside the success gate still only
    TRUNCATES ("stuck").
  - EXPLORE mode (mode 0): NO TIME PRESSURE and TERMINAL FAILURES. The
    step reward is the SAME saturating bonus plus the same ungated
    smoothness (r in [-IG_TERMINAL_PENALTY - max smoothness,
    +IG_EXPLORE_SPEED_PREF_K], i.e. [-6.0, +2.0] under the shipped W=0.5
    (the explore K was raised 0.2 -> 2.0, user 2026-09-21; the goal leg
    keeps IG_SPEED_PREF_K=0.2); exactly
    0.0 parked at delta 0 and +explore K in lane at or above v*), the FIRST
    off-road / crossing / collision step (and an invalid step) TERMINATES
    paying -IG_TERMINAL_PENALTY - 0.5*|delta-steer|, and the 100 m path
    arrival terminates with NO bonus (the ordinary bonus). The cap
    truncates as a safety bound. ORDERING: invalid > failure > arrival >
    cap.

Mode semantics (mirrors GatedReward): mode 1 = goal (image-goal task:
sustained stop on target), mode 0 = explore (drive EXPLORE_GOAL_DIST_M of
cumulative path). The mode is drawn once per reset.
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
from vae_nav.image_goal_reward import (  # noqa: E402
    EXPLORE_SPEED_PREF_K, IN_LANE_BONUS, SPEED_PREF_K, STOP_WINDOW_STEPS,
    STUCK_SPEED_MPS, STUCK_STEPS, TARGET_SPEED_KMH,
    TERMINAL_PENALTY, TwoModeImageGoalReward, image_goal_reward)

ACT = np.zeros(2, np.float32)
# One metre of forward motion per step along the lane's forward vector, at
# TARGET_SPEED (20 km/h): the in-lane driving state. The fixture drives AT the
# target speed and commands steer 0 from the reset's 0.0 (delta 0), so the
# velocity term is zero and the smoothness term is 0 and the arithmetic below
# stays exact; the smoothness tests drive NONZERO actions.
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

# The explore arm's 100 m arrival target, shortened so a stub walk reaches it
# in a handful of steps. The reward does not depend on the value: arrival
# terminates and pays nothing whatever the distance is.
EXPLORE_ARRIVAL_M = 10.0


class _StubEnv(gym.Env):
    """Steps `advance` (metres) along +x each step, so disp is real."""

    metadata = {}

    def __init__(self, info=None, done_from_env=False, advance=(1.0, 0.0),
                 explore_prob=None):            # explore_prob: ignored here --
        self.observation_space = spaces.Dict({})  # the WRAPPER draws the mode
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


def _wrap(info=None, mode=None, max_steps=1000, done_from_env=False,
          advance=(1.0, 0.0), seed=0):
    """Build the two-mode wrapper around a stub, mode forced unless None.

    Tests that need a shorter explore distance patch
    igm.EXPLORE_GOAL_DIST_M with monkeypatch FIRST -- the reward code reads
    the module global at call time, so the patch bites for the whole test.
    """
    env = _StubEnv(info, done_from_env, advance)
    w = TwoModeImageGoalReward(env, explore_prob=0.0, force_mode=mode,
                               max_steps=max_steps, seed=seed)
    w.reset()
    return w, env


def _stopped(**extra):
    """A goal-mode fixture: parked ON the target, on heading."""
    info = {"hero_pos": (0.0, 0.0), "speed_kmh": 0.0,
            "route_distance_to_goal_m": 1.0, "yaw_error_to_goal_deg": 0.0}
    info.update(extra)
    return _wrap(info, mode=1, advance=(0.0, 0.0))


# --------------------------------------------------- the mode draw + reward ----

def test_mode_drawn_and_rewarded_differently_at_v_star():
    # THE MODE SPLIT (user 2026-09-20, evening batch; explore K raised
    # 0.2 -> 2.0, user 2026-09-21). Same info, both modes, but the branches
    # differ: goal pays the base cost MINUS the saturating bonus (-0.8 in
    # lane at v*) while explore pays the bonus ALONE (+2.0 in lane at v*).
    # The reward is still the same pure function -- only the mode argument
    # differs.
    env_g = TwoModeImageGoalReward(_StubEnv(explore_prob=0.0),
                                   explore_prob=0.0, seed=0)
    env_g.reset()
    out_g = env_g.step(ACT)
    assert out_g[4]["mode"] == 1
    env_e = TwoModeImageGoalReward(_StubEnv(explore_prob=1.0),
                                   explore_prob=1.0, seed=0)
    env_e.reset()
    out_e = env_e.step(ACT)
    assert out_e[4]["mode"] == 0
    assert out_g[1] == pytest.approx(-1.0 + SPEED_PREF_K)
    assert out_g[1] == pytest.approx(-0.8)
    assert out_e[1] == pytest.approx(EXPLORE_SPEED_PREF_K)
    assert out_e[1] == pytest.approx(2.0)
    # the two legs do NOT share the coefficient any more: the explore leg
    # must not pick up the goal leg's 0.2 and the goal leg must not pick up
    # the explore leg's 2.0.
    assert EXPLORE_SPEED_PREF_K == pytest.approx(2.0)
    assert SPEED_PREF_K == pytest.approx(0.2)


def test_explore_prob_zero_and_one_draw_every_reset():
    for prob, want in ((0.0, 1), (1.0, 0)):
        w, _ = _wrap(mode=None)
        w.explore_prob, w.force_mode = prob, None
        w._rng = np.random.default_rng(0)
        for _ in range(4):
            w.reset()
            assert w.mode == want


def test_force_mode_overrides_the_draw():
    w, _ = _wrap(mode=1)
    assert w.mode == 1
    w2, _ = _wrap(mode=0)
    assert w2.mode == 0


def test_goal_mode_publishes_the_ig_info_keys():
    w, _ = _wrap(mode=1)
    _, _, _, _, info = w.step(ACT)
    assert info["is_success"] is False
    assert info["has_collided"] is False
    assert info["collision_impulse"] == 0.0
    assert "speed_penalty" in info["ig_reward_components"]
    assert info["arrived_this_step"] is False


def test_explore_mode_publishes_path_and_mode():
    w, _ = _wrap(mode=0)
    _, _, _, _, info = w.step(ACT)
    assert info["mode"] == 0
    assert info["explore_dist_m"] == pytest.approx(1.0)


# ------------------------------------------------------------- the success ----

def test_goal_success_needs_sustained_stop():
    # 10 consecutive stopped-on-target steps terminate; 9 do not.
    w, _ = _stopped()
    for i in range(STOP_WINDOW_STEPS - 1):
        _, _, terminated, *_ = w.step(ACT)
        assert terminated is False, f"fired early at step {i + 1}"
    _, reward, terminated, truncated, info = w.step(ACT)
    assert terminated is True and truncated is False
    assert info["is_success"] is True
    assert info["arrived_this_step"] is True
    # THE SUCCESS STEP PAYS THE ORDINARY REWARD: a parked car is in lane
    # (moving nothing is not driving the wrong way) but earns none of the
    # saturating bonus at v = 0, so the parked-in-lane row is exactly the
    # base cost (the bonus form is user 2026-09-20 evening; the flat lane
    # bonus is REMOVED, user 2026-09-20).
    comp = info["ig_reward_components"]
    assert reward == pytest.approx(-1.0 + IN_LANE_BONUS)
    assert reward == pytest.approx(-1.0)
    assert comp["speed_penalty"] == pytest.approx(0.0)   # parked: no bonus
    assert (reward == pytest.approx(comp["base"] + comp["lane_bonus"]
                                    + comp["speed_penalty"]))


def test_goal_mode_still_truncates_at_the_cap_without_success():
    # A goal episode that never stops and never fails runs to the cap. With
    # the 2026-09-20 truncation counter, an in-lane driving episode never
    # fills it (5.556 m/s > STUCK_SPEED_MPS, never off-road), so the cap
    # keeps being the cause and ig_truncated stays None.
    w, _ = _wrap(mode=1, max_steps=5)
    for _ in range(5):
        w.step(ACT)
    obs, reward, terminated, truncated, info = w.step(ACT)
    assert truncated is True and terminated is False
    assert info["ig_truncated"] is None
    assert reward < 0.0


# --------------------------------- goal-mode failures NON-terminal (night) ----
# The user ruling (2026-09-20 night; Task 18 section A's termination is
# REVERTED): lane invasion / off-road / collision are NON-terminal again in
# goal mode. Every FAILING step (off_road OR lane_invasion_any OR impulse
# over the latch threshold) pays the flat -IG_TERMINAL_PENALTY on top of its
# ordinary row (smoothness already inside it) and the episode CONTINUES:
# terminated=False, truncated=False, ig_truncated stays None. Sustained
# off-road -7.5/step; of the collision steps only the impulse-firing one
# pays the extra -5 (later latched steps -2.5). Success/invalid steps pay
# no flat.

def test_goal_off_road_step_is_non_terminal_and_pays_minus_7_5():
    # -1.0 base - 1.5 off-road flat, plus the flat -5 on the failing step,
    # and the episode CONTINUES. terminated=False, truncated=False, NO
    # ig_truncated cause.
    w, _ = _wrap(mode=1, info={"off_road": True})
    _, reward, terminated, truncated, info = w.step(ACT)
    assert (terminated, truncated) == (False, False)
    assert info["ig_truncated"] is None
    assert reward == pytest.approx(-7.5)
    comp = info["ig_reward_components"]
    assert comp["base"] == pytest.approx(-1.0)
    assert comp["offroad_penalty"] == pytest.approx(-1.5)
    assert comp["speed_penalty"] == pytest.approx(0.0)   # not in lane
    assert comp["steer_smooth"] == pytest.approx(0.0)    # delta 0
    assert comp["terminal_penalty"] == pytest.approx(-TERMINAL_PENALTY)
    # THE SUM IDENTITY, extended with the terminal_penalty and steer_smooth
    # components on the failing step.
    assert (comp["base"] + comp["offroad_penalty"] + comp["speed_penalty"]
            + comp["lane_bonus"] + comp["collision_penalty"]
            + comp["steer_smooth"] + comp["terminal_penalty"]
            == pytest.approx(reward))
    assert comp["goal_reward_term"] + comp["terminal_penalty"] \
        == pytest.approx(reward)      # explore leg masked to 0
    # SUSTAINED off-road: the next step pays -7.5 again, still not terminal.
    _, reward2, terminated2, truncated2, info2 = w.step(ACT)
    assert (terminated2, truncated2) == (False, False)
    assert info2["ig_truncated"] is None
    assert reward2 == pytest.approx(-7.5)


def test_goal_crossing_step_is_non_terminal_and_pays_minus_6_0():
    # A crossed step keeps its ordinary -1.0 (crossing gates only L_t) and
    # pays the flat -5 on top, and the episode CONTINUES.
    w, _ = _wrap(mode=1, info={"lane_invasion_any": True})
    _, reward, terminated, truncated, info = w.step(ACT)
    assert (terminated, truncated) == (False, False)
    assert info["ig_truncated"] is None
    assert reward == pytest.approx(-6.0)
    comp = info["ig_reward_components"]
    assert comp["offroad_penalty"] == pytest.approx(0.0)
    assert comp["steer_smooth"] == pytest.approx(0.0)
    assert comp["terminal_penalty"] == pytest.approx(-TERMINAL_PENALTY)
    assert (comp["base"] + comp["offroad_penalty"] + comp["speed_penalty"]
            + comp["lane_bonus"] + comp["collision_penalty"]
            + comp["steer_smooth"] + comp["terminal_penalty"]
            == pytest.approx(reward))


def test_goal_collision_step_pays_minus_7_5_then_minus_2_5_while_latched():
    # The latching collision step: ordinary -2.5 (base + collision flat; the
    # latch sets has_collided -> L_t false -> bonus 0) plus the flat -5 --
    # then, while the latch holds with no new impulse, -2.5/step WITHOUT
    # the flat (only the impulse-firing step pays it).
    w, env = _wrap(mode=1)
    env.info["collision_impulse"] = 5.0
    _, reward, terminated, truncated, info = w.step(ACT)
    assert (terminated, truncated) == (False, False)
    assert info["ig_truncated"] is None
    assert info["has_collided"] is True
    assert reward == pytest.approx(-7.5)
    comp = info["ig_reward_components"]
    assert comp["collision_penalty"] == pytest.approx(-1.5)
    assert comp["speed_penalty"] == pytest.approx(0.0)
    assert comp["terminal_penalty"] == pytest.approx(-TERMINAL_PENALTY)
    assert (comp["base"] + comp["offroad_penalty"] + comp["speed_penalty"]
            + comp["lane_bonus"] + comp["collision_penalty"]
            + comp["steer_smooth"] + comp["terminal_penalty"]
            == pytest.approx(reward))
    env.info["collision_impulse"] = 0.0
    _, reward2, terminated2, truncated2, info2 = w.step(ACT)
    assert (terminated2, truncated2) == (False, False)
    assert reward2 == pytest.approx(-2.5)             # latched, no flat
    assert info2["ig_reward_components"]["terminal_penalty"] == 0.0
    assert info2["has_collided"] is True


def test_goal_off_road_plus_latched_pays_minus_9_0():
    # The worst SUSTAINED row: off-road and the collision latch stack with
    # the failing-step flat (-1.0 -1.5 -1.5 -5.0 = -9.0 at delta 0; max
    # smoothness takes it to -10.0, which sizes TQC_TARGET_MIN).
    w, env = _wrap(mode=1)
    env.info["collision_impulse"] = 5.0
    _, reward, *_rest = w.step(ACT)             # latch + failing step: -7.5
    assert reward == pytest.approx(-7.5)
    env.info["collision_impulse"] = 0.0
    env.info["off_road"] = True
    _, reward2, terminated, truncated, info = w.step(ACT)
    assert (terminated, truncated) == (False, False)
    assert reward2 == pytest.approx(-9.0)
    comp = info["ig_reward_components"]
    assert comp["offroad_penalty"] == pytest.approx(-1.5)
    assert comp["collision_penalty"] == pytest.approx(-1.5)
    assert comp["terminal_penalty"] == pytest.approx(-TERMINAL_PENALTY)


def test_goal_sub_threshold_impulse_still_drives():
    # Sensor noise is still not an impact: no latch, no termination, and the
    # clean in-lane step at v* pays -0.8.
    w, env = _wrap(mode=1)
    env.info["collision_impulse"] = 0.5
    _, reward, terminated, *_ = w.step(ACT)
    env.info["collision_impulse"] = 0.0
    assert w._has_collided is False
    assert terminated is False
    assert reward == pytest.approx(-1.0 + SPEED_PREF_K)


def test_stuck_truncates_at_100_low_speed_steps_outside_the_gate():
    # Parked 50 m from the goal: every step is low speed AND outside the
    # spatial/yaw gate, so the stuck counter runs 1..100 and the 100th
    # truncates, paying the parked-in-lane row (exactly the base cost: the
    # saturating bonus is 0 at v = 0).
    w, _ = _wrap(mode=1, info={"speed_kmh": 0.0,
                               "route_distance_to_goal_m": 50.0},
                 advance=(0.0, 0.0))
    for i in range(STUCK_STEPS - 1):
        _, _, terminated, truncated, info = w.step(ACT)
        assert (terminated, truncated, info["ig_truncated"]) \
            == (False, False, None), f"fired early at step {i + 1}"
    _, reward, terminated, truncated, info = w.step(ACT)
    assert truncated is True and terminated is False
    assert info["ig_truncated"] == "stuck"
    assert reward == pytest.approx(-1.0)            # the usual parked row


def test_parking_inside_the_gate_reaches_success_not_stuck_truncation():
    # THE RULING THE STOP-WINDOW DESIGN DEPENDS ON: a parked car on target
    # and on heading sits inside the success spatial/yaw gate, so the stuck
    # counter never counts a single step of the sustained stop -- the 10th
    # gate step TERMINATES with success, never stuck-truncating first (and
    # 100 gate steps would otherwise blow the 100-step stuck budget).
    w, _ = _stopped()
    for i in range(STOP_WINDOW_STEPS - 1):
        _, _, terminated, truncated, info = w.step(ACT)
        assert (terminated, truncated, info["ig_truncated"]) \
            == (False, False, None)
        assert w._stuck_steps == 0
    _, _, terminated, truncated, info = w.step(ACT)
    assert terminated is True and truncated is False
    assert info["is_success"] is True
    assert info["ig_truncated"] is None


def test_stuck_counter_skips_gate_steps():
    # 100 NON-CONSECUTIVE low-speed steps interleaved with gate steps never
    # truncate: every gate step resets the stuck counter (a gate step is
    # never a stuck step), so no run reaches STUCK_STEPS.
    w, env = _wrap(mode=1, info={"speed_kmh": 0.0,
                                 "route_distance_to_goal_m": 1.0},
                   advance=(0.0, 0.0))
    for i in range(STUCK_STEPS + 10):
        # in the gate this step (1.0 m), out of it the next (50 m) ...
        env.info["route_distance_to_goal_m"] = (
            1.0 if i % 2 == 0 else 50.0)
        _, _, terminated, truncated, info = w.step(ACT)
        assert (terminated, truncated, info["ig_truncated"]) \
            == (False, False, None), f"truncated/terminated at step {i + 1}"
        assert w._stuck_steps <= 1


def test_success_wins_a_step_that_is_also_failing():
    # THE ORDERING RULING: success is evaluated first. Park on target for 9
    # clean gate steps, then flag the car off-road on the 10th (a polygon
    # flicker on the finish line): the stop window reaches 10 AND the
    # failure condition fires on the SAME step, and success keeps the step
    # -- terminated=True by SUCCESS, no terminal flat, no truncation cause.
    w, env = _stopped()
    for _ in range(STOP_WINDOW_STEPS - 1):
        _, _, terminated, truncated, info = w.step(ACT)
        assert (terminated, truncated, info["ig_truncated"]) \
            == (False, False, None)
    env.info["off_road"] = True
    _, reward, terminated, truncated, info = w.step(ACT)
    assert terminated is True and truncated is False
    assert info["is_success"] is True
    assert info["ig_truncated"] is None
    # the step pays its ordinary (off-road) reward, NOT the terminal flat
    assert reward == pytest.approx(-2.5)
    assert info["ig_reward_components"]["terminal_penalty"] \
        == pytest.approx(0.0)


def test_goal_truncation_survives_a_failing_step():
    # With failures NON-terminal (user 2026-09-20 night), the stuck
    # truncation is the only thing that can end a parked episode. A parked
    # car 50 m out goes off-road on the step its stuck counter would fill:
    # the step still TRUNCATES (truncated=True, ig_truncated="stuck" --
    # failures no longer beat the truncation) and the failing step pays its
    # ordinary off-road row PLUS its failing-step flat (-2.5 - 5 = -7.5 at
    # delta 0; the truncation itself adds nothing).
    w, env = _wrap(mode=1, info={"speed_kmh": 0.0,
                                 "route_distance_to_goal_m": 50.0},
                   advance=(0.0, 0.0))
    for _ in range(STUCK_STEPS - 1):
        w.step(ACT)
    env.info["off_road"] = True
    _, reward, terminated, truncated, info = w.step(ACT)
    assert (terminated, truncated) == (False, True)
    assert info["ig_truncated"] == "stuck"
    assert reward == pytest.approx(-2.5 - TERMINAL_PENALTY)


def test_explore_mode_runs_no_truncation_counters():
    # EXPLORE MODE UNTOUCHED (user decision 2026-09-20): a parked explore
    # episode far from the goal would blow the stuck budget in 100 steps if
    # the counters leaked into explore; it must instead run clean past them,
    # and explore info carries no ig_truncated key at all.
    w, _ = _wrap(mode=0, info={"speed_kmh": 0.0,
                               "route_distance_to_goal_m": 50.0},
                 advance=(0.0, 0.0))
    for _ in range(STUCK_STEPS + 10):
        _, _, terminated, truncated, info = w.step(ACT)
        assert (terminated, truncated) == (False, False)
        assert "ig_truncated" not in info


def test_goal_truncation_constants_env_override(monkeypatch):
    # The stuck-truncation constants are env-overridable (reload-with-env,
    # the file's own pattern): with the counter shortened, the wrapper
    # truncates on the shortened budget -- and the finally restores the
    # ruled defaults. (IG_OFFROAD_TRUNC_STEPS is gone: the first off-road
    # step terminates, so there is no off-road budget to shorten.)
    monkeypatch.setenv("IG_STUCK_SPEED_MPS", "0.5")
    monkeypatch.setenv("IG_STUCK_STEPS", "3")
    importlib.reload(igm)
    try:
        assert igm.STUCK_SPEED_MPS == pytest.approx(0.5)
        assert igm.STUCK_STEPS == 3
        # the raised stuck speed threshold catches the 20 km/h fixture too:
        # 5.556 m/s is not under 0.5, so drive the PARKED row instead
        w, _ = _wrap(mode=1, info={"speed_kmh": 0.0,
                                   "route_distance_to_goal_m": 50.0},
                     advance=(0.0, 0.0))
        w.step(ACT)
        w.step(ACT)
        _, reward, terminated, truncated, info = w.step(ACT)
        assert (terminated, truncated, info["ig_truncated"]) \
            == (False, True, "stuck")
        assert reward == pytest.approx(-1.0)    # ordinary, no penalty
    finally:
        monkeypatch.delenv("IG_STUCK_SPEED_MPS", raising=False)
        monkeypatch.delenv("IG_STUCK_STEPS", raising=False)
        importlib.reload(igm)
    assert igm.STUCK_SPEED_MPS == pytest.approx(0.3)
    assert igm.STUCK_STEPS == 100


# --------------------------------------------------- the collision latch ----

def test_goal_latch_pays_the_flat_once_then_clears_on_reset():
    # GOAL MODE (user 2026-09-20 night): the latching collision step pays
    # -7.5 (ordinary -2.5 + the flat -5) but does NOT terminate; after the
    # reset the latch is cleared and a clean in-lane step at v* pays -0.8
    # again.
    w, env = _wrap(mode=1)
    env.info["collision_impulse"] = 5.0
    _, reward, terminated, truncated, info = w.step(ACT)
    assert (terminated, truncated) == (False, False)
    assert reward == pytest.approx(-7.5)
    assert info["has_collided"] is True
    w.reset()
    env.info["collision_impulse"] = 0.0
    assert w._has_collided is False
    assert w.step(ACT)[1] == pytest.approx(-1.0 + SPEED_PREF_K)


def test_latch_is_visible_in_explore_mode_and_now_terminal():
    # ONE latch for both modes: explore mode still latches
    # info["has_collided"] for the statistics, and since 2026-09-20 the
    # latching step TERMINATES the explore episode, paying the flat
    # terminal penalty exactly (the ordinary component is 0.0: a collided
    # car is not in lane).
    w, env = _wrap(mode=0)
    env.info["collision_impulse"] = 5.0
    _, reward, terminated, truncated, info = w.step(ACT)
    assert terminated is True and truncated is False
    assert info["explore_end"] == "collision"
    assert info["has_collided"] is True
    assert reward == pytest.approx(-TERMINAL_PENALTY)


# ------------------------------------------------------- explore arrival ----

def test_explore_arrival_at_path(monkeypatch):
    # Cumulative path >= EXPLORE_GOAL_DIST_M -> terminated, NO bonus: the
    # arrival step pays the ordinary explore reward, which at v* is exactly
    # +SPEED_PREF_K, and the episode's return is exactly the sum of its
    # step rewards.
    monkeypatch.setattr(igm, "EXPLORE_GOAL_DIST_M", EXPLORE_ARRIVAL_M)
    w, _ = _wrap(mode=0)
    rewards = []
    for _ in range(10):
        _, reward, terminated, truncated, info = w.step(ACT)
        rewards.append(reward)
        if terminated:
            assert info["explore_end"] == "reached"
            break
    assert terminated is True and truncated is False
    assert w._path_m == pytest.approx(EXPLORE_ARRIVAL_M)
    assert all(r == pytest.approx(EXPLORE_SPEED_PREF_K) for r in rewards)
    assert sum(rewards) == pytest.approx(10 * EXPLORE_SPEED_PREF_K)


def test_explore_arrival_at_the_real_100_m_target():
    # The shipped constant: 100 m of cumulative path, driven at v* (the stub
    # advances 10 m a step), terminates with explore_end="reached", NO bonus
    # and a return of exactly ten ordinary bonus steps.
    assert igm.EXPLORE_GOAL_DIST_M == pytest.approx(100.0)
    w, _ = _wrap(mode=0, advance=(10.0, 0.0))
    rewards = []
    for _ in range(10):
        _, reward, terminated, truncated, info = w.step(ACT)
        rewards.append(reward)
        if terminated:
            assert info["explore_end"] == "reached"
            break
    assert terminated is True and truncated is False
    assert sum(rewards) == pytest.approx(10 * EXPLORE_SPEED_PREF_K)


def test_explore_short_of_the_path_keeps_driving(monkeypatch):
    monkeypatch.setattr(igm, "EXPLORE_GOAL_DIST_M", EXPLORE_ARRIVAL_M)
    w, _ = _wrap(mode=0)
    for _ in range(9):
        _, _, terminated, *_ = w.step(ACT)
        assert terminated is False


def test_explore_end_is_cap_at_the_step_limit(monkeypatch):
    # The cap TRUNCATES (never terminates) and pays the ordinary step
    # reward, which with no base cost is +EXPLORE_SPEED_PREF_K at v*.
    monkeypatch.setattr(igm, "EXPLORE_GOAL_DIST_M", EXPLORE_ARRIVAL_M)
    w, _ = _wrap(mode=0, max_steps=3)
    for _ in range(2):
        w.step(ACT)
    _, reward, terminated, truncated, info = w.step(ACT)
    assert truncated is True and terminated is False
    assert info["explore_end"] == "cap"
    assert reward == pytest.approx(EXPLORE_SPEED_PREF_K)


def test_explore_invalid_terminates_and_pays_the_penalty():
    # simulator_invalid still terminates; it pays the flat terminal penalty
    # exactly -- the destroyed hero's speed is not read as an ordinary
    # reward on top.
    w, _ = _wrap(mode=0, info={"simulator_invalid": True})
    _, reward, terminated, truncated, info = w.step(ACT)
    assert terminated is True and truncated is False
    assert info["explore_end"] == "invalid"
    assert reward == pytest.approx(-TERMINAL_PENALTY)


# --------------------------------------------- the explore reward table ----
# (user decisions 2026-09-20: no base cost, no flat penalties, terminal
# failures; the ordinary explore row is the SAME saturating bonus the goal
# leg carries, user 2026-09-20 evening, but with the explore leg's OWN
# coefficient EXPLORE_SPEED_PREF_K = 2.0 since user 2026-09-21.)

def test_explore_in_lane_at_target_speed_pays_the_full_bonus():
    # The no-time-pressure row: a clean in-lane step at v* earns the whole
    # +EXPLORE_SPEED_PREF_K (raised 0.2 -> 2.0, user 2026-09-21).
    w, _ = _wrap(mode=0)
    _, reward, terminated, truncated, _ = w.step(ACT)
    assert reward == pytest.approx(EXPLORE_SPEED_PREF_K)
    assert reward == pytest.approx(2.0)
    assert (terminated, truncated) == (False, False)


def test_explore_bonus_saturates_above_v_star():
    # 2x v* (40 km/h): min(v, v*) clips, so the bonus is still exactly K --
    # never more.
    _, comp = image_goal_reward({**DRIVING, "speed_kmh": 40.0}, (1.0, 0.0),
                                False, mode=0)
    assert comp["speed_penalty"] == pytest.approx(EXPLORE_SPEED_PREF_K)
    r, _ = image_goal_reward({**DRIVING, "speed_kmh": 40.0}, (1.0, 0.0),
                             False, mode=0)
    assert r == pytest.approx(EXPLORE_SPEED_PREF_K)


def test_explore_half_speed_pays_half_the_bonus():
    # 0.5 x v* (10 km/h): the bonus is pro-rated, +K/2 = +1.0.
    r, _ = image_goal_reward({**DRIVING, "speed_kmh": 10.0}, (1.0, 0.0),
                             False, mode=0)
    assert r == pytest.approx(0.5 * EXPLORE_SPEED_PREF_K)
    assert r == pytest.approx(1.0)


def test_explore_crawling_in_lane_pays_plus_072():
    # 2 m/s against the 5.556 m/s target: +K * 2 / 5.556 = +0.72 (the
    # raised K, user 2026-09-21).
    _, comp = image_goal_reward({**DRIVING, "speed_kmh": 7.2}, (1.0, 0.0),
                                False, mode=0)
    assert comp["speed_penalty"] == pytest.approx(0.72, abs=1e-3)
    r, comp = image_goal_reward({**DRIVING, "speed_kmh": 7.2}, (1.0, 0.0),
                                False, mode=0)
    assert r == pytest.approx(EXPLORE_SPEED_PREF_K * 2.0 / (TARGET_SPEED_KMH / 3.6))
    assert r == pytest.approx(0.72, abs=1e-3)
    assert comp["terminal_penalty"] == 0.0
    assert comp["explore_end"] is None


def test_explore_parked_in_lane_pays_exactly_zero():
    # The carve-out keeps L_t True at standstill, but the saturating bonus
    # at v = 0 is exactly 0 -- a parked explore step earns nothing and pays
    # nothing (user 2026-09-20 evening: the |v - v*| penalty is gone).
    info = dict(DRIVING)
    info["speed_kmh"] = 0.0
    r, comp = image_goal_reward(info, (0.0, 0.0), False, mode=0)
    assert r == pytest.approx(0.0)
    assert comp["speed_penalty"] == pytest.approx(0.0)
    assert comp["in_lane"] is True


def test_explore_off_road_step_is_terminal_and_pays_exactly_minus_5():
    # THE FIRST off-road step terminates and pays exactly -TERMINAL_PENALTY:
    # in_lane is False by construction, so the ordinary component is
    # exactly 0.0 and no off-road flat penalty is stacked on top.
    w, _ = _wrap(mode=0, info={"off_road": True})
    _, reward, terminated, truncated, info = w.step(ACT)
    assert (terminated, truncated) == (True, False)
    assert info["explore_end"] == "off_road"
    assert reward == pytest.approx(-TERMINAL_PENALTY)
    assert reward == pytest.approx(-5.0)
    comp = info["ig_reward_components"]
    assert comp["in_lane"] is False
    assert comp["speed_penalty"] == pytest.approx(0.0)
    assert comp["off_road"] is True
    assert comp["terminal_penalty"] == pytest.approx(-TERMINAL_PENALTY)
    assert comp["explore_end"] == "off_road"
    assert comp["goal_reward_term"] == 0.0
    assert comp["explore_reward_term"] == pytest.approx(-TERMINAL_PENALTY)


def test_explore_crossing_step_is_terminal_and_pays_exactly_minus_5():
    w, _ = _wrap(mode=0, info={"lane_invasion_any": True})
    _, reward, terminated, truncated, info = w.step(ACT)
    assert (terminated, truncated) == (True, False)
    assert info["explore_end"] == "crossed"
    assert reward == pytest.approx(-TERMINAL_PENALTY)
    comp = info["ig_reward_components"]
    assert comp["speed_penalty"] == pytest.approx(0.0)
    assert comp["terminal_penalty"] == pytest.approx(-TERMINAL_PENALTY)


def test_explore_collision_step_is_terminal_and_pays_exactly_minus_5():
    w, env = _wrap(mode=0)
    env.info["collision_impulse"] = 5.0
    _, reward, terminated, truncated, info = w.step(ACT)
    assert (terminated, truncated) == (True, False)
    assert info["explore_end"] == "collision"
    assert reward == pytest.approx(-TERMINAL_PENALTY)
    comp = info["ig_reward_components"]
    assert comp["speed_penalty"] == pytest.approx(0.0)
    assert comp["terminal_penalty"] == pytest.approx(-TERMINAL_PENALTY)


def test_explore_sub_threshold_impulse_does_not_terminate():
    # Sensor noise is still not an impact: no latch, no termination, and
    # the clean in-lane step at v* pays the full +EXPLORE_SPEED_PREF_K.
    w, env = _wrap(mode=0)
    env.info["collision_impulse"] = 0.5
    _, reward, terminated, *_ = w.step(ACT)
    env.info["collision_impulse"] = 0.0
    assert w._has_collided is False
    assert terminated is False
    assert reward == pytest.approx(EXPLORE_SPEED_PREF_K)


# --------------------------------------------------- the explore ordering ----

def test_explore_failure_beats_arrival(monkeypatch):
    # A car that goes off-road on the step it passes the distance still went
    # off-road: the failure's terminal penalty is what the step pays and the
    # end is the failure, not "reached".
    monkeypatch.setattr(igm, "EXPLORE_GOAL_DIST_M", 1.0)
    w, _ = _wrap(mode=0, info={"off_road": True})
    _, reward, terminated, truncated, info = w.step(ACT)
    assert terminated is True and truncated is False
    assert info["explore_end"] == "off_road"
    assert reward == pytest.approx(-TERMINAL_PENALTY)


def test_explore_arrival_beats_the_cap(monkeypatch):
    # 5 m of path, cap at 10: arrival terminates at step 5 with NO bonus
    # and explore_end="reached", before the cap can fire.
    monkeypatch.setattr(igm, "EXPLORE_GOAL_DIST_M", 5.0)
    w, _ = _wrap(mode=0, max_steps=10)
    for _ in range(4):
        _, _, terminated, *_ = w.step(ACT)
        assert terminated is False
    _, reward, terminated, truncated, info = w.step(ACT)
    assert terminated is True and truncated is False
    assert info["explore_end"] == "reached"
    assert reward == pytest.approx(EXPLORE_SPEED_PREF_K)


def test_explore_invalid_beats_arrival(monkeypatch):
    # ORDERING: invalid > failure > reached. An invalid step that is also
    # the arrival step reports the invalid, not "reached".
    monkeypatch.setattr(igm, "EXPLORE_GOAL_DIST_M", 1.0)
    w, _ = _wrap(mode=0, info={"simulator_invalid": True})
    _, reward, terminated, _, info = w.step(ACT)
    assert terminated is True
    assert info["explore_end"] == "invalid"
    assert reward == pytest.approx(-TERMINAL_PENALTY)


# -------------------------------------------- steering smoothness (night) ----
# (user decision 2026-09-20 night: -IG_STEER_SMOOTH_W * |steer_t -
# steer_(t-1)| on EVERY step of BOTH legs, ungated -- parked and failing
# steps included; the wrapper tracks the previous action's steer, reset to
# 0.0; |steer| <= 1 a side, so |delta| <= 2 and the term is at most -1.0.)

def test_smoothness_sign_flip_pays_minus_one_in_both_legs():
    # A sign-flip step (steer -1 -> +1, |delta| = 2) pays -0.5*2 = -1.0 on
    # top of the ordinary rows, in BOTH legs.
    rg, compg = image_goal_reward(DRIVING, (1.0, 0.0), False, mode=1,
                                  steer_delta=2.0)
    assert compg["steer_smooth"] == pytest.approx(-1.0)
    assert rg == pytest.approx(-1.0 + SPEED_PREF_K - 1.0)      # -1.8
    assert rg < 0.0
    re_, compe = image_goal_reward(DRIVING, (1.0, 0.0), False, mode=0,
                                   steer_delta=2.0)
    assert compe["steer_smooth"] == pytest.approx(-1.0)
    assert re_ == pytest.approx(EXPLORE_SPEED_PREF_K - 1.0)    # +1.0


def test_smoothness_constant_steering_pays_zero():
    # steer_t == steer_(t-1): the delta is 0 and the term is exactly 0, in
    # both legs.
    rg, compg = image_goal_reward(DRIVING, (1.0, 0.0), False, mode=1,
                                  steer_delta=0.0)
    assert compg["steer_smooth"] == pytest.approx(0.0)
    assert rg == pytest.approx(-1.0 + SPEED_PREF_K)            # -0.8
    re_, compe = image_goal_reward(DRIVING, (1.0, 0.0), False, mode=0,
                                   steer_delta=0.0)
    assert compe["steer_smooth"] == pytest.approx(0.0)
    assert re_ == pytest.approx(EXPLORE_SPEED_PREF_K)          # +2.0


def test_smoothness_applies_while_parked_in_both_legs():
    # UNGATED (controller ruling): a parked step still pays -0.5*|delta| --
    # explore parked 0.0 - 0.5*1 = -0.5, goal parked -1.0 - 0.5 = -1.5.
    parked = {**DRIVING, "speed_kmh": 0.0}
    re_, compe = image_goal_reward(parked, (0.0, 0.0), False, mode=0,
                                   steer_delta=1.0)
    assert compe["steer_smooth"] == pytest.approx(-0.5)
    assert re_ == pytest.approx(-0.5)
    rg, compg = image_goal_reward(parked, (0.0, 0.0), False, mode=1,
                                  steer_delta=1.0)
    assert compg["steer_smooth"] == pytest.approx(-0.5)
    assert rg == pytest.approx(-1.5)


def test_smoothness_rides_on_the_explore_failing_step():
    # The failing explore step pays -TERMINAL_PENALTY - 0.5*|delta| (the
    # "exactly -5" construction is superseded, controller ruling). The
    # wrapper fixture commands steer -1 on the FIRST step after reset, so
    # delta = -1 - 0 = -1: -5 - 0.5 = -5.5, and the step still terminates.
    act = np.array([-1.0, 0.0], np.float32)
    w, _ = _wrap(mode=0, info={"off_road": True})
    _, reward, terminated, truncated, info = w.step(act)
    assert (terminated, truncated) == (True, False)
    assert info["explore_end"] == "off_road"
    assert reward == pytest.approx(-TERMINAL_PENALTY - 0.5)
    assert reward == pytest.approx(-5.5)
    comp = info["ig_reward_components"]
    assert comp["steer_smooth"] == pytest.approx(-0.5)
    assert comp["terminal_penalty"] == pytest.approx(-TERMINAL_PENALTY)


def test_smoothness_rides_on_the_goal_failing_step():
    # The goal failing step = ordinary (incl. smoothness) + -5 (controller
    # ruling). Steer -1 on the first off-road step: -2.5 - 0.5 - 5 = -8.0,
    # non-terminal.
    act = np.array([-1.0, 0.0], np.float32)
    w, _ = _wrap(mode=1, info={"off_road": True})
    _, reward, terminated, truncated, info = w.step(act)
    assert (terminated, truncated) == (False, False)
    assert info["ig_truncated"] is None
    assert reward == pytest.approx(-8.0)
    comp = info["ig_reward_components"]
    assert comp["steer_smooth"] == pytest.approx(-0.5)
    assert comp["terminal_penalty"] == pytest.approx(-TERMINAL_PENALTY)


def test_wrapper_tracks_prev_steer_across_steps_and_resets():
    # The wrapper tracks the PREVIOUS action's steer: the second identical
    # command pays no smoothness (delta 0), and a reset restores 0.0 so the
    # next episode's first step pays from the standstill command again.
    act = np.array([1.0, 0.0], np.float32)
    w, _ = _wrap(mode=1)
    r1 = w.step(act)[1]                     # delta 1.0: -1.0 - 1.5 + 0.2 - 0.5
    assert r1 == pytest.approx(-1.0 + SPEED_PREF_K - 0.5)      # -1.3
    r2 = w.step(act)[1]                     # same command: delta 0
    assert r2 == pytest.approx(-1.0 + SPEED_PREF_K)            # -0.8
    w.reset()
    assert w._prev_steer == 0.0
    r3 = w.step(act)[1]                     # delta 1.0 again
    assert r3 == pytest.approx(-1.0 + SPEED_PREF_K - 0.5)


def test_smoothness_weight_env_override_restores_pre_smoothness_rows(
        monkeypatch):
    # IG_STEER_SMOOTH_W=0 restores the pre-smoothness rows exactly
    # (reload-with-env, the file's own pattern; the finally restores the
    # shipped 0.5). The explore failing step pays exactly -5 again.
    monkeypatch.setenv("IG_STEER_SMOOTH_W", "0")
    importlib.reload(igm)
    try:
        assert igm.STEER_SMOOTH_W == pytest.approx(0.0)
        rg, compg = igm.image_goal_reward(DRIVING, (1.0, 0.0), False,
                                          mode=1, steer_delta=2.0)
        assert compg["steer_smooth"] == pytest.approx(0.0)
        assert rg == pytest.approx(-1.0 + igm.SPEED_PREF_K)    # -0.8
        re_, compe = igm.image_goal_reward({**DRIVING, "off_road": True},
                                           (1.0, 0.0), False, mode=0,
                                           steer_delta=2.0)
        assert compe["steer_smooth"] == pytest.approx(0.0)
        assert re_ == pytest.approx(-igm.TERMINAL_PENALTY)     # exactly -5
    finally:
        monkeypatch.delenv("IG_STEER_SMOOTH_W", raising=False)
        importlib.reload(igm)
    assert igm.STEER_SMOOTH_W == pytest.approx(0.5)


# ------------------------------------------------- the ReViND invariants ----

def test_goal_every_step_negative_and_explore_bonus_bounded():
    # THE SPLIT INVARIANT (user 2026-09-20, evening and night batches;
    # explore K raised 0.2 -> 2.0, user 2026-09-21). Walk the reward table
    # in BOTH modes: goal is r < 0 at EVERY step (the base cost dominates
    # the bonus; smoothness subtracts only); explore is
    # -TERMINAL_PENALTY - STEER_SMOOTH_W*2 <= r <= +EXPLORE_SPEED_PREF_K
    # (at delta 0 the floor is exactly -TERMINAL_PENALTY), with r > 0
    # possible (a failing cell pays exactly -TERMINAL_PENALTY at delta 0).
    saw_positive = False
    for speed in (0.0, 2.0, 10.0, 20.0, 40.0):
        for off in (False, True):
            for coll in (False, True):
                for crossed in (False, True):
                    info = dict(DRIVING)
                    info["speed_kmh"] = speed
                    info["off_road"] = off
                    info["lane_invasion_any"] = crossed
                    rg, _ = image_goal_reward(info, (1.0, 0.0), coll, mode=1)
                    assert rg < 0.0, (speed, off, coll, crossed)
                    re_, comp = image_goal_reward(info, (1.0, 0.0), coll,
                                                  mode=0)
                    assert (-TERMINAL_PENALTY - 2 * 0.5 <= re_
                            <= EXPLORE_SPEED_PREF_K), (speed, off, coll,
                                                       crossed)
                    saw_positive = saw_positive or re_ > 0.0
                    if off or crossed or coll:
                        assert re_ == pytest.approx(-TERMINAL_PENALTY)
                        assert comp["explore_end"] in ("off_road", "crossed",
                                                       "collision")
                    else:
                        assert comp["explore_end"] is None
    assert saw_positive          # the explore invariant allows r > 0


def test_goal_rows_carry_the_flats_and_the_bonus():
    # The GOAL TABLE (user 2026-09-20 flats + the evening saturating
    # bonus), asserted against the pure function's default (the single-mode
    # arm's call form): every value of the current revision. The
    # composition is a MASKED SUM: the goal leg is the reward and the
    # explore leg is masked to 0.
    v_star = TARGET_SPEED_KMH / 3.6
    table = [
        # (info overrides, has_collided, expected reward)
        ({}, False, -1.0 + SPEED_PREF_K),           # in-lane at v* (-0.8)
        ({"speed_kmh": 0.0}, False, -1.0),          # parked in lane
        ({"speed_kmh": 7.2}, False, -1.0 + SPEED_PREF_K * 2.0 / v_star),
                                                    # crawl 2 m/s (-0.928)
        ({"lane_invasion_any": True}, False, -1.0),  # crossing (gates L_t)
        ({"off_road": True}, False, -2.50),          # off-road
        ({}, True, -2.50),                           # post-collision
        ({"off_road": True}, True, -4.00),           # off-road+coll
    ]
    for overrides, coll, want in table:
        info = dict(DRIVING)
        info.update(overrides)
        r, comp = image_goal_reward(info, (1.0, 0.0), coll, mode=1)
        assert r == pytest.approx(want), (overrides, coll)
        assert r < 0.0                              # the ReViND invariant
        # and the DEFAULT call (no mode argument) is the same goal leg:
        r0, _ = image_goal_reward(info, (1.0, 0.0), coll)
        assert r0 == pytest.approx(want), (overrides, coll)
        assert "base" in comp and "terminal_penalty" not in comp
        assert comp["goal_reward_term"] == pytest.approx(want)
        assert comp["explore_reward_term"] == 0.0
        assert comp["goal_reward_term"] + comp["explore_reward_term"] \
            == pytest.approx(r)


def test_explore_rows_are_the_explore_leg_with_goal_masked():
    # THE MASKED SUM, explore side: the reward is exactly the explore leg
    # and the goal leg is masked to 0, in every row of the explore table.
    rows = [
        dict(DRIVING),                                      # at v*: +2.0
        {**DRIVING, "speed_kmh": 7.2},                      # crawl: +0.72
        {**DRIVING, "speed_kmh": 0.0},                      # parked: 0.0
        {**DRIVING, "off_road": True},                      # off-road: -5
        {**DRIVING, "lane_invasion_any": True},             # crossing: -5
        {**DRIVING, "collision_impulse": 5.0},              # collision: -5
    ]
    for info in rows:
        coll = bool(info.get("collision_impulse", 0.0) > 1.0)
        r, comp = image_goal_reward(info, (1.0, 0.0), coll, mode=0)
        assert comp["goal_reward_term"] == 0.0
        assert comp["explore_reward_term"] == pytest.approx(r)
        assert comp["goal_reward_term"] + comp["explore_reward_term"] \
            == pytest.approx(r)


# -------------------------------------------------- the ablation overrides ----

def test_ig_terminal_penalty_env_override(monkeypatch):
    # IG_TERMINAL_PENALTY stays env-overridable: with 10.0 in the env the
    # failing explore step pays exactly -10.0. Reload-with-env, matching the
    # IG_IN_LANE_BONUS test's pattern; the reload in the finally puts the
    # shipped default back for the later tests.
    monkeypatch.setenv("IG_TERMINAL_PENALTY", "10.0")
    importlib.reload(igm)
    try:
        assert igm.TERMINAL_PENALTY == pytest.approx(10.0)
        r, comp = igm.image_goal_reward({**DRIVING, "off_road": True},
                                        (1.0, 0.0), False, mode=0)
        assert r == pytest.approx(-10.0)
        assert comp["terminal_penalty"] == pytest.approx(-10.0)
        assert comp["explore_end"] == "off_road"
        # the pure goal leg still never reads it (the wrapper pays the flat
        # when it terminates on a failure):
        rg, compg = igm.image_goal_reward(dict(DRIVING), (1.0, 0.0), False,
                                          mode=1)
        assert rg == pytest.approx(-1.0 + igm.SPEED_PREF_K)
        assert "terminal_penalty" not in compg
    finally:
        monkeypatch.delenv("IG_TERMINAL_PENALTY", raising=False)
        importlib.reload(igm)
    assert igm.TERMINAL_PENALTY == pytest.approx(5.0)


def test_ig_speed_pref_k_env_override(monkeypatch):
    # The GOAL weight is env-overridable: with K = 0.5 the goal crawl row is
    # -1.0 + 0.5 * 2 / v* -- and the EXPLORE leg is untouched (it reads its
    # own IG_EXPLORE_SPEED_PREF_K, user 2026-09-21 split).
    monkeypatch.setenv("IG_SPEED_PREF_K", "0.5")
    importlib.reload(igm)
    try:
        assert igm.SPEED_PREF_K == pytest.approx(0.5)
        assert igm.EXPLORE_SPEED_PREF_K == pytest.approx(2.0)
        crawl = {**DRIVING, "speed_kmh": 7.2}
        re_, _ = igm.image_goal_reward(crawl, (1.0, 0.0), False, mode=0)
        assert re_ == pytest.approx(2.0 * 2.0 / (20.0 / 3.6))   # explore: 2.0
        rg, _ = igm.image_goal_reward(crawl, (1.0, 0.0), False, mode=1)
        assert rg == pytest.approx(-1.0 + 0.5 * 2.0 / (20.0 / 3.6))
    finally:
        monkeypatch.delenv("IG_SPEED_PREF_K", raising=False)
        importlib.reload(igm)
    assert igm.SPEED_PREF_K == pytest.approx(0.2)
    assert igm.EXPLORE_SPEED_PREF_K == pytest.approx(2.0)


def test_ig_explore_speed_pref_k_env_override(monkeypatch):
    # The EXPLORE weight is env-overridable too, independently: with
    # IG_EXPLORE_SPEED_PREF_K=0.2 the explore in-lane-at-v* row is restored
    # to the pre-raise +0.2 (user 2026-09-21), and the GOAL leg does NOT
    # pick up the explore constant even when it is pushed somewhere
    # impossible (7.0 would break the goal leg's ReViND invariant if the
    # goal call ever read it).
    monkeypatch.setenv("IG_EXPLORE_SPEED_PREF_K", "0.2")
    importlib.reload(igm)
    try:
        assert igm.EXPLORE_SPEED_PREF_K == pytest.approx(0.2)
        re_, compe = igm.image_goal_reward(dict(DRIVING), (1.0, 0.0), False,
                                           mode=0)
        assert compe["speed_penalty"] == pytest.approx(0.2)
        assert re_ == pytest.approx(0.2)
        # the goal leg keeps its own constant's row, exactly as before:
        rg, _ = igm.image_goal_reward(dict(DRIVING), (1.0, 0.0), False,
                                      mode=1)
        assert rg == pytest.approx(-1.0 + igm.SPEED_PREF_K)
        assert rg == pytest.approx(-0.8)
    finally:
        monkeypatch.delenv("IG_EXPLORE_SPEED_PREF_K", raising=False)
        importlib.reload(igm)
    assert igm.EXPLORE_SPEED_PREF_K == pytest.approx(2.0)


def test_goal_leg_never_reads_the_explore_constant(monkeypatch):
    # THE REGRESSION GUARD (user 2026-09-21 split): the goal leg's bonus
    # coefficient must stay IG_SPEED_PREF_K = 0.2 whatever the explore
    # constant is set to -- the goal leg's ReViND invariant (every row < 0)
    # depends on it. With the explore K pushed to 7.0 the goal in-lane row
    # is still exactly -0.8.
    monkeypatch.setenv("IG_EXPLORE_SPEED_PREF_K", "7.0")
    importlib.reload(igm)
    try:
        rg, compg = igm.image_goal_reward(dict(DRIVING), (1.0, 0.0), False,
                                          mode=1)
        assert rg == pytest.approx(-1.0 + igm.SPEED_PREF_K)
        assert rg == pytest.approx(-0.8)
        assert rg < 0.0
        assert compg["speed_penalty"] == pytest.approx(0.2)
        # and the default (single-mode) call form is the goal leg too:
        r0, _ = igm.image_goal_reward(dict(DRIVING), (1.0, 0.0), False)
        assert r0 == pytest.approx(-0.8)
    finally:
        monkeypatch.delenv("IG_EXPLORE_SPEED_PREF_K", raising=False)
        importlib.reload(igm)
    assert igm.EXPLORE_SPEED_PREF_K == pytest.approx(2.0)


def test_explore_return_bound_invariant():
    # THE EXPLORE BOUND (user 2026-09-21 raise): -6.0 <= r <= +2.0 -- or the
    # equivalent formula -TERMINAL_PENALTY - 2*STEER_SMOOTH_W <= r <=
    # EXPLORE_SPEED_PREF_K (worst: the failing step with max smoothness;
    # best: in lane at v*, delta 0).
    worst = -TERMINAL_PENALTY - 2 * igm.STEER_SMOOTH_W
    assert worst == pytest.approx(-6.0)
    best = EXPLORE_SPEED_PREF_K
    assert best == pytest.approx(2.0)
    # the two extreme rows, directly:
    r_max, _ = image_goal_reward(dict(DRIVING), (1.0, 0.0), False, mode=0)
    assert r_max == pytest.approx(best)
    r_min, _ = image_goal_reward({**DRIVING, "off_road": True}, (1.0, 0.0),
                                 False, mode=0, steer_delta=2.0)
    assert r_min == pytest.approx(worst)
    # the first-crossing step is unchanged: -5.0 - 0.5*|delta-steer| at
    # delta 1 from the reset's 0.0.
    r_cross, _ = image_goal_reward({**DRIVING, "lane_invasion_any": True},
                                   (1.0, 0.0), False, mode=0, steer_delta=1.0)
    assert r_cross == pytest.approx(-TERMINAL_PENALTY - 0.5)


def test_no_terminal_pays_a_bonus(monkeypatch):
    # No terminal anywhere pays a BONUS on top of the ordinary reward: the
    # goal success step, the explore 100 m step, an invalid step and the
    # cap step each pay exactly their ordinary row.
    w, _ = _stopped(max_steps=1000)                    # goal success step
    rewards = [w.step(ACT)[1] for _ in range(STOP_WINDOW_STEPS)]
    assert all(r == pytest.approx(-1.0) for r in rewards)

    monkeypatch.setattr(igm, "EXPLORE_GOAL_DIST_M", EXPLORE_ARRIVAL_M)
    w, _ = _wrap(mode=0)                                # explore arrival step
    rewards = [w.step(ACT)[1] for _ in range(10)]
    assert all(r == pytest.approx(EXPLORE_SPEED_PREF_K) for r in rewards)

    w, _ = _wrap(mode=0, info={"simulator_invalid": True})
    assert w.step(ACT)[1] == pytest.approx(-TERMINAL_PENALTY)

    w, _ = _wrap(mode=1, max_steps=3)                   # the cap step
    for _ in range(2):
        w.step(ACT)
    assert w.step(ACT)[1] < 0.0


def test_failures_are_terminal_only_in_explore():
    # THE NIGHT BATCH'S ASYMMETRY (user 2026-09-20 night): the same failure
    # event terminates EXPLORE (the leg owns its flat, minus smoothness)
    # but is NON-terminal in GOAL, where the flat rides on every failing
    # step and the episode continues. The rows they terminate/pay on differ
    # too (goal keeps its ordinary row under the flat; explore's ordinary
    # component is 0.0).
    w, _ = _wrap(mode=1, info={"lane_invasion_any": True})
    _, reward, terminated, truncated, _ = w.step(ACT)
    assert (terminated, truncated) == (False, False)
    assert reward == pytest.approx(-1.0 - TERMINAL_PENALTY)   # -6.0
    w, _ = _wrap(mode=0, info={"lane_invasion_any": True})
    _, reward, terminated, truncated, _ = w.step(ACT)
    assert (terminated, truncated) == (True, False)
    assert reward == pytest.approx(-TERMINAL_PENALTY)         # -5.0

    w, _ = _wrap(mode=1, info={"off_road": True})
    _, reward, terminated, truncated, _ = w.step(ACT)
    assert (terminated, truncated) == (False, False)
    assert reward == pytest.approx(-2.5 - TERMINAL_PENALTY)   # -7.5
    w, _ = _wrap(mode=0, info={"off_road": True})
    _, reward, terminated, truncated, _ = w.step(ACT)
    assert (terminated, truncated) == (True, False)
    assert reward == pytest.approx(-TERMINAL_PENALTY)


def test_the_envs_own_done_is_discarded_in_both_modes():
    for mode in (0, 1):
        w, _ = _wrap(mode=mode, done_from_env=True)
        assert w.step(ACT)[2] is False


# ------------------------------------------------------------- the banner ----

def test_the_banner_names_the_knobs(capsys):
    TwoModeImageGoalReward(_StubEnv(), explore_prob=0.3, force_mode=None,
                           seed=0)
    out = capsys.readouterr().out
    assert "explore_prob=0.3" in out
    assert "force_mode=None" in out
    assert "EXPLORE_GOAL_DIST_M" in out
    assert "IG_SPEED_PREF_K" in out
    # BOTH legs' coefficients on screen (user 2026-09-21 split): the goal
    # leg's 0.2 and the explore leg's own raised 2.0.
    assert "IG_EXPLORE_SPEED_PREF_K=2" in out
    assert "raised 0.2 -> 2.0" in out
    assert "IG_TERMINAL_PENALTY" in out
    # The flat values are named as such: the 2026-09-20 raise must be on
    # screen (the setting-that-changes-the-reward rule).
    assert "flat penalties off=1.5 coll=1.5" in out
    assert "raised 0.75 -> 1.5" in out
    # The evening batch (user 2026-09-20): the saturating bonus and the
    # stuck-only truncation are on screen with their constants; the night
    # batch's non-terminal failing-step flat and the ungated smoothness
    # must be there too.
    assert "saturating bonus" in out
    assert "failures NON-terminal" in out
    assert "goal-mode stuck truncation" in out
    assert f"IG_STUCK_STEPS={STUCK_STEPS}" in out
    assert f"IG_STEER_SMOOTH_W={0.5:g}" in out
    assert "EVERY step BOTH modes" in out


# ------------------------------------------------------------ the coherence ----

def _set(monkeypatch, **kw):
    monkeypatch.delenv("IMAGE_GOAL_TASK", raising=False)
    monkeypatch.delenv("GATED", raising=False)
    monkeypatch.delenv("OBS_WITH_STATE", raising=False)
    monkeypatch.delenv("OBS_WITH_MODE", raising=False)
    monkeypatch.delenv("OBS_WITH_COLLISION", raising=False)
    monkeypatch.delenv("PERCEPTION", raising=False)
    monkeypatch.delenv("VC1_READOUT", raising=False)
    monkeypatch.delenv("GATED_ONE_CRITIC", raising=False)
    monkeypatch.delenv("EPISODE_HER", raising=False)
    monkeypatch.delenv("TQC_TARGET_MIN", raising=False)
    monkeypatch.delenv("TQC_TARGET_MAX", raising=False)
    monkeypatch.delenv("NO_TIMELIMIT", raising=False)
    monkeypatch.delenv("LANE_FOLLOW", raising=False)
    monkeypatch.delenv("LANE_EXPLORE", raising=False)
    monkeypatch.delenv("TIME_TO_GOAL", raising=False)
    for k, v in kw.items():
        monkeypatch.setenv(k, v)
    import vae_nav.image_goal_reward as m
    importlib.reload(m)
    return m


TWO_MODE = dict(IMAGE_GOAL_TASK="1", GATED="1", OBS_WITH_STATE="1",
                OBS_WITH_MODE="1", PERCEPTION="vc1", VC1_READOUT="grid2",
                GATED_ONE_CRITIC="1", EPISODE_HER="1",
                TQC_TARGET_MIN="-300", TQC_TARGET_MAX="-1")


def test_coherence_two_mode_requires_state_and_bounds(monkeypatch):
    # IMAGE_GOAL_TASK=1 GATED=1 without OBS_WITH_STATE=1 -> SystemExit: the
    # pairing is refused until the mode-independent state block is on.
    m = _set(monkeypatch, **{**TWO_MODE, "OBS_WITH_STATE": "0"})
    with pytest.raises(SystemExit):
        m.image_goal_coherence()


def test_coherence_two_mode_requires_the_mode_bit(monkeypatch):
    m = _set(monkeypatch, **{**TWO_MODE, "OBS_WITH_MODE": "0"})
    with pytest.raises(SystemExit):
        m.image_goal_coherence()


def test_coherence_two_mode_requires_one_critic(monkeypatch):
    m = _set(monkeypatch, **{**TWO_MODE, "GATED_ONE_CRITIC": "0"})
    with pytest.raises(SystemExit):
        m.image_goal_coherence()


def test_coherence_two_mode_requires_her(monkeypatch):
    m = _set(monkeypatch, **{**TWO_MODE, "EPISODE_HER": "0"})
    with pytest.raises(SystemExit):
        m.image_goal_coherence()


def test_coherence_two_mode_requires_both_value_bounds(monkeypatch):
    for drop in ("TQC_TARGET_MIN", "TQC_TARGET_MAX"):
        kw = dict(TWO_MODE)
        del kw[drop]
        m = _set(monkeypatch, **kw)
        with pytest.raises(SystemExit):
            m.image_goal_coherence()


def test_coherence_two_mode_requires_the_vc1_readout(monkeypatch):
    m = _set(monkeypatch, **{**TWO_MODE, "VC1_READOUT": "cls9"})
    with pytest.raises(SystemExit):
        m.image_goal_coherence()


def test_coherence_two_mode_refuses_the_coll_key(monkeypatch):
    # The `coll` key belongs to the single-mode arm; the 2-mode pairing uses
    # `state`, and the two keys must never silently share a run.
    m = _set(monkeypatch, **{**TWO_MODE, "OBS_WITH_COLLISION": "1"})
    with pytest.raises(SystemExit):
        m.image_goal_coherence()


def test_coherence_two_mode_refuses_stacked_rewards(monkeypatch):
    for flag in ("LANE_FOLLOW", "LANE_EXPLORE", "TIME_TO_GOAL"):
        m = _set(monkeypatch, **{**TWO_MODE, flag: "1"})
        with pytest.raises(SystemExit):
            m.image_goal_coherence()


def test_coherence_two_mode_refuses_no_timelimit(monkeypatch):
    # NO_TIMELIMIT is trace collection, not the training run this pairing is
    # for; the wrapper's own cap would be the only clock left and the banner
    # would not match the trainer's.
    m = _set(monkeypatch, **{**TWO_MODE, "NO_TIMELIMIT": "1"})
    with pytest.raises(SystemExit):
        m.image_goal_coherence()


def test_coherence_two_mode_full_pairing_is_accepted(monkeypatch):
    m = _set(monkeypatch, **TWO_MODE)
    m.image_goal_coherence()          # no raise


def test_coherence_single_mode_unchanged(monkeypatch):
    # IMAGE_GOAL_TASK=1 GATED=0 without OBS_WITH_COLLISION=1 -> SystemExit
    # (old behaviour, verbatim).
    m = _set(monkeypatch, IMAGE_GOAL_TASK="1", GATED="0",
             OBS_WITH_COLLISION="0")
    with pytest.raises(SystemExit):
        m.image_goal_coherence()


def test_coherence_single_mode_still_refuses_gaps_and_stacking(monkeypatch):
    m = _set(monkeypatch, IMAGE_GOAL_TASK="1", GATED="0",
             OBS_WITH_COLLISION="1", PERCEPTION="mobilenet")
    with pytest.raises(SystemExit):
        m.image_goal_coherence()
    m = _set(monkeypatch, IMAGE_GOAL_TASK="1", GATED="0",
             OBS_WITH_COLLISION="1", PERCEPTION="vc1", LANE_FOLLOW="1")
    with pytest.raises(SystemExit):
        m.image_goal_coherence()
    m = _set(monkeypatch, IMAGE_GOAL_TASK="1", GATED="0",
             OBS_WITH_COLLISION="1", PERCEPTION="vc1")
    m.image_goal_coherence()          # the coherent single-mode run: accepted