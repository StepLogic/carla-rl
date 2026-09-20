"""The 2-mode image-goal arm: ONE pure function, TWO modes, TWO contracts.

THE INVARIANTS THE FILE GUARDS (user decision 2026-09-20 splits them):
  - GOAL mode (mode 1): the ReViND reward is UNCHANGED and FROZEN -- every
    step < 0, the sustained-stop success terminates and pays no bonus, and
    off-road / crossings / collisions only cost per step.
  - EXPLORE mode (mode 0): NO TIME PRESSURE and TERMINAL FAILURES. The step
    reward is the velocity preference alone (r <= 0, exactly 0.0 in lane at
    v*), the FIRST off-road / crossing / collision step (and an invalid
    step) TERMINATES paying exactly -IG_TERMINAL_PENALTY, and the 100 m path
    arrival terminates with NO bonus (0.0 at v*). The cap truncates as a
    safety bound. ORDERING: invalid > failure > arrival > cap.

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
    IN_LANE_BONUS, SPEED_PREF_K, STOP_WINDOW_STEPS, TARGET_SPEED_KMH,
    TERMINAL_PENALTY, TwoModeImageGoalReward, image_goal_reward)

ACT = np.zeros(2, np.float32)
# One metre of forward motion per step along the lane's forward vector, at
# TARGET_SPEED (20 km/h): the in-lane driving state. The fixture drives AT the
# target speed so the velocity term is zero and the arithmetic below stays
# exact.
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
    # THE MODE SPLIT (user 2026-09-20). Same info, both modes, but the
    # branches differ now: goal keeps the ReViND step (-1.0 at v*) while
    # explore pays the speed preference ALONE, which is exactly 0.0 in lane
    # at v*. The reward is still the same pure function -- only the mode
    # argument differs.
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
    assert out_g[1] == pytest.approx(-1.0)
    assert out_e[1] == pytest.approx(0.0)


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
    # (moving nothing is not driving the wrong way) and pays the velocity
    # preference for being stopped, which is v = 0 against 20 km/h. The flat
    # lane bonus is REMOVED (user 2026-09-20), so the step is -2.1111
    # (K = 0.2, user 2026-09-20).
    comp = info["ig_reward_components"]
    assert reward == pytest.approx(
        -1.0 + IN_LANE_BONUS - SPEED_PREF_K * TARGET_SPEED_KMH / 3.6)
    assert reward == pytest.approx(
        -1.0 - SPEED_PREF_K * TARGET_SPEED_KMH / 3.6)
    assert reward == comp["base"] + comp["lane_bonus"] + comp["speed_penalty"]


def test_goal_mode_still_truncates_at_the_cap_without_success():
    # A goal episode that never stops runs to the cap like any other.
    w, _ = _wrap(mode=1, max_steps=5)
    for _ in range(5):
        w.step(ACT)
    obs, reward, terminated, truncated, _ = w.step(ACT)
    assert truncated is True and terminated is False
    assert reward < 0.0


# --------------------------------------------------- the collision latch ----

def test_latch_persists_until_reset():
    w, env = _wrap(mode=1)
    env.info["collision_impulse"] = 5.0
    w.step(ACT)                                # the impact step
    env.info["collision_impulse"] = 0.0        # back on the road, undamaged
    assert w.step(ACT)[1] == pytest.approx(-1.75)    # STILL paying
    w.reset()
    assert w.step(ACT)[1] == pytest.approx(-1.0)    # cleared


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
    # 0.0, and the episode's return is exactly the sum of its step rewards.
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
    assert all(r == pytest.approx(0.0) for r in rewards)
    assert sum(rewards) == pytest.approx(0.0)


def test_explore_arrival_at_the_real_100_m_target():
    # The shipped constant: 100 m of cumulative path, driven at v* (the stub
    # advances 10 m a step), terminates with explore_end="reached", NO bonus
    # and a return of exactly 0.0 -- the sum of ten zero steps.
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
    assert sum(rewards) == pytest.approx(0.0)


def test_explore_short_of_the_path_keeps_driving(monkeypatch):
    monkeypatch.setattr(igm, "EXPLORE_GOAL_DIST_M", EXPLORE_ARRIVAL_M)
    w, _ = _wrap(mode=0)
    for _ in range(9):
        _, _, terminated, *_ = w.step(ACT)
        assert terminated is False


def test_explore_end_is_cap_at_the_step_limit(monkeypatch):
    # The cap TRUNCATES (never terminates) and pays the ordinary step
    # reward, which with no base cost is 0.0 at v* -- a truncated clean
    # episode returns ~0.
    monkeypatch.setattr(igm, "EXPLORE_GOAL_DIST_M", EXPLORE_ARRIVAL_M)
    w, _ = _wrap(mode=0, max_steps=3)
    for _ in range(2):
        w.step(ACT)
    _, reward, terminated, truncated, info = w.step(ACT)
    assert truncated is True and terminated is False
    assert info["explore_end"] == "cap"
    assert reward == pytest.approx(0.0)


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
# (user decision 2026-09-20: no base cost, no flat penalties, terminal
# failures; every ordinary explore row is the velocity preference alone.)

def test_explore_in_lane_at_target_speed_pays_exactly_zero():
    # The no-time-pressure row: a clean in-lane step at v* costs nothing.
    w, _ = _wrap(mode=0)
    _, reward, terminated, truncated, _ = w.step(ACT)
    assert reward == pytest.approx(0.0)
    assert (terminated, truncated) == (False, False)


def test_explore_crawling_in_lane_pays_minus_0711():
    # 2 m/s against the 5.556 m/s target: -K * |2 - 5.556| = -0.711.
    _, comp = image_goal_reward({**DRIVING, "speed_kmh": 7.2}, (1.0, 0.0),
                                False, mode=0)
    assert comp["speed_penalty"] == pytest.approx(-0.7111, abs=1e-3)
    r, comp = image_goal_reward({**DRIVING, "speed_kmh": 7.2}, (1.0, 0.0),
                                False, mode=0)
    assert r == pytest.approx(-SPEED_PREF_K * abs(2.0 - TARGET_SPEED_KMH / 3.6))
    assert r == pytest.approx(-0.711, abs=1e-3)
    assert comp["terminal_penalty"] == 0.0
    assert comp["explore_end"] is None


def test_explore_parked_in_lane_pays_minus_1111():
    # The carve-out keeps L_t True at standstill, so a parked car in lane
    # pays -K * v* = -0.2 * 5.556 = -1.111.
    info = dict(DRIVING)
    info["speed_kmh"] = 0.0
    r, comp = image_goal_reward(info, (0.0, 0.0), False, mode=0)
    assert r == pytest.approx(-SPEED_PREF_K * TARGET_SPEED_KMH / 3.6)
    assert r == pytest.approx(-1.111, abs=1e-3)


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
    # the clean in-lane step at v* pays 0.0.
    w, env = _wrap(mode=0)
    env.info["collision_impulse"] = 0.5
    _, reward, terminated, *_ = w.step(ACT)
    env.info["collision_impulse"] = 0.0
    assert w._has_collided is False
    assert terminated is False
    assert reward == pytest.approx(0.0)


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
    assert reward == pytest.approx(0.0)


def test_explore_invalid_beats_arrival(monkeypatch):
    # ORDERING: invalid > failure > reached. An invalid step that is also
    # the arrival step reports the invalid, not "reached".
    monkeypatch.setattr(igm, "EXPLORE_GOAL_DIST_M", 1.0)
    w, _ = _wrap(mode=0, info={"simulator_invalid": True})
    _, reward, terminated, _, info = w.step(ACT)
    assert terminated is True
    assert info["explore_end"] == "invalid"
    assert reward == pytest.approx(-TERMINAL_PENALTY)


# ------------------------------------------------- the ReViND invariants ----

def test_goal_every_step_negative_and_explore_every_step_nonpositive():
    # THE SPLIT INVARIANT (user 2026-09-20). Walk the reward table in BOTH
    # modes: goal is r < 0 at EVERY step; explore is r <= 0 (exactly 0.0 in
    # lane at v*; a failure cell pays exactly -TERMINAL_PENALTY).
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
                    assert re_ <= 0.0, (speed, off, coll, crossed)
                    if off or crossed or coll:
                        assert re_ == pytest.approx(-TERMINAL_PENALTY)
                        assert comp["explore_end"] in ("off_road", "crossed",
                                                       "collision")
                    else:
                        assert comp["explore_end"] is None


def test_goal_rows_are_unchanged_by_the_mode_split():
    # The GOAL TABLE, asserted against the pure function's default (the
    # single-mode arm's call form): every previous value still holds. The
    # composition is a MASKED SUM: the goal leg is the reward and the
    # explore leg is masked to 0.
    table = [
        # (info overrides, has_collided, expected reward)
        ({}, False, -1.0),                                  # in-lane at v*
        ({"speed_kmh": 0.0}, False, -1.0 - 0.2 * (20.0 / 3.6)),   # parked
        ({"lane_invasion_any": True}, False, -1.0),         # crossing
        ({"off_road": True}, False, -1.75),                 # off-road
        ({}, True, -1.75),                                  # post-collision
        ({"off_road": True}, True, -2.50),                  # off-road+coll
    ]
    for overrides, coll, want in table:
        info = dict(DRIVING)
        info.update(overrides)
        r, comp = image_goal_reward(info, (1.0, 0.0), coll, mode=1)
        assert r == pytest.approx(want), (overrides, coll)
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
        dict(DRIVING),                                      # at v*: 0.0
        {**DRIVING, "speed_kmh": 7.2},                      # crawl: -0.711
        {**DRIVING, "speed_kmh": 0.0},                      # parked: -1.111
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
        # and the goal branch never reads it:
        rg, compg = igm.image_goal_reward(dict(DRIVING), (1.0, 0.0), False,
                                          mode=1)
        assert rg == pytest.approx(-1.0)
        assert "terminal_penalty" not in compg
    finally:
        monkeypatch.delenv("IG_TERMINAL_PENALTY", raising=False)
        importlib.reload(igm)
    assert igm.TERMINAL_PENALTY == pytest.approx(5.0)


def test_ig_speed_pref_k_env_override(monkeypatch):
    # The weight itself is env-overridable in both branches: with K = 0.5
    # the explore crawl row is -0.5 * |2 - 5.556| and the goal crawl row
    # moves by the same term on top of its base cost.
    monkeypatch.setenv("IG_SPEED_PREF_K", "0.5")
    importlib.reload(igm)
    try:
        assert igm.SPEED_PREF_K == pytest.approx(0.5)
        crawl = {**DRIVING, "speed_kmh": 7.2}
        re_, _ = igm.image_goal_reward(crawl, (1.0, 0.0), False, mode=0)
        assert re_ == pytest.approx(-0.5 * abs(2.0 - 20.0 / 3.6))
        rg, _ = igm.image_goal_reward(crawl, (1.0, 0.0), False, mode=1)
        assert rg == pytest.approx(-1.0 - 0.5 * abs(2.0 - 20.0 / 3.6))
    finally:
        monkeypatch.delenv("IG_SPEED_PREF_K", raising=False)
        importlib.reload(igm)
    assert igm.SPEED_PREF_K == pytest.approx(0.2)


def test_no_arrival_bonus_anywhere(monkeypatch):
    # No terminal anywhere pays a bonus: the goal success step, the explore
    # 100 m step, an invalid step and the cap step each pay their ordinary
    # (goal < 0, explore <= 0) reward.
    w, _ = _stopped(max_steps=1000)                    # goal success step
    rewards = [w.step(ACT)[1] for _ in range(STOP_WINDOW_STEPS)]
    assert all(r < 0.0 for r in rewards)
    assert rewards[-1] < 0.0

    monkeypatch.setattr(igm, "EXPLORE_GOAL_DIST_M", EXPLORE_ARRIVAL_M)
    w, _ = _wrap(mode=0)                                # explore arrival step
    rewards = [w.step(ACT)[1] for _ in range(10)]
    assert all(r <= 0.0 for r in rewards)
    assert rewards[-1] == pytest.approx(0.0)

    w, _ = _wrap(mode=0, info={"simulator_invalid": True})
    assert w.step(ACT)[1] == pytest.approx(-TERMINAL_PENALTY)

    w, _ = _wrap(mode=1, max_steps=3)                   # the cap step
    for _ in range(2):
        w.step(ACT)
    assert w.step(ACT)[1] < 0.0


def test_crossing_is_terminal_in_explore_but_not_in_goal():
    # GOAL (unchanged): a crossing costs its step and continues.
    w, _ = _wrap(mode=1, info={"lane_invasion_any": True})
    _, reward, terminated, truncated, _ = w.step(ACT)
    assert (terminated, truncated) == (False, False)
    assert reward == pytest.approx(-1.0)
    # EXPLORE (user 2026-09-20): the FIRST crossing terminates.
    w, _ = _wrap(mode=0, info={"lane_invasion_any": True})
    _, reward, terminated, truncated, _ = w.step(ACT)
    assert (terminated, truncated) == (True, False)
    assert reward == pytest.approx(-TERMINAL_PENALTY)


def test_off_road_is_terminal_in_explore_but_not_in_goal():
    # GOAL (unchanged): off-road costs -1.75 per step and continues.
    w, _ = _wrap(mode=1, info={"off_road": True})
    _, reward, terminated, *_ = w.step(ACT)
    assert terminated is False
    assert reward == pytest.approx(-1.75)
    # EXPLORE: the FIRST off-road step terminates at the flat penalty.
    w, _ = _wrap(mode=0, info={"off_road": True})
    _, reward, terminated, *_ = w.step(ACT)
    assert terminated is True
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
    assert "IG_TERMINAL_PENALTY" in out
    assert "UNCHANGED" in out          # goal mode: frozen, named as such


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