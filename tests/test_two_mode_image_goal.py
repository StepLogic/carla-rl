"""The 2-mode image-goal arm: ONE reward, TWO modes, NO explore bonus.

THE INVARIANT THE WHOLE FILE GUARDS: both modes call the same
image_goal_reward(), whose every step is < 0 (2026-09-20 velocity preference
included), and the explore mode's 100 m path arrival TERMINATES but pays
NOTHING -- the reward on that step is the ordinary step reward, so the episode
return is exactly the sum of its step rewards. Termination buys the end of the
negative stream and nothing else, in either mode.

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
    TwoModeImageGoalReward)

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

def test_mode_drawn_and_rewarded_identically():
    # THE UNIFIED CLAIM. Same info, both modes: the reward is IDENTICAL and
    # only the mode key differs. explore_prob=0 draws goal every reset,
    # explore_prob=1 draws explore every reset.
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
    assert out_g[1] == out_e[1] == pytest.approx(-1.0)


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
    # lane bonus is REMOVED (user 2026-09-20), so the step is -1.5556.
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


def test_latch_persists_in_explore_mode_too():
    # ONE reward means ONE collision latch: the persistent penalty is mode
    # independent, so explore mode must carry it as well.
    w, env = _wrap(mode=0)
    env.info["collision_impulse"] = 5.0
    w.step(ACT)
    env.info["collision_impulse"] = 0.0
    assert w.step(ACT)[1] == pytest.approx(-1.75)


# ------------------------------------------------------- explore arrival ----

def test_explore_arrival_at_path(monkeypatch):
    # Cumulative path >= EXPLORE_GOAL_DIST_M -> terminated, NO bonus, the
    # reward on the arrival step is the ordinary step reward, and the
    # episode's return is exactly the sum of its step rewards.
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
    assert all(r == pytest.approx(-1.0) for r in rewards)
    assert sum(rewards) == pytest.approx(-1.0 * len(rewards))


def test_explore_short_of_the_path_keeps_driving(monkeypatch):
    monkeypatch.setattr(igm, "EXPLORE_GOAL_DIST_M", EXPLORE_ARRIVAL_M)
    w, _ = _wrap(mode=0)
    for _ in range(9):
        _, _, terminated, *_ = w.step(ACT)
        assert terminated is False


def test_explore_end_is_cap_at_the_step_limit(monkeypatch):
    monkeypatch.setattr(igm, "EXPLORE_GOAL_DIST_M", EXPLORE_ARRIVAL_M)
    w, _ = _wrap(mode=0, max_steps=3)
    for _ in range(2):
        w.step(ACT)
    _, _, terminated, truncated, info = w.step(ACT)
    assert truncated is True and terminated is False
    assert info["explore_end"] == "cap"


def test_explore_invalid_terminates():
    w, _ = _wrap(mode=0, info={"simulator_invalid": True})
    _, _, terminated, *_ = w.step(ACT)
    assert terminated is True


# -------------------------------------------------- the ReViND invariant ----

def test_no_arrival_bonus_anywhere(monkeypatch):
    # Every reward in both modes < 0 even on terminal steps: the goal success
    # step, the explore 100 m step, an invalid step and the cap step each pay
    # the ordinary negative reward.
    w, _ = _stopped(max_steps=1000)                    # goal success step
    rewards = [w.step(ACT)[1] for _ in range(STOP_WINDOW_STEPS)]
    assert all(r < 0.0 for r in rewards)
    assert rewards[-1] < 0.0

    monkeypatch.setattr(igm, "EXPLORE_GOAL_DIST_M", EXPLORE_ARRIVAL_M)
    w, _ = _wrap(mode=0)                                # explore arrival step
    rewards = [w.step(ACT)[1] for _ in range(10)]
    assert all(r < 0.0 for r in rewards)

    w, _ = _wrap(mode=0, info={"simulator_invalid": True})
    assert w.step(ACT)[1] < 0.0

    w, _ = _wrap(mode=1, max_steps=3)                   # the cap step
    for _ in range(2):
        w.step(ACT)
    assert w.step(ACT)[1] < 0.0


def test_crossing_is_not_terminal_in_either_mode():
    for mode in (0, 1):
        w, _ = _wrap(mode=mode, info={"lane_invasion_any": True})
        _, reward, terminated, truncated, _ = w.step(ACT)
        assert (terminated, truncated) == (False, False), f"mode {mode}"
        assert reward == pytest.approx(-1.0)


def test_off_road_is_not_terminal_in_either_mode():
    for mode in (0, 1):
        w, _ = _wrap(mode=mode, info={"off_road": True})
        _, reward, terminated, *_ = w.step(ACT)
        assert terminated is False
        assert reward == pytest.approx(-1.75)


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