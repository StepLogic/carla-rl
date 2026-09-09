"""GatedReward: one env serving two modes, the bit drawn once per episode.

THIS FILE EXISTS BECAUSE THE WRAPPER HAD NEVER RUN. gated_reward.py was written
as a design sketch and never touched CARLA, so every contract below was an
assumption until now.

THE CONTRACT THAT MATTERS MOST is that the mode is constant for a whole episode.
The gated learner keeps one critic per mode, and a critic fits the return
distribution of its own mode. A mid-episode switch would splice two return
distributions into one bootstrap target, which no downstream assertion would
catch -- the loss curve would stay plausible while both critics learned a
blend.
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

from vae_nav.gated_reward import EXPLORE, GOAL, GatedReward  # noqa: E402


class _StubEnv(gym.Env):
    """Scripted info, never terminates on its own.

    `done_from_env` models the env's OWN termination, which fires on any lane
    crossing. Explore mode is supposed to discard it and re-derive termination
    with the junction exemption; goal mode is supposed to honour it.
    """
    metadata = {}

    def __init__(self, info=None, done_from_env=False):
        self.observation_space = spaces.Dict({})
        self.action_space = spaces.Box(
            np.array([-0.5, 0.0], np.float32), np.array([0.5, 0.6], np.float32))
        self.info = dict(info or {})
        self.info.setdefault("hero_pos", (0.0, 0.0))
        self.info.setdefault("speed_factor", 0.0)
        self.done_from_env = done_from_env

    def reset(self, **kw):
        return np.zeros(4, np.float32), dict(self.info)

    def step(self, action):
        return (np.zeros(4, np.float32), 0.0,
                bool(self.done_from_env), False, dict(self.info))


def _wrap(force_mode=None, info=None, max_steps=5, done_from_env=False,
          explore_prob=0.5, seed=0):
    return GatedReward(_StubEnv(info, done_from_env), explore_prob=explore_prob,
                       force_mode=force_mode, max_steps=max_steps, seed=seed)


ACT = np.zeros(2, np.float32)


# ------------------------------------------------------------ the mode ----

def test_forced_goal_mode_is_reported():
    assert _wrap(GOAL).reset()[1]["mode"] == GOAL


def test_forced_explore_mode_is_reported():
    assert _wrap(EXPLORE).reset()[1]["mode"] == EXPLORE


def test_explore_prob_zero_always_draws_goal():
    for seed in range(5):
        assert _wrap(explore_prob=0.0, seed=seed).reset()[1]["mode"] == GOAL


def test_explore_prob_one_always_draws_explore():
    for seed in range(5):
        assert _wrap(explore_prob=1.0, seed=seed).reset()[1]["mode"] == EXPLORE


def test_mode_is_constant_for_the_whole_episode():
    w = _wrap(explore_prob=0.5, max_steps=100)
    drawn = w.reset()[1]["mode"]
    for _ in range(10):
        assert w.step(ACT)[4]["mode"] == drawn


def test_mode_is_redrawn_on_reset():
    # Over many resets at p=0.5 both modes must appear, or the draw is stuck.
    w = _wrap(explore_prob=0.5, seed=1)
    seen = {w.reset()[1]["mode"] for _ in range(40)}
    assert seen == {GOAL, EXPLORE}


def test_forced_mode_survives_repeated_resets():
    w = _wrap(EXPLORE)
    assert {w.reset()[1]["mode"] for _ in range(10)} == {EXPLORE}


def test_the_draw_is_reproducible_for_a_seed():
    a = [_wrap(explore_prob=0.5, seed=7).reset()[1]["mode"] for _ in range(5)]
    b = [_wrap(explore_prob=0.5, seed=7).reset()[1]["mode"] for _ in range(5)]
    assert a == b


# ---------------------------------------------------------- termination ----

def test_collision_terminates_in_explore_mode():
    w = _wrap(EXPLORE, {"collision": True})
    w.reset()
    assert w.step(ACT)[2] is True


def test_broken_line_crossed_inside_a_junction_does_not_end_explore_mode():
    # The env's own done fires on ANY lane crossing, which would end an episode
    # every time the car turned through a junction. Explore mode re-derives
    # termination with the junction exemption: a Broken line crossed inside a
    # junction IS the turn.
    w = _wrap(EXPLORE, {"lane_any": True, "in_junction": True, "off_road": False,
                        "lane_marking_types": ["Broken"]}, done_from_env=True)
    w.reset()
    assert w.step(ACT)[2] is False


def test_the_same_broken_line_on_open_road_does_end_explore_mode():
    # Same marking, no junction: that is the vehicle drifting out of lane.
    w = _wrap(EXPLORE, {"lane_any": True, "in_junction": False, "off_road": False,
                        "lane_marking_types": ["Broken"]})
    w.reset()
    assert w.step(ACT)[2] is True


def test_a_solid_line_is_fatal_even_inside_a_junction():
    w = _wrap(EXPLORE, {"lane_any": True, "in_junction": True, "off_road": False,
                        "lane_marking_types": ["Solid"]})
    w.reset()
    assert w.step(ACT)[2] is True


def test_unknown_markings_fail_closed():
    # NOT A BUG, AND I NEARLY FILED IT AS ONE. Absent marking data is not
    # evidence of a legal crossing, so the rule terminates. A gated run whose
    # env stops publishing lane_marking_types would therefore get shorter
    # explore episodes, not silently permissive ones -- which is the safe
    # direction, but worth knowing when explore episode length drops.
    w = _wrap(EXPLORE, {"lane_any": True, "in_junction": True, "off_road": False})
    w.reset()
    assert w.step(ACT)[2] is True


def test_off_road_terminates_in_explore_mode():
    w = _wrap(EXPLORE, {"lane_any": True, "in_junction": True, "off_road": True})
    w.reset()
    assert w.step(ACT)[2] is True


def test_goal_mode_honours_the_env_termination():
    # Goal mode does NOT re-derive: the env's own goal-arm rules stand.
    w = _wrap(GOAL, done_from_env=True)
    w.reset()
    assert w.step(ACT)[2] is True


def test_explore_mode_truncates_at_max_steps():
    w = _wrap(EXPLORE, max_steps=3)
    w.reset()
    assert [w.step(ACT)[3] for _ in range(3)] == [False, False, True]


def test_truncation_does_not_fire_when_already_terminated():
    # A step cannot be both; the learner's mask reads terminated only.
    w = _wrap(EXPLORE, {"collision": True}, max_steps=1)
    w.reset()
    _, _, terminated, truncated, _ = w.step(ACT)
    assert (terminated, truncated) == (True, False)


def test_the_step_counter_resets_between_episodes():
    w = _wrap(EXPLORE, max_steps=2)
    w.reset()
    w.step(ACT)
    w.reset()
    assert w.step(ACT)[3] is False


# --------------------------------------------------------------- reward ----

def test_reward_is_finite_and_published_in_both_modes():
    for mode in (GOAL, EXPLORE):
        w = _wrap(mode, {"delta_distance": 1.0, "projected_distance": 0.5})
        w.reset()
        _, reward, _, _, info = w.step(ACT)
        assert np.isfinite(reward), mode
        assert info["gated_reward"] == reward


def test_a_nan_in_the_info_does_not_propagate_into_the_reward():
    # nan_to_num is applied on purpose; a single nan would poison the critic
    # target and, through the shared encoder, the actor as well.
    w = _wrap(GOAL, {"delta_distance": float("nan"),
                     "projected_distance": float("nan")})
    w.reset()
    assert np.isfinite(w.step(ACT)[1])


def test_the_two_modes_do_not_return_the_same_reward():
    # If they did, the gate would be decorative and two critics pointless.
    info = {"delta_distance": 5.0, "projected_distance": 5.0}
    rewards = []
    for mode in (GOAL, EXPLORE):
        w = _wrap(mode, info)
        w.reset()
        rewards.append(w.step(ACT)[1])
    assert rewards[0] != rewards[1], rewards


def test_displacement_is_measured_between_consecutive_steps():
    # The explore reward's progress term is built from hero_pos deltas. If the
    # previous position were never updated, displacement would grow without
    # bound and progress would be a ramp rather than a per-step quantity.
    env = _StubEnv({"hero_pos": (0.0, 0.0)})
    w = GatedReward(env, force_mode=EXPLORE, max_steps=50, seed=0)
    w.reset()
    env.info["hero_pos"] = (1.0, 0.0)
    w.step(ACT)
    first = w._prev.copy()
    env.info["hero_pos"] = (2.0, 0.0)
    w.step(ACT)
    assert np.allclose(first, [1.0, 0.0])
    assert np.allclose(w._prev, [2.0, 0.0])
