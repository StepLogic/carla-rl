"""LANE_INVASION_NOOP: crossing a marking costs nothing and ends nothing.

WHAT THIS TURNS OFF, AND WHY THAT IS WORTH TESTING RATHER THAN ASSUMING.
lane_explore_reward.py's own docstring states that the lane-departure penalty is
"the ONLY thing discouraging" a car that drives tight circles, because the
progress term pays the same for circling as for driving down the road. Removing
it removes that discouragement, so the tests below pin exactly how much is
removed and exactly what is left.

WHAT IS LEFT: off-road stays fatal. It is not a lane crossing, and it is the
only remaining term that makes staying on the road worth anything. Collisions
also still bill in explore mode.

WHERE IT DOES NOTHING: goal mode under the SORB reward never looked at lane
state -- only arrival and the wedge end an episode -- so the flag is inert
there. That is asserted too, because "already a no-op" is a claim worth
holding still.
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

ACT = np.zeros(2, np.float32)


def _reload(noop):
    os.environ["LANE_INVASION_NOOP"] = "1" if noop else "0"
    import vae_nav.gated_reward as m
    return importlib.reload(m)


class _StubEnv(gym.Env):
    metadata = {}

    def __init__(self, info=None):
        self.observation_space = spaces.Dict({})
        self.action_space = spaces.Box(
            np.array([-0.5, 0.0], np.float32), np.array([0.5, 0.6], np.float32))
        self.info = dict(info or {})
        self.info.setdefault("hero_pos", (0.0, 0.0))
        self.info.setdefault("speed_factor", 0.0)

    def reset(self, **kw):
        return np.zeros(4, np.float32), dict(self.info)

    def step(self, a):
        return np.zeros(4, np.float32), 0.0, True, False, dict(self.info)


def _explore(m, info):
    w = m.GatedReward(_StubEnv(info), force_mode=m.EXPLORE, max_steps=50, seed=0)
    w.reset()
    return w.step(ACT)


def _goal(m, info):
    w = m.GatedReward(_StubEnv(info), force_mode=m.GOAL, goal_kind="sorb",
                      goal_max_steps=50, step_cost=1.0, seed=0)
    w.reset()
    return w.step(ACT)


CROSS = {"lane_any": True, "in_junction": False, "off_road": False,
         "lane_marking_types": ["Solid"]}


# ------------------------------------------------- explore: the change ----

def test_a_lane_crossing_terminates_when_the_flag_is_off():
    # The behaviour being removed, pinned so the change is visible.
    m = _reload(False)
    assert _explore(m, CROSS)[2] is True


def test_a_lane_crossing_does_not_terminate_when_the_flag_is_on():
    m = _reload(True)
    assert _explore(m, CROSS)[2] is False


def test_a_lane_crossing_costs_nothing_when_the_flag_is_on():
    m = _reload(True)
    crossed = _explore(m, CROSS)[1]
    clean = _explore(m, {})[1]
    assert crossed == pytest.approx(clean)


def test_a_lane_crossing_costs_five_when_the_flag_is_off():
    m = _reload(False)
    crossed = _explore(m, CROSS)[1]
    clean = _explore(m, {})[1]
    assert clean - crossed == pytest.approx(m.LANE_DEPARTURE_PENALTY)


def test_a_junction_crossing_is_unaffected_either_way():
    # It was already forgiven; the flag must not change that.
    junction = {"lane_any": True, "in_junction": True, "off_road": False,
                "lane_marking_types": ["Broken"]}
    for noop in (False, True):
        assert _explore(_reload(noop), junction)[2] is False, noop


# ------------------------------------------------- explore: what remains ----

def test_off_road_is_still_fatal_with_the_flag_on():
    # THE BACKSTOP. Without this the explore arm has no reason to stay on the
    # road at all, because the progress term pays the same for circling.
    m = _reload(True)
    assert _explore(m, {"off_road": True})[2] is True


def test_off_road_still_bills_with_the_flag_on():
    m = _reload(True)
    off = _explore(m, {"off_road": True})[1]
    clean = _explore(m, {})[1]
    assert clean - off == pytest.approx(m.LANE_DEPARTURE_PENALTY)


def test_a_collision_still_terminates_and_bills_with_the_flag_on():
    m = _reload(True)
    _, reward, terminated, _, _ = _explore(m, {"collision": True})
    assert terminated is True
    assert reward < _explore(m, {})[1]


def test_off_road_while_also_crossing_a_line_is_still_fatal():
    m = _reload(True)
    both = {"lane_any": True, "off_road": True, "in_junction": False,
            "lane_marking_types": ["Solid"]}
    assert _explore(m, both)[2] is True


# ---------------------------------------------------- goal: already inert ----

def test_goal_mode_ignores_lane_crossings_regardless_of_the_flag():
    # SORB never looked at lane state. Asserted so "already a no-op" cannot
    # quietly stop being true.
    for noop in (False, True):
        m = _reload(noop)
        assert _goal(m, CROSS)[2] is False, noop
        assert _goal(m, CROSS)[1] == pytest.approx(-1.0), noop


def test_goal_mode_ignores_off_road_too():
    # Under minimum time the penalty for any failure IS the time it costs.
    for noop in (False, True):
        m = _reload(noop)
        assert _goal(m, {"off_road": True})[2] is False, noop


def test_the_flag_defaults_to_off():
    os.environ.pop("LANE_INVASION_NOOP", None)
    import vae_nav.gated_reward as m
    assert importlib.reload(m).LANE_INVASION_NOOP is False


def test_the_banner_states_the_no_op_and_what_survives(capsys):
    # A setting that changes the reward must be visible on screen. Without it
    # the only live confirmation is /proc/<pid>/environ.
    m = _reload(True)
    m.GatedReward(_StubEnv(), force_mode=m.EXPLORE, max_steps=50, seed=0)
    out = capsys.readouterr().out
    assert "LANE INVASION IS A NO-OP" in out
    assert "OFF-ROAD IS STILL FATAL" in out
    assert "carla_lanefollow_nogoal" in out


def test_the_banner_states_the_penalty_when_the_flag_is_off(capsys):
    m = _reload(False)
    m.GatedReward(_StubEnv(), force_mode=m.EXPLORE, max_steps=50, seed=0)
    out = capsys.readouterr().out
    assert "lane invasion: penalty" in out
    assert "junction exemption" in out
