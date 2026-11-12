"""v7's termination combination (user 2026-09-16), pinned so a flag drift cannot pass silently.

THE v7 RULE, in one place:
  goal mode     ARRIVAL and COLLISION terminate. Lane departure does NOT (GATED_GOAL_LANE_TERMINAL=0).
  explore mode  OFF-ROAD terminates (EXPLORE_OFF_ROAD_FATAL=1).
  penalties     every driving fault is 5.0 in both modes.

WHY PIN IT. These are all environment flags read at IMPORT, and the module's own history is of a lane rule that
silently evaluated False while the startup banner announced it as active. A test that reloads under the real flags is
the only thing that shows the combination is what the launcher claims.

The fixture reloads and RESTORES, following tests/test_gated_lane_terminal.py -- a reload without restoration leaked
into eleven tests in other modules there, which is the signature of getting this wrong.
"""
import importlib
import os

import pytest

BASE = {"arrived": False, "wedged": False, "collision": False,
        "in_junction": False, "off_road": False, "lane_any": False}


@pytest.fixture
def g():
    """gated_reward reloaded under v7's flags, then put back exactly as found."""
    env = {"COLLISION_TERMINAL": "1", "GATED_GOAL_LANE_TERMINAL": "0",
           "WEDGE_TRUNCATE": "0", "EXPLORE_OFF_ROAD_FATAL": "1",
           "GOAL_LANE_PENALTY": "5.0", "GOAL_OFF_ROAD_PENALTY": "5.0",
           "EXPLORE_OFF_ROAD_PENALTY": "5.0", "LANE_INVASION_NOOP": "0",
           "STEP_COST": "1.0"}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    import vae_nav.gated_reward as mod
    yield importlib.reload(mod)
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    importlib.reload(mod)


def test_arrival_terminates_goal_mode(g):
    _r, terminated, truncated, arrived = g.goal_reward_sorb({**BASE, "arrived": True}, t=5)
    assert (terminated, truncated, arrived) == (True, False, True)


def test_collision_terminates_goal_mode_and_is_billed(g):
    r, terminated, _trunc, arrived = g.goal_reward_sorb({**BASE, "collision": True}, t=10)
    assert terminated is True and arrived is False
    # billed the remaining clock exactly, so ending early is worth nothing
    assert r == pytest.approx(-1.0 * (1 + (g.GOAL_MAX_STEPS - 10)))


def test_lane_departure_does_not_terminate_goal_mode(g):
    """The v7 change: a crossing is charged, not fatal."""
    r, terminated, _trunc, _arr = g.goal_reward_sorb(
        {**BASE, "lane_any": True, "lane_marking_types": ["Solid"]}, t=10)
    assert terminated is False
    assert r == pytest.approx(-1.0 - 5.0)          # step cost plus the 5.0 lane fine


def test_off_road_is_charged_five_in_goal_mode_without_terminating(g):
    r, terminated, _trunc, _arr = g.goal_reward_sorb({**BASE, "off_road": True}, t=3)
    assert terminated is False
    assert r == pytest.approx(-1.0 - 5.0)


def test_off_road_is_fatal_in_explore_mode(g):
    assert g.explore_lane_fatal(False, [], True) is True


def test_goal_mode_still_truncates_at_the_cap(g):
    _r, terminated, truncated, _arr = g.goal_reward_sorb(dict(BASE), t=g.GOAL_MAX_STEPS)
    assert terminated is False and truncated is True
