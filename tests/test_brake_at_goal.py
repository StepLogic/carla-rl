"""Arrival counts only if the car is slow enough.

User 2026-10-14: "the agent should brake near goal". Expressed as the ARRIVAL
CONDITION rather than a speed penalty, and that choice is the whole point:
negative throttle is REVERSE now and the progress term is ungated, so slowing
already pays negatively. The reward as it stands wants maximum forward speed
forever; withholding the bonus is the only thing that gives a reason to stop.

Above the threshold, being inside the goal radius is not arriving: no bonus,
and the episode CONTINUES, so the car has to come back down to collect it.
"""
import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

FWD = np.array([0.4, 0.0])


def _mod(max_speed):
    os.environ["GOAL_ARRIVAL_MAX_SPEED"] = str(max_speed)
    import vae_nav.gated_reward as g
    importlib.reload(g)
    g.PROGRESS_MODE, g.PROGRESS_SCALE = "heading", 1.0
    g.GOAL_EXPLORE_ARRIVAL_BONUS = 50.0
    g.EXPLORE_COLLISION_FATAL = True
    return g


def _info(speed, arrived=True, **kw):
    d = {"hero_yaw": 0.0, "speed_factor": 0.5, "action_smoothness_penalty": 0.0,
         "collision": False, "off_road": False, "lane_any": False,
         "lane_marking_types": [], "in_junction": False,
         "speed_kmh": speed, "arrived": arrived}
    d.update(kw)
    return d


@pytest.fixture(autouse=True)
def _restore():
    yield
    os.environ.pop("GOAL_ARRIVAL_MAX_SPEED", None)
    importlib.reload(importlib.import_module("vae_nav.gated_reward"))


def test_arriving_slowly_pays_the_bonus_and_ends_the_episode():
    g = _mod(5.0)
    r, term, trunc, arrived = g.goal_reward_explore(_info(2.0), FWD, 10, 1000)
    base, _ = g.explore_reward(_info(2.0), FWD)
    assert arrived and term and not trunc
    assert r == pytest.approx(base + 50.0)


def test_blasting_through_at_speed_is_NOT_arriving():
    """The radius is reached but the car is doing 20 km/h: no bonus, and the
    episode continues so it can come back and stop."""
    g = _mod(5.0)
    r, term, trunc, arrived = g.goal_reward_explore(_info(20.0), FWD, 10, 1000)
    base, _ = g.explore_reward(_info(20.0), FWD)
    assert not arrived and not term
    assert r == pytest.approx(base), "no bonus at speed"


def test_the_threshold_is_inclusive():
    g = _mod(5.0)
    assert g.goal_reward_explore(_info(5.0), FWD, 10, 1000)[3] is True
    assert g.goal_reward_explore(_info(5.01), FWD, 10, 1000)[3] is False


def test_zero_disables_it_which_is_v52s_behaviour():
    """v52 pays on arrival at any speed; it must stay reproducible."""
    g = _mod(0.0)
    assert g.goal_reward_explore(_info(40.0), FWD, 10, 1000)[3] is True


def test_the_default_is_disabled():
    os.environ.pop("GOAL_ARRIVAL_MAX_SPEED", None)
    import vae_nav.gated_reward as g
    importlib.reload(g)
    assert g.GOAL_ARRIVAL_MAX_SPEED == 0.0


def test_a_fatal_step_still_fails_however_slow():
    """Stopping on top of a wall is not an arrival."""
    g = _mod(5.0)
    _r, term, _tr, arrived = g.goal_reward_explore(
        _info(0.0, collision=True), FWD, 10, 1000)
    assert term and not arrived


def test_not_being_at_the_goal_is_unaffected_by_the_threshold():
    g = _mod(5.0)
    for speed in (0.0, 3.0, 30.0):
        assert g.goal_reward_explore(_info(speed, arrived=False),
                                     FWD, 10, 1000)[3] is False
