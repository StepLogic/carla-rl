"""The gated lane rule must actually FIRE, in both modes.

THE BUG THIS PINS. gated_reward read `info["lane_any"]`, which core_state never
publishes -- it publishes `lane_invasion_any`, `lane_invasion_illegal` and
`lane_invasion`, and `lane_any` exists only as a key INSIDE `done_reasons`. So
every lane check in that module silently evaluated False and no crossing ever
terminated anything, in either mode, while the startup banner announced the rule
as active. lane_explore_reward.py:256 had the fallback; the gated copy did not.

A missing key that defaults False is the worst shape of bug available here: it
disables a safety rule, changes no output format, raises nothing, and leaves the
logs claiming the rule is on.

The pricing asymmetry is deliberate and is also pinned below:
  goal    billed in TIME, no fine, so V keeps meaning steps-to-goal;
  explore a fine is fine -- its return is a coverage rate, not a distance.
"""
import importlib
import os

import pytest


BASE = {"arrived": False, "wedged": False, "collision": False,
        "in_junction": False, "off_road": False}


@pytest.fixture
def g():
    """Reload gated_reward under this arm's flags, then PUT IT BACK.

    The module reads its flags at import, so exercising them needs a reload --
    and a reload without restoration leaks into every test that runs afterwards
    in the same process. That is not hypothetical: the first version of this
    file failed eleven tests in other modules while passing in isolation, which
    is the signature of exactly this mistake.
    """
    env = {"GATED_GOAL_LANE_TERMINAL": "1", "COLLISION_TERMINAL": "1",
           "LANE_INVASION_NOOP": "0", "STEP_COST": "1.0"}
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


def test_goal_terminates_on_an_illegal_crossing_and_bills_it(g):
    """Solid @ road: terminal, and charged every remaining step."""
    r, term, trunc, arrived = g.goal_reward_sorb(
        dict(BASE, lane_invasion_any=True, lane_marking_types=["Solid"]),
        10, 750, 1.0)
    assert term and not trunc and not arrived
    # this step plus the 740 left on the clock -- identical to timing out, which
    # is what stops the terminal becoming an escape.
    assert r == pytest.approx(-1.0 * (1 + 750 - 10))


def test_goal_forgives_a_junction_crossing(g):
    """Broken INSIDE a junction is the turn, not a departure."""
    r, term, _t, _a = g.goal_reward_sorb(
        dict(BASE, lane_invasion_any=True, in_junction=True,
             lane_marking_types=["Broken"]), 10, 750, 1.0)
    assert not term and r == pytest.approx(-1.0)


def test_goal_clean_road_costs_one_step(g):
    r, term, _t, _a = g.goal_reward_sorb(dict(BASE), 10, 750, 1.0)
    assert not term and r == pytest.approx(-1.0)


def test_the_key_that_was_missing_is_the_one_read(g):
    """Reading only `lane_any` is what broke it -- prove the fallback is used."""
    only_real_key = dict(BASE, lane_invasion_any=True,
                         lane_marking_types=["Solid"])
    assert "lane_any" not in only_real_key
    _r, term, _t, _a = g.goal_reward_sorb(only_real_key, 1, 750, 1.0)
    assert term, ("a crossing reported under core_state's ACTUAL key did not "
                  "terminate -- the lane rule is a no-op again")


def test_explore_also_terminates_on_a_crossing(g):
    """Both modes fatal, but explore is allowed a fine."""
    _r, fatal = g.explore_reward(
        dict(BASE, lane_invasion_any=True, lane_marking_types=["Solid"]),
        disp=(0.3, 0.0))
    assert fatal


def test_goal_never_takes_a_lane_fine(g):
    """A penalty would make V mean 'steps to goal PLUS a fine'."""
    r, _term, _t, _a = g.goal_reward_sorb(
        dict(BASE, lane_invasion_any=True, lane_marking_types=["Solid"]),
        10, 750, 1.0)
    # Exactly the time bill, with no extra constant folded in.
    assert r == pytest.approx(-741.0)


# ---------------------------------------------------------------------------
# THE SHARED FILTER MUST MATCH THE ENV CONFIG'S WEDGE.
#
# test_wedge_lane_filter.py pins a LOCAL REIMPLEMENTATION of the wedge, not
# lane_departure_is_fatal itself, so the real function was free to drift from
# the rule it claims to share -- and it had: NONE was fatal here and forgiven
# there. Measured cost of that divergence: the scripted PID oracle failed 64% of
# its episodes on lane_any in the gated arm. These pin the REAL function.
# ---------------------------------------------------------------------------
from vae_nav.lane_explore_reward import lane_departure_is_fatal  # noqa: E402

NONE_M = ["carla.libcarla.LaneMarkingType.NONE"]
BROKEN_M = ["carla.libcarla.LaneMarkingType.Broken"]
SOLID_M = ["carla.libcarla.LaneMarkingType.Solid"]


@pytest.mark.parametrize("in_junction", [False, True])
def test_unpainted_is_never_fatal(in_junction):
    """No lane discipline exists at a line that is not painted."""
    assert not lane_departure_is_fatal(True, NONE_M, in_junction, False)


def test_broken_is_fatal_on_road_and_forgiven_at_a_junction():
    assert lane_departure_is_fatal(True, BROKEN_M, False, False)
    assert not lane_departure_is_fatal(True, BROKEN_M, True, False)


def test_illegal_markings_are_fatal_anywhere():
    assert lane_departure_is_fatal(True, SOLID_M, True, False)


def test_unknown_markings_fail_closed():
    """Absent data is not evidence of a legal crossing."""
    assert lane_departure_is_fatal(True, [], False, False)


def test_off_road_is_always_fatal():
    assert lane_departure_is_fatal(False, NONE_M, True, True)


def test_a_mixed_crossing_is_judged_by_the_painted_line():
    assert lane_departure_is_fatal(True, BROKEN_M + NONE_M, False, False)
