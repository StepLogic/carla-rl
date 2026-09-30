"""LANE_UNKNOWN_MARKS_FATAL=0: a crossing with NO marking data, ON the road,
does not end the episode (user 2026-09-30: "no marking data at all and offroad
false - non terminal").

Pins the three things that could go wrong: (a) the freed case is ONLY
no-data-and-on-road, (b) off_road still ends the episode regardless, and (c)
the default is unchanged, so every earlier run's rule is intact.
"""
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))


def _rule(flag):
    """lane_departure_is_fatal with the env var set. The module reads the flag
    per CALL, so a reload is not strictly needed -- it is done anyway so the
    test still passes if the read is ever hoisted to import time."""
    if flag is None:
        os.environ.pop("LANE_UNKNOWN_MARKS_FATAL", None)
    else:
        os.environ["LANE_UNKNOWN_MARKS_FATAL"] = flag
    import vae_nav.lane_explore_reward as m
    return importlib.reload(m).lane_departure_is_fatal


@pytest.fixture(autouse=True)
def _clean():
    yield
    os.environ.pop("LANE_UNKNOWN_MARKS_FATAL", None)
    import vae_nav.lane_explore_reward as m
    importlib.reload(m)


# (marks, in_junction, off_road) -> fatal? , for each flag setting
CASES = [
    # no marking data, on the road: THE CASE THE FLAG CHANGES
    ([],          False, False, True,  False),
    (None,        False, False, True,  False),
    # no marking data but OFF ROAD: still fatal either way
    ([],          False, True,  True,  True),
    # unpainted edge: already non-fatal before the flag existed
    (["NONE"],    False, False, False, False),
    # painted line on open road: fatal either way -- the flag must not leak
    (["Broken"],  False, False, True,  True),
    (["Solid"],   False, False, True,  True),
    # painted line inside a junction: the turn, non-fatal either way
    (["Broken"],  True,  False, False, False),
    # an ILLEGAL marking is fatal even in a junction, either way
    (["Solid"],   True,  False, True,  True),
]


@pytest.mark.parametrize("marks,junc,off,fatal_on,fatal_off", CASES)
def test_rule(marks, junc, off, fatal_on, fatal_off):
    assert _rule("1")(True, marks, junc, off, strict=False) is fatal_on
    assert _rule("0")(True, marks, junc, off, strict=False) is fatal_off


def test_default_is_the_old_behaviour():
    """No env var set must equal LANE_UNKNOWN_MARKS_FATAL=1."""
    assert _rule(None)(True, [], False, False, strict=False) is True


def test_no_crossing_is_never_fatal_under_either_setting():
    for flag in ("0", "1"):
        assert _rule(flag)(False, [], False, False, strict=False) is False


def test_explore_arm_sees_the_same_verdict():
    """gated_reward.explore_lane_fatal delegates to the shared rule, so the
    explorer and the env wedge cannot disagree about an unreported marking."""
    os.environ["LANE_UNKNOWN_MARKS_FATAL"] = "0"
    import vae_nav.lane_explore_reward as lm
    importlib.reload(lm)
    import vae_nav.gated_reward as gm
    importlib.reload(gm)
    assert gm.explore_lane_fatal(True, [], False, False) is False
    assert gm.explore_lane_fatal(True, [], True, False) is True     # off-road
    assert gm.explore_lane_fatal(True, ["Broken"], False, False) is True
