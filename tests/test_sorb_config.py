"""The run script's constants must agree with each other.

Every bug this catches has actually happened in this project: a step cap that
disagreed with the reward wrapper's cap, a kappa mis-scaled against the reward
so the critic silently stopped learning, and a value bound that did not match
the achievable return range.
"""
import re
import pathlib

import pytest

SCRIPT = pathlib.Path("run_sorb_agent.sh")


def exports():
    """Parse `export NAME=value` pairs from the run script.

    Handles SEVERAL pairs on one `export` line -- the script groups related
    settings that way (`export GOAL_DIST_MIN=2.0 GOAL_DIST_START=30.0 ...`),
    and a parser that took only the first would silently drop the rest and
    make the tests below pass vacuously or KeyError.
    """
    out = {}
    for line in SCRIPT.read_text().splitlines():
        m = re.match(r"^export\s+(.*)", line)
        if not m:
            continue
        body = m.group(1).split("#")[0]          # drop trailing comments
        for pair in re.finditer(r"([A-Z_]+)=(\S+)", body):
            out[pair.group(1)] = pair.group(2).strip('"')
    return out


def test_parser_reads_every_pair_on_a_shared_export_line():
    """Guards the parser itself: a silent miss makes every test below vacuous."""
    e = exports()
    for name in ("GOAL_DIST_MIN", "GOAL_DIST_START", "GOAL_DIST_TARGET",
                 "MAX_STEER", "MAX_THROTTLE", "ENV_TOWNS"):
        assert name in e, f"{name} was not parsed from the run script"


def test_script_exists():
    assert SCRIPT.exists(), f"{SCRIPT} not found"


def test_episode_caps_agree():
    e = exports()
    assert e["TTG_MAX_STEPS"] == e["MAX_EPISODE_STEPS"], (
        "TTG_MAX_STEPS and MAX_EPISODE_STEPS must match; a shorter outer "
        "TimeLimit would cut episodes before the reward wrapper's own cap and "
        "make the reported return shorter than the task allows")


def test_value_bound_matches_achievable_return():
    e = exports()
    floor = -float(e["STEP_COST"]) * int(e["TTG_MAX_STEPS"])
    assert float(e["TQC_TARGET_MIN"]) == pytest.approx(floor), (
        f"TQC_TARGET_MIN={e['TQC_TARGET_MIN']} but the achievable return floor "
        f"is {floor}. A clip inside the achievable range biases the critic; a "
        "clip outside it does nothing.")
    assert float(e["TQC_TARGET_MAX"]) == pytest.approx(0.0)


def test_horizon_is_reachable_within_the_step_cap():
    """H must be traversable with headroom, or episodes time out for reasons
    unrelated to the policy."""
    e = exports()
    metres_per_step = 0.283          # MEASURED over 75 recorded eval episodes
    steps_for_h = float(e["GOAL_DIST_TARGET"]) / metres_per_step
    assert int(e["TTG_MAX_STEPS"]) >= 2.0 * steps_for_h, (
        f"cap {e['TTG_MAX_STEPS']} gives only "
        f"{int(e['TTG_MAX_STEPS']) / steps_for_h:.1f}x the {steps_for_h:.0f} "
        "steps a goal at H needs under perfect driving")


def test_curriculum_is_inert():
    """A fixed band means nothing may advance, demote, or stall-step-back."""
    e = exports()
    assert e["GOAL_DIST_START"] == e["GOAL_DIST_TARGET"], (
        "start must equal target so the curriculum cannot advance")
    assert float(e["GOAL_DIST_MIN"]) < float(e["GOAL_DIST_TARGET"])


def test_wedge_is_off_and_discount_is_one():
    e = exports()
    assert e["LANE_WEDGE"] == "0", (
        "an absorbing failure makes V mean 'distance OR death'; the graph "
        "reads V as travel time")
    assert float(e["DISCOUNT"]) == 1.0


def test_her_off_and_expert_on():
    e = exports()
    assert e["EPISODE_HER"] == "0"
    assert float(e["EXPERT_PROB"]) == pytest.approx(0.2)


def test_heldout_towns_are_not_trained_on():
    e = exports()
    trained = set(e["ENV_TOWNS"].split(","))
    assert not (trained & {"Town03", "Town10HD"}), (
        f"held-out towns appear in ENV_TOWNS={e['ENV_TOWNS']}")


def test_goal_ceiling_within_distance_cap():
    e = exports()
    assert float(e["GOAL_DIST_TARGET"]) < float(e["MAX_DISTANCE_M"]), (
        "the env's own distance terminator would end episodes before the goal")
