"""The explorer run script's constants must agree with each other.

Same pattern as test_sorb_config.py, and for the same reason: every failure it
catches has happened in this project already -- a mismatched step cap, a
mis-scaled kappa, a gamma that does not suit the return, a reward that quietly
reintroduced a goal.
"""
import re
import pathlib

import pytest

SCRIPT = pathlib.Path("run_lane_explorer.sh")


def exports():
    """Parse `export NAME=value` pairs, several per line, comments stripped."""
    out = {}
    for line in SCRIPT.read_text().splitlines():
        m = re.match(r"^export\s+(.*)", line)
        if not m:
            continue
        for pair in re.finditer(r"([A-Z_]+)=(\S+)", m.group(1).split("#")[0]):
            out[pair.group(1)] = pair.group(2).strip('"')
    return out


def test_script_exists_and_parses():
    assert SCRIPT.exists()
    e = exports()
    for name in ("LANE_EXPLORE", "PERCEPTION", "K_NOVELTY", "DISCOUNT",
                 "NOVELTY_PERSIST", "LANE_EXPLORE_MAX_STEPS"):
        assert name in e, f"{name} not parsed"


def test_the_explore_arm_is_selected():
    assert exports()["LANE_EXPLORE"] == "1"


def test_every_goal_reward_is_off():
    """A stale export leaking in would silently make this goal-conditioned,
    and junction ambiguity would then move as the goal moves."""
    e = exports()
    assert e["TIME_TO_GOAL"] == "0"
    assert e["LANE_FOLLOW"] == "0"
    for k in ("K_DIST", "K_SPEED", "K_LIVING"):
        assert float(e[k]) == 0.0


def test_no_goal_machinery_is_active():
    """No goal means no expert, no hindsight, no goal chaining."""
    e = exports()
    assert float(e["EXPERT_PROB"]) == 0.0
    assert e["EPISODE_HER"] == "0"


def test_episode_caps_agree():
    e = exports()
    assert e["LANE_EXPLORE_MAX_STEPS"] == e["MAX_EPISODE_STEPS"], (
        "a shorter outer TimeLimit would cut episodes before the wrapper's cap")


def test_novelty_persists_across_episodes():
    """LOAD-BEARING, not a tuning knob.

    Turn diversity emerges through the critic preferring the branch with unseen
    road. With a per-episode reset every episode starts blank, every branch
    looks equally unseen, and that pressure does not exist.
    """
    assert exports()["NOVELTY_PERSIST"] == "1"


def test_novelty_ceiling_stays_under_the_progress_term():
    """Or a car driving tight circles through fresh headings out-earns one
    driving forward. The bonus is clipped to [0,1] before scaling, so
    K_NOVELTY IS the ceiling."""
    metres_per_step = 0.283          # MEASURED over 75 eval episodes
    assert float(exports()["K_NOVELTY"]) < metres_per_step


def test_discount_is_below_one():
    """The explore return is a discounted coverage-rate, not a cost-to-go.

    The goal arm's gamma=1 is bounded only because every episode terminates at
    a goal or a step cap with a fixed per-step cost; that argument does not
    transfer to a reward with a positive progress term.
    """
    assert 0.0 < float(exports()["DISCOUNT"]) < 1.0


def test_perception_is_mobilenet_and_pooled():
    e = exports()
    assert e["PERCEPTION"] == "mobilenet"
    assert int(e["MOBILENET_POOL"]) > 0, (
        "the unpooled flatten is 5400-d, i.e. 43 GB of replay at 1M capacity")


def test_junction_crossings_are_forgiven():
    """LANE_STRICT=1 would end the episode on every turn, and junctions would
    read as catastrophic rather than ambiguous."""
    assert exports()["LANE_STRICT"] == "0"


def test_wedge_is_off():
    """The explorer's lane rule already terminates; a wedge on top would freeze
    the vehicle instead and burn the rest of the episode."""
    assert exports()["LANE_WEDGE"] == "0"


def test_distance_cap_exceeds_what_an_episode_can_travel():
    """500 steps at 0.283 m/step is ~140 m; the env's own distance terminator
    must not fire first."""
    e = exports()
    reachable = int(e["LANE_EXPLORE_MAX_STEPS"]) * 0.283
    assert float(e["MAX_DISTANCE_M"]) > reachable


def test_heldout_towns_are_not_trained_on():
    trained = set(exports()["ENV_TOWNS"].split(","))
    assert not (trained & {"Town03", "Town10HD"})
