"""vae_nav.route_turns.turn_runs: the direction of every junction turn on a planner route,
counted exactly as the probe counts them (train_carla_sac._route_turns: a turn is a run
of RoadOption LEFT/RIGHT)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from vae_nav.route_turns import turn_runs  # noqa: E402


class Opt:                                   # stands in for carla's RoadOption enum
    def __init__(self, name):
        self.name = name


def test_no_turns():
    assert turn_runs([]) == {"seq": "", "turns": 0, "mixed_runs": 0}
    assert turn_runs(["LANEFOLLOW", "STRAIGHT", "LANEFOLLOW"])["turns"] == 0


def test_each_run_of_left_or_right_is_one_turn():
    r = turn_runs(["LANEFOLLOW", "LEFT", "LEFT", "LANEFOLLOW", "RIGHT", "RIGHT", "STRAIGHT"])
    assert r == {"seq": "LR", "turns": 2, "mixed_runs": 0}


def test_lane_changes_are_not_turns():
    assert turn_runs(["CHANGELANELEFT", "LANEFOLLOW", "CHANGELANERIGHT"])["turns"] == 0


def test_adjacent_left_right_is_one_run_like_the_probe_and_is_flagged():
    r = turn_runs(["LEFT", "LEFT", "RIGHT", "LANEFOLLOW"])
    assert r == {"seq": "L", "turns": 1, "mixed_runs": 1}


def test_accepts_enum_like_objects():
    r = turn_runs([Opt("RIGHT"), Opt("LANEFOLLOW"), Opt("LEFT")])
    assert r["seq"] == "RL" and r["turns"] == 2
