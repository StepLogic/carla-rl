"""A MISSED GOAL ENDS THE EPISODE (user 2026-11-15: "ok. do it").

With the goal kept fixed (GOAL_KEEP_ON_PASS=1) a car that drove past its goal kept driving away from it
for up to 1,250 decisions, charged for the retreat and the lane cost on every one: measured on c0, goal
mode netted -0.28 per decision even counting its +100 arrivals. GOAL_MISS_END_M ends such an episode as a
failure once the car is that many metres further from the goal than it has been.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from vae_nav import gated_reward as gr  # noqa: E402


def test_not_missed_while_approaching_or_holding_distance():
    assert gr.goal_missed(dist_m=12.0, best_m=12.0, miss_m=10.0) is False
    assert gr.goal_missed(dist_m=3.0, best_m=0.8, miss_m=10.0) is False      # just past it, still close


def test_missed_once_the_car_is_miss_m_further_than_its_closest():
    assert gr.goal_missed(dist_m=10.9, best_m=0.8, miss_m=10.0) is True      # drove through and on
    assert gr.goal_missed(dist_m=22.5, best_m=12.0, miss_m=10.0) is True     # took the other branch


def test_off_by_default_and_before_any_distance_is_known():
    assert gr.goal_missed(dist_m=100.0, best_m=1.0, miss_m=0.0) is False
    assert gr.goal_missed(dist_m=100.0, best_m=None, miss_m=10.0) is False
    assert gr.goal_missed(dist_m=None, best_m=1.0, miss_m=10.0) is False


def test_the_end_is_named_missed_and_a_fault_on_the_same_step_still_wins():
    assert gr.goal_explore_end({"goal_missed": True}, arrived=False, truncated=False) == "missed"
    assert gr.goal_explore_end({"goal_missed": True}, arrived=True, truncated=False) == "reached"
    assert gr.goal_explore_end({"goal_missed": True, "collision": True}, False, False) in ("collision", "missed")


def test_goal_distance_is_the_straight_line_from_the_car_to_the_goal():
    assert gr.goal_distance_m({"hero_pos": [0.0, 0.0, 0.3], "goal_pos": [3.0, 4.0, 0.0]}) == 5.0
    assert gr.goal_distance_m({"hero_pos": [0.0, 0.0]}) is None
