"""NoMaD map-following rules, checked against navigation_policies' NoMaD.eval_action."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nomad_follow import localise_all, plan_finished, select_subgoal, window  # noqa: E402


def test_window_clips_at_both_ends():
    assert window(0, 3, 20) == (0, 4)
    assert window(10, 3, 20) == (7, 14)
    assert window(19, 3, 20) == (16, 20)


def test_subgoal_is_closest_when_near_else_next():
    # window starts at node 7; node 9 is closest
    assert select_subgoal([6.0, 4.0, 2.5, 5.0], start=7) == (9, 2)      # < 3: aim at the closest node itself
    assert select_subgoal([6.0, 4.0, 3.5, 5.0], start=7) == (9, 3)      # >= 3: aim at the next node
    assert select_subgoal([6.0, 4.0, 5.0, 3.2], start=7) == (10, 3)     # next node clipped to the window end


def test_stop_rule_and_global_localisation():
    assert plan_finished(18, 19) and plan_finished(19, 19) and not plan_finished(17, 19)
    assert localise_all([4.0, 1.5, 2.0]) == 1
