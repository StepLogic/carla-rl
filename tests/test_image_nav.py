"""Pose-free map navigation (vae_nav.image_nav), the Habitat procedure: arrival by
an image-match competition, a 150-step leg timeout, up to 3 replans, then the
real goal."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from vae_nav.image_nav import ImageNavigator, top1  # noqa: E402


def sc(**kw):
    s = np.zeros(10)
    for k, v in kw.items():
        s[int(k[1:])] = v
    return s


def test_the_subgoal_is_reached_after_two_consecutive_winning_steps():
    nav = ImageNavigator([0, 1, 2, 3], thr=0.5)
    assert nav.current() == 1
    assert nav.step(sc(n0=0.4, n1=0.6, n2=0.3)) is None          # one win
    assert nav.step(sc(n0=0.4, n1=0.6, n2=0.3)) == "switch"      # two in a row
    assert nav.current() == 2 and nav.switches == 1


def test_a_win_must_beat_both_neighbours_and_clear_the_threshold():
    nav = ImageNavigator([0, 1, 2, 3], thr=0.5)
    for s in (sc(n0=0.7, n1=0.6, n2=0.3),     # the previous node still wins
              sc(n0=0.4, n1=0.6, n2=0.65),    # the next node already wins
              sc(n0=0.1, n1=0.45, n2=0.1)):   # best, but below threshold
        assert nav.step(s) is None
    assert nav.step(sc(n1=0.6)) is None                          # a fresh run of 1
    assert nav.current() == 1


def test_an_interrupted_run_starts_over():
    nav = ImageNavigator([0, 1, 2], thr=0.5)
    nav.step(sc(n1=0.6)); nav.step(sc(n1=0.4)); nav.step(sc(n1=0.6))
    assert nav.current() == 1
    assert nav.step(sc(n1=0.6)) == "switch"


def test_the_last_node_reached_hands_over_the_real_goal():
    nav = ImageNavigator([0, 1], thr=0.5)
    nav.step(sc(n1=0.6)); assert nav.step(sc(n1=0.6)) == "switch"
    assert nav.final and nav.current() is None and nav.step(sc(n1=0.9)) is None


def test_a_leg_times_out_after_150_steps():
    nav = ImageNavigator([0, 1, 2], thr=0.5)
    out = [nav.step(sc(n1=0.1)) for _ in range(150)]
    assert out[:149] == [None] * 149 and out[149] == "timeout"


def test_three_replans_then_the_real_goal():
    nav = ImageNavigator([0, 1, 2], thr=0.5, leg_steps=1)
    for r in range(3):
        assert nav.step(sc()) == "timeout"
        assert nav.replan([5, 6, 7]) and nav.current() == 6 and nav.replans == r + 1
    assert nav.step(sc()) == "timeout"
    assert not nav.replan([5, 6, 7]) and nav.final


def test_no_route_on_replan_gives_up_at_once():
    nav = ImageNavigator([0, 1, 2], thr=0.5, leg_steps=1)
    nav.step(sc())
    assert not nav.replan(None) and nav.final


def test_top1_has_no_threshold_or_radius():
    assert top1([0.1, 0.05, 0.12]) == 2


def test_the_benchmark_navigates_by_image_without_reading_pose():
    src = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    assert 'EVAL_MAP_LOCALISE = os.environ.get("EVAL_MAP_LOCALISE", "pose")' in src
    body = src.split("def run_eval(")[1].split("\ndef ")[0]
    img = body.split("if img_scores is not None:")[1].split("elif EVAL_MAP_JOINT_LOCALISE")[0]
    # the start / goal choice uses only image scores; pose appears only in the RECORD
    choice = img.split("img_loc = {")[0]
    assert "hp" not in choice and "goal_pos" not in choice.replace('_gp = info.get("goal_pos")', "")
    step = body.split("if nav is not None and img_scores is not None:")[1].split("elif nav is not None and nav.update(")[0]
    assert "pos" not in step.replace("map_row[\"map_goal_node\"]", "")       # no pose in the loop
    assert "nav.step(img_scores(" in step and "top1(img_scores(" in step
