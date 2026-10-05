"""GATED_GOAL_REWARD=explore: explore mode's road-following reward plus a goal bonus.

User 2026-10-05: "can i use same reward for explore mode and add goal reached reward".

WHY. Same v43 policy, same town: explore mode drives a median 189 m before an episode
ends; goal mode with a far goal ends after 9 m, mostly OFF-ROAD. When the goal sits
90-180 degrees behind the car the median is 3 m and no episode succeeds. The goal arm's
dense reward paid closing the STRAIGHT-LINE distance, so it steered at the goal through
whatever lay in between. Explore's dense reward pays only forward progress along the
road, so it never asks for that. Under this kind the goal is reached for the bonus; the
dense signal never points off the road.

WHY ARRIVAL ENDS THE EPISODE (no chaining). With DISCOUNT=0.99 the value of continuing
to drive is bounded near 0.3/step / 0.01 = 30. A bonus above that strictly beats
driving past the goal, so termination costs nothing. Chaining would also break the
success log and the curriculum, which read success off the final step.

HER IS REFUSED for this kind until it has its own recost path: HER recomputes hindsight
rewards per kind, and without one it would price hindsight rows differently from live
ones -- the bug that mis-costed 83% of hindsight rows in v13-v15.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))


@pytest.fixture
def gr(monkeypatch):
    from vae_nav import gated_reward as g
    monkeypatch.setattr(g, "PROGRESS_MODE", "heading")
    monkeypatch.setattr(g, "PROGRESS_SCALE", 1.0)
    monkeypatch.setattr(g, "GOAL_EXPLORE_ARRIVAL_BONUS", 50.0)
    monkeypatch.setattr(g, "EXPLORE_ARRIVAL_BONUS", 10.0)
    monkeypatch.setattr(g, "EXPLORE_COLLISION_FATAL", True)
    monkeypatch.setattr(g, "EXPLORE_OFF_ROAD_FATAL", True)
    monkeypatch.setattr(g, "EXPLORE_LANE_MODE", "terminal")
    return g


def _info(**kw):
    base = {"hero_yaw": 0.0, "speed_factor": 0.5, "action_smoothness_penalty": 0.0,
            "collision": False, "off_road": False, "lane_any": False,
            "lane_marking_types": [], "in_junction": False}
    base.update(kw)
    return base


FWD = np.array([0.3, 0.0])     # 0.3 m straight ahead along the heading


def test_the_step_reward_is_exactly_explore_modes(gr):
    r, term, trunc, arrived = gr.goal_reward_explore(_info(), FWD, t=5, max_steps=100)
    want, fatal = gr.explore_reward(_info(), FWD)
    assert r == pytest.approx(want) and not fatal
    assert (term, trunc, arrived) == (False, False, False)


def test_the_reward_does_not_depend_on_where_the_goal_is(gr):
    """THE POINT. No term may pull towards the goal: two steps differing only in
    the goal position must pay the same."""
    a = gr.goal_reward_explore(_info(goal_pos=[100.0, 0.0]), FWD, t=5, max_steps=100)
    b = gr.goal_reward_explore(_info(goal_pos=[-100.0, 40.0]), FWD, t=5, max_steps=100)
    assert a == b


def test_driving_backwards_is_not_paid_even_towards_a_goal_behind(gr):
    """A straight-line reward would pay reversing towards a goal behind the car.
    This one pays forward progress along the heading only."""
    back = np.array([-0.3, 0.0])
    r_fwd, *_ = gr.goal_reward_explore(_info(goal_pos=[-50.0, 0.0]), FWD, 5, 100)
    r_back, *_ = gr.goal_reward_explore(_info(goal_pos=[-50.0, 0.0]), back, 5, 100)
    assert r_fwd > r_back


def test_arrival_pays_the_bonus_and_ends_the_episode(gr):
    r, term, trunc, arrived = gr.goal_reward_explore(
        _info(arrived=True), FWD, t=40, max_steps=100)
    base, _ = gr.explore_reward(_info(), FWD)
    assert arrived and term and not trunc
    assert r == pytest.approx(base + 50.0)


def test_arriving_on_a_fatal_step_is_a_failure_not_a_success(gr):
    """Explore's own rule: a car that crosses a line on the step it reaches the
    target still crossed a line."""
    r, term, trunc, arrived = gr.goal_reward_explore(
        _info(arrived=True, collision=True), FWD, t=40, max_steps=100)
    assert term and not arrived
    base, _ = gr.explore_reward(_info(collision=True), FWD)
    assert r == pytest.approx(base)                 # no bonus


def test_off_road_ends_the_episode_without_success(gr):
    _r, term, _tr, arrived = gr.goal_reward_explore(
        _info(off_road=True), FWD, t=5, max_steps=100)
    assert term and not arrived


def test_the_clock_truncates_rather_than_terminates(gr):
    """Running out of time is not a failure state the critic should treat as
    absorbing; it bootstraps."""
    _r, term, trunc, arrived = gr.goal_reward_explore(_info(), FWD, t=100, max_steps=100)
    assert trunc and not term and not arrived


def test_the_bonus_beats_the_discounted_value_of_driving_on(gr):
    """The no-chaining argument, checked: at gamma 0.99 the most driving on can be
    worth is the per-step progress over (1 - gamma)."""
    per_step, _ = gr.explore_reward(_info(), FWD)
    assert gr.GOAL_EXPLORE_ARRIVAL_BONUS > per_step / (1.0 - 0.99)


def test_the_kind_is_accepted_by_the_validator():
    from vae_nav.gated_config import GOAL_REWARD_KINDS
    assert "explore" in GOAL_REWARD_KINDS


_SNAP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "clean_fix1", "results", "run_env",
                     "carla_gated_v50_segvae_lagrange_t05.env")


def _real_env(**over):
    """v50's actual snapshot, switched to this reward: proves the REAL launch
    config passes, not a hand-made minimal one."""
    if not os.path.exists(_SNAP):
        pytest.skip("v50 snapshot not present")
    env = {}
    for line in open(_SNAP):
        k, _, v = line.rstrip("\n").partition("=")
        if k:
            env[k] = v
    # run_gated.sh exports these AFTER writing the snapshot, so the file never
    # holds them. Without GATED=1, check_gated_run returns on its first line and
    # every assertion below passes for nothing -- which is exactly what the first
    # version of these tests did.
    env.update(GATED="1", OBS_WITH_MODE="1", OBS_WITH_GOAL="1",
               OBS_WITH_VECTOR="1", EXPLORE_DISCOUNT="0.99", LANE_EXPLORE="0")
    env.update(GATED_GOAL_REWARD="explore", DISCOUNT="0.99", LAGRANGE_LANE="0",
               GATED_GOAL_LANE_TERMINAL="0")
    env.update(over)
    return env


def test_the_validator_actually_runs_on_this_env():
    """Guard against the vacuous pass: an unknown kind must be REJECTED, which
    only happens if the validator gets past its GATED early-return."""
    from vae_nav.gated_config import check_gated_run
    with pytest.raises(ValueError, match="GATED_GOAL_REWARD"):
        check_gated_run(_real_env(GATED_GOAL_REWARD="nonsense", EPISODE_HER="0"))


def test_the_real_v51_config_is_accepted():
    from vae_nav.gated_config import check_gated_run
    check_gated_run(_real_env(EPISODE_HER="0"))


def test_her_is_refused_for_this_kind():
    from vae_nav.gated_config import check_gated_run
    with pytest.raises((SystemExit, ValueError), match="HER"):
        check_gated_run(_real_env(EPISODE_HER="1"))


# ------------------------------------------------------------ end labels --
def test_a_painted_crossing_is_labelled_crossing_not_other(gr):
    """v51's first launch logged other=0.83, cross=0.00: the generic classifier
    could not see explore's fatal-crossing rule. The lane invasions this run is
    about were being filed as 'other'."""
    assert gr.goal_explore_end(_info(lane_any=True), False, False) == "crossing"


@pytest.mark.parametrize("info_kw,arrived,trunc,want", [
    (dict(), True, False, "reached"),
    (dict(), False, True, "clock"),
    (dict(collision=True), False, False, "collision"),
    (dict(off_road=True), False, False, "off_road"),
    (dict(collision=True, off_road=True), False, False, "collision"),
])
def test_end_labels_follow_explore_modes_order(gr, info_kw, arrived, trunc, want):
    assert gr.goal_explore_end(_info(**info_kw), arrived, trunc) == want



# -------------------------------------------- one setup, two arrivals --
# User 2026-10-05: "keep same setup for goal mode and explore mode; the only
# difference is goal mode arrival is when the agent arrives at the goal".

def test_both_modes_pay_the_same_reward_on_the_same_step(gr):
    """Identical per-step reward whenever neither mode is arriving."""
    for kw in (dict(), dict(off_road=True), dict(collision=True),
               dict(lane_any=True, lane_marking_types=["Solid"])):
        g_r, *_ = gr.goal_reward_explore(_info(**kw), FWD, t=5, max_steps=100)
        e_r, _ = gr.explore_reward(_info(**kw), FWD)
        assert g_r == pytest.approx(e_r), kw


def test_the_bonus_is_the_ONE_thing_besides_arrival_that_differs(gr):
    """User: "keep the bonus different". Goal mode pays its own bonus on reaching
    the goal; explore pays EXPLORE_ARRIVAL_BONUS at 100 m. Two knobs, each read by
    one mode only."""
    r, *_ = gr.goal_reward_explore(_info(arrived=True), FWD, t=5, max_steps=100)
    base, _ = gr.explore_reward(_info(), FWD)
    assert r - base == pytest.approx(gr.GOAL_EXPLORE_ARRIVAL_BONUS)
    assert gr.GOAL_EXPLORE_ARRIVAL_BONUS != gr.EXPLORE_ARRIVAL_BONUS


def test_goal_mode_uses_the_explore_episode_cap():
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "src", "vae_nav", "gated_reward.py")).read()
    # The STEP branch, not the startup banner (which also tests goal_kind).
    start = src.index('elif self.mode == GOAL and self.goal_kind == "explore":')
    call = src.index("goal_reward_explore(", start)
    args = src[call:src.index(")", call)]
    assert "self.max_steps" in args and "goal_max_steps" not in args, args
