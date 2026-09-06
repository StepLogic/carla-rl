"""Goal-free lane following, against the upstream reward it is taken from.

Upstream (train_env_config_v4_image_goal.py:compute_reward, forked from
nisheeth-golakiya/carla-rl):

    reward = -4e-3 - steer_change
    if speed_factor < 1.0:  reward += delta_distance + projected_distance
    -3.0 collision   -5.0 lane departure   -3.0 overspeed
    +5.0 goal        +2.0 checkpoint       -3.0 wrong junction

These tests pin that the goal-free half is reproduced exactly and the
goal-dependent half is absent -- the second is what makes the signal topology
rather than route.
"""
import gymnasium as gym
import numpy as np
import pytest

from vae_nav.lane_explore_reward import (BASE_PENALTY, COLLISION_PENALTY,
                                         LANE_DEPARTURE_PENALTY,
                                         LaneExploreReward,
                                         lane_departure_is_fatal)

BROKEN = ["carla.libcarla.LaneMarkingType.Broken"]
SOLID = ["carla.libcarla.LaneMarkingType.Solid"]
NONE_MARK = ["carla.libcarla.LaneMarkingType.NONE"]
CURB = ["carla.libcarla.LaneMarkingType.Curb"]


class FakeEnv(gym.Env):
    def __init__(self, steps):
        self.steps = steps          # list of info dicts
        self.t = -1
        box = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = box
        self.action_space = box

    def _info(self):
        base = {"hero_pos": np.zeros(2), "wp_forward": np.array([1.0, 0.0]),
                "action_smoothness_penalty": 0.0, "speed_factor": 0.5,
                "collision": False, "off_road": False, "done_speed": False,
                "lane_any": False, "in_junction": False,
                "lane_marking_types": []}
        if 0 <= self.t < len(self.steps):
            base.update(self.steps[self.t])
        return base

    def reset(self, **kw):
        # t = -1 so the FIRST step() returns steps[0]. Setting t = 0 here would
        # make reset consume the first scripted step and shift every assertion
        # by one -- which it did, until the tests caught it.
        self.t = -1
        return np.zeros(2, np.float32), self._info()

    def step(self, a):
        self.t += 1
        return np.zeros(2, np.float32), 999.0, True, False, self._info()


def run(steps):
    env = LaneExploreReward(FakeEnv(steps), max_steps=1000)
    env.reset()
    out = []
    for _ in range(len(steps)):
        _o, r, term, trunc, info = env.step(np.zeros(2, np.float32))
        out.append((r, term, trunc, info))
        if term or trunc:
            break
    return out


# -- the goal-free reward terms -------------------------------------------

def test_idle_step_pays_only_the_base_penalty():
    (r, *_), = run([{"hero_pos": np.zeros(2)}])
    assert r == pytest.approx(-BASE_PENALTY)


def test_progress_along_the_lane_is_rewarded():
    """1 m along the lane direction earns 1 m, minus the base penalty."""
    (r, *_), = run([{"hero_pos": np.array([1.0, 0.0])}])
    assert r == pytest.approx(1.0 - BASE_PENALTY)


def test_movement_across_the_lane_earns_nothing():
    """Projection onto the road: lateral displacement is not progress."""
    (r, *_), = run([{"hero_pos": np.array([0.0, 1.0])}])
    assert r == pytest.approx(-BASE_PENALTY)


def test_driving_backwards_along_the_lane_is_negative():
    (r, *_), = run([{"hero_pos": np.array([-1.0, 0.0])}])
    assert r == pytest.approx(-1.0 - BASE_PENALTY)


def test_speed_gate_withholds_progress_above_target():
    """Upstream pays the progress term only when speed_factor < 1.0."""
    (r, *_), = run([{"hero_pos": np.array([1.0, 0.0]), "speed_factor": 1.5}])
    assert r == pytest.approx(-BASE_PENALTY), "over target speed pays no progress"


def test_steer_change_is_penalised_unweighted():
    """Upstream subtracts steer_change raw, with no coefficient."""
    (r, *_), = run([{"action_smoothness_penalty": 0.3}])
    assert r == pytest.approx(-BASE_PENALTY - 0.3)


def test_collision_costs_three_and_ends_driving():
    r, term, _trunc, _i = run([{"collision": True}])[0]
    assert r == pytest.approx(-BASE_PENALTY - COLLISION_PENALTY)
    assert term is True, "you cannot keep driving after a collision"


def test_lane_departure_costs_five_and_ends_driving():
    r, term, _t, _i = run([{"lane_any": True, "lane_marking_types": SOLID}])[0]
    assert r == pytest.approx(-BASE_PENALTY - LANE_DEPARTURE_PENALTY)
    assert term is True


# -- the goal-dependent half must be ABSENT --------------------------------

def test_no_goal_bonus_is_ever_paid():
    """+5.0 goal and +2.0 checkpoint are goal-dependent and must not appear."""
    steps = [{"is_success": True, "arrived": True, "done_dist": True,
              "passed_chckp": True, "hero_pos": np.zeros(2)}]
    (r, *_), = run(steps)
    assert r == pytest.approx(-BASE_PENALTY), (
        "a goal bonus leaked in; the reward would then be route-dependent and "
        "junction ambiguity would move as the goal moves")


def test_no_progress_toward_a_goal_term():
    """delta_distance is dropped: only progress along the LANE counts."""
    steps = [{"hero_pos": np.array([0.0, 1.0]), "delta_distance": 10.0}]
    (r, *_), = run(steps)
    assert r == pytest.approx(-BASE_PENALTY)


# -- the junction rule, which is what makes junctions ambiguous -------------

def test_junction_crossing_is_forgiven_and_does_not_end_the_episode():
    """THE POINT. If turning through a junction terminated, junctions would
    read as catastrophic rather than ambiguous and the explorer could never
    learn to turn."""
    r, term, _t, _i = run([{"lane_any": True, "lane_marking_types": BROKEN,
                            "in_junction": True,
                            "hero_pos": np.array([1.0, 0.0])}])[0]
    assert term is False
    assert r == pytest.approx(1.0 - BASE_PENALTY), "no departure penalty"


def test_same_crossing_on_open_road_is_fatal():
    r, term, _t, _i = run([{"lane_any": True, "lane_marking_types": BROKEN,
                            "in_junction": False}])[0]
    assert term is True
    assert r == pytest.approx(-BASE_PENALTY - LANE_DEPARTURE_PENALTY)


@pytest.mark.parametrize("marks", [SOLID, CURB])
@pytest.mark.parametrize("in_junction", [True, False])
def test_illegal_markings_are_fatal_even_in_a_junction(marks, in_junction):
    assert lane_departure_is_fatal(True, marks, in_junction, False)


def test_unmarked_crossing_inside_a_junction_is_forgiven():
    assert not lane_departure_is_fatal(True, NONE_MARK, True, False)


def test_unknown_marking_fails_closed():
    """Absent data is not evidence of a legal crossing."""
    assert lane_departure_is_fatal(True, [], True, False)


@pytest.mark.parametrize("in_junction", [True, False])
def test_off_road_is_always_fatal(in_junction):
    assert lane_departure_is_fatal(False, BROKEN, in_junction, True)


def test_clean_driving_is_never_fatal():
    assert not lane_departure_is_fatal(False, [], False, False)


def test_illegal_set_matches_core_state():
    """The explorer and the reward's lane signal must agree on 'violation'."""
    import re
    from vae_nav.lane_explore_reward import ILLEGAL_MARKS
    src = open("src/vae_nav/core_state.py").read()
    m = re.search(r"illegal\s*=\s*\{([^}]*)\}", src)
    assert m
    core = {s.strip().strip('"\'') for s in m.group(1).split(",") if s.strip()}
    assert core == ILLEGAL_MARKS


def test_step_cap_truncates_without_terminating():
    env = LaneExploreReward(FakeEnv([{} for _ in range(50)]), max_steps=5)
    env.reset()
    for _ in range(5):
        _o, _r, term, trunc, _i = env.step(np.zeros(2, np.float32))
    assert trunc is True and term is False


def test_env_done_is_discarded():
    """FakeEnv returns done=True every step; only our rules may end it."""
    out = run([{}, {}, {}])
    assert all(term is False for _r, term, _t, _i in out)
