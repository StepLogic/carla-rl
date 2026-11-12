"""Goal-mode lane invasion: TERMINAL and billed the remaining clock, in the
counting regime (user 2026-09-26: "make lane invasion terminal and bill
remaining steps in goal mode").

WHAT WAS WRONG. GATED_GOAL_LANE_TERMINAL existed, but goal_reward_sorb's counting
branch (GOAL_TRUNCATE_COUNTS=1, every run since v9) returned BEFORE the legacy
branch that reads `departed`, so the flag was silently inert.

WHAT THESE PIN
  - the bill is the legacy exact one, -step_cost * (1 + remaining), mask 0;
  - it fires only on a FATAL departure (the junction-forgiving rule), never on a
    junction turn or an unpainted road edge, and never on off-road alone;
  - arrival still wins;
  - HER_RECOST mirrors it: the relabelled leg is cut at the same step, so no
    hindsight goal is drawn from beyond a crossing the real episode never got
    past, and the logged end reason says "crossing";
  - with the flag off, nothing changes.

Constants are monkeypatched ON THE MODULES production reads, as in
test_episode_her_recost: environment writes after import would be dead.
"""
import inspect
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from vae_nav import gated_reward as gr  # noqa: E402
from vae_nav.episode_her import EpisodeRelabelBuffer  # noqa: E402

STEP_COST = 1.0
MAX_STEPS = 1000
FINE = 10.0
TRIP = 20

BROKEN = ["carla.LaneMarkingType.Broken"]
NONE_ONLY = ["carla.LaneMarkingType.NONE"]


@pytest.fixture
def counting(monkeypatch):
    """The counting regime with lane terminal ON, pinned on the modules."""
    from vae_nav import lane_explore_reward as ler
    monkeypatch.setattr(ler, "LANE_ANY_FATAL", False)   # LANE_STRICT off
    monkeypatch.setattr(gr, "GOAL_TRUNCATE_COUNTS", True)
    monkeypatch.setattr(gr, "LANE_TERMINAL", True)
    monkeypatch.setattr(gr, "STEP_COST", STEP_COST)
    monkeypatch.setattr(gr, "GOAL_ARRIVAL_BONUS", 10.0)
    monkeypatch.setattr(gr, "GOAL_OFF_ROAD_PENALTY", FINE)
    monkeypatch.setattr(gr, "GOAL_LANE_PENALTY", FINE)
    monkeypatch.setattr(gr, "GOAL_COLLISION_PENALTY", FINE)
    monkeypatch.setattr(gr, "GOAL_OFFROAD_TRUNCATE_STEPS", TRIP)
    monkeypatch.setattr(gr, "GOAL_COLLISION_TRUNCATE_STEPS", TRIP)
    monkeypatch.setattr(gr, "GOAL_CROSSING_TRUNCATE_STEPS", TRIP)
    monkeypatch.setattr(gr, "GOAL_MAX_STEPS", MAX_STEPS)
    monkeypatch.setattr(gr, "GOAL_SPEED_BONUS", 0.0)
    return monkeypatch


def sorb(info, t):
    return gr.goal_reward_sorb(info, t, max_steps=MAX_STEPS,
                               step_cost=STEP_COST)


def crossing(marks=BROKEN, junction=False, **extra):
    return dict(lane_any=True, lane_marking_types=marks, in_junction=junction,
                **extra)


# ---- the reward ----------------------------------------------------------

def test_fatal_crossing_terminates_with_the_exact_bill(counting):
    r, term, trunc, arrived = sorb(crossing(), t=100)
    assert term is True and trunc is False and arrived is False
    assert r == pytest.approx(-STEP_COST * (1 + (MAX_STEPS - 100)))   # -901


@pytest.mark.parametrize("t", [0, 1, 500, 999, MAX_STEPS])
def test_bill_equals_the_remaining_clock(counting, t):
    """A crossing is exactly as bad as sitting there until time runs out."""
    r, *_ = sorb(crossing(), t=t)
    assert r == pytest.approx(-STEP_COST * (1 + max(0, MAX_STEPS - t)))


def test_the_fine_is_not_multiplied_into_the_bill(counting):
    """11 x 1000 would sit far below the goal critic's value floor and be
    clipped, breaking exactness; the bill is step cost only."""
    r, *_ = sorb(crossing(), t=0)
    assert r == pytest.approx(-(MAX_STEPS + 1) * STEP_COST)
    assert r > -(STEP_COST + FINE) * MAX_STEPS


def test_junction_turn_is_not_terminal(counting):
    r, term, trunc, _ = sorb(crossing(junction=True), t=10)
    assert term is False
    assert r == pytest.approx(-STEP_COST - FINE)   # the normal fined step


def test_unpainted_road_edge_is_not_terminal(counting):
    """NONE-as-fatal took the scripted oracle to 0.000 at 100 m goals."""
    _, term, _, _ = sorb(crossing(marks=NONE_ONLY), t=10)
    assert term is False


def test_missing_marking_data_fails_closed(counting):
    _, term, _, _ = sorb(crossing(marks=[]), t=10)
    assert term is True


def test_off_road_alone_is_not_lane_terminal(counting):
    """This is the LANE rule; off-road keeps its own fine and step cap."""
    _, term, _, _ = sorb(dict(lane_any=False, off_road=True), t=10)
    assert term is False


def test_arrival_wins_over_a_fatal_crossing(counting):
    r, term, _, arrived = sorb(crossing(is_success=True), t=10)
    assert arrived is True and term is True
    assert r == pytest.approx(10.0)


def test_flag_off_leaves_the_counting_regime_unchanged(counting):
    counting.setattr(gr, "LANE_TERMINAL", False)
    r, term, trunc, _ = sorb(crossing(), t=100)
    assert term is False and trunc is False
    assert r == pytest.approx(-STEP_COST - FINE)


def test_helper_matches_the_explore_rule_on_every_lane_case(counting):
    """One definition of 'departed' across goal and explore modes."""
    cases = [(True, BROKEN, False), (True, BROKEN, True),
             (True, NONE_ONLY, False), (True, [], False), (False, BROKEN, False)]
    for lane_any, marks, junc in cases:
        info = dict(lane_any=lane_any, lane_marking_types=marks,
                    in_junction=junc)
        assert gr.goal_lane_departed(info) == gr.explore_lane_fatal(
            lane_any, marks, False, in_junction=junc, off_road_fatal=False), \
            (lane_any, marks, junc)


# ---- the hindsight mirror -------------------------------------------------

class FakeBuffer:
    def __init__(self):
        self.rows = []

    def insert(self, row, mode=None, segment=None):
        self.rows.append((row, mode))

    def __len__(self):
        return len(self.rows)


def goal_block(fields):
    return np.asarray(fields["goal_pos"], np.float32) - np.asarray(
        fields["pos"], np.float32)


def make_her():
    buf = FakeBuffer()
    her = EpisodeRelabelBuffer(
        buf, goal_block, goal_radius=2.0, step_cost=STEP_COST,
        time_to_goal=True, gated=True, recost_counting=True,
        k_goal=600.0, k_collision=5.0, k_lane=5.0, k_living=0.5, k_dist=1.0,
        seed=0)
    return her, buf


def drive(her, n, fatal_at=(), raw_lane_at=()):
    for t in range(n):
        pos = (float(t), 0.0)
        txn = {"observations": {"goal": np.zeros(2, np.float32),
                                "latent": np.full(4, float(t), np.float32)},
               "next_observations": {"goal": np.zeros(2, np.float32),
                                     "latent": np.full(4, t + 1.0, np.float32)},
               "actions": np.zeros(2, np.float32),
               "rewards": np.float32(-STEP_COST),
               "masks": np.float32(1.0), "dones": np.float32(0.0)}
        her.record(txn, {"pos": pos}, pos, hit=False,
                   lane=(t in fatal_at or t in raw_lane_at),
                   lane_fatal=t in fatal_at,
                   speed_kmh=20.0, target_speed_kmh=20.0)
    her.end_leg()


def test_her_cuts_the_leg_at_the_fatal_crossing(counting):
    her, _ = make_her()
    drive(her, 30, fatal_at=(12,))
    assert her._counting_end(0, 30) == 13


def test_no_hindsight_goal_from_beyond_the_crossing(counting):
    """A goal past step 12 would pay an arrival the real episode never had."""
    her, _ = make_her()
    drive(her, 30, fatal_at=(12,))
    her.flush_episode()
    assert her.last_pairs, "no hindsight rows; the test proves nothing"
    assert max(i for i, _ in her.last_pairs) < 12
    assert max(tp for _, tp in her.last_pairs) <= 12
    assert her.n_recost_truncated >= 1


def test_logged_end_reason_is_crossing(counting):
    her, _ = make_her()
    drive(her, 30, fatal_at=(12,))
    her.flush_episode()
    assert her.last_end_reason == "crossing"


def test_a_forgiven_crossing_does_not_cut_the_leg(counting):
    """A junction turn still pays its fine but must not end anything."""
    her, _ = make_her()
    drive(her, 30, raw_lane_at=(12,))
    assert her._counting_end(0, 30) == 30


def test_her_flag_off_does_not_cut(counting):
    counting.setattr(gr, "LANE_TERMINAL", False)
    her, _ = make_her()                      # reads the flag at construction
    drive(her, 30, fatal_at=(12,))
    assert her._counting_end(0, 30) == 30


def test_image_goal_record_accepts_the_flag():
    """The trainer passes lane_fatal to every relabel buffer; the image-goal
    subclass overrides record() without **kwargs, so without this it would
    raise TypeError on every step of every image-goal run."""
    from vae_nav.image_goal_her import ImageGoalRelabelBuffer
    assert "lane_fatal" in inspect.signature(
        ImageGoalRelabelBuffer.record).parameters
