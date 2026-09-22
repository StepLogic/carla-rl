"""HER_RECOST: hindsight rows recosted under the goal arm's counting regime.

The min-time branch of EpisodeRelabelBuffer was consistent while the goal
reward WAS minimum time (v7/v8). v9's GOAL_TRUNCATE_COUNTS regime pays
-1/step, -fines on fault steps and +arrival on arrival, and truncates at
cumulative fault-step counts -- the min-time mirror silently fell behind and
measured ~50% of the goal side of the v9mb buffer, diluting the 100.0 fines
~2x. These tests pin the recosted relabelled reward to goal_reward_sorb's
counting branch, event for event, and pin the min-time path to staying put
when the flag is off.

The counting constants are monkeypatched ON THE MODULE the buffer reads
(single source of truth, as in production) rather than set via environment:
other test modules may already have imported gated_reward, so environment
writes here would be silently dead.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from vae_nav.episode_her import EpisodeRelabelBuffer  # noqa: E402

STEP_COST = 1.0
RADIUS = 2.0
BONUS = 10.0
FINE = 100.0
TRIP = 20


@pytest.fixture
def counting(monkeypatch):
    """The counting regime the buffer must mirror, pinned on the module."""
    from vae_nav import gated_reward as gr
    monkeypatch.setattr(gr, "GOAL_TRUNCATE_COUNTS", True)
    monkeypatch.setattr(gr, "STEP_COST", STEP_COST)
    monkeypatch.setattr(gr, "GOAL_ARRIVAL_BONUS", BONUS)
    monkeypatch.setattr(gr, "GOAL_OFF_ROAD_PENALTY", FINE)
    monkeypatch.setattr(gr, "GOAL_LANE_PENALTY", FINE)
    monkeypatch.setattr(gr, "GOAL_COLLISION_PENALTY", FINE)
    monkeypatch.setattr(gr, "GOAL_OFFROAD_TRUNCATE_STEPS", TRIP)
    monkeypatch.setattr(gr, "GOAL_COLLISION_TRUNCATE_STEPS", TRIP)
    monkeypatch.setattr(gr, "GOAL_CROSSING_TRUNCATE_STEPS", TRIP)
    monkeypatch.setattr(gr, "GOAL_MAX_STEPS", 1000)
    monkeypatch.setattr(gr, "GOAL_SPEED_BONUS", 0.0)


class FakeBuffer:
    def __init__(self):
        self.rows = []

    def insert(self, row, mode=None, segment=None):
        self.rows.append((row, mode))

    def __len__(self):
        return len(self.rows)


def goal_block(fields):
    """Stand-in for VAEGoalWrapper.goal_block: only the goal vector matters."""
    return np.asarray(fields["goal_pos"], np.float32) - np.asarray(
        fields["pos"], np.float32)


def make_buffer(**kw):
    buf = FakeBuffer()
    her = EpisodeRelabelBuffer(
        buf, goal_block, goal_radius=RADIUS,
        step_cost=STEP_COST, time_to_goal=True, gated=True,
        recost_counting=True,
        k_goal=600.0, k_collision=5.0, k_lane=5.0, k_living=0.5, k_dist=1.0,
        seed=0, **kw)
    return her, buf


def drive(her, n, lane_steps=(), offroad_steps=(), hit_steps=(), end_legs=(),
          speeds=None):
    """A straight run east, one metre per step, with per-step fault flags.

    speeds: optional list of (speed_kmh, target_speed_kmh) per step; the
    default is a uniform 22 km/h against a 20 km/h target (the speed the
    pre-bonus tests were written with)."""
    for t in range(n):
        pos = (float(t), 0.0)
        skmh, tkmh = speeds[t] if speeds is not None else (22.0, 20.0)
        txn = {"observations": {"goal": np.zeros(2, np.float32),
                                "latent": np.full(4, float(t), np.float32)},
               "next_observations": {"goal": np.zeros(2, np.float32),
                                     "latent": np.full(4, t + 1.0, np.float32)},
               "actions": np.zeros(2, np.float32),
               "rewards": np.float32(-STEP_COST),
               "masks": np.float32(1.0),
               "dones": np.float32(0.0)}
        her.record(txn, {"pos": pos}, pos,
                   hit=t in hit_steps,
                   lane=t in lane_steps,
                   off_road=t in offroad_steps,
                   speed_kmh=skmh, target_speed_kmh=tkmh)
        if t in end_legs:
            her.end_leg()
    her.end_leg()          # close the final leg, as flush would expect


def test_reward_matches_the_counting_branch_step_for_step(counting):
    """The recosted row equals goal_reward_sorb's counting branch, row for row."""
    her, buf = make_buffer()
    drive(her, 12, lane_steps=(5,), offroad_steps=(7,), hit_steps=(9,))
    her.flush_episode()
    assert her.last_pairs, "no hindsight rows produced; the test proves nothing"

    pos = np.array([[float(t), 0.0] for t in range(12)])
    n_fines = {5: 1, 7: 1, 9: 1}
    for (i, tp), (row, _m) in zip(her.last_pairs, buf.rows[12:]):
        d_next = float(np.linalg.norm(pos[min(i + 1, 11)] - pos[tp]))
        if d_next < RADIUS:
            expected = BONUS                     # arrival pays ONLY the bonus
            assert float(row["masks"]) == 0.0
            assert float(row["dones"]) == 1.0
        else:
            expected = -STEP_COST - FINE * n_fines.get(i, 0)
            assert float(row["masks"]) == 1.0
            assert float(row["dones"]) == 0.0
        assert float(row["rewards"]) == pytest.approx(expected), (i, tp)


def test_fines_actually_fire(counting):
    """At least one fault step paid its fine, and stacked faults add."""
    her, buf = make_buffer()
    drive(her, 12, lane_steps=(3, 4, 5), hit_steps=(5,))
    her.flush_episode()
    pos = np.array([[float(t), 0.0] for t in range(12)])
    n_fines = {3: 1, 4: 1, 5: 2}
    saw_fine = False
    for (i, tp), (row, _m) in zip(her.last_pairs, buf.rows[12:]):
        d_next = float(np.linalg.norm(pos[min(i + 1, 11)] - pos[tp]))
        if d_next < RADIUS:
            continue
        expected = -STEP_COST - FINE * n_fines.get(i, 0)
        assert float(row["rewards"]) == pytest.approx(expected)
        saw_fine = saw_fine or i in n_fines
    assert saw_fine, "no non-arrival fault-step row was drawn; nothing fired"


def test_non_arrival_rows_bootstrap_on(counting):
    her, buf = make_buffer()
    drive(her, 12)
    her.flush_episode()
    for row, _m in buf.rows[12:]:
        if float(row["dones"]) == 1.0:
            assert float(row["masks"]) == 0.0       # arrival: terminal
        else:
            assert float(row["masks"]) == 1.0       # fault steps: no mask cut


def test_tripped_count_cuts_the_leg(counting):
    """20 crossing steps trip the regime: no goal is drawn past the trip."""
    her, buf = make_buffer()
    drive(her, 40, lane_steps=range(20))            # trip at step 19
    her.flush_episode()

    assert her.n_recost_truncated == 1
    assert her.last_pairs, "no hindsight rows produced"
    assert all(tp < 20 for _i, tp in her.last_pairs), (
        "a hindsight goal was drawn past the tripped step -- the relabelled "
        "episode never had time to reach it, so its arrival bonus would be "
        "false supervision")
    # The tripped step and everything after it generate no rows.
    assert all(i < 19 for i, _tp in her.last_pairs)


def test_counts_are_leg_scoped(counting):
    """A tripped leg must not starve the next one: counts start over per leg."""
    her, buf = make_buffer()
    drive(her, 45, lane_steps=range(20), end_legs=(29,))
    her.flush_episode()

    later = [(i, tp) for i, tp in her.last_pairs if i >= 30]
    assert later, "leg 2 produced no relabels -- leg 1's counts leaked across"
    assert her.n_recost_truncated == 1
    assert all(i < 19 for i, _tp in her.last_pairs if i < 30)


def test_offroad_alone_trips(counting):
    her, buf = make_buffer()
    drive(her, 40, offroad_steps=range(20))
    her.flush_episode()
    assert her.n_recost_truncated == 1
    assert all(i < 19 for i, _tp in her.last_pairs)


def test_min_time_rows_unchanged_without_recost():
    """HER_RECOST off must leave the shipped min-time mirror byte-identical."""
    buf = FakeBuffer()
    her = EpisodeRelabelBuffer(
        buf, goal_block, goal_radius=RADIUS, step_cost=0.1,
        time_to_goal=True, k_goal=600.0, k_collision=5.0, k_lane=5.0,
        k_living=0.5, k_dist=1.0, seed=0)
    drive(her, 12)
    her.flush_episode()
    for row, _m in buf.rows[12:]:
        r = float(row["rewards"])
        assert r in (0.0, pytest.approx(-0.1)), r
        assert float(row["dones"]) == (1.0 if r == 0.0 else 0.0)


def test_recost_requires_gated_and_time_to_goal(counting):
    with pytest.raises(ValueError):
        EpisodeRelabelBuffer(
            FakeBuffer(), goal_block, goal_radius=RADIUS, step_cost=STEP_COST,
            time_to_goal=True, gated=False, recost_counting=True, k_goal=1.0,
            k_collision=1.0, seed=0)


def test_recost_requires_the_counting_regime(monkeypatch):
    from vae_nav import gated_reward as gr
    monkeypatch.setattr(gr, "GOAL_TRUNCATE_COUNTS", False)
    with pytest.raises(SystemExit):
        EpisodeRelabelBuffer(
            FakeBuffer(), goal_block, goal_radius=RADIUS, step_cost=STEP_COST,
            time_to_goal=True, gated=True, recost_counting=True, k_goal=1.0,
            k_collision=1.0, seed=0)


def test_recost_refuses_a_step_cost_mismatch(counting):
    with pytest.raises(SystemExit):
        EpisodeRelabelBuffer(
            FakeBuffer(), goal_block, goal_radius=RADIUS, step_cost=0.1,
            time_to_goal=True, gated=True, recost_counting=True, k_goal=1.0,
            k_collision=1.0, seed=0)

# -- v10: the target-velocity bonus, mirrored event for event ------------------

BONUS_W = 0.5


@pytest.fixture
def bonus(counting, monkeypatch):
    from vae_nav import gated_reward as gr
    monkeypatch.setattr(gr, "GOAL_SPEED_BONUS", BONUS_W)


def _row_expectations(her, buf, n_pos):
    """Map each inserted (i, tp) pair to its row, for reward assertions."""
    return zip(her.last_pairs, buf.rows[n_pos:])


def test_her_bonus_peaks_at_target(bonus):
    """Every clean non-arrival row driven at the target speed pays
    -step_cost + 0.5."""
    her, buf = make_buffer()
    drive(her, 12, speeds=[(20.0, 20.0)] * 12)
    her.flush_episode()
    assert her.last_pairs
    for (_i, _tp), (row, _m) in _row_expectations(her, buf, 12):
        if float(row["dones"]) == 1.0:
            assert float(row["rewards"]) == BONUS          # arrival only
        else:
            assert float(row["rewards"]) == pytest.approx(
                -STEP_COST + BONUS_W)


def test_her_bonus_tapers_and_floors(bonus):
    """Half target pays 0.25, standing still and 3x target pay 0."""
    her, buf = make_buffer()
    drive(her, 12, speeds=[(20.0, 20.0), (10.0, 20.0), (0.0, 20.0),
                           (60.0, 20.0)] + [(20.0, 20.0)] * 8)
    her.flush_episode()
    expect = {0: 0.5, 1: 0.25, 2: 0.0, 3: 0.0}
    for (i, _tp), (row, _m) in _row_expectations(her, buf, 12):
        if float(row["dones"]) == 1.0:
            continue
        assert float(row["rewards"]) == pytest.approx(
            -STEP_COST + expect.get(i, BONUS_W)), (i, row["rewards"])


def test_her_bonus_blocked_on_fault_steps(bonus):
    """A fault step pays its fines and NO bonus, however fast it was going."""
    her, buf = make_buffer()
    drive(her, 12, lane_steps=(2,), offroad_steps=(5,), hit_steps=(8,),
          speeds=[(20.0, 20.0)] * 12)
    her.flush_episode()
    for (i, _tp), (row, _m) in _row_expectations(her, buf, 12):
        if float(row["dones"]) == 1.0:
            continue
        n_fines = sum(1 for s in (2, 5, 8) if s == i)
        expected = -STEP_COST + (0.0 if n_fines else BONUS_W) - FINE * n_fines
        assert float(row["rewards"]) == pytest.approx(expected), (i, row["rewards"])


def test_her_bonus_matches_the_reward_event_for_event(bonus):
    """The recosted row equals goal_reward_sorb's counting branch, row for
    row, WITH the bonus active -- the single-source-of-truth guarantee."""
    from vae_nav import gated_reward as gr

    her, buf = make_buffer()
    speeds = [(18.0, 20.0), (20.0, 20.0), (5.0, 20.0), (40.0, 20.0)] * 3
    drive(her, 12, lane_steps=(6,), hit_steps=(9,), speeds=speeds)
    her.flush_episode()
    pos = np.array([[float(t), 0.0] for t in range(12)])
    for (i, tp), (row, _m) in _row_expectations(her, buf, 12):
        info = {"off_road": False, "lane_any": i == 6, "collision": i == 9,
                "speed_kmh": speeds[i][0], "target_speed_kmh": speeds[i][1]}
        d_next = float(np.linalg.norm(pos[min(i + 1, 11)] - pos[tp]))
        if d_next < RADIUS:
            info["arrived"] = True
        r, term, trunc, _ = gr.goal_reward_sorb(
            info, i, max_steps=1000, step_cost=STEP_COST,
            offroad_steps=0, collision_steps=0, crossing_steps=0)
        assert float(row["rewards"]) == pytest.approx(r), (i, tp)
        assert float(row["dones"]) == (1.0 if term else 0.0)


def test_her_bonus_absent_when_weight_zero(counting):
    """GOAL_SPEED_BONUS=0 (every pre-v10 run) must leave the rows as they
    were: -1/step - fines, regardless of the staged speeds."""
    her, buf = make_buffer()
    drive(her, 12, speeds=[(20.0, 20.0)] * 12)   # would peak if weight leaked
    her.flush_episode()
    for (_i, _tp), (row, _m) in _row_expectations(her, buf, 12):
        if float(row["dones"]) == 1.0:
            assert float(row["rewards"]) == BONUS
        else:
            assert float(row["rewards"]) == pytest.approx(-STEP_COST)
