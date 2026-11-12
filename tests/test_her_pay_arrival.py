"""HER must produce ARRIVAL rows, or it cannot teach a min-time critic distance.

MEASURED 2026-10-01 (v42): `her=2014 paying=0` after 30 episodes. A hindsight
goal for step i is drawn 5-50 m away (HER_MIN_DISP=5) and the row pays only if
step i+1 lies within goal_radius=2.0 m of it. One step moves ~0.28 m, so
d_next >= 4.7 m > 2 m: NO hindsight row can ever pay. Every relabelled row is
-1/step bootstrapping that never terminates -- unanchored TD at gamma=1, the
opposite of what SORB's relabelling is for.

HER_PAY_ARRIVAL=1: for every hindsight goal tp drawn, also emit the row that
actually reaches it, (tp-1, tp). Its next position IS the goal, so it pays the
arrival reward and terminates exactly as an on-policy arrival does. Off by
default: every existing recipe is byte-identical.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
from vae_nav.episode_her import EpisodeRelabelBuffer  # noqa: E402

STEP_M, RADIUS, MIN_DISP, MAX_DISP, BONUS, FINE = 0.28, 2.0, 5.0, 50.0, 0.0, 10.0


@pytest.fixture
def counting(monkeypatch):
    from vae_nav import gated_reward as gr
    for k, v in dict(GOAL_TRUNCATE_COUNTS=True, STEP_COST=1.0, GOAL_ARRIVAL_BONUS=BONUS,
                     GOAL_OFF_ROAD_PENALTY=FINE, GOAL_LANE_PENALTY=FINE,
                     GOAL_COLLISION_PENALTY=FINE, GOAL_OFFROAD_TRUNCATE_STEPS=1,
                     GOAL_COLLISION_TRUNCATE_STEPS=1, GOAL_CROSSING_TRUNCATE_STEPS=20,
                     GOAL_MAX_STEPS=500, GOAL_SPEED_BONUS=0.0).items():
        monkeypatch.setattr(gr, k, v)


class FakeBuffer:
    def __init__(self): self.rows = []
    def insert(self, row, mode=None, segment=None): self.rows.append(row)
    def __len__(self): return len(self.rows)


def goal_block(fields):
    return np.asarray(fields["goal_pos"], np.float32) - np.asarray(fields["pos"], np.float32)


def run(monkeypatch, pay, n=300, offroad_at=None):
    monkeypatch.setenv("HER_PAY_ARRIVAL", "1" if pay else "0")
    buf = FakeBuffer()
    her = EpisodeRelabelBuffer(buf, goal_block, goal_radius=RADIUS, step_cost=1.0,
                               time_to_goal=True, gated=True, recost_counting=True,
                               min_disp=MIN_DISP, max_disp=MAX_DISP, seed=0,
                               k_goal=600.0, k_collision=5.0)
    for t in range(n):
        pos = (t * STEP_M, 0.0)
        txn = {"observations": {"goal": np.zeros(2, np.float32), "latent": np.zeros(4, np.float32)},
               "next_observations": {"goal": np.zeros(2, np.float32), "latent": np.zeros(4, np.float32)},
               "actions": np.zeros(2, np.float32), "rewards": np.float32(-1.0),
               "masks": np.float32(1.0), "dones": np.float32(0.0)}
        her.record(txn, {"pos": pos}, pos, hit=False, lane=False,
                   off_road=(t == offroad_at), speed_kmh=20.0, target_speed_kmh=20.0)
    her.end_leg()
    n_onpolicy = len(buf.rows)
    her.flush_episode()
    return her, buf.rows[n_onpolicy:]


def test_without_the_flag_no_hindsight_row_can_pay(counting, monkeypatch):
    her, rows = run(monkeypatch, pay=False)
    assert len(rows) > 100
    assert her.n_paying == 0                      # the v42 bug, by construction


def test_every_drawn_goal_gets_exactly_one_paying_arrival_row(counting, monkeypatch):
    her, rows = run(monkeypatch, pay=True)
    goals = {tp for _i, tp in her.last_pairs}
    arrivals = [(p, r) for p, r in zip(her.last_pairs, rows) if p[0] == p[1] - 1]
    assert her.n_paying == len(goals) > 0
    assert {p[1] for p, _r in arrivals} == goals
    for _p, r in arrivals:                         # arrival: bonus, terminal
        assert float(r["rewards"]) == BONUS
        assert float(r["masks"]) == 0.0 and float(r["dones"]) == 1.0


def test_the_original_rows_are_unchanged(counting, monkeypatch):
    her0, rows0 = run(monkeypatch, pay=False)
    her1, rows1 = run(monkeypatch, pay=True)
    k = len(rows0)
    assert her1.last_pairs[:k] == her0.last_pairs        # same draws, same order
    for a, b in zip(rows0, rows1[:k]):
        assert float(a["rewards"]) == float(b["rewards"])
        assert float(a["masks"]) == float(b["masks"])


def test_arrival_rows_stay_inside_a_leg_cut_by_a_fault(counting, monkeypatch):
    # off-road at t=150 trips the 1-step cap: the relabelled episode ends there,
    # so no hindsight row -- arrival rows included -- may sit at or past it.
    her, _rows = run(monkeypatch, pay=True, offroad_at=150)
    assert her.last_pairs and max(i for i, _tp in her.last_pairs) < 150
    assert max(tp for _i, tp in her.last_pairs) <= 150
