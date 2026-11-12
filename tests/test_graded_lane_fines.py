"""GOAL_LANE_GRADED (v18): the goal arm's lane fine follows the marking crossed.

User 2026-09-27, after "any suggestions to make agent transfer to other towns and
respect driving rules?": "proceed". The old rule fined RAW lane_any at
GOAL_LANE_PENALTY per step -- a legal junction turn, a legal lane change and a
solid-line crossing all cost the same, while JUNCTION_GOAL_FRAC=0.5 made half the
goals require a junction.

Pinned here:
  - the grading itself, mark by mark, and that ungraded is the old rule exactly;
  - the 20-step counter and the velocity bonus's clean gate use the same verdict;
  - DIFFERENTIAL: every hindsight row equals the goal WRAPPER's full reward
    (goal_reward_sorb + POTENTIAL_K progress, v18's recipe) under graded fines,
    with the relabeller fed exactly what the trainer stages;
  - a mutant that forgets to stage the graded verdicts is caught.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from vae_nav import gated_reward as gr  # noqa: E402
from vae_nav.episode_her import EpisodeRelabelBuffer  # noqa: E402

M = "carla.LaneMarkingType."
BROKEN, SOLID, DOUBLE, NONE = [M + "Broken"], [M + "Solid"], [M + "SolidSolid"], [M + "NONE"]
FINE, BROKEN_FINE, TRIP, RADIUS = 10.0, 2.0, 20, 2.0


@pytest.fixture
def v18(monkeypatch):
    for k, v in dict(GOAL_TRUNCATE_COUNTS=True, LANE_TERMINAL=False,
                     LANE_GRADED=True, GOAL_LANE_PENALTY_BROKEN=BROKEN_FINE,
                     STEP_COST=1.0, GOAL_ARRIVAL_BONUS=10.0,
                     GOAL_OFF_ROAD_PENALTY=FINE, GOAL_LANE_PENALTY=FINE,
                     GOAL_COLLISION_PENALTY=FINE,
                     GOAL_OFFROAD_TRUNCATE_STEPS=TRIP,
                     GOAL_COLLISION_TRUNCATE_STEPS=TRIP,
                     GOAL_CROSSING_TRUNCATE_STEPS=TRIP, GOAL_MAX_STEPS=1000,
                     GOAL_SPEED_BONUS=0.75, GOAL_REWARD_KIND="sorb",
                     POTENTIAL_K=1.0, PROGRESS_MAX_STEP_M=5.0).items():
        monkeypatch.setattr(gr, k, v)
    return monkeypatch


def cross(marks, junction=False, **kw):
    return dict(lane_any=True, lane_marking_types=marks, in_junction=junction, **kw)


# ---- the grading ------------------------------------------------------------

@pytest.mark.parametrize("info,fine", [
    (cross(BROKEN, junction=True), 0.0),       # the turn
    (cross(NONE), 0.0),                        # road edge: off_road fines it
    (cross(BROKEN), BROKEN_FINE),              # a lane change
    (cross(SOLID), FINE),                      # illegal
    (cross(DOUBLE), FINE),
    (cross(SOLID, junction=True), FINE),       # illegal even in a junction
    (cross([]), FINE),                         # no data: fails closed
    ({"lane_any": False}, 0.0),
])
def test_fine_by_marking(v18, info, fine):
    assert gr.goal_lane_fine(info) == pytest.approx(fine)
    assert gr.goal_lane_fault(info) == (fine > 0)


def test_ungraded_is_the_old_rule_exactly(v18):
    v18.setattr(gr, "LANE_GRADED", False)
    for info in (cross(BROKEN, junction=True), cross(NONE), cross(BROKEN),
                 cross(SOLID), cross([])):
        assert gr.goal_lane_fine(info) == FINE
        assert gr.goal_lane_fault(info) is True
    assert gr.goal_lane_fine({"lane_any": False}) == 0.0


def test_zero_penalty_pays_nothing_graded_or_not(v18):
    v18.setattr(gr, "GOAL_LANE_PENALTY", 0.0)
    assert gr.goal_lane_fine(cross(SOLID)) == 0.0
    v18.setattr(gr, "LANE_GRADED", False)
    assert gr.goal_lane_fine(cross(SOLID)) == 0.0


def test_legal_turn_keeps_the_velocity_bonus(v18):
    """A junction turn at target speed is a CLEAN step: no fine, full bonus."""
    base = {"speed_kmh": 20.0, "target_speed_kmh": 20.0}
    r_turn, *_ = gr.goal_reward_sorb(dict(base, **cross(BROKEN, junction=True)),
                                     5, max_steps=1000, step_cost=1.0)
    r_plain, *_ = gr.goal_reward_sorb(dict(base), 5, max_steps=1000, step_cost=1.0)
    assert r_turn == pytest.approx(r_plain) == pytest.approx(-1.0 + 0.75)
    r_solid, *_ = gr.goal_reward_sorb(dict(base, **cross(SOLID)), 5,
                                      max_steps=1000, step_cost=1.0)
    assert r_solid == pytest.approx(-1.0 - FINE)          # fined, no bonus


# ---- the differential against the FULL wrapper reward ------------------------

class Buf:
    def __init__(self):
        self.rows = []

    def insert(self, row, mode=None, segment=None):
        self.rows.append(row)

    def __len__(self):
        return len(self.rows)


class Road:
    def reset(self, o, g, origin_z=0.0, goal_z=0.0):
        self.g = np.asarray(g, float)
        return abs(self.g[0] - o[0])

    def remaining(self, xy):
        return abs(self.g[0] - np.asarray(xy)[0])


def gb(f):
    return np.asarray(f["goal_pos"], np.float32) - np.asarray(f["pos"], np.float32)


KINDS = [cross(BROKEN, junction=True), cross(NONE), cross(BROKEN), cross(SOLID)]


def gen(rng):
    n = int(rng.integers(30, 90))
    infos, legs = [], []
    for t in range(n):
        info = {"speed_kmh": float(rng.uniform(5, 30)), "target_speed_kmh": 20.0}
        if rng.random() < 0.25:
            info.update(KINDS[int(rng.integers(len(KINDS)))])
        if rng.random() < 0.05:
            info["off_road"] = True
        if rng.random() < 0.03:
            info["collision"] = True
        infos.append(info)
        if rng.random() < 0.05 and t < n - 1:
            legs.append(t)
    x = np.cumsum(rng.uniform(0.2, 1.4, n))
    return infos, [(float(v), 0.0) for v in x], legs


def run(seed, stage=True):
    rng = np.random.default_rng(seed)
    infos, pos, legs = gen(rng)
    buf = Buf()
    her = EpisodeRelabelBuffer(buf, gb, goal_radius=RADIUS, step_cost=1.0,
                               time_to_goal=True, gated=True, recost_counting=True,
                               k_goal=600.0, k_collision=5.0, k_lane=5.0,
                               k_living=0.5, k_dist=1.0, seed=seed,
                               route_provider=lambda: Road())
    for t, (info, p) in enumerate(zip(infos, pos)):
        txn = {"observations": {"goal": np.zeros(2, np.float32),
                                "latent": np.full(4, float(t), np.float32)},
               "next_observations": {"goal": np.zeros(2, np.float32),
                                     "latent": np.full(4, t + 1.0, np.float32)},
               "actions": np.zeros(2, np.float32), "rewards": np.float32(0.0),
               "masks": np.float32(1.0), "dones": np.float32(0.0)}
        kw = dict(lane_fine=gr.goal_lane_fine(info),
                  lane_fault=gr.goal_lane_fault(info)) if stage else {}
        her.record(txn, {"pos": p}, p, hit=bool(info.get("collision", False)),
                   lane=bool(info.get("lane_any", False)),
                   off_road=bool(info.get("off_road", False)),
                   speed_kmh=info["speed_kmh"],
                   target_speed_kmh=info["target_speed_kmh"],
                   lane_fatal=gr.goal_lane_departed(info), z=0.0, **kw)
        if t in legs:
            her.end_leg()
    her.end_leg()
    n = len(infos)
    her.flush_episode()
    return infos, pos, legs, her, buf.rows[n:]


def leg_start(i, legs):
    s = 0
    for e in sorted(legs):
        if e < i:
            s = e + 1
    return s


def wrapper_reward(infos, pos, ls, i, tp):
    """GatedReward.step's goal reward: goal_reward_sorb + POTENTIAL_K progress,
    with the counters accumulated by goal_lane_fault exactly as the wrapper does."""
    g = pos[tp]
    j = min(i + 1, len(pos) - 1)
    info = dict(infos[i])
    info["is_success"] = bool(np.linalg.norm(np.asarray(pos[j]) - np.asarray(g)) < RADIUS)
    seg = infos[ls:i + 1]
    r, term, trunc, arrived = gr.goal_reward_sorb(
        info, i - ls, max_steps=1000, step_cost=1.0,
        offroad_steps=sum(bool(s.get("off_road")) for s in seg),
        collision_steps=sum(bool(s.get("collision")) for s in seg),
        crossing_steps=sum(gr.goal_lane_fault(s) for s in seg))
    if gr.POTENTIAL_K and not (term or trunc):
        r += gr.POTENTIAL_K * (abs(g[0] - pos[i][0]) - abs(g[0] - pos[j][0]))
    return r, term, trunc, arrived


def audit(stage=True, seeds=200):
    n = bad = fined_broken = free_turns = 0
    for seed in range(seeds):
        infos, pos, legs, her, rows = run(seed, stage)
        for (i, tp), row in zip(her.last_pairs, rows):
            r, term, trunc, arrived = wrapper_reward(infos, pos, leg_start(i, legs), i, tp)
            n += 1
            if (abs(float(row["rewards"]) - r) > 1e-4
                    or float(row["masks"]) != (0.0 if term else 1.0) or trunc):
                bad += 1
            f = gr.goal_lane_fine(infos[i])
            fined_broken += int(f == BROKEN_FINE)
            free_turns += int(bool(infos[i].get("lane_any")) and f == 0.0)
    return n, bad, fined_broken, free_turns


def test_every_hindsight_row_matches_the_wrapper_under_graded_fines(v18):
    n, bad, fb, ft = audit()
    assert bad == 0, f"{bad}/{n} hindsight rows disagree with the wrapper"
    assert n > 3000 and fb > 100 and ft > 100, (n, fb, ft)


def test_forgetting_to_stage_the_verdicts_is_caught(v18):
    """Without lane_fine/lane_fault staged, the relabeller falls back to the raw
    rule -- which under grading is a different reward. The differential sees it."""
    n, bad, *_ = audit(stage=False, seeds=80)
    assert bad > 0.05 * n, (bad, n)


# ---- multi-town (v18 rotates Town05 <-> Town01) --------------------------------

def test_town_switch_between_flushes_swaps_the_planner(v18):
    """The relabeller must re-ask the provider each flush: a rotation replaces the
    env and its map between episodes, and a held planner would keep measuring on
    the previous town's roads."""
    class Tagged(Road):
        def __init__(self, tag):
            self.tag, self.plans = tag, 0

        def reset(self, o, g, origin_z=0.0, goal_z=0.0):
            self.plans += 1
            return super().reset(o, g, origin_z, goal_z)
    towns = {"Town05": Tagged("Town05"), "Town01": Tagged("Town01")}
    current = {"town": "Town05"}
    her = EpisodeRelabelBuffer(Buf(), gb, goal_radius=RADIUS, step_cost=1.0,
                               time_to_goal=True, gated=True, recost_counting=True,
                               k_goal=600.0, k_collision=5.0, k_lane=5.0,
                               k_living=0.5, k_dist=1.0, seed=0,
                               route_provider=lambda: towns[current["town"]])

    def episode():
        for t in range(40):
            p = (float(t) * 0.9, 0.0)
            txn = {"observations": {"goal": np.zeros(2, np.float32),
                                    "latent": np.full(4, float(t), np.float32)},
                   "next_observations": {"goal": np.zeros(2, np.float32),
                                         "latent": np.full(4, t + 1.0, np.float32)},
                   "actions": np.zeros(2, np.float32), "rewards": np.float32(0.0),
                   "masks": np.float32(1.0), "dones": np.float32(0.0)}
            her.record(txn, {"pos": p}, p, hit=False, lane=False, off_road=False,
                       speed_kmh=20.0, target_speed_kmh=20.0, lane_fatal=False,
                       z=0.0, lane_fine=0.0, lane_fault=False)
        her.end_leg()
        her.flush_episode()

    episode()
    before = towns["Town01"].plans
    assert towns["Town05"].plans > 0 and before == 0
    current["town"] = "Town01"                 # the rotation
    episode()
    assert towns["Town01"].plans > 0, "the second town's episode was planned on Town05"
