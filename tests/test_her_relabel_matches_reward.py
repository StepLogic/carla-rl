"""DIFFERENTIAL: every hindsight row pays what goal_reward_sorb pays for it.

User 2026-09-26: "Ensure HER relabels correctly" (v15, goal lane terminal on).

The other HER tests pin the mirror against hand-written expected values. This one
uses the REAL reward function as the oracle instead: for every relabelled row
(i, tp) it rebuilds the info the goal wrapper would have seen had the goal been
the hindsight goal pos[tp], calls vae_nav.gated_reward.goal_reward_sorb on it,
and requires the row's reward and mask to match. A mirror that drifts from the
reward in ANY term -- fines, speed bonus, arrival, the lane-terminal bill, the
junction exemption -- fails here without anyone having to foresee which.

The constants are v15's, from results/run_env/carla_gated_her_v15_laneterm_vc1.env
plus run_gated.sh's exports: counting regime, lane terminal ON, step cost 1,
arrival 10, fines 10/10/10, caps 20/20/20, clock 1000, velocity bonus 0.75,
LANE_STRICT off.

EPISODES ARE GENERATED THE WAY THE ENV PRODUCES THEM: an on-policy episode ENDS at
its first fatal crossing (the reward terminates it), goal arrivals close legs
(chaining), and every other fault is a per-step event. A second family injects a
fatal crossing MID-leg -- impossible on-policy except on an arrival step -- to
prove the leg is cut there rather than relying on the episode happening to end.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from vae_nav import gated_reward as gr  # noqa: E402
from vae_nav.episode_her import EpisodeRelabelBuffer  # noqa: E402

STEP_COST, BONUS, FINE, TRIP, CLOCK, SPEED_BONUS = 1.0, 10.0, 10.0, 20, 1000, 0.75
RADIUS = 2.0

# Marking sets and whether goal_lane_departed calls them fatal off-junction.
MARKS = {
    "broken_road": (["carla.LaneMarkingType.Broken"], False),   # fatal
    "broken_junc": (["carla.LaneMarkingType.Broken"], True),    # forgiven
    "none_edge":   (["carla.LaneMarkingType.NONE"], False),     # forgiven
    "solid_road":  (["carla.LaneMarkingType.Solid"], False),    # fatal
}


@pytest.fixture
def v15(monkeypatch):
    from vae_nav import lane_explore_reward as ler
    monkeypatch.setattr(ler, "LANE_ANY_FATAL", False)
    for k, v in dict(GOAL_TRUNCATE_COUNTS=True, LANE_TERMINAL=True,
                     STEP_COST=STEP_COST, GOAL_ARRIVAL_BONUS=BONUS,
                     GOAL_OFF_ROAD_PENALTY=FINE, GOAL_LANE_PENALTY=FINE,
                     GOAL_COLLISION_PENALTY=FINE,
                     GOAL_OFFROAD_TRUNCATE_STEPS=TRIP,
                     GOAL_COLLISION_TRUNCATE_STEPS=TRIP,
                     GOAL_CROSSING_TRUNCATE_STEPS=TRIP,
                     GOAL_MAX_STEPS=CLOCK, GOAL_SPEED_BONUS=SPEED_BONUS).items():
        monkeypatch.setattr(gr, k, v)


class FakeBuffer:
    def __init__(self):
        self.rows = []

    def insert(self, row, mode=None, segment=None):
        self.rows.append(row)

    def __len__(self):
        return len(self.rows)


def goal_block(fields):
    return np.asarray(fields["goal_pos"], np.float32) - np.asarray(
        fields["pos"], np.float32)


def gen_episode(rng, mid_fatal=False):
    """Per-step infos, positions and leg ends, shaped like the live env's."""
    n = int(rng.integers(25, 90))
    infos, legs = [], []
    for t in range(n):
        info = {"speed_kmh": float(rng.uniform(5, 30)),
                "target_speed_kmh": 20.0}
        u = rng.random()
        if u < 0.10:
            kind = rng.choice(["broken_junc", "none_edge"])   # forgiven
            marks, junc = MARKS[kind]
            info.update(lane_any=True, lane_marking_types=marks, in_junction=junc)
        if rng.random() < 0.06:
            info["off_road"] = True
        if rng.random() < 0.03:
            info["collision"] = True
        infos.append(info)
        if rng.random() < 0.05 and t < n - 1:
            legs.append(t)              # a goal arrival closes a leg here
    # ON-POLICY: the episode ends at its first fatal crossing, so the fatal
    # step is the LAST step.
    kind = rng.choice(["broken_road", "solid_road"])
    marks, junc = MARKS[kind]
    infos[-1].update(lane_any=True, lane_marking_types=marks, in_junction=junc)
    if mid_fatal:
        k = int(rng.integers(3, n - 3))
        infos[k].update(lane_any=True, lane_marking_types=MARKS["solid_road"][0],
                        in_junction=False)
    pos = [(float(t) * 0.9, 0.0) for t in range(n)]
    return infos, pos, legs


def record_like_the_trainer(her, infos, pos, legs):
    """The trainer's record call: the SAME flags, lane_fatal from the SAME
    helper the reward uses (train_carla_tqc _rec_kwargs)."""
    for t, (info, p) in enumerate(zip(infos, pos)):
        txn = {"observations": {"goal": np.zeros(2, np.float32),
                                "latent": np.full(4, float(t), np.float32)},
               "next_observations": {"goal": np.zeros(2, np.float32),
                                     "latent": np.full(4, t + 1.0, np.float32)},
               "actions": np.zeros(2, np.float32),
               "rewards": np.float32(0.0), "masks": np.float32(1.0),
               "dones": np.float32(0.0)}
        her.record(txn, {"pos": p}, p,
                   hit=bool(info.get("collision", False)),
                   lane=bool(info.get("lane_any", False)),
                   off_road=bool(info.get("off_road", False)),
                   speed_kmh=info["speed_kmh"],
                   target_speed_kmh=info["target_speed_kmh"],
                   lane_fatal=gr.goal_lane_departed(info))
        if t in legs:
            her.end_leg()
    her.end_leg()


def oracle(infos, pos, leg_start, i, tp):
    """What goal_reward_sorb pays for step i under hindsight goal pos[tp]."""
    info = dict(infos[i])
    info["is_success"] = bool(np.linalg.norm(
        np.asarray(pos[min(i + 1, len(pos) - 1)]) - np.asarray(pos[tp])) < RADIUS)
    seg = infos[leg_start:i + 1]
    return gr.goal_reward_sorb(
        info, i - leg_start, max_steps=CLOCK, step_cost=STEP_COST,
        offroad_steps=sum(bool(s.get("off_road")) for s in seg),
        collision_steps=sum(bool(s.get("collision")) for s in seg),
        crossing_steps=sum(bool(s.get("lane_any")) for s in seg))


def leg_start_of(i, legs):
    s = 0
    for e in sorted(legs):
        if e < i:
            s = e + 1
    return s


def run(seed, mid_fatal):
    rng = np.random.default_rng(seed)
    infos, pos, legs = gen_episode(rng, mid_fatal=mid_fatal)
    buf = FakeBuffer()
    her = EpisodeRelabelBuffer(
        buf, goal_block, goal_radius=RADIUS, step_cost=STEP_COST,
        time_to_goal=True, gated=True, recost_counting=True,
        k_goal=600.0, k_collision=5.0, k_lane=5.0, k_living=0.5, k_dist=1.0,
        seed=seed)
    record_like_the_trainer(her, infos, pos, legs)
    n = len(infos)
    her.flush_episode()
    return infos, pos, legs, her, buf.rows[n:]


@pytest.mark.parametrize("mid_fatal", [False, True])
def test_every_hindsight_row_pays_what_the_reward_pays(v15, mid_fatal):
    compared = arrivals = fined = bonus = 0
    for seed in range(300):
        infos, pos, legs, her, rows = run(seed, mid_fatal)
        assert len(rows) == len(her.last_pairs)
        for (i, tp), row in zip(her.last_pairs, rows):
            ls = leg_start_of(i, legs)
            r, term, trunc, arrived = oracle(infos, pos, ls, i, tp)
            assert float(row["rewards"]) == pytest.approx(r, abs=1e-4), (
                f"seed {seed} row {i}->{tp}: HER paid {float(row['rewards'])}, "
                f"goal_reward_sorb pays {r}")
            assert float(row["masks"]) == (0.0 if term else 1.0), (seed, i, tp)
            # No row may sit where the real episode had already ended.
            assert not trunc, f"seed {seed}: row {i} lies past a tripped cap"
            assert not (term and not arrived), (
                f"seed {seed}: row {i} sits ON a fatal crossing -- the relabelled "
                f"episode had already ended there")
            compared += 1
            arrivals += int(arrived)
            fined += int(r < -STEP_COST - 1e-6 and not arrived)
            bonus += int(r > -STEP_COST + 1e-6 and not arrived)
    # Coverage: a differential test that compared nothing interesting proves
    # nothing. Every reward term must have been exercised.
    assert compared > 3000, compared
    assert arrivals > 50, arrivals
    assert fined > 200, fined
    assert bonus > 200, bonus


def test_no_hindsight_goal_crosses_a_mid_leg_fatal_step(v15):
    """A fatal crossing inside a leg ends the relabelled episode there: no row
    at or after it, and no goal drawn from beyond it."""
    checked = 0
    for seed in range(300):
        infos, pos, legs, her, rows = run(seed, mid_fatal=True)
        fatal = [t for t, s in enumerate(infos) if gr.goal_lane_departed(s)]
        for (i, tp) in her.last_pairs:
            ls = leg_start_of(i, legs)
            first = next((f for f in fatal if f >= ls), None)
            if first is not None:
                assert i < first, (seed, i, first)
                assert tp <= first, (seed, tp, first)
                checked += 1
    assert checked > 1000, checked


def test_the_trainer_and_the_reward_share_one_verdict(v15):
    """The flag the trainer stages IS the reward's termination test: on every
    generated step, lane_fatal == 'goal_reward_sorb terminates without arrival'."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        infos, _, _ = gen_episode(rng, mid_fatal=True)
        for t, info in enumerate(infos):
            _, term, _, arrived = gr.goal_reward_sorb(
                dict(info), t, max_steps=CLOCK, step_cost=STEP_COST)
            assert gr.goal_lane_departed(info) == (term and not arrived), (t, info)
