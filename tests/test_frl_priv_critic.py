"""The critic on the TRUE STATE ALONE (torch_rl/frl/carla: priv_vector, PrivObs, the privileged critic,
the per-decision lane cost; user 2026-11-14: "lets try priveledge critic with per steering and lane
error", "images go to only actor", "add goal and current location etc to critic input").

Offline, a critic on image features preferred steering unrelated to the expert's (full lock in 52% of
states); the same critic on the true state with a per-step lane cost preferred corrective steering
(81% sign agreement with the expert) and braked for the goal. These tests pin the live wiring: what
the critic sees, that the actor never sees it, and what the stored reward is charged.
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
os.environ.setdefault("MONO_CHW", "1")
os.environ.setdefault("SPEED_OBS", "1")
torch = pytest.importorskip("torch")

from torch_rl.frl import carla as fc  # noqa: E402

EMB, ST = 12, 2


def _info(lat=0.3, head_deg=5.0, kmh=10.0, goal=(13.0, 4.0), hero=(10.0, 4.0), yaw=0.0):
    lp = np.zeros(fc.LANE_PRIV_DIM, np.float32)
    lp[0], lp[2], lp[3], lp[12] = lat, math.sin(math.radians(head_deg)), math.cos(math.radians(head_deg)), 1.0
    return {"lane_priv": lp, "speed_kmh": kmh, "goal_pos": np.array(goal), "hero_pos": np.array(hero),
            "hero_yaw": yaw}


def test_the_true_state_holds_the_lane_the_speed_the_mode_and_the_goal_in_the_cars_frame():
    v = fc.priv_vector(_info(), mode=1)
    assert v.shape == (fc.PRIV_DIM,) and v[0] == pytest.approx(0.3)
    assert v[fc.LANE_PRIV_DIM] == pytest.approx(0.5) and v[fc.LANE_PRIV_DIM + 1] == 1.0
    assert v[fc.LANE_PRIV_DIM + 2] == pytest.approx(0.3)          # 3 m ahead / 10
    assert v[fc.LANE_PRIV_DIM + 3] == pytest.approx(0.0) and v[fc.LANE_PRIV_DIM + 4] == 1.0
    # the car turned 90 degrees: the same goal is now to its side, not ahead
    side = fc.priv_vector(_info(yaw=math.pi / 2), mode=1)
    assert abs(side[fc.LANE_PRIV_DIM + 2]) < 1e-6 and abs(side[fc.LANE_PRIV_DIM + 3]) == pytest.approx(0.3)


def test_in_explore_mode_the_goal_columns_are_zero():
    v = fc.priv_vector(_info(), mode=0)
    assert v[fc.LANE_PRIV_DIM + 1] == 0.0
    assert np.all(v[fc.LANE_PRIV_DIM + 2:fc.LANE_PRIV_DIM + 5] == 0.0)   # user: "but during exploration zero"


def test_without_the_envs_lane_geometry_it_fails_loudly():
    with pytest.raises(KeyError, match="OBS_PRIV"):
        fc.priv_vector({"speed_kmh": 0.0}, mode=0)


def test_the_lane_cost_charges_offset_and_heading_per_tick():
    v = fc.priv_vector(_info(lat=-0.3, head_deg=-5.0), mode=0)
    assert fc.lane_step_cost(v, 0.5, 2) == pytest.approx(0.5 * 2 * (0.3 + 0.5), abs=1e-4)
    assert fc.lane_step_cost(fc.priv_vector(_info(lat=0.0, head_deg=0.0), 0), 0.5, 2) == pytest.approx(0.0, abs=1e-6)


def _obs(priv, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"embeddings": torch.randn(1, EMB, generator=g), "states": torch.randn(1, ST, generator=g),
            "priv": torch.as_tensor(priv)[None]}


def test_the_replay_keeps_the_true_state_and_stores_the_reward_minus_the_lane_cost():
    rb = fc.FeatureReplay(8, EMB, ST, 2, "cpu", priv_dim=fc.PRIV_DIM, lane_cost=0.5, ticks=2)
    a, b = fc.priv_vector(_info(lat=0.0, head_deg=0.0), 0), fc.priv_vector(_info(lat=0.3, head_deg=5.0), 0)
    rb.add(_obs(a), np.zeros(2, np.float32), 1.0, 1.0, _obs(b, 1))
    out = rb.sample(4)
    assert torch.allclose(out["observations"]["priv"][0], torch.as_tensor(a))
    assert torch.allclose(out["next_observations"]["priv"][0], torch.as_tensor(b))
    assert float(out["rewards"][0]) == pytest.approx(1.0 - 0.8, abs=1e-4)   # charged at the state reached
    with pytest.raises(ValueError):
        fc.FeatureReplay(8, EMB, ST, 2, "cpu", lane_cost=0.5)               # no true state to price it from


def test_the_critic_reads_the_true_state_and_nothing_from_the_images():
    lr = fc.make_learner(EMB, ST, 2, "cpu", seed=0, priv_dim=fc.PRIV_DIM)
    o = _obs(fc.priv_vector(_info(), 1))
    act = torch.zeros(1, 2)
    q = lr.critic(o, act)
    other_images = dict(o, embeddings=o["embeddings"] + 5.0, states=o["states"] - 3.0)
    assert torch.equal(lr.critic(other_images, act), q)                      # images do not reach it
    other_state = dict(o, priv=o["priv"] + 1.0)
    assert not torch.equal(lr.critic(other_state, act), q)
    assert lr.dims["priv_dim"] == fc.PRIV_DIM
    # ... and the ACTOR reads images only: the true state does not move it
    assert torch.equal(lr.actor.sample(other_state)[2], lr.actor.sample(o)[2])


def test_sac_updates_run_on_true_state_batches():
    lr = fc.make_learner(EMB, ST, 2, "cpu", seed=0, priv_dim=fc.PRIV_DIM)
    rb = fc.FeatureReplay(64, EMB, ST, 2, "cpu", priv_dim=fc.PRIV_DIM, lane_cost=0.5, ticks=2)
    rng = np.random.default_rng(0)
    for i in range(40):
        p = fc.priv_vector(_info(lat=float(rng.normal(0, 0.3)), head_deg=float(rng.normal(0, 5))), i % 2)
        rb.add(_obs(p, i), rng.uniform(-1, 1, 2).astype(np.float32), 1.0, 1.0, _obs(p, i + 1))
    before = [q.clone() for q in lr.critic.parameters()]
    info = lr.update(rb.sample(64), utd_ratio=8)
    assert np.isfinite(info["critic_loss"]) and np.isfinite(info["actor_loss"])
    assert any(not torch.equal(a, b) for a, b in zip(before, lr.critic.parameters()))
    assert np.isfinite(fc.critic_only_update(lr, rb.sample(64), 8)["critic_loss"])


class _Trunk:
    emb_dim, state_dim, hidden_dim = EMB, ST, 4

    def zero_hidden(self):
        return 0

    def step(self, row, hidden):
        return {"embeddings": torch.zeros(1, EMB), "states": torch.as_tensor([[float(row[0]), 0.0]])}, hidden + 1


def test_the_wrapped_trunk_adds_the_true_state_of_the_row_it_is_stepping():
    t = fc.PrivObs(_Trunk())
    assert t.emb_dim == EMB and t.priv_dim == fc.PRIV_DIM            # everything else is the trunk's
    row_of = t.watching(lambda obs, info: np.array([obs["mode"]], np.float32))
    o, h = t.step(row_of({"mode": 1}, _info(lat=0.4)), t.zero_hidden())
    assert h == 1 and float(o["priv"][0, 0]) == pytest.approx(0.4) and float(o["priv"][0, fc.LANE_PRIV_DIM + 1]) == 1.0
    o2, _ = t.step(row_of({"mode": 0}, _info(lat=-0.2)), h)
    assert float(o2["priv"][0, 0]) == pytest.approx(-0.2) and float(o2["priv"][0, fc.LANE_PRIV_DIM + 4]) == 0.0
    with pytest.raises(RuntimeError):
        fc.PrivObs(_Trunk()).step(np.zeros(1), 0)


def test_inside_a_junction_the_lane_cost_charges_the_offset_but_not_the_heading():
    """A correct turn has a large heading error against the lane it is leaving. Charging it made the
    first run slow down and time out; the lane rule forgives junctions and so does the cost."""
    info = _info(lat=0.3, head_deg=40.0)
    on_lane = fc.lane_step_cost(fc.priv_vector(info, 0), 0.1, 2)
    info["lane_priv"][11] = 1.0                                   # the waypoint is inside a junction
    in_junction = fc.lane_step_cost(fc.priv_vector(info, 0), 0.1, 2)
    assert on_lane == pytest.approx(0.1 * 2 * (0.3 + 4.0), abs=1e-4)
    assert in_junction == pytest.approx(0.1 * 2 * 0.3, abs=1e-4)


# --- the critic sees what the reward pays (user 2026-11-15: "how can we make the actor and critic agree",
#     "ok fix it"): the explore distance toward the 200 m bonus, and the ROUTE distance to the goal ---

def test_explore_mode_carries_the_distance_toward_the_bonus_and_still_no_goal():
    info = dict(_info(), explore_frac=0.4, route_remaining_m=17.0)
    v = fc.priv_vector(info, mode=0)
    assert v[fc.LANE_PRIV_DIM + 5] == pytest.approx(0.4)          # 80 m of the 200 m
    assert np.all(v[fc.LANE_PRIV_DIM + 2:fc.LANE_PRIV_DIM + 5] == 0.0)   # goal columns stay zero
    assert v[fc.LANE_PRIV_DIM + 6] == 0.0                         # and no route distance in explore


def test_goal_mode_carries_the_route_distance_and_no_explore_distance():
    info = dict(_info(), explore_frac=0.4, route_remaining_m=17.0)
    v = fc.priv_vector(info, mode=1)
    assert v[fc.LANE_PRIV_DIM + 6] == pytest.approx(1.7)          # 17 m of road / 10
    assert v[fc.LANE_PRIV_DIM + 5] == 0.0
    # no route published (ROUTE_POTENTIAL off, or the reset step): zero, not a crash
    assert fc.priv_vector(dict(_info(), route_remaining_m=None), mode=1)[fc.LANE_PRIV_DIM + 6] == 0.0


class _FakeActor:
    def __init__(self, steer):
        self.steer = steer

    def sample(self, obs):
        n = obs["priv"].shape[0]
        a = torch.tensor([[self.steer, 0.5]]).repeat(n, 1)
        return a, torch.zeros(n), a


class _FakeLearner:
    """A critic that prefers steering = `best` in every state, and an actor that steers `steer`."""

    def __init__(self, steer, best):
        self.actor, self.best, self.device = _FakeActor(steer), best, "cpu"

    def critic(self, obs, a):
        return torch.stack([-(a[:, 0] - self.best) ** 2] * 2, 0)


def test_steer_agreement_is_one_when_the_critic_prefers_the_actors_side_and_zero_when_opposite():
    batch = {"observations": {"priv": torch.zeros(16, fc.PRIV_DIM)}}
    same = fc.steer_agreement(_FakeLearner(0.3, 0.5), batch)
    assert same["steer_agree"] == 1.0 and same["steer_gap"] == pytest.approx(0.2, abs=0.06)
    opp = fc.steer_agreement(_FakeLearner(0.3, -0.5), batch)
    assert opp["steer_agree"] == 0.0 and opp["steer_gap"] == pytest.approx(0.8, abs=0.06)


def test_steer_agreement_ignores_states_where_the_actor_drives_straight():
    batch = {"observations": {"priv": torch.zeros(16, fc.PRIV_DIM)}}
    out = fc.steer_agreement(_FakeLearner(0.0, 0.5), batch)
    assert math.isnan(out["steer_agree"]) and out["steer_gap"] == pytest.approx(0.5, abs=0.06)
