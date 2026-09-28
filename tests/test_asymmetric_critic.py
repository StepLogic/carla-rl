"""Asymmetric goal critic (OBS_PRIV=1, user 2026-09-27).

The design: a PRIVILEGED goal critic sees the lane block and drives the actor
on goal rows; the IMAGE-ONLY goal critic (agent._critic) trains on the same
batches with the same reward and is the distance the map builder reads; the
actor and the explore critic never see the block. These tests pin each of
those properties, plus the lane geometry and the checkpoint round trip.
"""
import math
import os
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from vae_nav.lane_priv import PRIV_DIM, lane_priv, strip_priv  # noqa: E402


# ---- lane geometry ---------------------------------------------------------
def _wp(x=0.0, y=0.0, yaw_deg=0.0, width=3.5, left="Broken", right="Solid",
        junction=False):
    return NS(transform=NS(location=NS(x=x, y=y), rotation=NS(yaw=yaw_deg)),
              lane_width=width, is_junction=junction,
              left_lane_marking=NS(type=f"LaneMarkingType.{left}"),
              right_lane_marking=NS(type=f"LaneMarkingType.{right}"))


def test_lateral_offset_is_left_positive_in_carlas_left_handed_frame():
    # yaw 0: forward = +x, right = +y (CARLA is left-handed), so y = -1 is 1 m LEFT
    p = lane_priv((0.0, -1.0), 0.0, _wp())
    assert p[0] == pytest.approx(1.0)
    assert p[1] == pytest.approx(1.0 / 1.75)
    assert p[5] == pytest.approx(0.75)        # to the left edge
    assert p[6] == pytest.approx(2.75)        # to the right edge
    assert p[12] == 1.0


def test_heading_error_and_rotated_lane():
    # lane pointing +y (yaw 90): right = (-1, 0), so x = +1 is 1 m LEFT
    p = lane_priv((1.0, 0.0), math.radians(100.0), _wp(yaw_deg=90.0))
    assert p[0] == pytest.approx(1.0, abs=1e-6)
    assert math.atan2(p[2], p[3]) == pytest.approx(math.radians(10.0), abs=1e-6)


def test_markings_junction_and_missing_waypoint():
    p = lane_priv((0.0, 0.0), 0.0, _wp(left="NONE", right="SolidSolid",
                                        junction=True))
    assert (p[7], p[8], p[9], p[10], p[11]) == (0.0, 1.0, 1.0, 0.0, 1.0)
    assert not lane_priv((0.0, 0.0), 0.0, None).any()
    assert lane_priv((0.0, 0.0), 0.0, _wp()).shape == (PRIV_DIM,)


def test_strip_priv_obs_and_batch():
    o = {"latent": 1, "priv": 2}
    assert strip_priv(o) == {"latent": 1}
    b = {"observations": dict(o), "next_observations": dict(o), "rewards": 0}
    s = strip_priv(b)
    assert "priv" not in s["observations"] and "priv" not in s["next_observations"]
    assert "priv" in b["observations"]        # the input is not mutated
    assert strip_priv({"latent": 1}) == {"latent": 1}


# ---- learner -------------------------------------------------------------
jax = pytest.importorskip("jax")
pytest.importorskip("jaxrl2")
spaces = pytest.importorskip("gymnasium").spaces
from flax.core.frozen_dict import freeze, unfreeze  # noqa: E402

from vae_nav.gated_tqc_learner import GatedTQCLearner  # noqa: E402
from vae_nav.policy_io import agent_state, load_agent_state  # noqa: E402

BASE = {"goal": spaces.Box(-np.inf, np.inf, (4,), np.float32),
        "latent": spaces.Box(-np.inf, np.inf, (16,), np.float32),
        "mode": spaces.Box(0.0, 1.0, (1,), np.float32)}
OBS_P = spaces.Dict({**BASE, "priv": spaces.Box(-np.inf, np.inf, (PRIV_DIM,),
                                                np.float32)})
OBS = spaces.Dict(BASE)
ACT = spaces.Box(np.array([-0.5, 0.0], np.float32),
                 np.array([0.5, 0.4], np.float32))


def _agent(space, seed=0):
    return GatedTQCLearner(seed=seed, observations=space.sample(),
                           actions=ACT.sample(), discount=0.99, discount_e=0.99,
                           hidden_dims=(32, 32), num_qs=2, num_quantiles=5,
                           top_quantiles_to_drop_per_net=1, target_entropy=-2.0,
                           action_space=ACT, value_floor=-100.0,
                           value_ceiling=10.0)


def _obs(rng, n, mode, priv):
    o = {"goal": rng.normal(size=(n, 4)).astype(np.float32),
         "latent": rng.normal(size=(n, 16)).astype(np.float32),
         "mode": np.full((n, 1), float(mode), np.float32)}
    if priv:
        o["priv"] = rng.normal(size=(n, PRIV_DIM)).astype(np.float32)
    return o


def _batch(seed, priv, n=16):
    rng = np.random.default_rng(seed)
    obs = _obs(rng, n, 1, priv)
    nxt = _obs(rng, n, 1, priv)
    for o in (obs, nxt):
        o["mode"][n // 2:] = 0.0              # half goal, half explore rows
    return freeze({"observations": obs, "next_observations": nxt,
                   "actions": rng.uniform(-0.4, 0.4, (n, 2)).astype(np.float32),
                   "rewards": rng.normal(size=(n,)).astype(np.float32),
                   "masks": np.ones((n,), np.float32),
                   "dones": np.zeros((n,), np.float32)})


def _has_priv_layer(params):
    return any("priv" in "/".join(str(getattr(k, "key", k)) for k in path)
               for path, _ in jax.tree_util.tree_flatten_with_path(params)[0])


def test_only_the_privileged_critic_has_a_priv_input_layer():
    a = _agent(OBS_P)
    assert a.asymmetric and a._critic_p is not None
    assert _has_priv_layer(a._critic_p.params)
    assert not _has_priv_layer(a._critic.params)       # image-only: map distance
    assert not _has_priv_layer(a._critic_e.params)
    assert not _has_priv_layer(a._actor.params)


def test_actor_ignores_priv_values():
    a = _agent(OBS_P)
    rng = np.random.default_rng(1)
    o = {k: v[0] for k, v in _obs(rng, 1, 1, True).items()}
    o2 = dict(o, priv=o["priv"] + 100.0)
    assert np.array_equal(np.asarray(a.eval_actions(o)),
                          np.asarray(a.eval_actions(o2)))


def test_update_trains_all_three_and_the_actor_stays_priv_free():
    a = _agent(OBS_P)
    p0 = jax.tree_util.tree_leaves(a._critic_p.params)[0].copy()
    c0 = jax.tree_util.tree_leaves(a._critic.params)[0].copy()
    for i in range(3):
        info = a.update(_batch(i, priv=True))
    for k in ("q_p", "critic_loss_p", "td_errors_p", "q", "q_e"):
        assert np.isfinite(float(info[k])), k
    assert not np.array_equal(p0, jax.tree_util.tree_leaves(a._critic_p.params)[0])
    assert not np.array_equal(c0, jax.tree_util.tree_leaves(a._critic.params)[0])
    assert not _has_priv_layer(a._actor.params)


def test_privileged_critic_output_depends_on_priv():
    a = _agent(OBS_P)
    b = _batch(0, priv=True)
    obs = unfreeze(b["observations"])
    q1 = a._critic_p.apply_fn({"params": a._critic_p.params}, obs, b["actions"])
    obs["priv"] = obs["priv"] + 5.0
    q2 = a._critic_p.apply_fn({"params": a._critic_p.params}, obs, b["actions"])
    assert not np.allclose(np.asarray(q1), np.asarray(q2))


def test_without_priv_nothing_changes():
    a = _agent(OBS)
    assert not a.asymmetric and a._critic_p is None
    info = a.update(_batch(0, priv=False))
    assert "q_p" not in info and np.isfinite(float(info["q"]))
    assert "critic_p" not in agent_state(a)


def test_checkpoint_round_trip_and_refusal():
    a = _agent(OBS_P)
    a.update(_batch(0, priv=True))
    st = agent_state(a)
    assert "critic_p" in st and "target_critic_p" in st
    b = load_agent_state(_agent(OBS_P, seed=3), st)
    for x, y in zip(jax.tree_util.tree_leaves(st["critic_p"]),
                    jax.tree_util.tree_leaves(b._critic_p.params)):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    st_plain = {k: v for k, v in st.items()
                if k not in ("critic_p", "target_critic_p")}
    with pytest.raises(SystemExit, match="privileged goal critic"):
        load_agent_state(_agent(OBS_P, seed=4), st_plain)
