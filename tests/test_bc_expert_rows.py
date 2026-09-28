"""Behaviour cloning on expert rows (BC_WEIGHT, user 2026-09-28 "i prefer 2")."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
jax = pytest.importorskip("jax")
pytest.importorskip("jaxrl2")
spaces = pytest.importorskip("gymnasium").spaces
from flax.core.frozen_dict import freeze  # noqa: E402

from vae_nav.gated_tqc_learner import GatedTQCLearner  # noqa: E402
from vae_nav.lane_priv import PRIV_DIM  # noqa: E402

ACT = spaces.Box(np.array([-0.5, 0.0], np.float32), np.array([0.5, 0.4], np.float32))
TARGET = np.array([0.3, 0.35], np.float32)


def _space(priv):
    d = {"goal": spaces.Box(-np.inf, np.inf, (4,), np.float32),
         "latent": spaces.Box(-np.inf, np.inf, (16,), np.float32),
         "mode": spaces.Box(0.0, 1.0, (1,), np.float32)}
    if priv:
        d["priv"] = spaces.Box(-np.inf, np.inf, (PRIV_DIM,), np.float32)
    return spaces.Dict(d)


def _agent(priv=False, w=0.0):
    return GatedTQCLearner(seed=0, observations=_space(priv).sample(),
                           actions=ACT.sample(), discount=0.99, discount_e=0.99,
                           hidden_dims=(32, 32), num_qs=2, num_quantiles=5,
                           top_quantiles_to_drop_per_net=1, target_entropy=-2.0,
                           action_space=ACT, value_floor=-100.0, value_ceiling=10.0,
                           bc_weight=w)


def _batch(seed, priv, expert, n=32):
    rng = np.random.default_rng(seed)

    def obs():
        o = {"goal": rng.normal(size=(n, 4)).astype(np.float32),
             "latent": rng.normal(size=(n, 16)).astype(np.float32),
             "mode": np.ones((n, 1), np.float32)}
        o["mode"][n // 2:] = 0.0
        if priv:
            o["priv"] = rng.normal(size=(n, PRIV_DIM)).astype(np.float32)
        if expert is not None:
            o["expert"] = np.full((n, 1), float(expert), np.float32)
        return o
    return freeze({"observations": obs(), "next_observations": obs(),
                   "actions": np.tile(TARGET, (n, 1)),
                   "rewards": np.zeros((n,), np.float32),
                   "masks": np.ones((n,), np.float32),
                   "dones": np.zeros((n,), np.float32)})


def _no_expert_layer(params):
    return not any("expert" in "/".join(str(getattr(k, "key", k)) for k in p)
                   for p, _ in jax.tree_util.tree_flatten_with_path(params)[0])


@pytest.mark.parametrize("priv", [False, True])
def test_bc_pulls_actor_to_expert_actions(priv):
    a = _agent(priv, w=50.0)
    probe = {k: v[0] for k, v in _batch(99, priv, None, n=4)["observations"].items()}
    before = np.abs(np.asarray(a.eval_actions(probe)) - TARGET).sum()
    losses = []
    for i in range(60):
        info = a.update(_batch(i, priv, expert=1.0))
        losses.append(float(info["bc_loss"]))
    after = np.abs(np.asarray(a.eval_actions(probe)) - TARGET).sum()
    assert after < 0.5 * before and losses[-1] < losses[0]
    assert float(info["bc_frac"]) == 1.0
    for net in (a._actor, a._critic, a._critic_e) + ((a._critic_p,) if priv else ()):
        assert _no_expert_layer(net.params)


def test_non_expert_rows_carry_no_bc():
    a = _agent(w=50.0)
    info = a.update(_batch(0, False, expert=0.0))
    assert float(info["bc_frac"]) == 0.0 and float(info["bc_loss"]) == 0.0


def test_bc_off_without_the_key():
    info = _agent().update(_batch(0, False, expert=None))
    assert "bc_loss" not in info


def test_buffer_with_expert_key_feeds_the_learner():
    """The trainer's path: buffer space = env space + 'expert'; learner built
    from the env space alone; sampled batches go straight into update()."""
    from vae_nav.mode_split_buffer import ModeSplitBuffer
    env_space = _space(False)
    buf_space = spaces.Dict({**env_space.spaces,
                             "expert": spaces.Box(0.0, 1.0, (1,), np.float32)})
    buf = ModeSplitBuffer(buf_space, ACT, 400)
    rng = np.random.default_rng(0)
    for i in range(120):
        m = i % 2
        o = {k: rng.normal(size=v.shape).astype(np.float32)
             for k, v in env_space.spaces.items()}
        o["mode"] = np.array([float(m)], np.float32)
        o["expert"] = np.array([float(i % 4 == 0)], np.float32)
        buf.insert(dict(observations=o, next_observations=o, actions=TARGET.copy(),
                        rewards=np.float32(-1), masks=np.float32(1),
                        dones=np.float32(0)), mode=m)
    a = _agent(w=1.0)
    info = a.update(buf.sample(32))
    assert 0.0 < float(info["bc_frac"]) < 1.0 and np.isfinite(float(info["bc_loss"]))
