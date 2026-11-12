"""EXPLORE_VALUE_FLOOR / EXPLORE_VALUE_CEILING: bound the explore critic's target.

THE FAILURE THIS CLOSES, measured on v49. The explore critic was deliberately left
unbounded. Its Q estimate went 20 -> 1,393 -> 11,801 between episodes 1,580 and 3,080,
against a realisable maximum of about 110 (100 m of progress at 1 per metre plus a
10-point bonus), and the explore arm's driving collapsed from 52 m per episode to 8 m
in the same window. Over the SAME episodes the goal critic -- same network, same
buffer, but clipped -- stayed at 38-65. The clip was the only difference.

These run the real jitted update.
"""
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

ACT = spaces.Box(np.array([-1.0, 0.0], np.float32), np.array([1.0, 1.0], np.float32))


def _learner(**kw):
    from vae_nav.gated_tqc_learner import GatedTQCLearner
    space = spaces.Dict({"goal": spaces.Box(-np.inf, np.inf, (4,), np.float32),
                         "latent": spaces.Box(-np.inf, np.inf, (16,), np.float32),
                         "mode": spaces.Box(0.0, 1.0, (1,), np.float32)})
    return GatedTQCLearner(seed=0, observations=space.sample(), actions=ACT.sample(),
                           discount=0.99, discount_e=0.99, hidden_dims=(32, 32),
                           num_qs=2, num_quantiles=5, top_quantiles_to_drop_per_net=1,
                           target_entropy=-2.0, action_space=ACT,
                           value_floor=-500.0, value_ceiling=50.0, **kw)


def _batch(seed, reward, n=32):
    rng = np.random.default_rng(seed)

    def obs():
        o = {"goal": rng.normal(size=(n, 4)).astype(np.float32),
             "latent": rng.normal(size=(n, 16)).astype(np.float32),
             "mode": np.ones((n, 1), np.float32)}
        o["mode"][n // 2:] = 0.0                    # half goal, half explore
        return o
    return freeze({"observations": obs(), "next_observations": obs(),
                   "actions": rng.uniform(0, 1, (n, 2)).astype(np.float32),
                   "rewards": np.full((n,), reward, np.float32),
                   "masks": np.ones((n,), np.float32),
                   "dones": np.zeros((n,), np.float32)})


def _q_explore(a, b):
    obs = {k: np.asarray(v) for k, v in b["observations"].items()}
    qs = np.asarray(a._critic_e.apply_fn({"params": a._critic_e.params}, obs,
                                         np.asarray(b["actions"])))
    return float(qs[:, len(b["rewards"]) // 2:].mean())   # explore rows only


def _flat(tree):
    return np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(tree)])


def test_unbounded_by_default_so_every_existing_run_is_unchanged():
    a = _learner()
    assert a.value_floor_e is None and a.value_ceiling_e is None


def test_a_ceiling_holds_the_explore_q_down_under_a_large_reward():
    """A reward far above what the bound allows: the bounded critic's estimate
    must stay near the ceiling while the unbounded one climbs past it.

    THE FIRST VERSION OF THIS TEST PROVED NOTHING. At the default learning rate,
    60 updates left both critics near 2 -- below the ceiling of 5, so it never
    bound. And the quantile loss is Huber, whose gradient saturates for large
    errors: a target of 5 and a target of 52 both read as "far above" to a critic
    sitting at 2, so the two move identically until one of them reaches 5. The
    unbounded critic has to be driven PAST the ceiling for a difference to exist."""
    kw = dict(critic_lr=1e-2)
    free = _learner(**kw)
    held = _learner(value_floor_e=-100.0, value_ceiling_e=5.0, **kw)
    for i in range(300):
        b = _batch(i, reward=50.0)
        free.update(b)
        held.update(b)
    probe = _batch(999, reward=50.0)
    q_free, q_held = _q_explore(free, probe), _q_explore(held, probe)
    assert q_free > 15.0, f"the unbounded critic never climbed ({q_free:.1f})"
    assert q_held < 8.0, f"the bound did not hold ({q_held:.1f})"


def test_the_explore_bound_never_touches_the_goal_critic():
    """One bound per arm. The goal critic must be bit-identical whatever the
    explore bound is."""
    a, b = _learner(), _learner(value_floor_e=-100.0, value_ceiling_e=5.0)
    for i in range(5):
        batch = _batch(i, reward=50.0)
        a.update(batch)
        b.update(batch)
    np.testing.assert_array_equal(_flat(a._critic.params), _flat(b._critic.params))


def test_the_banner_reports_the_bound_actually_in_effect(capsys):
    """It said UNCLIPPED, hard-coded, after the bound existed."""
    _learner(value_floor_e=-200.0, value_ceiling_e=150.0)
    out = capsys.readouterr().out
    assert "explore critic clip [-200.0, 150.0]" in out
    assert "explore critic UNCLIPPED" not in out
