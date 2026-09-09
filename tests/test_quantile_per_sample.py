"""quantile_huber_loss_per_sample: the same loss, one reduction short.

WHY THIS LIVES IN carla-rl AND NOT IN THE VENDORED jaxrl2. The gated learner
needs a per-row quantile loss so a mode mask can drop rows rather than scale the
whole loss. The obvious place is beside quantile_huber_loss inside
dependencies/jaxrl2, but that directory is a nested repository placed on
sys.path by absolute path in easy-install.pth, so it is SHARED by every session
on this machine and cannot be isolated in a worktree. Defining the function here
keeps the vendored fork unpatched, keeps this code under carla-rl's own history,
and removes any chance of a half-written edit reaching another session's
training run. It costs nothing: the function needs only `taus` from the vendored
module, which it imports read-only.

THE LOAD-BEARING TEST is test_mean_of_per_sample_equals_the_reduced_loss. If it
drifts, the gated critics are silently optimising a different objective from
every other TQC run in this repo, and nothing else would reveal it.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("jaxrl2")

from jaxrl2.agents.tqc.quantiles import quantile_huber_loss  # noqa: E402

from vae_nav.quantile_per_sample import quantile_huber_loss_per_sample  # noqa: E402


def _fixture(batch=4, n_critics=2, n_quantiles=5, kept=3, seed=0):
    rng = np.random.default_rng(seed)
    cur = jnp.asarray(rng.normal(size=(n_critics, batch, n_quantiles)), jnp.float32)
    tgt = jnp.asarray(rng.normal(size=(batch, kept)), jnp.float32)
    return cur, tgt


def test_per_sample_returns_one_value_per_batch_row():
    cur, tgt = _fixture()
    assert quantile_huber_loss_per_sample(cur, tgt).shape == (4,)


def test_mean_of_per_sample_equals_the_reduced_loss():
    cur, tgt = _fixture()
    per_mean = float(quantile_huber_loss_per_sample(cur, tgt).mean())
    reduced = float(quantile_huber_loss(cur, tgt))
    assert np.isclose(per_mean, reduced, atol=1e-5), (per_mean, reduced)


def test_equivalence_holds_across_shapes_and_seeds():
    # One fixture agreeing could be an accident of broadcasting; several cannot.
    for seed in range(4):
        for batch, n_critics, n_quantiles, kept in (
                (1, 1, 3, 2), (8, 5, 25, 23), (3, 2, 7, 7), (16, 5, 25, 21)):
            cur, tgt = _fixture(batch, n_critics, n_quantiles, kept, seed)
            per_mean = float(quantile_huber_loss_per_sample(cur, tgt).mean())
            reduced = float(quantile_huber_loss(cur, tgt))
            assert np.isclose(per_mean, reduced, atol=1e-5), (
                seed, batch, n_critics, n_quantiles, kept, per_mean, reduced)


def test_equivalence_holds_at_the_production_shape():
    # NUM_QS=5, N_QUANTILES=25, TOP_QUANTILES_DROP=2 per net -> kept = 5*25-10.
    cur, tgt = _fixture(batch=32, n_critics=5, n_quantiles=25, kept=115, seed=11)
    assert np.isclose(float(quantile_huber_loss_per_sample(cur, tgt).mean()),
                      float(quantile_huber_loss(cur, tgt)), atol=1e-5)


def test_kappa_is_honoured():
    # kappa is DERIVED from the reward scale in this repo. If it were ignored,
    # the gated critics would silently use 1.0 while the plain path uses the
    # derived value, and the two arms would not be comparable.
    cur, tgt = _fixture()
    a = float(quantile_huber_loss_per_sample(cur, tgt, kappa=1.0).mean())
    b = float(quantile_huber_loss_per_sample(cur, tgt, kappa=5.0).mean())
    assert not np.isclose(a, b, atol=1e-6)


def test_kappa_matches_the_reduced_form_too():
    cur, tgt = _fixture()
    for kappa in (0.5, 2.0, 7.5):
        assert np.isclose(
            float(quantile_huber_loss_per_sample(cur, tgt, kappa=kappa).mean()),
            float(quantile_huber_loss(cur, tgt, kappa=kappa)), atol=1e-5), kappa


def test_masking_a_row_removes_its_contribution():
    cur, tgt = _fixture()
    per = quantile_huber_loss_per_sample(cur, tgt)
    mask = jnp.asarray([1.0, 1.0, 0.0, 0.0])
    masked = float((per * mask).sum() / jnp.maximum(mask.sum(), 1.0))
    assert np.isclose(masked, float(per[:2].mean()), atol=1e-6)


def test_rows_are_independent():
    # If the reduction leaked across the batch a mask could not isolate a mode,
    # and each critic would be trained partly on the other's rows.
    cur, tgt = _fixture()
    base = quantile_huber_loss_per_sample(cur, tgt)
    moved = quantile_huber_loss_per_sample(cur, tgt.at[3].add(10.0))
    assert np.allclose(np.asarray(base[:3]), np.asarray(moved[:3]), atol=1e-6)
    assert not np.isclose(float(base[3]), float(moved[3]), atol=1e-6)


def test_loss_is_non_negative():
    cur, tgt = _fixture(batch=6, seed=3)
    assert bool((quantile_huber_loss_per_sample(cur, tgt) >= 0).all())


def test_a_perfect_prediction_costs_almost_nothing():
    # Sanity on the sign and scale: predicting the target exactly should sit at
    # the floor, well below a prediction that is far off.
    cur = jnp.zeros((2, 4, 5), jnp.float32)
    exact = quantile_huber_loss_per_sample(cur, jnp.zeros((4, 3), jnp.float32))
    far = quantile_huber_loss_per_sample(cur, jnp.full((4, 3), 5.0, jnp.float32))
    assert float(exact.mean()) < float(far.mean())
