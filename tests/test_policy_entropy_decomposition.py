"""The trace's policy-entropy telemetry must equal the TRUE entropy of the policy.

v43's actor is jaxrl2's TanhMultivariateNormalDiag: a learned MultivariateNormalDiag
pushed through Chain([affine rescale to [low, high], Block(Tanh)]). Its entropy is
    H[a] = H_gauss  +  E[log|det d(rescale)/du|]  +  E[sum log(1 - tanh(z)^2)]
           (exact)     (exact, a constant)          (no closed form)
Two bugs shipped in the first version, both caught on a 60-sample smoke trace:
  * unwrapping `.distribution` in a LOOP went past the learned MVNDiag into the unit
    Normal inside it -> H_gauss read a constant 2.838 with log_std = [0, 0];
  * arctanh was applied to the ACTION-space mode, which for the [0,1] throttle is not
    the pre-tanh variable at all.
The decisive check is against ground truth: -mean(log_prob(a)) over samples drawn from
the distribution IS its entropy, and the decomposition must reproduce it.
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import vae_nav.jax_compat  # noqa: E402,F401  (distrax 0.1.5 needs jax.core.Var)
from jaxrl2.networks.normal_tanh_policy import TanhMultivariateNormalDiag  # noqa: E402
from vae_nav.policy_entropy import policy_entropy  # noqa: E402

LOW, HIGH = np.array([-1.0, 0.0], np.float32), np.array([1.0, 1.0], np.float32)  # v43's box


def _dist(loc, log_std):
    return TanhMultivariateNormalDiag(loc=jnp.asarray(loc, jnp.float32),
                                      scale_diag=jnp.exp(jnp.asarray(log_std, jnp.float32)),
                                      low=jnp.asarray(LOW), high=jnp.asarray(HIGH))


def _true_entropy(d, n=200_000, seed=0):
    a, lp = d.sample_and_log_prob(seed=jax.random.PRNGKey(seed), sample_shape=(n,))
    return float(-np.mean(np.asarray(lp)))


def test_gaussian_part_is_the_learned_scale_not_the_unit_normal():
    e1 = policy_entropy(_dist([0.0, 0.0], [-1.0, -2.0]))
    e2 = policy_entropy(_dist([0.0, 0.0], [0.5, 0.3]))
    want1 = float(np.sum(0.5 * np.log(2 * np.pi * np.e) + np.array([-1.0, -2.0])))
    assert e1["ent_gauss"] == pytest.approx(want1, abs=1e-5)
    assert e1["ent_gauss"] != pytest.approx(e2["ent_gauss"])        # NOT a constant
    assert e1["log_std"] == pytest.approx([-1.0, -2.0], abs=1e-5)  # not [0, 0]


def test_rescale_term_is_the_exact_constant():
    e = policy_entropy(_dist([0.0, 0.0], [0.0, 0.0]))
    assert e["ent_rescale"] == pytest.approx(float(np.sum(np.log(0.5 * (HIGH - LOW)))), abs=1e-6)
    assert e["ent_rescale"] == pytest.approx(-0.6931, abs=1e-3)       # the trainer's banner value


@pytest.mark.parametrize("loc,log_std", [([0.0, 0.0], [0.0, 0.0]),
                                         ([0.8, -1.5], [-1.0, -0.5]),
                                         ([2.0, 1.0], [-2.0, 0.3])])
def test_total_matches_ground_truth_entropy(loc, log_std):
    d = _dist(loc, log_std)
    e = policy_entropy(d, n_mc=100_000, seed=1)
    assert e["ent_total"] == pytest.approx(_true_entropy(d), abs=0.03)


def test_analytic_part_excludes_only_the_tanh_term():
    e = policy_entropy(_dist([0.3, -0.2], [-0.7, -1.1]), n_mc=2048)
    assert e["ent_analytic"] == pytest.approx(e["ent_gauss"] + e["ent_rescale"], abs=1e-6)
    assert e["ent_total"] == pytest.approx(e["ent_analytic"] + e["ent_tanh_mc"], abs=1e-6)
    assert e["ent_tanh_mc"] < 0.0                                       # tanh only contracts
