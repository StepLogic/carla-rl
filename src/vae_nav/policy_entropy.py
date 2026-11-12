"""Entropy of a jaxrl2 TanhMultivariateNormalDiag policy, split into its parts.

The actor is a learned diagonal Gaussian u ~ N(loc, diag(std^2)) pushed through
Chain([affine rescale to [low, high], Block(Tanh)]). Change of variables gives
    H[a] = H[u] + E_u[ log|det J(u)| ]
         = ent_gauss  (closed form: sum(0.5*log(2*pi*e) + log_std))
         + ent_rescale (closed form: sum(log(0.5*(high-low))), a constant)
         + ent_tanh   (E[sum log(1 - tanh(u)^2)], no closed form -> Monte Carlo)
`ent_analytic` = ent_gauss + ent_rescale is the exact closed-form part;
`ent_tanh_at_mean` is the tanh term evaluated at loc (a deterministic proxy);
`ent_total` = ent_analytic + ent_tanh_mc is the true action-space entropy.

Read `dist.distribution` exactly ONCE: a distrax MultivariateNormalDiag is itself a
Transformed of a unit Normal, so unwrapping further reads a constant 2.838 nats.
"""
import jax
import jax.numpy as jnp
import numpy as np


def _log1m_tanh2(z):
    # log(1 - tanh(z)^2), numerically stable (same form distrax.Tanh uses)
    return 2.0 * (jnp.log(2.0) - z - jax.nn.softplus(-2.0 * z))


def policy_entropy(dist, n_mc=256, seed=0):
    """-> dict of floats for the FIRST batch element of `dist`."""
    base = dist.distribution                       # the learned MVNDiag, once
    loc = np.asarray(base.loc, np.float64)
    std = np.asarray(base.stddev(), np.float64)
    d = loc.shape[-1]
    loc, std = loc.reshape(-1, d)[0], std.reshape(-1, d)[0]
    log_std = np.log(std)

    ent_gauss = float(np.sum(0.5 * np.log(2 * np.pi * np.e) + log_std))
    z = loc + std * np.asarray(jax.random.normal(jax.random.PRNGKey(seed), (n_mc, d)))
    z = jnp.asarray(z, jnp.float32)
    tanh_mc = float(jnp.mean(jnp.sum(_log1m_tanh2(z), -1)))
    fldj = float(jnp.mean(dist.bijector.forward_log_det_jacobian(z)))
    ent_rescale = fldj - tanh_mc                   # the affine part (constant)
    return {
        "ent_gauss": ent_gauss,
        "ent_rescale": ent_rescale,
        "ent_analytic": ent_gauss + ent_rescale,
        "ent_tanh_at_mean": float(jnp.sum(_log1m_tanh2(jnp.asarray(loc, jnp.float32)))),
        "ent_tanh_mc": tanh_mc,
        "ent_total": ent_gauss + ent_rescale + tanh_mc,
        "loc": loc.astype(np.float32).tolist(),
        "log_std": log_std.astype(np.float32).tolist(),
    }
