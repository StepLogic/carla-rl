"""Per-sample quantile Huber loss, for mode-masked critics.

WHY THIS EXISTS. The gated learner trains two critics on disjoint halves of one
batch, selected by a mode bit. Masking requires a PER-ROW loss: jaxrl2's
`quantile_huber_loss` reduces the batch axis away with `.mean()`, so multiplying
its scalar output by a mask would scale the whole loss rather than drop rows,
and each critic would still be trained on the other mode's transitions.

WHY IT LIVES HERE RATHER THAN BESIDE THE FUNCTION IT MIRRORS. The natural home
is `jaxrl2/agents/tqc/quantiles.py`, but `dependencies/jaxrl2` is a nested
repository that `easy-install.pth` places on `sys.path` by ABSOLUTE path. It is
therefore shared by every session on this machine and cannot be isolated in a
git worktree. Keeping the addition here leaves the vendored fork unpatched,
keeps the code under carla-rl's own history, and means a half-written edit can
never reach another session's training run. The only thing it borrows is
`taus`, imported read-only.

THE INVARIANT, pinned by tests/test_quantile_per_sample.py: for every shape and
kappa, `quantile_huber_loss_per_sample(...).mean()` equals
`quantile_huber_loss(...)`. This is the same loss, one reduction short. If that
ever drifts, the gated critics are optimising a different objective from every
other TQC run in this repo and no other assertion would notice.
"""
import jax.numpy as jnp

from jaxrl2.agents.tqc.quantiles import taus


def quantile_huber_loss_per_sample(current_quantiles: jnp.ndarray,
                                   target_quantiles: jnp.ndarray,
                                   kappa: float = 1.0) -> jnp.ndarray:
    """Quantile Huber loss reduced to (batch,) rather than to a scalar.

    current: (n_critics, batch, n_quantiles)   target: (batch, kept)
    returns: (batch,)
    """
    n_quantiles = current_quantiles.shape[-1]
    t = taus(n_quantiles)

    # (n_critics, batch, n_quantiles, kept)
    delta = target_quantiles[None, :, None, :] - current_quantiles[..., None]
    abs_delta = jnp.abs(delta)
    huber = jnp.where(abs_delta <= kappa,
                      0.5 * delta ** 2,
                      kappa * (abs_delta - 0.5 * kappa))
    # |tau - 1{delta < 0}|: delta < 0 means the prediction OVERSHOT the target,
    # which a high-tau head tolerates and a low-tau head punishes.
    weight = jnp.abs(t[None, None, :, None] - (delta < 0).astype(jnp.float32))

    # SUM over the current-quantile axis exactly as the reduced form does, then
    # mean over critics and targets ONLY. The batch axis survives; that is the
    # whole difference between this and quantile_huber_loss.
    return (weight * huber / kappa).sum(axis=2).mean(axis=(0, 2))


def masked_mean(per_sample: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Mean of `per_sample` over the rows `mask` selects.

    The denominator is clamped at 1 so a batch containing none of a mode yields
    a finite zero loss rather than a nan. That case is real: at the very start
    of training only one mode has been visited, and a nan here would propagate
    through the shared encoder and destroy the other critic too.
    """
    return (per_sample * mask).sum() / jnp.maximum(mask.sum(), 1.0)
