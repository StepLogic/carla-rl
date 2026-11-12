#!/usr/bin/env python3
"""TQC maths in torch, a line-by-line port of jaxrl2/jaxrl2/agents/tqc/quantiles.py.

The JAX version is the reference implementation this repo's arms are trained
with; tests/test_tqc_loss_parity.py pins value AND gradient against it. Read the
JAX file's docstrings for why each choice matters (bin midpoints, pooling before
truncation, Huber over L2, the elementwise floor and ceiling).
"""
from typing import Optional

import torch


def taus(n_quantiles: int, device=None) -> torch.Tensor:
    """Quantile fractions at bin midpoints: (i + 0.5) / M."""
    return (torch.arange(n_quantiles, dtype=torch.float32, device=device) + 0.5) / n_quantiles


def truncated_targets(next_quantiles: torch.Tensor, top_drop: int) -> torch.Tensor:
    """(n_critics, batch, n_quantiles) -> (batch, n_critics * n_quantiles - top_drop).

    Pool every critic's quantiles, sort ascending, drop the largest k.
    """
    n_critics, batch, n_quantiles = next_quantiles.shape
    pooled = next_quantiles.permute(1, 0, 2).reshape(batch, n_critics * n_quantiles)
    ordered, _ = torch.sort(pooled, dim=1)
    keep = n_critics * n_quantiles - int(top_drop)
    return ordered[:, :keep]


def _pairwise(current: torch.Tensor, target: torch.Tensor, kappa: float):
    n_quantiles = current.shape[-1]
    t = taus(n_quantiles, current.device)
    delta = target[None, :, None, :] - current[..., None]
    abs_delta = delta.abs()
    huber = torch.where(abs_delta <= kappa, 0.5 * delta ** 2, kappa * (abs_delta - 0.5 * kappa))
    weight = (t[None, None, :, None] - (delta < 0).to(current.dtype)).abs()
    return weight * huber / kappa


def quantile_huber_loss(current: torch.Tensor, target: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """Sum over the current-quantile axis, mean over critics, batch and targets."""
    return _pairwise(current, target, kappa).sum(dim=2).mean()


def quantile_huber_loss_per_sample(current: torch.Tensor, target: torch.Tensor,
                                   kappa: float = 1.0) -> torch.Tensor:
    """quantile_huber_loss before the batch reduction -> (batch,)."""
    return _pairwise(current, target, kappa).sum(dim=2).mean(dim=(0, 2))


def tqc_target(next_quantiles: torch.Tensor, rewards: torch.Tensor, masks: torch.Tensor,
               discount: float, top_drop: int, alpha, next_log_probs: torch.Tensor,
               value_floor: Optional[float] = None,
               value_ceiling: Optional[float] = None) -> torch.Tensor:
    """r + gamma * mask * (truncated next quantiles - alpha * log pi), clipped elementwise."""
    kept = truncated_targets(next_quantiles, top_drop)
    soft = kept - alpha * next_log_probs[:, None]
    target = rewards[:, None] + discount * masks[:, None] * soft
    if value_floor is not None:
        target = torch.clamp_min(target, value_floor)
    if value_ceiling is not None:
        target = torch.clamp_max(target, value_ceiling)
    return target
