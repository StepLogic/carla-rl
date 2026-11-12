#!/usr/bin/env python3
"""REDQ target for the image-goal arm: expectation SAC with 10 critics, min over 2.

User 2026-10-12: "I want to move away from TQC"; "USE 10 critics for SAC". The
choice between the three candidate operators was put to them explicitly and this
is the one they picked:

  REDQ (Chen et al. 2021)   target = min over a RANDOM SUBSET of M of the N critics
  truncated mean (TQC's)    pool and drop the k highest -- already implemented in
                            tqc_loss.py, and at n_quantiles=1 it is a one-line
                            config change, which is why it was offered
  full min over N           rejected: at N=10 and gamma 0.99 both this side and the
                            peer expect it to idle, the failure our drop=2 TQC
                            critic already shows at short horizons

WHY THE SUBSET IS DRAWN ONCE PER UPDATE, NOT PER ROW. The pessimism REDQ buys
comes from evaluating a whole update under one draw: resampling per row averages
the draw away across a 256-row batch and the ensemble stops doing anything that a
single critic with more noise would not. The caller draws with sample_subset() and
passes the indices in, so the draw is visible in the training loop rather than
hidden in here -- and so a test can pin a subset instead of seeding a generator.

WHY ALL N CRITICS REGRESS ONTO THE SHARED TARGET. That is REDQ: the subset picks
the target, every head then fits it. Giving each head its own min would make the
heads collapse toward each other, and their disagreement is the only reason to
carry ten of them.

The value floor and ceiling are a DIVERGENCE GUARD on the backed-up value, not a
reward clip, and they are applied last -- after the entropy term and after the
bootstrap -- exactly as tqc_loss.tqc_target applies them. Our gamma=1.0 goal
critic needed them; the explore critic drifted 20 -> 11,801 when left unclipped.
An inverted pair raises: the peer's older config defaulted value_ceiling to -1,
and with a reward that can be positive every target pins at the ceiling and the
run looks alive while learning nothing.
"""
from typing import Optional, Union

import torch

Alpha = Union[float, torch.Tensor]


def sample_subset(n_critics: int, subset: int, device=None,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """`subset` distinct critic indices out of `n_critics`, uniformly.

    WITHOUT REPLACEMENT. Drawn with replacement, a "min over 2" is a min over 1
    on 1/N of updates, at random, for no reason anyone chose.
    """
    n_critics, subset = int(n_critics), int(subset)
    if not 1 <= subset <= n_critics:
        raise ValueError(
            f"subset={subset} must be between 1 and n_critics={n_critics}; "
            "subset == n_critics is the full min, subset == 1 is a single "
            "random critic, and REDQ's own recipe at N=10 is 2")
    perm = torch.randperm(n_critics, generator=generator,
                          device=None if generator is None else generator.device)
    return perm[:subset].to(device) if device is not None else perm[:subset]


def _scalars(q: torch.Tensor) -> torch.Tensor:
    """(n_critics, batch, 1) -> (n_critics, batch); (n_critics, batch) unchanged.

    QuantileCritics returns the quantile axis even when there is one quantile,
    and a bare scalar head does not. Without this both shapes reach the min and
    the elementwise subtraction below broadcasts into (n, B, B) silently.
    """
    if q.dim() == 3:
        if q.shape[-1] != 1:
            raise ValueError(
                f"redq_target wants ONE value per critic, got n_quantiles="
                f"{q.shape[-1]}. Quantile critics belong in tqc_loss; this arm "
                "is expectation SAC (build QuantileCritics with n_quantiles=1).")
        return q[..., 0]
    if q.dim() != 2:
        raise ValueError(f"expected (n_critics, batch[, 1]), got {tuple(q.shape)}")
    return q


@torch.no_grad()
def redq_target(next_q: torch.Tensor, rewards: torch.Tensor, masks: torch.Tensor,
                discount: float, alpha: Alpha, next_log_probs: torch.Tensor,
                subset_idx: torch.Tensor,
                value_floor: Optional[float] = None,
                value_ceiling: Optional[float] = None) -> torch.Tensor:
    """r + gamma * mask * (min_{i in subset} Q_i(s', a') - alpha * log pi(a'|s')).

    next_q          (n_critics, batch[, 1]) target-net values at the sampled next action
    masks           0 on a terminal row, 1 otherwise -- a 0 must kill the bootstrap
    subset_idx      which critics set the target, from sample_subset(); ONE draw
                    for the whole batch
    -> (batch,), never carrying gradient (it is a target).
    """
    if (value_floor is not None and value_ceiling is not None
            and value_ceiling < value_floor):
        raise ValueError(
            f"value_ceiling={value_ceiling} is below value_floor={value_floor}. "
            "These bound the BACKED-UP VALUE as a divergence guard, so an "
            "inverted pair pins every target: the peer's older config defaulted "
            "the ceiling to -1 and, with a reward that can be positive, the run "
            "looked alive while learning nothing.")

    q = _scalars(next_q)
    if subset_idx.numel() == 0:
        raise ValueError("subset_idx is empty; nothing would set the target")
    chosen = q.index_select(0, subset_idx.to(q.device))
    soft = chosen.min(dim=0).values - alpha * next_log_probs
    target = rewards + discount * masks * soft
    if value_floor is not None:
        target = torch.clamp_min(target, value_floor)
    if value_ceiling is not None:
        target = torch.clamp_max(target, value_ceiling)
    return target


def critic_loss_per_sample(current: torch.Tensor, target: torch.Tensor
                           ) -> torch.Tensor:
    """Mean squared error of every critic against the shared target -> (batch,).

    Per sample, not reduced over the batch: the gated learner weights each row by
    the per-mode mask from the replay, so the batch axis has to survive.
    """
    q = _scalars(current)
    return ((q - target[None, :]) ** 2).mean(dim=0)
