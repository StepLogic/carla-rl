#!/usr/bin/env python3
"""The gated learner as expectation SAC with 10 critics and a REDQ target.

User 2026-10-12: "I want to move away from TQC if it gives less performance as
regular SAC", then "USE 10 critics for SAC", and after being shown the three
candidate target operators they chose REDQ -- min over a random subset of 2 of
the 10. The dual critic stays ("keep same critics since reward is similar"): two
ensembles, two targets, two temperatures, masked by the replay's per-episode mode.

HOW LITTLE THIS FILE IS, AND WHY THAT MATTERS. Everything that makes the gated
learner work -- the GRU over 40-step chunks with an 8-step burn-in, the mode
masks, the per-mode actor Q-normalisation, the explore-actor warm-up, skipping a
temperature step for an absent mode, the save/load contract -- is unchanged and
not duplicated here. Only the critic algorithm differs, and in the parent that
lives in exactly two methods. A second copy of an 850-line learner would drift
from the original within a week and neither copy would be trustworthy.

WHAT CHANGES, PRECISELY:
  n_quantiles          25 -> 1. QuantileCritics then holds n_critics scalar heads
                       and every downstream reduction (the mean over dim (0, 2),
                       the actor's Q term) already tolerates a width-1 axis.
  n_critics            5 -> 10, the user's number.
  target               truncated mean over pooled quantiles -> min over a random
                       subset of `subset_size` critics (torch_rl/redq_loss.py).
  critic loss          quantile Huber -> plain MSE against the shared target.
  top_quantiles_to_drop_per_net   ignored, and refused if set, so a config cannot
                       claim a TQC setting that silently does nothing.

ONE DRAW PER HEAD PER UPDATE. The goal and explore ensembles are separate
networks, so each draws its own subset. The draw happens in _critic_target, which
the parent calls once per head per update -- so "per update" is a property of the
call site, and the test that pins it calls the method twice and checks the draws
are independent rather than cached.
"""
from dataclasses import dataclass
from typing import Optional

import torch

from torch_rl import redq_loss
from torch_rl.gated_recurrent_tqc import (GatedRecurrentTQC,
                                          GatedRecurrentTQCConfig)


@dataclass
class GatedRecurrentSACConfig(GatedRecurrentTQCConfig):
    """The parent's config with the SAC defaults and one new knob.

    n_quantiles is forced to 1 in __post_init__ rather than merely defaulted, so
    a stray 25 in an old config.json cannot turn this arm back into quantile
    regression without anyone noticing.
    """
    n_critics: int = 10
    n_quantiles: int = 1
    subset_size: int = 2          # REDQ's M at N=10
    top_quantiles_to_drop_per_net: int = 0

    def __post_init__(self) -> None:
        if int(self.n_quantiles) != 1:
            raise ValueError(
                f"n_quantiles={self.n_quantiles}: this arm is EXPECTATION SAC, "
                "one value per critic (user 2026-10-12, 'do not use TQC'). "
                "Quantile critics belong in GatedRecurrentTQC.")
        if int(self.top_quantiles_to_drop_per_net) != 0:
            raise ValueError(
                "top_quantiles_to_drop_per_net is TQC's truncation and is not "
                "read here; leave it 0 rather than configuring something that "
                "does nothing. The pessimism knob in this arm is subset_size.")
        if not 1 <= int(self.subset_size) <= int(self.n_critics):
            raise ValueError(
                f"subset_size={self.subset_size} must be between 1 and "
                f"n_critics={self.n_critics}")
        super().__post_init__()


class GatedRecurrentSAC(GatedRecurrentTQC):
    """GatedRecurrentTQC with the REDQ target and an MSE critic loss."""

    def __init__(self, cfg: GatedRecurrentSACConfig, **kw) -> None:
        super().__init__(cfg, **kw)
        # A dedicated generator, so the subset draws neither consume nor depend
        # on the global RNG the parent seeds for construction order -- otherwise
        # adding an update would shift every later network init.
        self._subset_rng = torch.Generator()
        self._subset_rng.manual_seed(int(cfg.seed) + 1)
        self._last_subset = {}

    # ------------------------------------------------------------- the seams --
    def _critic_target(self, next_q, rewards, masks, discount, alpha,
                       next_log_probs, value_floor, value_ceiling, head):
        cfg = self.cfg
        idx = redq_loss.sample_subset(cfg.n_critics, cfg.subset_size,
                                      generator=self._subset_rng)
        self._last_subset[head] = idx            # telemetry only
        return redq_loss.redq_target(
            next_q, rewards, masks, discount=discount, alpha=alpha,
            next_log_probs=next_log_probs, subset_idx=idx,
            value_floor=value_floor, value_ceiling=value_ceiling)

    def _critic_loss_per_sample(self, qs, target, head):
        return redq_loss.critic_loss_per_sample(qs, target)

    # ------------------------------------------------------------- telemetry --
    def describe(self) -> str:
        cfg = self.cfg
        return (f"GatedRecurrentSAC: expectation SAC, {cfg.n_critics} critics per "
                f"head, target = min over a random subset of {cfg.subset_size} "
                f"(REDQ), MSE critic loss, NO quantiles | gamma {cfg.discount} "
                f"(goal) / {cfg.discount_e} (explore) | goal target clip "
                f"[{cfg.value_floor}, {cfg.value_ceiling}], explore "
                f"[{cfg.value_floor_e}, {cfg.value_ceiling_e}]")


def sac_config_from_tqc(cfg: GatedRecurrentTQCConfig, n_critics: int = 10,
                        subset_size: int = 2,
                        value_floor_e: Optional[float] = None,
                        value_ceiling_e: Optional[float] = None
                        ) -> GatedRecurrentSACConfig:
    """Same arm, SAC critics: copy every shared field across so a comparison run
    differs ONLY in the critic algorithm.

    This exists because the comparison the user asked for is only worth running
    if nothing else moved, and hand-copying thirty fields is how a second
    difference gets in.
    """
    fields = {f: getattr(cfg, f) for f in cfg.__dataclass_fields__}
    fields.update(n_critics=int(n_critics), n_quantiles=1,
                  subset_size=int(subset_size),
                  top_quantiles_to_drop_per_net=0,
                  value_floor_e=value_floor_e, value_ceiling_e=value_ceiling_e)
    return GatedRecurrentSACConfig(**fields)
