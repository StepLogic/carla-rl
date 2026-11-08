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
import os
from dataclasses import dataclass
from typing import Optional

import torch

from torch_rl import redq_loss
from torch_rl.gated_recurrent_tqc import (_HALF_LOG_2PI_E, GatedRecurrentTQC,
                                          GatedRecurrentTQCConfig, _bound_frac,
                                          _masked_mean)
from torch_rl.iql import awr_weights, expectile_loss
from torch_rl.nets import QuantileCritics, soft_update


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
    # algo "iql" (user 2026-11-06, torch_rl/iql.py): the critic and the actor are
    # queried only at replay actions. expectile tau for V, beta for the advantage
    # weights (on A / the running Q std), and the weight clip.
    algo: str = "sac"
    iql_expectile: float = 0.7
    iql_beta: float = 3.0
    iql_max_weight: float = 100.0

    def __post_init__(self) -> None:
        if self.algo not in ("sac", "iql"):
            raise ValueError(f"algo={self.algo!r}: 'sac' or 'iql'")
        if self.algo == "iql":
            if not self.one_critic:
                raise ValueError("algo='iql' is implemented for one_critic only "
                                 "(this arm's configuration)")
            if not 0.5 <= float(self.iql_expectile) < 1.0:
                raise ValueError(f"iql_expectile={self.iql_expectile}: in [0.5, 1); "
                                 "0.5 is the mean, toward 1 the best replay action")
            if float(self.actor_bc_alpha) > 0.0:
                raise ValueError("actor_bc_alpha is SAC's in-loss cloning (TD3+BC); "
                                 "IQL's actor is already advantage-weighted regression")
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
        # IQL's state value V(s): the critic's network without the action input,
        # one head. Built after every parent module, so the parent's construction
        # RNG order -- and a SAC agent -- is unchanged. Saved beside params.pt
        # (iql_value.pt), so the checkpoint format is too, and a warm start from
        # a SAC checkpoint (the clone) leaves V at its initialisation.
        self.value_g = self.opt_value = None
        if cfg.algo == "iql":
            self.value_g = QuantileCritics(cfg.hidden + len(cfg.goal_cols), 0, 1, 1, cfg.mlp,
                                           dropout=cfg.critic_dropout,
                                           layer_norm=cfg.critic_layer_norm).to(self.device)
            self.opt_value = torch.optim.Adam(self.value_g.parameters(), lr=cfg.lr)

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

    # ------------------------------------------------------------------ IQL --
    def _update_impl(self, obs, keep, actions, rewards, masks, valid, mode, goal=None):
        if self.cfg.algo != "iql":
            return super()._update_impl(obs, keep, actions, rewards, masks, valid, mode, goal)
        return self._update_iql(obs, keep, actions, rewards, masks, valid, goal)

    def _update_iql(self, obs, keep, actions, rewards, masks, valid, goal=None):
        """One IQL step on a replay chunk (torch_rl/iql.py). Every critic and
        value query is at the REPLAY action or at no action; the policy is never
        sampled. Shares the parent's GRU roll, burn-in, state view, targets and
        soft updates; one critic only (the config refuses anything else)."""
        cfg, K = self.cfg, int(self.cfg.burn_in)
        B, T = valid.shape
        N = B * T
        m = valid.reshape(N)
        if not bool((m.sum() > 0).item()):
            return self._skip_metrics()
        zero = torch.zeros((), device=obs.device)

        s_all = self._roll(self.compress, self.act_embed, self.core, obs, keep, goal)
        with torch.no_grad():
            s_all_t = self._roll(self.compress_targ, self.act_embed_targ, self.core_targ, obs,
                                 keep, goal)
        H = cfg.hidden
        s = s_all[:, :-1].reshape(N, H)
        s_t = s_all_t[:, :-1].reshape(N, H)
        s_next_t = s_all_t[:, 1:].reshape(N, H)
        st = self._state(obs[:, K:-1]).reshape(N, cfg.state_dim)
        st_next = self._state(obs[:, K + 1:]).reshape(N, cfg.state_dim)
        a = actions.reshape(N, cfg.action_dim)
        r, msk = rewards.reshape(N), masks.reshape(N)
        no_a = a.new_zeros((N, 0))

        # Q_targ at the replay action: min over a random REDQ subset, as SAC's target.
        with torch.no_grad():
            q_t_all = self._q(self.critics_g_targ, s_t, st, self.goal_cols, a)[..., 0]
            idx = redq_loss.sample_subset(cfg.n_critics, cfg.subset_size,
                                          generator=self._subset_rng)
            self._last_subset["goal"] = idx
            q_t = q_t_all.index_select(0, idx.to(q_t_all.device)).min(dim=0).values

        # V(s): an upper expectile of Q_targ(s, a_replay). On the detached trunk:
        # only the critic loss shapes the encoder and the GRU, as in SAC.
        v = self._q(self.value_g, s.detach(), st, self.goal_cols, no_a)[0, :, 0]
        v_loss = _masked_mean(expectile_loss(q_t - v, cfg.iql_expectile), m)

        # Q(s, a_replay) regressed on r + gamma * mask * V(s'): no next action.
        with torch.no_grad():
            v_next = self._q(self.value_g, s_next_t, st_next, self.goal_cols, no_a)[0, :, 0]
            target = r + float(cfg.discount) * msk * v_next
            if cfg.value_floor is not None:
                target = target.clamp(min=float(cfg.value_floor))
            if cfg.value_ceiling is not None:
                target = target.clamp(max=float(cfg.value_ceiling))
            at_floor = (_masked_mean(_bound_frac(target, cfg.value_floor, upper=False), m)
                        if cfg.value_floor is not None else zero)
            at_ceiling = (_masked_mean(_bound_frac(target, cfg.value_ceiling, upper=True), m)
                          if cfg.value_ceiling is not None else zero)
        qs = self._q(self.critics_g, s, st, self.goal_cols, a)
        q_loss = _masked_mean(self._critic_loss_per_sample(qs, target, head="goal"), m)
        self.opt_critic.zero_grad(set_to_none=True)
        self.opt_value.zero_grad(set_to_none=True)
        (q_loss + v_loss).backward()
        self.opt_critic.step()
        self.opt_value.step()

        # Actor: advantage-weighted regression on the replay action.
        actor_in = torch.cat([s.detach(), st], dim=-1)
        mean_u, log_std = self._actor_forward(actor_in)
        with torch.no_grad():
            adv = q_t - v.detach()
            scale = self._actor_q_scale(q_t, m, self.q_scale_g, self.q_scale_init_g)
            w = awr_weights(adv, cfg.iql_beta, scale, cfg.iql_max_weight)
            entropy = log_std.sum(dim=-1) + cfg.action_dim * _HALF_LOG_2PI_E
            det_a = torch.tanh(mean_u)
            diag = {
                # IQL samples no action: the analytic base-Gaussian entropy stands
                # in for SAC's sampled -log pi in the [Diag] line.
                "neg_logp_g": _masked_mean(entropy, m),
                "sat_steer": _masked_mean((det_a[..., 0].abs() > 0.99).float(), m),
                "sat_throttle": _masked_mean((det_a[..., 1].abs() > 0.99).float(), m),
                "mean_abs_u_steer": _masked_mean(mean_u[..., 0].abs(), m),
                "mean_abs_u_throttle": _masked_mean(mean_u[..., 1].abs(), m),
                "log_std_g": _masked_mean(log_std.mean(dim=-1), m),
                "enc_alive": self._enc_alive(obs),
                "pretanh_pen": zero,
            }
        if self.warmup or not self._actor_turn:
            # critic and value only, as in SAC's warm-up and its off-turn
            # updates under ACTOR_UPDATE_EVERY
            actor_loss = zero
        else:
            u = torch.atanh(a.clamp(-0.995, 0.995))
            nll = (0.5 * ((u - mean_u) / log_std.exp()) ** 2 + log_std).sum(dim=-1)
            actor_loss = _masked_mean(w * nll, m)
            self.opt_actor.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.opt_actor.step()

        with torch.no_grad():
            self.n_updates.add_(1)
        for targ, src in ((self.compress_targ, self.compress), (self.act_embed_targ, self.act_embed),
                          (self.core_targ, self.core), (self.critics_g_targ, self.critics_g)):
            if src is not None:
                soft_update(targ, src, cfg.tau)

        return {"critic_loss_g": q_loss.detach(), "critic_loss_e": zero,
                "actor_loss": actor_loss.detach(), "temp_loss_g": zero, "temp_loss_e": zero,
                "alpha_g": self.log_alpha_g.detach().exp(),
                "alpha_e": self.log_alpha_e.detach().exp(),
                "q_g": _masked_mean(qs.detach().mean(dim=(0, 2)), m), "q_e": zero,
                "q_scale_g": self.q_scale_g.clone(), "q_scale_e": self.q_scale_e.clone(),
                "entropy_g": _masked_mean(entropy, m), "entropy_e": zero,
                "has_g": (m.sum() > 0).float(), "has_e": zero, **diag,
                "goal_frac": torch.ones((), device=obs.device), "valid_frac": valid.mean(),
                "target_at_floor": at_floor, "target_at_ceiling": at_ceiling, "skipped": zero,
                "v_loss": v_loss.detach(), "v_mean": _masked_mean(v.detach(), m),
                "adv_mean": _masked_mean(adv, m), "awr_w_mean": _masked_mean(w, m),
                "awr_w_clip_frac": _masked_mean((w >= float(cfg.iql_max_weight)).float(), m)}

    # ----------------------------------------------------------- checkpoints --
    def save(self, out_dir: str) -> str:
        path = super().save(out_dir)
        if self.value_g is not None:
            torch.save({"value_g": self.value_g.state_dict(),
                        "opt_value": self.opt_value.state_dict()},
                       os.path.join(out_dir, "iql_value.pt"))
        return path

    def load(self, in_dir: str, fresh_critics_on_mismatch: bool = False) -> None:
        super().load(in_dir, fresh_critics_on_mismatch=fresh_critics_on_mismatch)
        p = os.path.join(in_dir, "iql_value.pt")
        if self.value_g is not None and os.path.exists(p):
            blob = torch.load(p, map_location=self.device, weights_only=True)
            self.value_g.load_state_dict(blob["value_g"])
            self.opt_value.load_state_dict(blob["opt_value"])
            for group in self.opt_value.param_groups:      # the configured rate, not the saved one
                group["lr"] = float(self.cfg.lr)

    # ------------------------------------------------------------- telemetry --
    def describe(self) -> str:
        p = self.cfg.actor_log_std
        pin = ("" if p is None else
               f" | actor log-std PINNED at "
               f"{', '.join(f'{v:g}' for v in (p if isinstance(p, tuple) else (p,)))}"
               f": only the mean action trains")
        every = int(self.cfg.actor_update_every)
        delay = ("" if every == 1 else
                 f" | actor (and temperature) updated every {every} critic updates")
        a = float(self.cfg.actor_bc_alpha)
        td3bc = ("" if a <= 0.0 else
                 f" | actor: TD3+BC in-loss cloning, critic term x {a:g}/mean|Q| plus "
                 f"mean (tanh(mu) - expert label)^2, one optimizer")
        fixed = float(self.cfg.fixed_alpha)
        temp = ("" if fixed <= 0.0 else
                f" | temperature FIXED at {fixed:g} (never updated, re-applied on load)")
        return self._describe_learner() + pin + delay + td3bc + temp

    def _describe_learner(self) -> str:
        cfg = self.cfg
        if cfg.algo == "iql":
            return (f"RecurrentIQL, ONE CRITIC for both reward modes: {cfg.n_critics} "
                    f"critics on REPLAY actions, V(s) = expectile {cfg.iql_expectile:g} of "
                    f"min over a random subset of {cfg.subset_size} target critics, Q -> r + "
                    f"gamma V(s') | gamma {cfg.discount} | actor: advantage-weighted "
                    f"regression, w = exp({cfg.iql_beta:g} A / Q-std) clipped at "
                    f"{cfg.iql_max_weight:g} | NO temperature, NO policy sample in the "
                    f"update | the mode bit is a critic INPUT (state col {cfg.mode_index})")
        if cfg.one_critic:
            # The banner must not claim two heads when there is one: this line
            # is what a log is read back for months later.
            return (f"RecurrentSAC, ONE CRITIC for both reward modes: "
                    f"{cfg.n_critics} critics, target = min over a random subset "
                    f"of {cfg.subset_size} (REDQ), MSE critic loss, NO quantiles "
                    f"| gamma {cfg.discount} | one temperature | the mode bit is "
                    f"a critic INPUT (state col {cfg.mode_index}) | target clip "
                    f"[{cfg.value_floor}, {cfg.value_ceiling}] | the two modes "
                    f"differ only in what the env pays")
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
