"""IQL for learner C, ported from FastRLAP's own implementation (user 2026-11-14: "ensure you borrow
IQL from fastrlap's and use the new proven learner").

Source read 2026-11-14: kylestach/fastrlap-release, jaxrl5/jaxrl5/agents/iql/iql_learner.py. The
five places it differs from textbook IQL are reproduced deliberately and pinned by
tests/test_frl_iql.py, because each would be a silent difference in a port:

  1. expectile weighting: `weight = where(diff > 0, expectile, 1 - expectile)`, loss `weight * diff**2`,
     with `diff = q - v`;
  2. the V target is the MIN over TARGET-Q heads and carries no gradient (the critic is untouched by
     the V step);
  3. `target_value` is a HARD COPY of `value` after each V step -- no Polyak on the value target;
  4. the critic loss SUMS the squared error over the Q heads and averages over the batch;
  5. the AWR actor weight is `min(exp((q - v) * temperature), 100)` with `q` from the TARGET critic at
     the DATASET action and `v` the MEAN over value heads, used as a weighted log-likelihood.
Update order: V -> ACTOR -> Q, so the actor sees the freshly updated V and the old target critic.

FastRLAP defaults kept: expectile 0.8, temperature 0.1, discount 0.99, tau 0.005, actor_lr 1e-3,
critic_lr = value_lr = 3e-4, hidden (256, 256). Two deliberate differences, both recorded:
  * the critic is learner C's EnsembleQ (10 heads by default here, FastRLAP's IQL file defaults to 2)
    because that is the benchmark-validated critic this project uses everywhere else;
  * the actor is learner C's TanhNormalActor, not FastRLAP's fixed-std unsquashed Normal, because the
    SAC stage that follows needs a squashed stochastic actor and the whole point is to hand it over.
"""
import copy
import math
from typing import Dict, Sequence

import torch
from torch import nn

from torch_rl.frl.networks import MLP, EnsembleQ, TanhNormalActor, _log1m_tanh_sq, xavier_


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    """FastRLAP's asymmetric square: tau on the positive side, 1 - tau on the negative."""
    weight = torch.where(diff > 0, float(expectile), 1.0 - float(expectile))
    return weight * diff ** 2


def tanh_normal_log_prob(actor: TanhNormalActor, obs: torch.Tensor, action: torch.Tensor,
                         eps: float = 1e-6) -> torch.Tensor:
    """log pi(a|s) of learner C's squashed actor for a GIVEN action -- what AWR needs and `sample`
    cannot give. The tanh is inverted and the change-of-variables term subtracted with the same
    stable form the actor itself uses. Expert actions sit in (-1, 1) but touch the bound, so they
    are clamped before atanh. Implemented here on purpose: torch_rl/frl/networks.py is the
    benchmark-validated learner and this port does not modify it."""
    mean, log_std = actor.dist_params(obs)
    std = log_std.exp()
    u = torch.atanh(action.clamp(-1.0 + eps, 1.0 - eps))
    return (-0.5 * ((u - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
            - _log1m_tanh_sq(u)).sum(-1)


class EnsembleV(nn.Module):
    """`num_vs` independent state-value heads -> (num_vs, batch). FastRLAP's IQL ensembles V too
    (its own comment there reads "TODO: Why even ensemble v?"); the default is one head."""

    def __init__(self, obs_dim: int, num_vs: int = 1, hidden_dims: Sequence[int] = (256, 256),
                 use_layer_norm: bool = False):
        super().__init__()
        self.num_vs = int(num_vs)
        self.heads = nn.ModuleList()
        for _ in range(self.num_vs):
            base = MLP(obs_dim, hidden_dims, activate_final=True, use_layer_norm=use_layer_norm)
            self.heads.append(nn.Sequential(base, xavier_(nn.Linear(base.out_dim, 1))))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.stack([h(obs).squeeze(-1) for h in self.heads], dim=0)


class IQLLearner:
    """Offline critic (and actor) pretraining on expert transitions, to hand to SAC."""

    def __init__(self, obs_dim: int, action_dim: int, num_qs: int = 10, num_vs: int = 1,
                 hidden_dims: Sequence[int] = (256, 256), expectile: float = 0.8,
                 temperature: float = 0.1, discount: float = 0.99, tau: float = 0.005,
                 actor_lr: float = 1e-3, critic_lr: float = 3e-4, value_lr: float = 3e-4,
                 critic_layer_norm: bool = True, device: str = "cpu", seed: int = 0,
                 batched_heads: bool = False, awr_clip: float = 100.0):
        torch.manual_seed(int(seed))
        self.device = torch.device(device)
        self.expectile, self.temperature = float(expectile), float(temperature)
        self.discount, self.tau, self.awr_clip = float(discount), float(tau), float(awr_clip)
        self.actor = TanhNormalActor(obs_dim, action_dim, hidden_dims).to(self.device)
        self.critic = EnsembleQ(obs_dim, action_dim, num_qs=num_qs, hidden_dims=hidden_dims,
                                use_layer_norm=critic_layer_norm,
                                batched=bool(batched_heads)).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.value = EnsembleV(obs_dim, num_vs=num_vs, hidden_dims=hidden_dims,
                               use_layer_norm=critic_layer_norm).to(self.device)
        self.target_value = copy.deepcopy(self.value).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=value_lr)

    # ---- the three steps, in FastRLAP's order ----------------------------------------------------

    def update_v(self, b: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """V regresses on the MIN over TARGET-Q heads with the expectile loss. No critic gradient,
        then target_value is hard-copied from value (FastRLAP does not Polyak the value target)."""
        with torch.no_grad():
            q = self.target_critic(b["observations"], b["actions"]).min(dim=0).values
        v = self.value(b["observations"])
        loss = expectile_loss(q.unsqueeze(0) - v, self.expectile).mean()
        self.value_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.value_opt.step()
        self.target_value.load_state_dict(self.value.state_dict())      # HARD copy
        return {"value_loss": float(loss.detach()), "v_mean": float(v.mean().detach()),
                "q_mean": float(q.mean())}

    def update_actor(self, b: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Advantage-weighted log-likelihood of the DATASET action; weights outside the gradient."""
        with torch.no_grad():
            v = self.value(b["observations"]).mean(dim=0)               # MEAN over value heads
            q = self.target_critic(b["observations"], b["actions"]).min(dim=0).values
            adv = q - v
            w = torch.clamp(torch.exp(adv * self.temperature), max=self.awr_clip)
        logp = tanh_normal_log_prob(self.actor, b["observations"], b["actions"])
        loss = -(w * logp).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_opt.step()
        return {"actor_loss": float(loss.detach()), "adv_mean": float(adv.mean()),
                "awr_weight_mean": float(w.mean()), "awr_weight_max": float(w.max()),
                "logp_mean": float(logp.mean().detach())}

    def td_target(self, b: Dict[str, torch.Tensor]) -> torch.Tensor:
        """r + gamma * mask * min_heads V_target(s'), FastRLAP's masks = 1 - terminal."""
        next_v = self.target_value(b["next_observations"]).min(dim=0).values
        return b["rewards"] + self.discount * b["masks"] * next_v

    def update_q(self, b: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Squared TD error SUMMED over heads, then Polyak the target critic."""
        with torch.no_grad():
            target_q = self.td_target(b)
        qs = self.critic(b["observations"], b["actions"])
        per_head = ((qs - target_q.unsqueeze(0)) ** 2).mean(dim=-1)     # (num_qs,)
        loss = per_head.sum()
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_opt.step()
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                tp.mul_(1.0 - self.tau).add_(p, alpha=self.tau)
        return {"critic_loss": float(loss.detach()),
                "critic_loss_per_head": float(per_head.mean().detach()),
                "target_q_mean": float(target_q.mean())}

    def update(self, b: Dict[str, torch.Tensor]) -> Dict[str, float]:
        b = {k: (v if torch.is_tensor(v) else torch.as_tensor(v)).to(self.device)
             for k, v in b.items()}
        info = {}
        info.update(self.update_v(b))
        info.update(self.update_actor(b))
        info.update(self.update_q(b))
        return info

    # ---- handover ---------------------------------------------------------------------------------

    def state_dict(self) -> Dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(), "value": self.value.state_dict(),
                "target_value": self.target_value.state_dict()}

    def load_state_dict(self, st: Dict, strict: bool = True) -> None:
        self.actor.load_state_dict(st["actor"], strict=strict)
        self.critic.load_state_dict(st["critic"], strict=strict)
        self.target_critic.load_state_dict(st["target_critic"], strict=strict)
        self.value.load_state_dict(st["value"], strict=strict)
        self.target_value.load_state_dict(st["target_value"], strict=strict)

    def q_over_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Q for one state against a grid of actions -> (grid,), the min over heads. The gate metric:
        how much Q varies across the steering grid, against |Q| itself. With c0's features a trained
        SAC critic spanned ~1% of its own value, which is why its actor's gradient was noise."""
        with torch.no_grad():
            o = obs.unsqueeze(0).expand(actions.shape[0], -1)
            return self.critic(o, actions).min(dim=0).values
