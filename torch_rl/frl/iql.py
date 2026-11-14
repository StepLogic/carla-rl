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

THE HANDOVER (user 2026-11-14: "also pretrain critic with IQL first"): `iql_warmup` runs these same
three steps on the SAC learner's OWN critic, target critic and actor (`IQLLearner.on_learner`) before
its first online step, so there is nothing to copy across afterwards -- and no second IQL to drift
from this one. Only the networks' weights carry into SAC: the warm-up has its own optimizers, and the
learner's optimizers, temperature and entropy target are left exactly as they were built.
"""
import copy
import math
from typing import Callable, Dict, Optional, Sequence

import torch
from torch import nn

from torch_rl.frl.learner import _to
from torch_rl.frl.networks import (MLP, EnsembleQ, TanhNormalActor, _latent, _log1m_tanh_sq,
                                   xavier_)


class Compressed(nn.Module):
    """obs -> a LayerNorm+tanh latent -> the wrapped head. Learner C's Multiplexer does this and the
    first IQL pretraining dropped it "for more capacity", which was a mistake with a measurement
    behind it: with the raw observation the action is 2 of 3,267 inputs (0.06%), its block L2 is 0.41
    against the observation's ~21, and shuffling the action moved Q no more than shuffling two
    arbitrary observation features (ratio 0.89). Compressed to 50 dims the action is 2 of 52.
    The encoder trains with whichever head wraps it, so the critic's encoder is shaped by the critic
    loss (DrQ's rule) and is copied into the target along with it."""

    def __init__(self, obs_dim: int, latent: int, inner: nn.Module, hidden: int = 0):
        super().__init__()
        # hidden > 0 makes the compression an MLP (user 2026-11-14: "compress to mlp 10 and add
        # actions"): obs -> hidden -> ReLU -> latent, LayerNorm, tanh; the action is concatenated
        # AFTER it by the wrapped head. hidden = 0 is the single layer learner C's Multiplexer uses,
        # and keeps the parameter names earlier checkpoints were saved with.
        if int(hidden) > 0:
            self.enc = nn.Sequential(xavier_(nn.Linear(int(obs_dim), int(hidden))), nn.ReLU(),
                                     _latent(int(hidden), int(latent)))
        else:
            self.enc = _latent(int(obs_dim), int(latent))
        self.inner = inner

    def forward(self, obs: torch.Tensor, *rest):
        return self.inner(self.enc(obs), *rest)

    def dist_params(self, obs: torch.Tensor):
        return self.inner.dist_params(self.enc(obs))

    @property
    def num_qs(self) -> int:
        return getattr(self.inner, "num_qs", 1)


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


class _ActorView(nn.Module):
    """Learner C's actor (learner._Actor: observation trunk -> TanhNormalActor) as IQLLearner reads
    an actor: `dist_params` of the OBSERVATION. It wraps the learner's own module -- no copy -- so
    its parameters are the learner's, and a step through the view is a step on the SAC actor."""

    def __init__(self, actor: nn.Module):
        super().__init__()
        if not (hasattr(actor, "trunk") and hasattr(getattr(actor, "pi", None), "dist_params")):
            raise ValueError(
                f"IQL on the learner's own networks needs learner C's own actor (observation trunk "
                f"-> TanhNormalActor, FRL_ACTOR=frl); {type(actor).__name__} gives no "
                f"log-probability of a dataset action, so the AWR step cannot train it")
        self.actor = actor

    def dist_params(self, obs):
        return self.actor.pi.dist_params(self.actor.trunk(obs))


class _V(nn.Module):
    """A state-value ensemble behind its own observation trunk (learner C's observations are a
    dict that only a Multiplexer reads)."""

    def __init__(self, trunk: nn.Module, v: nn.Module):
        super().__init__()
        self.trunk, self.v = trunk, v

    def forward(self, obs) -> torch.Tensor:
        return self.v(self.trunk(obs))


def _to_device(obj, device):
    """A batch onto the device. Observations may be a dict (learner C's embeddings + states)."""
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return (obj if torch.is_tensor(obj) else torch.as_tensor(obj)).to(device)


class IQLLearner:
    """Offline critic (and actor) pretraining on expert transitions, to hand to SAC."""

    def __init__(self, obs_dim: int, action_dim: int, num_qs: int = 10, num_vs: int = 1,
                 hidden_dims: Sequence[int] = (256, 256), expectile: float = 0.8,
                 temperature: float = 0.1, discount: float = 0.99, tau: float = 0.005,
                 actor_lr: float = 1e-3, critic_lr: float = 3e-4, value_lr: float = 3e-4,
                 critic_layer_norm: bool = True, device: str = "cpu", seed: int = 0,
                 batched_heads: bool = False, awr_clip: float = 100.0, latent: int = 0,
                 latent_hidden: int = 0):
        torch.manual_seed(int(seed))
        self.device = torch.device(device)
        self.expectile, self.temperature = float(expectile), float(temperature)
        self.discount, self.tau, self.awr_clip = float(discount), float(tau), float(awr_clip)
        # latent > 0 compresses the observation first, so the action is not drowned in it; each head
        # gets its OWN encoder, trained by its own loss.
        self.latent = int(latent)
        d = self.latent or int(obs_dim)

        def wrap(inner):
            return (Compressed(obs_dim, self.latent, inner, hidden=int(latent_hidden))
                    if self.latent else inner).to(self.device)

        self.actor = wrap(TanhNormalActor(d, action_dim, hidden_dims))
        self.critic = wrap(EnsembleQ(d, action_dim, num_qs=num_qs, hidden_dims=hidden_dims,
                                     use_layer_norm=critic_layer_norm, batched=bool(batched_heads)))
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.value = wrap(EnsembleV(d, num_vs=num_vs, hidden_dims=hidden_dims,
                                    use_layer_norm=critic_layer_norm))
        self.target_value = copy.deepcopy(self.value).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=value_lr)

    @classmethod
    def on_learner(cls, learner, *, expectile: float = 0.8, temperature: float = 0.1,
                   awr_clip: float = 100.0, actor_lr: float = 1e-3, critic_lr: float = 3e-4,
                   action_eps: float = 0.05,
                   value_lr: float = 3e-4, hidden_dims: Sequence[int] = (256, 256),
                   num_vs: int = 1) -> "IQLLearner":
        """THIS class's three steps on a SAC learner's OWN networks (torch_rl.frl.learner.SACLearner,
        learner C). Nothing is built here except the value net, so the rules that run are the ones
        pinned by tests/test_frl_iql.py and used by every offline critic measurement:

          * critic and target critic: the learner's own objects. V's target and the AWR advantage
            are therefore the min over ALL its target heads -- REDQ's subset (num_min_qs) belongs to
            the SAC target and is not drawn here;
          * actor: the learner's own, through a view that gives log pi of a dataset action;
          * value: new, behind ITS OWN copy of the critic's observation trunk (so the critic's
            compression is shaped by the critic loss alone, as with `Compressed`), with LayerNorm
            iff the critic's heads have it;
          * discount and tau: the learner's;
          * optimizers: NEW ones, at FastRLAP's IQL rates. The learner's own actor, critic and
            temperature optimizers, its temperature and its entropy target are never touched, so
            the SAC stage starts as if IQL-pretrained weights had been loaded into a fresh learner.
        """
        self = cls.__new__(cls)
        self.device = learner.device
        self.expectile, self.temperature = float(expectile), float(temperature)
        self.discount, self.tau = float(learner.discount), float(learner.tau)
        self.awr_clip = float(awr_clip)
        # clipped dataset actions (2.5% of expert rows: throttle exactly -1) must not become a
        # pre-tanh target of 7.25; 0.05 caps it at 1.83 (review of f0a0958)
        self.action_eps = float(action_eps)
        self.latent = 0
        self.actor = _ActorView(learner.actor)
        self.critic, self.target_critic = learner.critic, learner.target_critic
        # what a Q head reads besides the action: the trunk's output (or the raw state)
        first = next(m for m in learner.critic.q.modules() if isinstance(m, nn.Linear))
        in_dim = int(first.in_features) - int(learner.action_dim)
        layer_norm = any(isinstance(m, nn.LayerNorm) for m in learner.critic.q.modules())
        self.value = _V(copy.deepcopy(learner.critic.trunk),
                        EnsembleV(in_dim, num_vs=num_vs, hidden_dims=hidden_dims,
                                  use_layer_norm=layer_norm)).to(self.device)
        self.target_value = copy.deepcopy(self.value).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        return self

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
        logp = tanh_normal_log_prob(self.actor, b["observations"], b["actions"],
                                    eps=getattr(self, "action_eps", 1e-6))
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

    def update(self, b: Dict) -> Dict[str, float]:
        b = _to_device(b, self.device)
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


def iql_warmup(learner, sample: Callable[[], Dict], updates: int, *, expectile: float = 0.8,
               temperature: float = 0.1, awr_clip: float = 100.0, actor_lr: float = 1e-3,
               critic_lr: float = 3e-4, value_lr: float = 3e-4,
               hidden_dims: Sequence[int] = (256, 256),
               log: Optional[Callable[[str], None]] = None, log_every: int = 5000
               ) -> Dict[str, float]:
    """`updates` IQL updates on learner C's OWN critic, target critic and actor from recorded rows,
    BEFORE its first online step (user 2026-11-14: "also pretrain critic with IQL first").

    Each update is IQLLearner.update on IQLLearner.on_learner(learner) -- V, then the actor, then
    Q, by FastRLAP's rules -- so nothing is handed over afterwards: the SAC loop simply continues
    on the same networks. What does NOT carry over is optimizer state: the learner's own
    optimizers, temperature and entropy target are untouched (recorded rows never tune the
    temperature, user rule 2026-11-07), and the value net and the warm-up's three optimizers are
    discarded on return.

    `sample()` -> a batch in the replay's format ({"observations": {...}, "next_observations":
    {...}, "actions", "rewards", "masks"}, as FeatureReplay and ExpertReplay give it). Returns the
    last update's info. With `log`, a line "[IQL] u=<u>/<n> key=value ..." goes to it every
    `log_every` updates and at the end. updates <= 0 does nothing and returns {}.
    """
    n = int(updates)
    if n <= 0:
        return {}
    iql = IQLLearner.on_learner(learner, expectile=expectile, temperature=temperature,
                                awr_clip=awr_clip, actor_lr=actor_lr, critic_lr=critic_lr,
                                value_lr=value_lr, hidden_dims=hidden_dims)
    every = max(1, int(log_every))
    info: Dict[str, float] = {}
    for u in range(1, n + 1):
        b = _to(sample(), learner.device)
        for key in ("actions", "rewards", "masks"):   # float32, as learner C reads them
            b[key] = b[key].float()
        info = iql.update(b)
        if log is not None and (u % every == 0 or u == n):
            log(f"[IQL] u={u}/{n} " + " ".join(f"{k}={v:.4g}" for k, v in info.items()))
    for opt in (iql.actor_opt, iql.critic_opt):           # no gradient outlives the warm-up
        opt.zero_grad(set_to_none=True)
    return info
