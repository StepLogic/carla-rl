"""Learner C: a PyTorch port of jaxrl5's SACLearner and DrQLearner as FastRLAP uses them.

Reference (kylestach/fastrlap-release, jaxrl5/jaxrl5/agents/): sac/sac_learner.py,
sac/temperature.py, drq/drq_learner.py, drq/augmentations.py; configs in
fastrlap-ros/offroad_learning/config/{redq,drq}_config.py.

The update, as the reference does it:
  * the batch is split into `utd_ratio` contiguous minibatches; the critic is updated on each,
    with a Polyak step of the target after every critic update; the actor, then the temperature,
    are updated once, on the LAST minibatch;
  * critic target: next action sampled from the current actor; `num_min_qs` of the TARGET heads
    drawn without replacement once per critic update; y = r + gamma * mask * min(Q'); the
    entropy term -gamma * mask * alpha * log pi(a'|s') only with `backup_entropy`;
  * critic loss: mean over heads and batch of (Q_i - y)^2;
  * actor loss: mean(alpha * log pi(a|s) - MEAN over ALL heads Q(s, a)), the critic fixed;
  * temperature: a log-alpha parameter, loss alpha * (H - H_target), H = -mean log pi of the
    actor step; target entropy defaults to -action_dim / 2;
  * DrQ: before every update the critic's conv encoder is copied into the actor (whose encoder
    output is stop-gradient), then observations and next observations get independent random
    crops (pad 4, edge values).
"""
import copy
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from torch_rl.frl.networks import EnsembleQ, Multiplexer, TanhNormalActor

Obs = Union[torch.Tensor, Dict[str, torch.Tensor]]


# ---- the pure rules -------------------------------------------------------------------------

def subsample_heads(num_qs: int, k: Optional[int], generator: Optional[torch.Generator] = None
                    ) -> torch.Tensor:
    """`k` distinct head indices out of `num_qs` (REDQ's subset); all heads when k is None."""
    if k is None:
        return torch.arange(int(num_qs))
    if not 1 <= int(k) <= int(num_qs):
        raise ValueError(f"num_min_qs={k} must be in 1..num_qs={num_qs}")
    return torch.randperm(int(num_qs), generator=generator)[: int(k)]


def critic_target(next_qs: torch.Tensor, rewards: torch.Tensor, masks: torch.Tensor,
                  discount: float, alpha: float, next_log_probs: torch.Tensor,
                  backup_entropy: bool) -> torch.Tensor:
    """next_qs (k, B) from the chosen target heads -> y (B,)."""
    y = rewards + discount * masks * next_qs.min(dim=0).values
    if backup_entropy:
        y = y - discount * masks * alpha * next_log_probs
    return y


def actor_loss(log_probs: torch.Tensor, qs: torch.Tensor, alpha: float) -> torch.Tensor:
    """qs (num_qs, B): the actor maximises the MEAN over all heads."""
    return (alpha * log_probs - qs.mean(dim=0)).mean()


def temperature_loss(log_temp: torch.Tensor, entropy: float, target_entropy: float
                     ) -> torch.Tensor:
    return log_temp.exp() * (float(entropy) - float(target_entropy))


def random_crop(x: torch.Tensor, pad: int = 4, offsets: Optional[torch.Tensor] = None,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """DrQ's crop: pad H and W by `pad` with edge values, crop back at an offset in [0, 2*pad]
    drawn per sample. `offsets` (B, 2) = (row, col) overrides the draw (tests). Any dtype."""
    n, _, h, w = x.shape
    if offsets is None:
        offsets = torch.randint(0, 2 * pad + 1, (n, 2), generator=generator)
    offsets = offsets.to(x.device)
    iy = (offsets[:, 0:1] + torch.arange(h, device=x.device) - pad).clamp(0, h - 1)    # (n, h)
    ix = (offsets[:, 1:2] + torch.arange(w, device=x.device) - pad).clamp(0, w - 1)    # (n, w)
    rows = torch.gather(x, 2, iy[:, None, :, None].expand(n, x.shape[1], h, w))
    return torch.gather(rows, 3, ix[:, None, None, :].expand(n, x.shape[1], h, w))


def _slice(batch, lo: int, hi: int):
    if isinstance(batch, dict):
        return {k: _slice(v, lo, hi) for k, v in batch.items()}
    return batch[lo:hi]


def _to(obj, device):
    if isinstance(obj, dict):
        return {k: _to(v, device) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)
    return obj.to(device)


# ---- the modules ----------------------------------------------------------------------------

class _Critic(nn.Module):
    def __init__(self, trunk: nn.Module, in_dim: int, action_dim: int, num_qs: int,
                 hidden_dims: Sequence[int], use_layer_norm: bool, dropout_rate):
        super().__init__()
        self.trunk = trunk
        self.q = EnsembleQ(in_dim, action_dim, num_qs, hidden_dims, use_layer_norm, dropout_rate)

    def forward(self, obs: Obs, action: torch.Tensor, heads=None) -> torch.Tensor:
        return self.q(self.trunk(obs), action, heads=heads)


class _Actor(nn.Module):
    def __init__(self, trunk: nn.Module, in_dim: int, action_dim: int,
                 hidden_dims: Sequence[int]):
        super().__init__()
        self.trunk = trunk
        self.pi = TanhNormalActor(in_dim, action_dim, hidden_dims)

    def sample(self, obs: Obs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.pi.sample(self.trunk(obs))


class SACLearner:
    """jaxrl5 SACLearner: raw states (`obs_dim`), or precomputed embeddings (`embedding_dim`,
    optional `state_dim`) through the multiplexer's Dense -> LayerNorm -> tanh latent."""

    def __init__(self, obs_dim: Optional[int] = None, action_dim: int = 2, *,
                 embedding_dim: Optional[int] = None, state_dim: int = 0, latent_dim: int = 50,
                 actor_lr: float = 3e-4, critic_lr: float = 3e-4, temp_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256), discount: float = 0.99,
                 tau: float = 0.005, num_qs: int = 2, num_min_qs: Optional[int] = None,
                 critic_dropout_rate: Optional[float] = None, critic_layer_norm: bool = False,
                 target_entropy: Optional[float] = None, init_temperature: float = 1.0,
                 backup_entropy: bool = True, seed: int = 0, device: str = "cpu",
                 **trunk_kwargs):
        self.device = torch.device(device)
        torch.manual_seed(int(seed))                  # reproducible initialisation
        self.gen = torch.Generator().manual_seed(int(seed))
        self.action_dim = int(action_dim)
        self.discount, self.tau = float(discount), float(tau)
        self.num_qs = int(num_qs)
        self.num_min_qs = None if num_min_qs is None else int(num_min_qs)
        self.backup_entropy = bool(backup_entropy)
        self.target_entropy = (-self.action_dim / 2.0 if target_entropy is None
                               else float(target_entropy))
        actor_trunk, critic_trunk, in_dim = self._make_trunks(
            obs_dim=obs_dim, embedding_dim=embedding_dim, state_dim=state_dim,
            latent_dim=latent_dim, **trunk_kwargs)
        self.actor = _Actor(actor_trunk, in_dim, self.action_dim, hidden_dims).to(self.device)
        self.critic = _Critic(critic_trunk, in_dim, self.action_dim, self.num_qs, hidden_dims,
                              critic_layer_norm, critic_dropout_rate).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)
        self.log_temp = torch.tensor(float(np.log(init_temperature)), device=self.device,
                                     requires_grad=True)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.temp_opt = torch.optim.Adam([self.log_temp], lr=temp_lr)

    def _make_trunks(self, obs_dim, embedding_dim, state_dim, latent_dim):
        if embedding_dim:
            a = Multiplexer(embedding_dim=embedding_dim, state_dim=state_dim,
                            latent_dim=latent_dim, stop_gradient=True)
            c = Multiplexer(embedding_dim=embedding_dim, state_dim=state_dim,
                            latent_dim=latent_dim, stop_gradient=True)
            return a, c, a.out_dim
        if obs_dim is None:
            raise ValueError("SACLearner needs obs_dim (raw states) or embedding_dim")
        return nn.Identity(), nn.Identity(), int(obs_dim)

    @property
    def alpha(self) -> float:
        return float(self.log_temp.detach().exp())

    # -- updates
    def update(self, batch: Dict, utd_ratio: int, update_temperature: bool = True) -> Dict:
        batch = _to(batch, self.device)
        n = int(batch["rewards"].shape[0])
        if n % int(utd_ratio):
            raise ValueError(f"batch of {n} does not split into utd_ratio={utd_ratio} parts")
        bs = n // int(utd_ratio)
        for i in range(int(utd_ratio)):
            mb = _slice(batch, i * bs, (i + 1) * bs)
            critic_info = self.update_critic(mb)
        actor_info = self.update_actor(mb)                 # the LAST minibatch, as the reference
        # NaN entropy = the reference minibatch held no rows of the agent's own (all expert rows
        # pinned in the replay): the temperature is left alone for this update.
        temp_info = (self.update_temperature(actor_info["entropy"])
                     if update_temperature and not np.isnan(actor_info["entropy"]) else {})
        return {**actor_info, **critic_info, **temp_info}

    def update_critic(self, mb: Dict) -> Dict:
        with torch.no_grad():
            next_a, next_logp, _ = self.actor.sample(mb["next_observations"])
            heads = subsample_heads(self.num_qs, self.num_min_qs, self.gen)
            next_qs = self.target_critic(mb["next_observations"], next_a, heads=heads)
            y = critic_target(next_qs, mb["rewards"].float(), mb["masks"].float(),
                              self.discount, self.alpha, next_logp, self.backup_entropy)
        qs = self.critic(mb["observations"], mb["actions"].float())
        loss = ((qs - y[None, :]) ** 2).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_opt.step()
        self._polyak()
        return {"critic_loss": float(loss), "q": float(qs.mean())}

    def update_actor(self, mb: Dict) -> Dict:
        crit = list(self.critic.parameters())
        for p in crit:
            p.requires_grad_(False)                        # gradients reach the actor only
        try:
            a, logp, _ = self.actor.sample(mb["observations"])
            qs = self.critic(mb["observations"], a)
            loss = actor_loss(logp, qs, self.alpha)
            self.actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.actor_opt.step()
        finally:
            for p in crit:
                p.requires_grad_(True)
        # The entropy that tunes the temperature: over the minibatch, or -- when the batch marks
        # which rows the agent collected itself ("online", expert rows pinned in the replay) --
        # over its own rows only: recorded rows never move the temperature (user rule 2026-11-07).
        lp = logp.detach()
        if "online" in mb:
            lp = lp[mb["online"].to(lp.device).bool()]
        entropy = float(-lp.mean()) if lp.numel() else float("nan")
        return {"actor_loss": float(loss), "entropy": entropy}

    def update_temperature(self, entropy: float) -> Dict:
        loss = temperature_loss(self.log_temp, entropy, self.target_entropy)
        self.temp_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.temp_opt.step()
        return {"temperature": self.alpha, "temperature_loss": float(loss)}

    @torch.no_grad()
    def _polyak(self) -> None:
        for p, pt in zip(self.critic.parameters(), self.target_critic.parameters()):
            pt.mul_(1.0 - self.tau).add_(self.tau * p)

    # -- acting
    @torch.no_grad()
    def sample_actions(self, obs) -> np.ndarray:
        return self.actor.sample(_to(obs, self.device))[0].cpu().numpy()

    @torch.no_grad()
    def eval_actions(self, obs) -> np.ndarray:
        return self.actor.sample(_to(obs, self.device))[2].cpu().numpy()


class DrQLearner(SACLearner):
    """jaxrl5 DrQLearner: pixels (B, C, H, W) through a d4pg encoder owned by the critic, copied
    into the actor before every update, with random-crop augmentation of both observations."""

    def __init__(self, pixel_shape: Tuple[int, int, int], action_dim: int = 2, *,
                 state_dim: int = 0, latent_dim: int = 50, encoder_kwargs: Optional[Dict] = None,
                 crop_pad: int = 4, **sac_kwargs):
        self.crop_pad = int(crop_pad)
        super().__init__(None, action_dim, state_dim=state_dim, latent_dim=latent_dim,
                         pixel_shape=pixel_shape, encoder_kwargs=encoder_kwargs, **sac_kwargs)

    def _make_trunks(self, obs_dim, embedding_dim, state_dim, latent_dim, pixel_shape=None,
                     encoder_kwargs=None):
        a = Multiplexer(pixel_shape=pixel_shape, state_dim=state_dim, latent_dim=latent_dim,
                        stop_gradient=True, encoder_kwargs=encoder_kwargs)
        c = Multiplexer(pixel_shape=pixel_shape, state_dim=state_dim, latent_dim=latent_dim,
                        stop_gradient=False, encoder_kwargs=encoder_kwargs)
        return a, c, c.out_dim

    def _share_encoder(self) -> None:
        self.actor.trunk.encoder.load_state_dict(self.critic.trunk.encoder.state_dict())

    def _augment(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(obs)
        out["pixels"] = random_crop(obs["pixels"], self.crop_pad, generator=self.gen)
        return out

    def update(self, batch: Dict, utd_ratio: int, update_temperature: bool = True) -> Dict:
        self._share_encoder()
        batch = dict(_to(batch, self.device))
        batch["observations"] = self._augment(batch["observations"])
        batch["next_observations"] = self._augment(batch["next_observations"])
        return SACLearner.update(self, batch, utd_ratio, update_temperature)
