#!/usr/bin/env python3
"""Networks for the recurrent TQC: shared GRU, squashed-Gaussian actor, and a
stacked quantile-critic ensemble (one batched matmul instead of N small ones).

The GRU is nn.GRU rather than a GRUCell loop on purpose: chunks never span two
episodes (torch_rl/episode_buffer.py), so there is no mid-chunk hidden reset and
cuDNN's fused multi-step kernel applies.
"""
import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class GRUCore(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.rnn = nn.GRU(int(in_dim), self.hidden, num_layers=1, batch_first=True)

    def forward(self, seq: torch.Tensor, h0: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """seq (B, L, in_dim) -> (B, L, hidden), h (1, B, hidden)."""
        return self.rnn(seq, h0)

    def step(self, x: torch.Tensor, h: Optional[torch.Tensor] = None
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One step: x (B, in_dim) -> (B, hidden), h (1, B, hidden)."""
        out, h = self.rnn(x[:, None, :], h)
        return out[:, 0], h


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: Sequence[int] = (256, 256),
                 log_std_bounds: Tuple[float, float] = (LOG_STD_MIN, LOG_STD_MAX),
                 in_cols: Optional[Sequence[int]] = None) -> None:
        """`in_cols` (privileged critic, 2026-11-07): the input columns the actor
        reads, taken from whatever wider vector the caller hands it; `state_dim`
        is then their count. A non-persistent buffer, so the state dict -- and
        every checkpoint -- is the same either way. None = read it all."""
        super().__init__()
        if in_cols is None:
            self.in_cols_t = None
        else:
            self.register_buffer("in_cols_t", torch.as_tensor(list(in_cols), dtype=torch.long),
                                 persistent=False)
        self.action_dim = int(action_dim)
        lo = torch.as_tensor(log_std_bounds[0], dtype=torch.float32)
        hi = torch.as_tensor(log_std_bounds[1], dtype=torch.float32)
        self.per_dim = bool(lo.dim() or hi.dim())
        if not self.per_dim:
            self.lo, self.hi = float(lo), float(hi)
        else:
            # One bound per action (ACTOR_LOG_STD="-3.70,-0.43"). Non-persistent
            # buffers: they follow .to(device) but stay out of the state dict, so
            # every checkpoint loads either way.
            self.lo = self.hi = None
            self.register_buffer("lo_t", lo.expand(self.action_dim).clone(), persistent=False)
            self.register_buffer("hi_t", hi.expand(self.action_dim).clone(), persistent=False)
        self.trunk = _mlp(state_dim, hidden, 2 * self.action_dim)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.in_cols_t is not None:
            s = s.index_select(-1, self.in_cols_t)
        mean, log_std = self.trunk(s).chunk(2, dim=-1)
        if self.per_dim:
            return mean, torch.minimum(torch.maximum(log_std, self.lo_t.to(log_std.dtype)),
                                       self.hi_t.to(log_std.dtype))
        return mean, log_std.clamp(self.lo, self.hi)

    def sample(self, s: torch.Tensor, noise_scale: float = 1.0
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """`noise_scale` shrinks the draw towards the mean: 1.0 is training's
        own sampling, 0.0 is tanh(mean) exactly. It exists for the eval, which
        scores tanh(mean) while the return was earned by samples; sweeping the
        scale separates "the mean action is bad" from "the task is hard".
        The returned log_prob is the density of the UNSCALED policy at the drawn
        point, so it is only the behaviour log-density at scale 1.0 -- nothing
        but the learner (always scale 1.0) consumes it.
        """
        if torch.is_tensor(noise_scale):
            # PER ROW (noise-free training episodes): (N, 1) or (N,) against
            # an (N, action_dim) mean; a 0 row is exactly tanh(mean).
            if bool((noise_scale < 0).any()):
                raise ValueError("noise_scale must be >= 0 in every row")
            if noise_scale.dim() == 1:
                noise_scale = noise_scale[:, None]
        else:
            noise_scale = float(noise_scale)
            if noise_scale < 0.0:
                raise ValueError(f"noise_scale={noise_scale} must be >= 0")
        mean, log_std = self(s)
        std = log_std.exp()
        noise = torch.randn_like(mean) * (noise_scale.to(mean) if torch.is_tensor(noise_scale)
                                          else noise_scale)
        pre = mean + std * noise
        action = torch.tanh(pre)
        log_prob = (-0.5 * noise ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        log_prob = log_prob - torch.log(1.0 - action ** 2 + 1e-6).sum(-1)
        return action, log_prob

    def deterministic(self, s: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self(s)[0])


class QuantileCritics(nn.Module):
    """N critics as stacked parameters: forward is einsum over the critic axis."""

    def __init__(self, state_dim: int, action_dim: int, n_critics: int = 5,
                 n_quantiles: int = 25, hidden: Sequence[int] = (256, 256),
                 dropout: float = 0.0, layer_norm: bool = False,
                 proj_dim: int = 0) -> None:
        """DroQ-style regularisation (dropout, layer_norm) is off by default.

        Linear -> Dropout -> LayerNorm -> ReLU on every HIDDEN layer, the
        paper's order, and the part that makes a high replay ratio stable --
        this arm already had REDQ (min over a random subset of the ensemble).
        Both are PER CRITIC: the ensemble is stacked parameters, so the norm's
        affine terms carry a critic axis as well, or all heads would share one
        normalisation and stop being independent estimators.

        `proj_dim` > 0 (user 2026-11-09): DrQ's / SAC-AE's input head, per
        critic -- the state goes through Linear(state_dim -> proj_dim) ->
        LayerNorm -> tanh and the ACTION is concatenated to that feature, so
        the action is 2 of proj_dim + 2 first-layer inputs instead of 2 of
        state_dim + 2 (3652 here, where it drove < 0.4% of the layer's input).
        0 = every earlier critic, parameter for parameter.
        """
        super().__init__()
        self.n_critics, self.n_quantiles = int(n_critics), int(n_quantiles)
        self.dropout, self.layer_norm = float(dropout), bool(layer_norm)
        self.proj_dim = int(proj_dim)
        if self.proj_dim > 0:
            pw = torch.empty(self.n_critics, int(state_dim), self.proj_dim)
            for i in range(self.n_critics):           # per critic, read as (d_in, d_out)
                nn.init.kaiming_uniform_(pw[i].T, a=math.sqrt(5))
            self.proj_w = nn.Parameter(pw)
            self.proj_b = nn.Parameter(torch.zeros(self.n_critics, self.proj_dim))
            self.proj_ln_w = nn.Parameter(torch.ones(self.n_critics, self.proj_dim))
            self.proj_ln_b = nn.Parameter(torch.zeros(self.n_critics, self.proj_dim))
        in_state = self.proj_dim if self.proj_dim > 0 else int(state_dim)
        dims = [in_state + int(action_dim)] + list(hidden) + [self.n_quantiles]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            w = torch.empty(self.n_critics, d_in, d_out)
            b = torch.zeros(self.n_critics, d_out)
            for i in range(self.n_critics):           # per-critic init, so heads differ
                # w[i] is stored (d_in, d_out) for the batched einsum below, but
                # nn.init assumes a 2-D weight is (out_features, in_features) and
                # reads fan_in from dim 1. Init through the transposed view so
                # fan_in is read from d_in; without this every non-square layer
                # is scaled by the wrong fan (sqrt(d_out) instead of sqrt(d_in)).
                nn.init.kaiming_uniform_(w[i].T, a=math.sqrt(5))
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(b))
        if self.layer_norm:
            # One affine pair per HIDDEN layer (the head is left raw), each with
            # a critic axis.
            self.ln_w = nn.ParameterList(
                [nn.Parameter(torch.ones(self.n_critics, d)) for d in dims[1:-1]])
            self.ln_b = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.n_critics, d)) for d in dims[1:-1]])

    def _reg(self, x: torch.Tensor, k: int, idx: Optional[int] = None) -> torch.Tensor:
        """Dropout then LayerNorm on hidden layer k, for one critic or all."""
        if self.dropout > 0.0:
            x = nn.functional.dropout(x, self.dropout, self.training)
        if self.layer_norm:
            x = nn.functional.layer_norm(x, (x.shape[-1],))
            w, b = self.ln_w[k], self.ln_b[k]
            x = x * (w[idx] if idx is not None else w[:, None, :]) \
                + (b[idx] if idx is not None else b[:, None, :])
        return x

    def project(self, s: torch.Tensor, idx: Optional[int] = None) -> torch.Tensor:
        """DrQ's head: tanh(LayerNorm(s W + b)) per critic -> (n_critics, B, proj_dim),
        or (B, proj_dim) for critic `idx`."""
        if idx is None:
            z = torch.einsum("bi,nio->nbo", s, self.proj_w) + self.proj_b[:, None, :]
            w, b = self.proj_ln_w[:, None, :], self.proj_ln_b[:, None, :]
        else:
            z = s @ self.proj_w[idx] + self.proj_b[idx]
            w, b = self.proj_ln_w[idx], self.proj_ln_b[idx]
        z = nn.functional.layer_norm(z, (z.shape[-1],)) * w + b
        return torch.tanh(z)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if self.proj_dim > 0:
            x = torch.cat([self.project(s), a[None].expand(self.n_critics, -1, -1)], dim=-1)
        else:
            x = torch.cat([s, a], dim=-1)[None].expand(self.n_critics, -1, -1)
        for k, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = torch.einsum("nbi,nio->nbo", x, w) + b[:, None, :]
            if k < len(self.weights) - 1:
                x = torch.relu(self._reg(x, k))
        return x

    def single_forward(self, i: int, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.project(s, idx=i) if self.proj_dim > 0 else s, a], dim=-1)
        for k, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w[i] + b[i]
            if k < len(self.weights) - 1:
                x = torch.relu(self._reg(x, k, idx=i))
        return x


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.mul_(1.0 - tau).add_(tau * sp)
    for tb, sb in zip(target.buffers(), source.buffers()):
        tb.copy_(sb)
