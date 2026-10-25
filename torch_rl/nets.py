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
                 log_std_bounds: Tuple[float, float] = (LOG_STD_MIN, LOG_STD_MAX)) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.lo, self.hi = float(log_std_bounds[0]), float(log_std_bounds[1])
        self.trunk = _mlp(state_dim, hidden, 2 * self.action_dim)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.trunk(s).chunk(2, dim=-1)
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
        noise_scale = float(noise_scale)
        if noise_scale < 0.0:
            raise ValueError(f"noise_scale={noise_scale} must be >= 0")
        mean, log_std = self(s)
        std = log_std.exp()
        noise = torch.randn_like(mean) * noise_scale
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
                 n_quantiles: int = 25, hidden: Sequence[int] = (256, 256)) -> None:
        super().__init__()
        self.n_critics, self.n_quantiles = int(n_critics), int(n_quantiles)
        dims = [int(state_dim) + int(action_dim)] + list(hidden) + [self.n_quantiles]
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

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)[None].expand(self.n_critics, -1, -1)
        for k, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = torch.einsum("nbi,nio->nbo", x, w) + b[:, None, :]
            if k < len(self.weights) - 1:
                x = torch.relu(x)
        return x

    def single_forward(self, i: int, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        for k, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w[i] + b[i]
            if k < len(self.weights) - 1:
                x = torch.relu(x)
        return x


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.mul_(1.0 - tau).add_(tau * sp)
    for tb, sb in zip(target.buffers(), source.buffers()):
        tb.copy_(sb)
