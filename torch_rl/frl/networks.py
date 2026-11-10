"""Networks of learner C: a PyTorch port of jaxrl5 as used by FastRLAP.

Reference (kylestach/fastrlap-release, jaxrl5/jaxrl5/): networks/mlp.py, state_action_value.py,
ensemble.py, encoders/d4pg_encoder.py, pixel_multiplexer.py, distributions/tanh_normal.py.

Matched details, each from the reference:
  * every Dense / Conv kernel Xavier-uniform, every bias zero (``default_init = xavier_uniform``;
    flax biases default to zeros);
  * MLP hidden layer = Dense -> (Dropout) -> (LayerNorm) -> ReLU, the last layer activated only
    with ``activate_final``; LayerNorm eps 1e-6 (flax's default, not PyTorch's 1e-5);
  * a Q head = MLP(activate_final=True) on [obs, action] -> Dense(1);
  * the actor = MLP(activate_final=True, no LayerNorm) -> Dense mean, Dense log-std clipped to
    [-20, 2] -> tanh-squashed diagonal Gaussian;
  * the d4pg encoder = Conv-ReLU per layer, VALID padding, flattened;
  * the multiplexer: pixels / 255 (always) -> encoder -> (stop-gradient for the actor) ->
    Dense(latent) -> LayerNorm -> tanh; precomputed embeddings skip the encoder; states get their
    own Dense(latent) -> LayerNorm -> tanh and are concatenated.
"""
import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LN_EPS = 1e-6           # flax.linen.LayerNorm default
LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def xavier_(module: nn.Module) -> nn.Module:
    """Xavier-uniform kernels and zero biases on every Linear / Conv2d inside `module`."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return module


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], activate_final: bool = False,
                 use_layer_norm: bool = False, dropout_rate: Optional[float] = None):
        super().__init__()
        layers, d = [], int(in_dim)
        for i, size in enumerate(hidden_dims):
            layers.append(nn.Linear(d, int(size)))
            d = int(size)
            if i + 1 < len(hidden_dims) or activate_final:
                if dropout_rate:
                    layers.append(nn.Dropout(float(dropout_rate)))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(d, eps=LN_EPS))
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.out_dim = d
        xavier_(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnsembleQ(nn.Module):
    """`num_qs` independent Q heads on [obs, action] -> (num_qs, batch)."""

    def __init__(self, obs_dim: int, action_dim: int, num_qs: int = 10,
                 hidden_dims: Sequence[int] = (256, 256), use_layer_norm: bool = False,
                 dropout_rate: Optional[float] = None):
        super().__init__()
        self.num_qs = int(num_qs)
        self.heads = nn.ModuleList()
        for _ in range(self.num_qs):
            base = MLP(obs_dim + action_dim, hidden_dims, activate_final=True,
                       use_layer_norm=use_layer_norm, dropout_rate=dropout_rate)
            self.heads.append(nn.Sequential(base, xavier_(nn.Linear(base.out_dim, 1))))

    def forward(self, obs: torch.Tensor, action: torch.Tensor,
                heads: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        idx = range(self.num_qs) if heads is None else [int(i) for i in heads]
        return torch.stack([self.heads[i](x).squeeze(-1) for i in idx], dim=0)


def _log1m_tanh_sq(u: torch.Tensor) -> torch.Tensor:
    """log(1 - tanh(u)^2), numerically stable (as TFP's tanh bijector)."""
    return 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))


class TanhNormalActor(nn.Module):
    def __init__(self, in_dim: int, action_dim: int, hidden_dims: Sequence[int] = (256, 256)):
        super().__init__()
        self.base = MLP(in_dim, hidden_dims, activate_final=True)
        self.mean = xavier_(nn.Linear(self.base.out_dim, action_dim))
        self.log_std = xavier_(nn.Linear(self.base.out_dim, action_dim))

    def dist_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.base(x)
        return self.mean(h), self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (action in (-1, 1), log pi(action), deterministic action tanh(mean))."""
        mean, log_std = self.dist_params(x)
        std = log_std.exp()
        u = mean + std * torch.randn_like(mean)                      # reparameterised
        log_prob = (-0.5 * ((u - mean) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
                    - _log1m_tanh_sq(u)).sum(-1)
        return torch.tanh(u), log_prob, torch.tanh(mean)


class D4PGEncoder(nn.Module):
    def __init__(self, in_channels: int, features: Sequence[int] = (32, 32, 32, 32),
                 filters: Sequence[int] = (3, 3, 3, 3), strides: Sequence[int] = (2, 2, 2, 2)):
        super().__init__()
        if not len(features) == len(filters) == len(strides):
            raise ValueError("features, filters and strides must have the same length")
        layers, c = [], int(in_channels)
        for f, k, s in zip(features, filters, strides):
            layers += [nn.Conv2d(c, int(f), int(k), stride=int(s), padding=0), nn.ReLU()]
            c = int(f)
        self.net = nn.Sequential(*layers)
        self._spec = (tuple(features), tuple(filters), tuple(strides))
        xavier_(self)

    def out_dim(self, h: int, w: int) -> int:
        features, filters, strides = self._spec
        for k, s in zip(filters, strides):
            h, w = (h - k) // s + 1, (w - k) // s + 1
        return int(features[-1]) * h * w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


def _latent(in_dim: int, latent_dim: int) -> nn.Sequential:
    return nn.Sequential(xavier_(nn.Linear(in_dim, latent_dim)),
                         nn.LayerNorm(latent_dim, eps=LN_EPS), nn.Tanh())


class Multiplexer(nn.Module):
    """jaxrl5's PixelMultiplexer for ONE image key: pixels (B, C, H, W) or precomputed
    embeddings (B, E), plus optional states (B, S) -> a tanh latent (B, latent [+ latent])."""

    def __init__(self, pixel_shape: Optional[Tuple[int, int, int]] = None,
                 embedding_dim: Optional[int] = None, state_dim: int = 0, latent_dim: int = 50,
                 stop_gradient: bool = False, encoder_kwargs: Optional[Dict] = None):
        super().__init__()
        if pixel_shape is None and not embedding_dim:
            raise ValueError("Multiplexer needs pixels or embeddings; for states alone, "
                             "feed the raw state to the networks (jaxrl5 SACLearner does)")
        self.stop_gradient = bool(stop_gradient)
        if pixel_shape is not None:
            c, h, w = pixel_shape
            self.encoder = D4PGEncoder(c, **(encoder_kwargs or {}))
            self.project = _latent(self.encoder.out_dim(h, w), latent_dim)
        else:
            self.encoder = None
            self.project = _latent(int(embedding_dim), latent_dim)
        self.state_project = _latent(int(state_dim), latent_dim) if state_dim else None
        self.out_dim = latent_dim * (2 if state_dim else 1)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.encoder is not None:
            x = self.encoder(obs["pixels"].float() / 255.0)
        else:
            x = obs["embeddings"].float()
        if self.stop_gradient:
            x = x.detach()                    # the actor never trains the conv layers
        z = self.project(x)
        if self.state_project is not None:
            z = torch.cat([z, self.state_project(obs["states"].float())], dim=-1)
        return z
