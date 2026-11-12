"""Learner C networks (torch_rl/frl, plan 2026-11-10): a PyTorch port of jaxrl5 as used by
FastRLAP (kylestach/fastrlap-release): networks/mlp.py, state_action_value.py, ensemble.py,
distributions/tanh_normal.py, encoders/d4pg_encoder.py, pixel_multiplexer.py."""
import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from torch_rl.frl.networks import (MLP, D4PGEncoder, EnsembleQ, Multiplexer,  # noqa: E402
                                   TanhNormalActor)


def test_mlp_orders_dense_layernorm_relu_and_activates_the_final_layer_only_if_asked():
    m = MLP(5, (7, 3), activate_final=False, use_layer_norm=True)
    kinds = [type(x).__name__ for x in m.net]
    # hidden: Linear -> LayerNorm -> ReLU ; final: Linear only
    assert kinds == ["Linear", "LayerNorm", "ReLU", "Linear"]
    m2 = MLP(5, (7, 3), activate_final=True, use_layer_norm=False)
    assert [type(x).__name__ for x in m2.net] == ["Linear", "ReLU", "Linear", "ReLU"]


def test_dense_layers_are_xavier_uniform_with_zero_bias():
    torch.manual_seed(0)
    m = MLP(400, (300,), activate_final=True)
    lin = m.net[0]
    bound = math.sqrt(6.0 / (400 + 300))
    assert float(lin.weight.abs().max()) <= bound + 1e-6
    assert float(lin.weight.abs().max()) > 0.9 * bound          # uniform, not normal/kaiming
    assert torch.count_nonzero(lin.bias) == 0


def test_ensemble_returns_one_value_per_head_and_heads_differ():
    torch.manual_seed(0)
    q = EnsembleQ(obs_dim=6, action_dim=2, num_qs=10, hidden_dims=(32, 32), use_layer_norm=True)
    out = q(torch.randn(5, 6), torch.randn(5, 2))
    assert out.shape == (10, 5)
    assert float(out.std(dim=0).mean()) > 0                     # independent initialisations
    sub = q(torch.randn(5, 6), torch.randn(5, 2), heads=torch.tensor([3, 7]))
    assert sub.shape == (2, 5)


def test_tanh_normal_actor_samples_in_bounds_with_the_change_of_variables_log_prob():
    torch.manual_seed(0)
    a = TanhNormalActor(4, action_dim=2, hidden_dims=(16, 16))
    obs = torch.randn(64, 4)
    act, logp, mean_act = a.sample(obs)
    assert act.shape == (64, 2) and logp.shape == (64,)
    assert float(act.abs().max()) < 1.0
    mean, log_std = a.dist_params(obs)
    assert float(log_std.min()) >= -20 and float(log_std.max()) <= 2
    assert torch.allclose(mean_act, torch.tanh(mean))
    # log pi(a) = log N(u; mean, std) - sum log(1 - tanh(u)^2), u = atanh(a)
    u = torch.atanh(act.clamp(-1 + 1e-6, 1 - 1e-6))
    ref = (torch.distributions.Normal(mean, log_std.exp()).log_prob(u)
           - torch.log(1 - act ** 2 + 1e-12)).sum(-1)
    assert torch.allclose(logp, ref, atol=2e-3)


def test_d4pg_encoder_matches_valid_conv_arithmetic():
    enc = D4PGEncoder(in_channels=9, features=(32, 32, 32, 32), filters=(3, 3, 3, 3),
                      strides=(2, 2, 2, 2))
    out = enc(torch.zeros(2, 9, 84, 84))
    # 84 -> 41 -> 20 -> 9 -> 4 with VALID 3x3 stride 2
    assert out.shape == (2, 32 * 4 * 4)
    assert enc.out_dim(84, 84) == 512


def test_multiplexer_scales_pixels_projects_to_a_tanh_latent_and_concatenates_states():
    torch.manual_seed(0)
    mux = Multiplexer(pixel_shape=(9, 84, 84), state_dim=3, latent_dim=50)
    obs = {"pixels": torch.full((2, 9, 84, 84), 255, dtype=torch.uint8),
           "states": torch.randn(2, 3)}
    z = mux(obs)
    assert z.shape == (2, 100)                                   # 50 pixel + 50 state latent
    assert float(z.abs().max()) <= 1.0
    # pixels ALWAYS enter as x / 255 (jaxrl5 casts to float32 and divides, whatever the dtype)
    z_float = mux({"pixels": torch.full((2, 9, 84, 84), 255.0), "states": obs["states"]})
    assert torch.allclose(z, z_float, atol=1e-6)


def test_multiplexer_stop_gradient_keeps_actor_loss_out_of_the_encoder():
    torch.manual_seed(0)
    mux = Multiplexer(pixel_shape=(3, 84, 84), state_dim=0, latent_dim=8, stop_gradient=True)
    mux({"pixels": torch.rand(2, 3, 84, 84) * 255}).sum().backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in mux.encoder.parameters())
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in mux.project.parameters())


def test_multiplexer_embeddings_mode_projects_precomputed_features():
    mux = Multiplexer(embedding_dim=64, state_dim=0, latent_dim=50)
    assert mux({"embeddings": torch.randn(4, 64)}).shape == (4, 50)
    with pytest.raises(ValueError):
        Multiplexer(state_dim=3, latent_dim=50)                  # nothing to encode: use raw states
