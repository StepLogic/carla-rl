"""The offline steering test's FastRLAP arm (user 2026-11-10: "Compressed to 50-d, compression
trained by the critic (FastRLAP)" -- "only this run").

scripts/offline_priv_critic.py's protocol is unchanged (data, no_arrival cut, rewards x10, batch,
updates, probe, c0's frozen actor as the target's next action). Only the critic changes: learner
C's own critic (torch_rl.frl) in FastRLAP's embeddings mode -- c0's frozen [GRU | binocular]
features through a Dense(50) -> LayerNorm -> tanh that the critic loss trains, [mode, speed]
through its own Dense(50) -> LayerNorm -> tanh, 10 LayerNorm Q heads, min over 2 target heads,
no entropy term in the target. These tests pin that glue; the learning code is learner C's."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")]
torch = pytest.importorskip("torch")

import offline_priv_critic as opc  # noqa: E402

H, MS, B = 512, 2, 3136


def _rows(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, H + MS + B, generator=g)


def test_an_image_row_splits_into_the_multiplexers_embeddings_and_states():
    x = _rows()
    obs = opc.split_image_row(x)
    assert torch.equal(obs["embeddings"], torch.cat([x[:, :H], x[:, H + MS:]], 1))
    assert torch.equal(obs["states"], x[:, H:H + MS])                 # [mode, speed]
    assert obs["embeddings"].shape == (4, H + B)


def test_the_arm_is_learner_cs_critic_with_fastrlaps_rules():
    L = opc.frl_critic(emb_dim=H + B, state_dim=MS, device="cpu", seed=0)
    proj = L.critic.trunk.project[0]
    assert (proj.in_features, proj.out_features) == (H + B, 50)        # compressed to 50-d
    assert L.critic.trunk.state_project[0].out_features == 50
    assert L.num_qs == 10 and L.num_min_qs == 2
    assert L.backup_entropy is False                                    # FastRLAP's target
    assert (L.discount, L.tau) == (opc.GAMMA, opc.TAU)
    assert L.critic_opt.param_groups[0]["lr"] == opc.LR
    assert any(isinstance(m, torch.nn.LayerNorm) for m in L.critic.q.modules())


def test_the_target_next_action_is_c0s_frozen_actor():
    actor = opc.C0NextActor(torch.Generator().manual_seed(0))
    mean = torch.tensor([[0.3, -0.2], [-1.0, 0.5]])
    a, logp, det = actor.sample({"c0_mean_u": mean, "c0_log_std": torch.full_like(mean, -20.0)})
    assert torch.allclose(a, torch.tanh(mean), atol=1e-6)               # std ~0: c0's mean
    assert torch.equal(det, torch.tanh(mean)) and logp.shape == (2,)


def test_a_critic_update_trains_the_compression_and_leaves_the_target_entropy_free():
    L = opc.frl_critic(emb_dim=H + B, state_dim=MS, device="cpu", seed=0)
    L.actor = opc.C0NextActor(torch.Generator().manual_seed(1))
    before = L.critic.trunk.project[0].weight.detach().clone()
    x, xn = _rows(8, 1), _rows(8, 2)
    nxt = opc.split_image_row(xn)
    nxt["c0_mean_u"] = torch.zeros(8, 2)
    nxt["c0_log_std"] = torch.zeros(8, 2)
    mb = {"observations": opc.split_image_row(x), "next_observations": nxt,
          "actions": torch.rand(8, 2) * 2 - 1, "rewards": torch.ones(8), "masks": torch.ones(8)}
    info = L.update_critic(mb)
    assert np.isfinite(info["critic_loss"])
    assert not torch.equal(before, L.critic.trunk.project[0].weight)  # the critic trains it


def test_the_measure_view_gives_measure_the_quantile_layout():
    L = opc.frl_critic(emb_dim=H + B, state_dim=MS, device="cpu", seed=0)
    q = opc.MeasureView(L.critic)(_rows(5), torch.zeros(5, 2))
    assert q.shape == (10, 5, 1)                                        # (heads, batch, 1)
