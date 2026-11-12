"""Learner C update rules (torch_rl/frl/learner.py, plan 2026-11-10): a PyTorch port of jaxrl5's
SACLearner and DrQLearner (kylestach/fastrlap-release, jaxrl5/jaxrl5/agents/{sac,drq}/)."""
import copy
import math
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from torch_rl.frl.learner import (DrQLearner, SACLearner, actor_loss,  # noqa: E402
                                  critic_target, random_crop, subsample_heads,
                                  temperature_loss)


# ---- the pure rules -------------------------------------------------------------------------

def test_critic_target_min_over_the_subset_with_and_without_the_entropy_backup():
    next_qs = torch.tensor([[1.0, 5.0], [3.0, 2.0]])       # (k heads, batch)
    r, m, lp = torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.0]), torch.tensor([2.0, 2.0])
    y = critic_target(next_qs, r, m, discount=0.9, alpha=0.5, next_log_probs=lp, backup_entropy=False)
    assert torch.allclose(y, torch.tensor([1.9, 1.0]))     # r + g*m*min ; a terminal row is r
    y2 = critic_target(next_qs, r, m, discount=0.9, alpha=0.5, next_log_probs=lp, backup_entropy=True)
    assert torch.allclose(y2, torch.tensor([1.0, 1.0]))    # minus g*m*alpha*logp


def test_subsample_draws_distinct_heads():
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        idx = subsample_heads(10, 2, generator=g)
        assert idx.numel() == 2 and len(set(idx.tolist())) == 2
        assert 0 <= int(idx.min()) and int(idx.max()) < 10
    assert subsample_heads(10, None, generator=g).tolist() == list(range(10))   # no subset: all


def test_actor_loss_uses_the_mean_of_all_heads_not_a_min():
    qs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0]])
    lp = torch.tensor([1.0, 1.0])
    assert math.isclose(float(actor_loss(lp, qs, alpha=0.1)), -2.4, abs_tol=1e-6)


def test_temperature_loss_value_and_direction():
    log_temp = torch.tensor(math.log(0.5), requires_grad=True)
    loss = temperature_loss(log_temp, entropy=1.0, target_entropy=-1.0)
    assert math.isclose(float(loss), 1.0, abs_tol=1e-6)    # alpha * (H - H_target)
    loss.backward()
    assert float(log_temp.grad) > 0                         # entropy above target -> alpha falls


def test_random_crop_pads_with_edge_values_and_keeps_the_shape():
    x = torch.arange(2 * 1 * 6 * 6, dtype=torch.float32).reshape(2, 1, 6, 6)
    same = random_crop(x, pad=2, offsets=torch.tensor([[2, 2], [2, 2]]))
    assert torch.equal(same, x)                             # offset = pad is the identity
    corner = random_crop(x, pad=2, offsets=torch.tensor([[0, 0], [0, 0]]))
    assert corner.shape == x.shape
    assert torch.all(corner[:, :, :3, :3] == x[:, :, :1, :1])   # edge-replicated corner block
    g = torch.Generator().manual_seed(1)
    big = torch.rand(64, 3, 16, 16)
    crops = random_crop(big, pad=4, generator=g)
    shifts = {tuple(torch.nonzero(torch.isclose(crops[i, 0], big[i, 0])).shape) for i in range(8)}
    assert crops.shape == big.shape and len(shifts) > 1    # independent draws per sample


# ---- the learner ----------------------------------------------------------------------------

def _batch(n, obs_dim=3, act_dim=1, gen=0):
    g = torch.Generator().manual_seed(gen)
    return {"observations": torch.randn(n, obs_dim, generator=g),
            "actions": torch.rand(n, act_dim, generator=g) * 2 - 1,
            "rewards": torch.randn(n, generator=g),
            "masks": torch.ones(n),
            "next_observations": torch.randn(n, obs_dim, generator=g)}


def test_default_target_entropy_is_minus_half_the_action_dim():
    agent = SACLearner(obs_dim=3, action_dim=4, seed=0)
    assert agent.target_entropy == -2.0


def test_update_steps_the_critic_utd_times_then_actor_and_temperature_once_on_the_last_minibatch():
    agent = SACLearner(obs_dim=3, action_dim=1, num_qs=4, num_min_qs=2, seed=0)
    seen = []
    orig_c, orig_a = agent.update_critic, agent.update_actor
    agent.update_critic = lambda mb: (seen.append(("c", float(mb["rewards"][0]))), orig_c(mb))[1]
    agent.update_actor = lambda mb: (seen.append(("a", float(mb["rewards"][0]))), orig_a(mb))[1]
    b = _batch(32)
    info = agent.update(b, utd_ratio=4)
    firsts = [float(b["rewards"][i * 8]) for i in range(4)]
    assert seen == [("c", firsts[0]), ("c", firsts[1]), ("c", firsts[2]), ("c", firsts[3]),
                    ("a", firsts[3])]
    assert "temperature" in info and "critic_loss" in info and "entropy" in info
    with pytest.raises(ValueError):
        agent.update(_batch(30), utd_ratio=4)               # batch must split evenly


def test_polyak_runs_after_every_critic_step():
    agent = SACLearner(obs_dim=3, action_dim=1, num_qs=2, critic_lr=0.0, actor_lr=0.0,
                       temp_lr=0.0, tau=0.1, seed=0)
    with torch.no_grad():
        for p in agent.target_critic.parameters():
            p.add_(1.0)                                     # target = critic + 1
    agent.update(_batch(16), utd_ratio=4)
    for p, pt in zip(agent.critic.parameters(), agent.target_critic.parameters()):
        assert torch.allclose(pt - p, torch.full_like(p, 0.9 ** 4), atol=1e-5)


def test_sac_learns_a_one_step_bandit():
    """Terminal one-step task, reward -(a - s)^2: the deterministic action must track s."""
    torch.manual_seed(0)
    agent = SACLearner(obs_dim=1, action_dim=1, num_qs=2, num_min_qs=2, hidden_dims=(64, 64),
                       actor_lr=1e-3, critic_lr=1e-3, temp_lr=1e-3, init_temperature=0.05,
                       backup_entropy=False, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(1500):
        s = rng.uniform(-0.6, 0.6, size=(64, 1)).astype(np.float32)
        a = rng.uniform(-1, 1, size=(64, 1)).astype(np.float32)
        b = {"observations": torch.from_numpy(s), "actions": torch.from_numpy(a),
             "rewards": torch.from_numpy(-(a - s)[:, 0] ** 2), "masks": torch.zeros(64),
             "next_observations": torch.from_numpy(s)}
        agent.update(b, utd_ratio=1)
    test_s = np.linspace(-0.5, 0.5, 11, dtype=np.float32)[:, None]
    act = agent.eval_actions(test_s)
    assert float(np.abs(act - test_s).mean()) < 0.1, act.ravel()


def test_drq_copies_the_critic_encoder_into_the_actor_and_augments_both_observations():
    agent = DrQLearner(pixel_shape=(3, 16, 16), action_dim=2, num_qs=2, seed=0,
                       encoder_kwargs={"features": (8, 8), "filters": (3, 3), "strides": (2, 2)},
                       critic_lr=0.0, actor_lr=1e-3, temp_lr=0.0, latent_dim=8, hidden_dims=(16,))
    with torch.no_grad():
        for p in agent.critic.trunk.encoder.parameters():
            p.add_(0.5)
    n = 8
    b = {"observations": {"pixels": torch.randint(0, 256, (n, 3, 16, 16), dtype=torch.uint8)},
         "actions": torch.rand(n, 2) * 2 - 1, "rewards": torch.zeros(n), "masks": torch.ones(n),
         "next_observations": {"pixels": torch.randint(0, 256, (n, 3, 16, 16), dtype=torch.uint8)}}
    seen = {}
    orig = SACLearner.update

    def spy(self, batch, utd_ratio, update_temperature=True):
        seen["obs"] = batch["observations"]["pixels"].clone()
        seen["next"] = batch["next_observations"]["pixels"].clone()
        return orig(self, batch, utd_ratio, update_temperature)
    SACLearner.update = spy
    try:
        agent.update(b, utd_ratio=1)
    finally:
        SACLearner.update = orig
    for pa, pc in zip(agent.actor.trunk.encoder.parameters(), agent.critic.trunk.encoder.parameters()):
        assert torch.equal(pa, pc)                          # shared before the update; critic lr 0
    assert seen["obs"].shape == b["observations"]["pixels"].shape
    assert not torch.equal(seen["obs"], b["observations"]["pixels"])        # cropped
    assert not torch.equal(seen["next"], b["next_observations"]["pixels"])
