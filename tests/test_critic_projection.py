"""DrQ's critic input head (user 2026-11-09: "Yes, then test locally" -- after "is it
because the state latent dwarfs the action in size?").

The critic was one MLP on [h (512) | state block (3138) | action (2)]: the action is
2 of 3652 first-layer inputs, and in every trained critic measured (the c0 fine-tune
and run Q, which had random actions) it drove < 0.4% of that layer's input variance
even for uniform-random actions. DrQ / SAC-AE never hand a critic the raw latent:
Linear -> feature_dim (50) -> LayerNorm -> tanh, and the action is concatenated to
that feature. critic_proj_dim > 0 gives every critic of the ensemble its own such
head (REDQ's critics stay independent); 0 is every earlier critic, unchanged.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")

from tests.test_one_critic import _cfg, _chunk  # noqa: E402
from torch_rl.gated_recurrent_sac import GatedRecurrentSAC  # noqa: E402
from torch_rl.nets import QuantileCritics  # noqa: E402

SRC = open(os.path.join(ROOT, "train_carla_sac.py")).read()


def test_off_is_the_old_critic_exactly():
    torch.manual_seed(0)
    old = QuantileCritics(30, 2, n_critics=3, n_quantiles=1, hidden=(8, 8), layer_norm=True)
    torch.manual_seed(0)
    new = QuantileCritics(30, 2, n_critics=3, n_quantiles=1, hidden=(8, 8), layer_norm=True,
                          proj_dim=0)
    assert set(old.state_dict()) == set(new.state_dict())
    s, a = torch.randn(5, 30), torch.rand(5, 2) * 2 - 1
    assert torch.equal(old(s, a), new(s, a))


def test_each_critic_reads_its_own_50_d_head_then_the_action():
    c = QuantileCritics(3650, 2, n_critics=10, n_quantiles=1, hidden=(256, 256),
                        layer_norm=True, proj_dim=50)
    assert tuple(c.proj_w.shape) == (10, 3650, 50)
    assert tuple(c.weights[0].shape) == (10, 52, 256)        # 50 features + 2 actions
    assert not torch.equal(c.proj_w[0], c.proj_w[1])         # independent heads
    out = c(torch.randn(7, 3650), torch.rand(7, 2) * 2 - 1)
    assert tuple(out.shape) == (10, 7, 1)
    one = c.single_forward(3, torch.randn(7, 3650), torch.rand(7, 2) * 2 - 1)
    assert tuple(one.shape) == (7, 1)


def test_the_head_is_linear_layernorm_tanh():
    c = QuantileCritics(40, 2, n_critics=2, n_quantiles=1, hidden=(8, 8), proj_dim=6)
    z = c.project(torch.randn(11, 40) * 50.0)                # any input scale
    assert tuple(z.shape) == (2, 11, 6)
    assert float(z.abs().max()) <= 1.0                       # tanh
    with torch.no_grad():
        c.proj_ln_w.fill_(1.0)
        c.proj_ln_b.zero_()
        pre = torch.atanh(c.project(torch.randn(11, 40)).clamp(-0.999999, 0.999999))
    assert torch.allclose(pre.mean(-1), torch.zeros(2, 11), atol=1e-3)   # LayerNorm


def _action_share(c, s):
    """The action's share of the FIRST MLP layer's input variance (uniform actions)."""
    a = torch.rand(s.shape[0], 2) * 2 - 1
    W = c.weights[0]
    if c.proj_dim:
        z = c.project(s)                                     # (n, B, p)
        ps = torch.einsum("nbi,nio->nbo", z, W[:, :c.proj_dim])
    else:
        ps = torch.einsum("bi,nio->nbo", s, W[:, :s.shape[1]])
    pa = torch.einsum("bi,nio->nbo", a, W[:, -2:])
    vs, va = ps.var(dim=1), pa.var(dim=1)
    return float((va / (va + vs)).mean())


def test_the_action_is_no_longer_dwarfed_at_initialisation():
    torch.manual_seed(1)
    s = torch.randn(512, 3650) * 0.4
    raw = QuantileCritics(3650, 2, n_critics=4, n_quantiles=1, hidden=(256, 256),
                          layer_norm=True)
    proj = QuantileCritics(3650, 2, n_critics=4, n_quantiles=1, hidden=(256, 256),
                           layer_norm=True, proj_dim=50)
    with torch.no_grad():
        r, p = _action_share(raw, s), _action_share(proj, s)
    assert r < 0.01            # 2 of 3652 inputs: ~0.1 %
    assert p > 0.03            # 2 of 52: a few %
    assert p > 10 * r


def test_the_agent_builds_trains_and_targets_the_head():
    agent = GatedRecurrentSAC(_cfg(critic_proj_dim=6, critic_layer_norm=True))
    assert agent.critics_g.proj_dim == 6
    assert agent.critics_g_targ.proj_dim == 6                # the target copy has it too
    before = agent.critics_g.proj_w.detach().clone()
    agent.update(_chunk(agent.cfg, [1, 0, 1, 0]))
    assert not torch.equal(before, agent.critics_g.proj_w.detach())   # the critic loss trains it


def test_save_load_and_a_clone_without_the_head_gives_fresh_critics(tmp_path):
    a = GatedRecurrentSAC(_cfg(critic_proj_dim=6, critic_layer_norm=True))
    a.save(str(tmp_path / "proj"))
    assert json.load(open(tmp_path / "proj" / "config.json"))["critic_proj_dim"] == 6
    b = GatedRecurrentSAC(_cfg(critic_proj_dim=6, critic_layer_norm=True))
    b.load(str(tmp_path / "proj"))                           # strict
    assert torch.equal(a.critics_g.proj_w, b.critics_g.proj_w)
    clone = GatedRecurrentSAC(_cfg())                        # c0-like: no LN, no head
    clone.save(str(tmp_path / "clone"))
    c = GatedRecurrentSAC(_cfg(critic_proj_dim=6, critic_layer_norm=True))
    fresh = c.critics_g.proj_w.detach().clone()
    c.load(str(tmp_path / "clone"), fresh_critics_on_mismatch=True)
    assert torch.equal(fresh, c.critics_g.proj_w)            # critics stay fresh
    assert torch.equal(clone.actor.trunk[0].weight, c.actor.trunk[0].weight)


def test_the_trainer_reads_the_setting_and_evals_follow_the_checkpoint(tmp_path):
    from train_carla_sac import checkpoint_critic_proj_dim
    assert 'critic_proj_dim=(env_int("CRITIC_PROJ_DIM", 0)' in SRC
    main_src = SRC[SRC.index("\ndef main():"):]
    assert "critic_proj_dim=checkpoint_critic_proj_dim(EVAL_FROM) if EVAL_FROM else None" in main_src
    (tmp_path / "config.json").write_text(json.dumps({"critic_proj_dim": 50}))
    assert checkpoint_critic_proj_dim(str(tmp_path)) == 50
    (tmp_path / "config.json").write_text(json.dumps({"hidden": 16}))    # before the field
    assert checkpoint_critic_proj_dim(str(tmp_path)) == 0
