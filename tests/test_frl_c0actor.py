"""c0's own actor head as learner C's actor (user 2026-11-11: "but the actor shares same frontend" ->
"proceed"). Learner C's actor reads the same frozen c0 features c0's head reads, so the RL actor can BE a
trainable copy of c0's head: at step 0 the RL policy is c0. The fine-tune then needs the critic to learn
alone first (FRL_ACTOR_FREEZE: a fresh critic must not push c0's actor) and the actor to drive from step 1
(FRL_RANDOM_START=0: no uniform-random warm-up filling the replay)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
os.environ.setdefault("MONO_CHW", "1")
os.environ.setdefault("SPEED_OBS", "1")
torch = pytest.importorskip("torch")

from torch_rl import pixel_rows as pxr  # noqa: E402
from torch_rl.frl import carla as fc  # noqa: E402
from torch_rl.gated_recurrent_sac import GatedRecurrentSAC, GatedRecurrentSACConfig  # noqa: E402
from tests.test_frl_carla import _FakeEnv, _row, _row_of  # noqa: E402

BINOC = pxr.BINOC_DIM


@pytest.fixture(scope="module")
def trunk(tmp_path_factory):
    sd = pxr.mono_state_dim(BINOC)
    cfg = GatedRecurrentSACConfig(
        mono_encoder="resnet", embed_dim=3, grid=pxr.MONO_IMG, action_dim=2, mono_head="ln_tanh",
        goal_tokens=False, prev_action=False, core_state_cols=pxr.mono_core_state_cols(BINOC),
        state_dim=sd, mode_index=0, one_critic=True, goal_cols=tuple(range(sd)), explore_cols=(),
        hidden=16, visual_dim=16, mlp=(8, 8), n_critics=2, device="cpu", compile=False, seed=0)
    d = str(tmp_path_factory.mktemp("c0h"))
    GatedRecurrentSAC(cfg).save(d)
    return fc.C0Trunk(d, "cpu")


def _c0_learner(trunk, seed=0):
    return fc.make_learner(trunk.emb_dim, trunk.state_dim, 2, "cpu", seed=seed, actor="c0", trunk=trunk)


@torch.no_grad()
def test_the_c0_head_actor_acts_exactly_like_c0(trunk):
    learner = _c0_learner(trunk)
    a = trunk.agent
    h = trunk.zero_hidden()
    for i in range(3):
        row = _row(i)
        obs, h_next = trunk.step(row, h)
        x = torch.as_tensor(np.asarray(row, np.float32)[None])
        s, _ = a.core.step(a._embed(a.compress, a.act_embed, x), h)
        inp = torch.cat([s, a._state(x)], dim=-1)                    # what c0's act() feeds its head
        assert torch.allclose(learner.actor.sample(obs)[2], a.actor.deterministic(inp), atol=1e-6)
        m1, ls1 = learner.actor.net(learner.actor.c0_input(obs))
        m0, ls0 = a.actor(inp)
        assert torch.allclose(m1, m0, atol=1e-6) and torch.allclose(ls1, ls0, atol=1e-6)
        h = h_next


def test_the_c0_head_actor_is_a_trainable_copy(trunk):
    learner = _c0_learner(trunk)
    assert all(p.requires_grad for p in learner.actor.parameters())
    before = [p.detach().clone() for p in trunk.agent.actor.parameters()]
    with torch.no_grad():
        for p in learner.actor.parameters():
            p.add_(1.0)
    assert all(torch.equal(b, p) for b, p in zip(before, trunk.agent.actor.parameters()))
    assert set(id(p) for p in learner.actor_opt.param_groups[0]["params"]) == \
        set(id(p) for p in learner.actor.parameters())                  # the optimiser trains the copy


def test_a_critic_only_update_moves_neither_actor_nor_temperature(trunk):
    learner = _c0_learner(trunk)
    rb = fc.FeatureReplay(50, trunk.emb_dim, trunk.state_dim, 2, "cpu", seed=0)
    h = trunk.zero_hidden()
    o, h = trunk.step(_row(0), h)
    for i in range(1, 12):
        n, h = trunk.step(_row(i), h)
        rb.add(o, np.array([0.1, 0.5], np.float32), 1.0, 1.0, n)
        o = n
    a0 = [p.detach().clone() for p in learner.actor.parameters()]
    c0 = [p.detach().clone() for p in learner.critic.parameters()]
    t0 = float(learner.log_temp)
    fc.critic_only_update(learner, rb.sample(8), utd=2)
    assert all(torch.equal(a, p) for a, p in zip(a0, learner.actor.parameters()))
    assert float(learner.log_temp) == t0
    assert any(not torch.equal(c, p) for c, p in zip(c0, learner.critic.parameters()))


def test_the_freeze_runs_critic_only_updates_first(trunk, tmp_path, monkeypatch):
    calls = []
    real = fc.critic_only_update
    monkeypatch.setattr(fc, "critic_only_update", lambda *a, **k: calls.append(1) or real(*a, **k))
    env = _FakeEnv()
    obs, info = env.reset()
    s = fc.LoopSettings(max_steps=6, start_training=2, utd=2, batch=4, replay=100, ckpt_every=6,
                        log_every=100, device="cpu", seed=0, actor="c0", actor_freeze=3,
                        random_start=0)
    learner = _c0_learner(trunk)
    a0 = [p.detach().clone() for p in learner.actor.parameters()]
    fc.train(env, obs, info, str(tmp_path / "run"), trunk=trunk, row_of=_row_of, settings=s,
             c0_dir=trunk.c0_dir, log=lambda m: None, learner=learner)
    assert len(calls) == 3                                              # steps 2, 3, 4
    assert any(not torch.equal(a, p) for a, p in zip(a0, learner.actor.parameters()))  # 5, 6


def test_without_a_random_start_the_actor_drives_from_the_first_step(trunk, tmp_path):
    learner = _c0_learner(trunk)
    with torch.no_grad():                                              # make the head (near) noiseless
        learner.actor.net.trunk[-1].bias[2:] = -20.0
        learner.actor.net.trunk[-1].weight[2:] = 0.0
    env = _FakeEnv()
    obs, info = env.reset()
    o0, _ = trunk.step(_row_of(obs, info), trunk.zero_hidden())
    with torch.no_grad():
        want = fc.box_map(learner.actor.sample(o0)[2][0].numpy(), env.action_space)
    s = fc.LoopSettings(max_steps=2, start_training=100, replay=10, ckpt_every=2, log_every=100,
                        device="cpu", seed=0, actor="c0", random_start=0)
    fc.train(env, obs, info, str(tmp_path / "run"), trunk=trunk, row_of=_row_of, settings=s,
             c0_dir=trunk.c0_dir, log=lambda m: None, learner=learner)
    assert np.allclose(env.actions[0], want, atol=1e-4)


def test_a_c0_actor_checkpoint_evaluates_as_itself(trunk, tmp_path):
    learner = _c0_learner(trunk)
    with torch.no_grad():
        for p in learner.actor.parameters():
            p.mul_(1.5)                                                 # no longer c0's weights
    d = str(tmp_path / "step_9")
    fc.save_checkpoint(d, learner, step=9, c0_dir=trunk.c0_dir, settings=fc.LoopSettings(actor="c0"))
    pol = fc.load_policy(d, "cpu")
    assert isinstance(pol.actor, fc.C0HeadActor)
    a1, _ = pol.act(_row(5), pol.reset_hidden(1), deterministic=True)
    obs, _ = trunk.step(_row(5), trunk.zero_hidden())
    with torch.no_grad():
        assert np.allclose(a1, learner.actor.sample(obs)[2][0].numpy(), atol=1e-6)


def test_settings_read_the_new_switches():
    s = fc.LoopSettings.from_env({"FRL_ACTOR": "c0", "FRL_ACTOR_FREEZE": "5000",
                                  "FRL_RANDOM_START": "0"})
    assert (s.actor, s.actor_freeze, s.random_start) == ("c0", 5000, 0)
    d = fc.LoopSettings.from_env({})
    assert (d.actor, d.actor_freeze, d.random_start) == ("frl", 0, 1)   # every earlier run
