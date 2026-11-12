"""ACTOR_BC_ALPHA: SAC with the cloning term INSIDE the actor's own loss, TD3+BC
style (user 2026-11-06: "SAC + in-loss cloning").

Why: the fine-tune's cloning step ran on its OWN Adam optimizer (bc_trainer.
make_bc_optimizer), and Adam is blind to a loss's scale, so the "cloning
weight" barely changed how hard cloning pulled; every SAC fine-tune drifted
along the critic's consistently tilted slope anyway. With ACTOR_BC_ALPHA = a > 0
the SAC actor loss is

    alpha_T * log pi  -  (a / mean|Q|) * Q(s, pi(s))  +  mean((tanh(mu(s_E)) - a_E)^2)

on one optimizer, a_E the expert's clean labels on expert states s_E drawn from
the source il_finetune_prepare sets, so `a` really sets how far SAC may leave
the expert (TD3+BC's a = 2.5). The trunk is frozen in these fine-tunes: the
cloning term trains the actor head. 0 = off, every earlier run.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")

from tests.test_one_critic import _cfg, _chunk  # noqa: E402
from torch_rl.gated_recurrent_sac import GatedRecurrentSAC  # noqa: E402

SRC = open(os.path.join(ROOT, "train_carla_sac.py")).read()


def _labelled(cfg, value=0.5, seed=7):
    bc = _chunk(cfg, [1, 0], seed=seed)
    bc["actions"] = torch.full_like(bc["actions"], value)
    return bc


def test_the_cloning_term_pulls_the_actor_to_the_labels_inside_the_sac_step():
    """Against a plain-SAC twin on the same updates: only the in-loss cloning
    term moves the noise-free action toward the labels."""
    def run(**kw):
        agent = GatedRecurrentSAC(_cfg(lr=3e-3, **kw))
        agent.set_trunk_frozen(True)
        bc = _labelled(agent.cfg)
        if kw.get("actor_bc_alpha"):
            agent.set_actor_bc_source(lambda: bc)
        before = agent.bc_mse(bc)
        for i in range(25):
            agent.update(_chunk(agent.cfg, [1, 0], seed=i))
        return before, agent.bc_mse(bc)

    before, cloned = run(actor_bc_alpha=1e-6)                 # Q term negligible
    _, plain = run()
    assert cloned < 0.75 * before
    assert cloned < plain


def test_without_the_mode_the_sac_step_ignores_the_labels():
    agent = GatedRecurrentSAC(_cfg())
    agent.set_trunk_frozen(True)
    bc = _labelled(agent.cfg)
    m = agent.update(_chunk(agent.cfg, [1, 0], seed=0))
    assert agent.cfg.actor_bc_alpha == 0.0 and "bc_mse" not in m


def test_the_update_reports_the_cloning_error():
    agent = GatedRecurrentSAC(_cfg(actor_bc_alpha=2.5))
    agent.set_trunk_frozen(True)
    bc = _labelled(agent.cfg)
    agent.set_actor_bc_source(lambda: bc)
    m = agent.update(_chunk(agent.cfg, [1, 0], seed=0))
    assert m["bc_mse"] > 0.0


def test_the_mode_needs_a_source():
    agent = GatedRecurrentSAC(_cfg(actor_bc_alpha=2.5))
    agent.set_trunk_frozen(True)
    with pytest.raises(RuntimeError):
        agent.update(_chunk(agent.cfg, [1, 0], seed=0))


def test_the_config_refuses_iql_and_negative_alpha():
    with pytest.raises(ValueError):
        _cfg(algo="iql", actor_bc_alpha=2.5)
    with pytest.raises(ValueError):
        _cfg(actor_bc_alpha=-1.0)


def test_prepare_sets_the_source_and_its_hook_takes_no_separate_cloning_step(tmp_path, monkeypatch):
    from tests.test_bc_trainer import _shards, _small_pixel_agent
    from torch_rl import pixel_rows as pxr
    from torch_rl.bc_trainer import il_finetune_prepare
    from torch_rl.episode_buffer import DeviceEpisodeBuffer
    import dataclasses
    _shards(tmp_path / "data")
    monkeypatch.setenv("REPLAY_DEVICE", "cpu")
    agent = GatedRecurrentSAC(dataclasses.replace(_small_pixel_agent(pxr.BINOC_DIM, 2).cfg,
                                                  actor_bc_alpha=2.5))
    agent.set_trunk_frozen(True)
    buf = DeviceEpisodeBuffer(pxr.mono_row_dim(), 2, 200, "cpu", goal_dim=pxr.PIX)
    hook = il_finetune_prepare(agent, buf, str(tmp_path / "data"), ["Town05"],
                               warmup_updates=1, chunk_b=2, chunk_t=4, w0=1.0,
                               decay_updates=10, bc_chunks=2, bc_len=4, aug=None,
                               log=lambda m: None)
    assert agent._actor_bc_source is not None
    actor0 = [p.detach().clone() for p in agent.actor.parameters()]
    hook()
    assert all(torch.equal(a, b) for a, b in zip(actor0, agent.actor.parameters()))


def test_the_banner_names_the_in_loss_cloning():
    agent = GatedRecurrentSAC(_cfg(actor_bc_alpha=2.5))
    assert "TD3+BC" in agent.describe()
    assert "TD3+BC" not in GatedRecurrentSAC(_cfg()).describe()


def test_the_training_script_reads_ACTOR_BC_ALPHA():
    assert 'actor_bc_alpha=env_float("ACTOR_BC_ALPHA", 0.0)' in SRC
