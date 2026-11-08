"""A warm start WITHOUT a preload (user 2026-11-08: "start from c0 and freeze encoder",
"also freeze GRU") keeps the loaded actor still for its first IL_ONLINE_FREEZE_UPDATES
online updates, so the fresh critic learns the clone's own driving before the clone
moves -- the warm-start rule (a warm start whose actor trained from the first update
lost its goal steering within 50k steps; fine-tune job 463440 fell 0.76 -> 0.10 within
~190 updates). Before this, IL_ONLINE_FREEZE_UPDATES worked only with IL_PRELOAD_ROOT
and was silently ignored otherwise.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")


def _agent_and_chunk():
    from tests.test_one_critic import _cfg, _chunk
    from torch_rl.gated_recurrent_sac import GatedRecurrentSAC
    torch.manual_seed(0)
    agent = GatedRecurrentSAC(_cfg())
    return agent, _chunk(agent.cfg, [1, 0])


def _actor(agent):
    return [p.detach().clone() for p in agent.actor.parameters()]


def _critic(agent):
    return [p.detach().clone() for p in agent.critics_g.parameters()]


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def test_the_actor_holds_for_the_first_n_updates_then_trains():
    from torch_rl.bc_trainer import actor_freeze_hook
    agent, ch = _agent_and_chunk()
    hook = actor_freeze_hook(agent, 2)
    a0, c0 = _actor(agent), _critic(agent)
    for _ in range(2):                          # the frozen updates
        agent.update(ch)
        hook()
    assert _same(_actor(agent), a0)             # the actor did not move
    assert not _same(_critic(agent), c0)        # the critic did
    agent.update(ch)                            # the first free update
    hook()
    assert not _same(_actor(agent), a0)


def test_zero_freezes_nothing():
    from torch_rl.bc_trainer import actor_freeze_hook
    agent, ch = _agent_and_chunk()
    hook = actor_freeze_hook(agent, 0)
    a0 = _actor(agent)
    agent.update(ch)
    hook()
    assert not _same(_actor(agent), a0)


def test_a_frozen_trunk_stays_frozen_after_the_release():
    from torch_rl.bc_trainer import actor_freeze_hook
    agent, ch = _agent_and_chunk()
    agent.set_trunk_frozen(True)
    hook = actor_freeze_hook(agent, 1)
    agent.update(ch)
    hook()                                      # released here
    assert not agent.warmup and agent.trunk_frozen
    assert not any(p.requires_grad for p in agent.core.parameters())


def test_the_trainer_uses_it_for_a_warm_start_without_a_preload():
    src = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    assert "actor_freeze_hook(agent, IL_ONLINE_FREEZE_UPDATES" in src
