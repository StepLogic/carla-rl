"""ACTOR_UPDATE_EVERY (user 2026-11-06). The pinned SAC fine-tune, run F, still
lost the clone's steering (step-50k eval 41 of 80 against c0's 63), and the
user chose "keep pure sac - but update actor less frequently": TD3's delayed
policy update. The critic trains on every update; the actor and the SAC
temperature step on every N-th, so the actor follows a critic that had N
updates per actor step. The fine-tune's cloning step goes with the actor step
and its fade counts actor steps, so per actor step the SAC term and the
cloning weight are run F's. N = 1 is every update, as before.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")

from tests.test_il_finetune import _ChunkBuf  # noqa: E402
from tests.test_one_critic import _cfg, _chunk  # noqa: E402
from torch_rl.gated_recurrent_sac import GatedRecurrentSAC  # noqa: E402

SRC = open(os.path.join(ROOT, "train_carla_sac.py")).read()


def _changed(module, before):
    return any(not torch.equal(a, b) for a, b in zip(before, module.parameters()))


def _actor_moves(agent, n):
    """For n updates, whether each one changed the actor's weights."""
    out = []
    for i in range(n):
        before = [p.detach().clone() for p in agent.actor.parameters()]
        agent.update(_chunk(agent.cfg, [1, 0], seed=i))
        out.append(_changed(agent.actor, before))
    return out


@pytest.mark.parametrize("algo", ["sac", "iql"])
def test_the_actor_steps_on_every_nth_update_only(algo):
    agent = GatedRecurrentSAC(_cfg(algo=algo, actor_update_every=3))
    assert _actor_moves(agent, 6) == [True, False, False, True, False, False]


@pytest.mark.parametrize("algo", ["sac", "iql"])
def test_the_critic_trains_on_every_update(algo):
    agent = GatedRecurrentSAC(_cfg(algo=algo, actor_update_every=3))
    for i in range(3):
        before = [p.detach().clone() for p in agent.critics_g.parameters()]
        agent.update(_chunk(agent.cfg, [1, 0], seed=i))
        assert _changed(agent.critics_g, before), i


def test_the_sac_temperature_steps_with_the_actor():
    agent = GatedRecurrentSAC(_cfg(actor_update_every=2))
    moved = []
    for i in range(4):
        t0 = float(agent.log_alpha_g.detach())
        agent.update(_chunk(agent.cfg, [1, 0], seed=i))
        moved.append(float(agent.log_alpha_g.detach()) != t0)
    assert moved == [True, False, True, False]


def test_one_is_every_update_as_before():
    agent = GatedRecurrentSAC(_cfg())
    assert agent.cfg.actor_update_every == 1
    assert _actor_moves(agent, 3) == [True, True, True]


def test_update_reports_whether_the_actor_stepped_and_keeps_its_last_diagnostics():
    agent = GatedRecurrentSAC(_cfg(actor_update_every=2))
    ms = [agent.update(_chunk(agent.cfg, [1, 0], seed=i)) for i in range(4)]
    assert [m["actor_stepped"] for m in ms] == [1.0, 0.0, 1.0, 0.0]
    assert agent.last_actor_stepped is False
    # an off-turn update reports the last actor step's numbers, not zeros
    assert ms[1]["actor_loss"] == ms[0]["actor_loss"] != 0.0
    assert ms[3]["log_std_g"] == ms[2]["log_std_g"] != 0.0


def test_the_cloning_step_goes_with_the_actor_step_and_fades_by_actor_steps():
    from torch_rl.bc_trainer import il_bc_hook, make_bc_optimizer
    agent = GatedRecurrentSAC(_cfg(actor_update_every=2))
    weights = []
    hook = il_bc_hook(agent, make_bc_optimizer(agent), _ChunkBuf(agent), n_chunks=2,
                      length=8, w0=1.0, decay_updates=4, on_weight=weights.append)
    for i in range(6):
        agent.update(_chunk(agent.cfg, [1, 0], seed=i))
        hook()
    assert weights == [1.0, 0.75, 0.5]          # three actor steps, three cloning steps


def test_the_config_refuses_zero():
    with pytest.raises(ValueError):
        _cfg(actor_update_every=0)


def test_the_banner_names_the_actor_interval():
    assert "every 4 critic updates" in GatedRecurrentSAC(_cfg(actor_update_every=4)).describe()
    assert "critic updates" not in GatedRecurrentSAC(_cfg()).describe()


def test_the_training_script_reads_ACTOR_UPDATE_EVERY():
    assert 'actor_update_every=env_int("ACTOR_UPDATE_EVERY", 1)' in SRC
