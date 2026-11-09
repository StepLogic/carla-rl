"""REWARD_SCALE (user 2026-11-08: "lets fix this with reward scale"): every reward the
learner sees -- live driving, HER relabels, preloaded demos -- is multiplied by
reward_scale inside the update, and the critic's value clip with it, so a scaled run
is exactly the unscaled run fed rewards and bounds multiplied by the same factor.

Why: with rewards ~0.27/step, SAC's entropy term (alpha x the drop in log pi when the
actor widens its noise, ~4 nats from c0's precision to near-uniform) outweighed the
driving reward at alpha 0.1 and wrecked c0 within ~80 updates. ACTOR_Q_NORM=0 goes
with it: the actor's Q / std(Q) normalisation cancels any reward scale.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")


def _agent(**kw):
    from tests.test_one_critic import _cfg
    from torch_rl.gated_recurrent_sac import GatedRecurrentSAC
    torch.manual_seed(0)
    return GatedRecurrentSAC(_cfg(**kw))


def _chunk(agent, reward_mult=1.0):
    from tests.test_one_critic import _chunk as base
    ch = base(agent.cfg, [1, 0])
    ch["rewards"] = ch["rewards"] * reward_mult
    return ch


def _params(agent):
    return [p.detach().clone() for m in (agent.critics_g, agent.actor) for p in m.parameters()]


def test_scaling_inside_equals_scaling_the_data_and_the_bounds():
    k = 10.0
    inside = _agent(reward_scale=k, value_floor=-100.0, value_ceiling=150.0)
    outside = _agent(reward_scale=1.0, value_floor=-100.0 * k, value_ceiling=150.0 * k)
    torch.manual_seed(1)
    inside.update(_chunk(inside))
    torch.manual_seed(1)
    outside.update(_chunk(outside, reward_mult=k))
    for a, b in zip(_params(inside), _params(outside)):
        assert torch.allclose(a, b, atol=1e-6)


def test_the_default_scale_changes_nothing():
    one = _agent(reward_scale=1.0)
    plain = _agent()
    torch.manual_seed(1)
    one.update(_chunk(one))
    torch.manual_seed(1)
    plain.update(_chunk(plain))
    for a, b in zip(_params(one), _params(plain)):
        assert torch.equal(a, b)


def test_without_q_norm_the_actor_scale_stays_one():
    agent = _agent(actor_q_norm=False)
    agent.update(_chunk(agent))
    assert float(agent.q_scale_g) == 1.0 and float(agent.q_scale_init_g) == 0.0


def test_the_learner_description_shows_the_scale():
    assert "reward x10" in _agent(reward_scale=10.0).describe()
    assert "reward x" not in _agent().describe()


def test_the_trainer_reads_both_settings():
    src = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    assert 'env_float("REWARD_SCALE", 1.0)' in src
    assert 'os.environ.get("ACTOR_Q_NORM", "1") == "1"' in src
