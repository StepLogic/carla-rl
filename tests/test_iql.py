"""IQL for the fine-tune (user 2026-11-06, "Lets try IQL").

Four SAC fine-tunes of the clone collapsed as soon as its actor could move, and
none of the fixes that left SAC's actor update in place helped (online critic
freeze, 50/50 replay, frozen encoder, temperature 0.001). SAC's actor climbs the
critic at actions the policy SAMPLES, and its critic target bootstraps from the
policy's next action -- both query the critic where the narrow data never was.
IQL (Kostrikov et al. 2021) queries it only at replay actions:

  V(s)    expectile tau of the target critics' Q(s, a_replay)
  Q(s,a)  regressed to r + gamma * mask * V(s')       (no next action)
  actor   advantage-weighted regression: weight exp(beta * A / scale), clipped,
          times the log-likelihood of the REPLAY action; A = Q_targ - V, scale =
          the learner's running Q std (actor_q_norm), so beta is scale-free.
"""
import math
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")

from tests.test_one_critic import _cfg, _chunk  # noqa: E402
from torch_rl.gated_recurrent_sac import GatedRecurrentSAC  # noqa: E402
from torch_rl.iql import awr_weights, expectile_loss  # noqa: E402


def _params(m):
    return [p.detach().clone() for p in m.parameters()]


def _moved(before, m):
    return any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))


# ------------------------------------------------------------ the two losses --
def test_expectile_loss_weights_errors_asymmetrically():
    diff = torch.tensor([2.0, -2.0])                    # Q - V
    out = expectile_loss(diff, tau=0.9)
    assert torch.allclose(out, torch.tensor([0.9 * 4.0, 0.1 * 4.0]))


def test_the_expectile_sits_above_the_mean_for_tau_above_a_half():
    q = torch.tensor([0.0, 0.0, 0.0, 10.0])
    v = torch.zeros((), requires_grad=True)
    opt = torch.optim.SGD([v], lr=0.05)
    for _ in range(2000):
        opt.zero_grad()
        expectile_loss(q - v, tau=0.9).mean().backward()
        opt.step()
    assert float(v) > float(q.mean()) + 1.0               # an upper expectile, not the mean


def test_awr_weights_grow_with_the_advantage_and_are_clipped():
    adv = torch.tensor([-1.0, 0.0, 1.0, 100.0])
    w = awr_weights(adv, beta=3.0, scale=torch.tensor(1.0), max_weight=100.0)
    assert torch.allclose(w[:3], torch.exp(torch.tensor([-3.0, 0.0, 3.0])))
    assert float(w[3]) == 100.0
    # the scale makes beta independent of the reward units
    w2 = awr_weights(adv * 10, beta=3.0, scale=torch.tensor(10.0), max_weight=100.0)
    assert torch.allclose(w, w2)


# ----------------------------------------------------------- the learner --
def test_the_iql_update_trains_value_critic_and_actor():
    agent = GatedRecurrentSAC(_cfg(algo="iql"))
    v0, c0, a0 = _params(agent.value_g), _params(agent.critics_g), _params(agent.actor)
    m = agent.update(_chunk(agent.cfg, [1, 0]))
    assert _moved(v0, agent.value_g) and _moved(c0, agent.critics_g) and _moved(a0, agent.actor)
    for k in ("v_loss", "v_mean", "adv_mean", "awr_w_mean", "critic_loss_g", "q_g"):
        assert k in m, k


def test_iql_never_samples_the_policy_inside_the_update():
    """The whole point: no critic query at a policy action. SAC's update samples
    the actor twice (the next action for the target, the new action for the
    actor loss); IQL's must not sample it at all."""
    agent = GatedRecurrentSAC(_cfg(algo="iql"))

    def _forbidden(*a, **k):
        raise AssertionError("IQL sampled the policy inside the update")
    agent._actor_sample = _forbidden
    agent.update(_chunk(agent.cfg, [1, 0]))


def test_iql_warmup_trains_value_and_critic_but_not_the_actor():
    agent = GatedRecurrentSAC(_cfg(algo="iql"))
    agent.set_warmup(True)
    v0, c0, a0 = _params(agent.value_g), _params(agent.critics_g), _params(agent.actor)
    agent.update(_chunk(agent.cfg, [1, 0]))
    assert _moved(v0, agent.value_g) and _moved(c0, agent.critics_g)
    assert not _moved(a0, agent.actor)


def test_iql_leaves_the_temperature_alone():
    agent = GatedRecurrentSAC(_cfg(algo="iql"))
    t0 = float(agent.log_alpha_g.detach())
    agent.update(_chunk(agent.cfg, [1, 0]))
    assert float(agent.log_alpha_g.detach()) == t0


def test_sac_agents_have_no_value_head_and_are_unchanged():
    agent = GatedRecurrentSAC(_cfg())
    assert agent.value_g is None


# ----------------------------------------------------------- checkpoints --
def test_the_value_head_is_saved_beside_the_checkpoint(tmp_path):
    agent = GatedRecurrentSAC(_cfg(algo="iql"))
    agent.update(_chunk(agent.cfg, [1, 0]))
    agent.save(str(tmp_path / "ck"))
    assert os.path.exists(tmp_path / "ck" / "iql_value.pt")
    other = GatedRecurrentSAC(_cfg(algo="iql"))
    other.load(str(tmp_path / "ck"))
    assert all(torch.equal(a, b) for a, b in zip(agent.value_g.parameters(),
                                                 other.value_g.parameters()))


def test_an_iql_agent_warm_starts_from_a_sac_checkpoint(tmp_path):
    """INIT_FROM the clone: c0's checkpoint has no value head; V starts fresh."""
    GatedRecurrentSAC(_cfg()).save(str(tmp_path / "clone"))
    agent = GatedRecurrentSAC(_cfg(algo="iql"))
    v0 = _params(agent.value_g)
    agent.load(str(tmp_path / "clone"))
    assert not _moved(v0, agent.value_g)


# ----------------------------------------------------------- the config --
def test_the_config_refuses_iql_settings_that_cannot_work():
    with pytest.raises(ValueError):
        _cfg(algo="awac")
    with pytest.raises(ValueError):
        _cfg(algo="iql", one=False)                       # one critic only
    with pytest.raises(ValueError):
        _cfg(algo="iql", iql_expectile=0.3)


def test_the_trainer_passes_the_algorithm_and_its_settings():
    s = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    assert 'algo=os.environ.get("ALGO", "sac")' in s
    assert 'iql_expectile=env_float("IQL_EXPECTILE", 0.7)' in s
    assert 'iql_beta=env_float("IQL_BETA", 3.0)' in s
    assert 'iql_max_weight=env_float("IQL_MAX_WEIGHT", 100.0)' in s
