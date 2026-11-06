"""ACTOR_LOG_STD (2026-11-06): pin the actor's log-std at one value, so a
fine-tune trains only the mean action.

Every fine-tune of the clone collapsed with its action noise growing on the
same replay states: SAC run C's log-std went to the +2 cap (pure noise), the
trunk-frozen SAC run's to about -1.2, IQL run D's from -2.0 to -0.8 (AWR fits
the noise to the EXECUTED actions: DART noise plus the policy's own). A pinned
log-std cannot grow; the mean still trains. Unset, the actor is unchanged.
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
from torch_rl.nets import LOG_STD_MAX  # noqa: E402

SRC = open(os.path.join(ROOT, "train_carla_sac.py")).read()


@pytest.mark.parametrize("algo", ["sac", "iql"])
def test_a_pinned_log_std_stays_put_while_the_mean_trains(algo):
    agent = GatedRecurrentSAC(_cfg(algo=algo, actor_log_std=-2.0))
    x = torch.randn(6, agent.cfg.hidden + agent.cfg.state_dim,
                    generator=torch.Generator().manual_seed(0))
    mean0, ls0 = agent._actor_forward(x)
    for seed in range(3):
        agent.update(_chunk(agent.cfg, [1, 0], seed=seed))
    mean1, ls1 = agent._actor_forward(x)
    assert torch.all(ls0 == -2.0) and torch.all(ls1 == -2.0)
    assert not torch.allclose(mean0, mean1)


def test_the_actor_is_unpinned_by_default():
    agent = GatedRecurrentSAC(_cfg())
    assert agent.cfg.actor_log_std is None
    assert agent.actor.hi == LOG_STD_MAX


def test_the_config_refuses_a_pin_outside_the_actor_bounds():
    with pytest.raises(ValueError):
        _cfg(actor_log_std=LOG_STD_MAX + 1.0)


def test_the_training_script_reads_ACTOR_LOG_STD():
    assert ('actor_log_std=(float(os.environ["ACTOR_LOG_STD"])\n'
            '                       if os.environ.get("ACTOR_LOG_STD") else None)') in SRC


def test_the_banner_says_the_noise_is_pinned():
    assert "PINNED" in GatedRecurrentSAC(_cfg(actor_log_std=-2.0)).describe()
    assert "PINNED" not in GatedRecurrentSAC(_cfg()).describe()
