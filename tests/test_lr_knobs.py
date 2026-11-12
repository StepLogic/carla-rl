"""LR and ALPHA_LR (v31, user 2026-11-01: updates every step, "i want maximum
learning", then "should we use a lower learner rate").

Nothing set a learning rate: the config's 3e-4 default drove the critic, the
actor AND both temperature optimisers. With UPDATE_EVERY=1 that is 20x more
gradient steps per transition than v30 (160 replayed samples per new
transition, against 8), so:

  * the critic and actor move 20x further per transition at the same rate --
    the configuration high-replay-ratio methods (REDQ, DroQ) add ensembles and
    dropout to survive, and this arm runs ONE critic;
  * the temperature collapses 20x sooner in step terms. v30's alpha fell
    0.075 -> 0.0005 by 650k; at 20x that is ~30k steps, so exploration would
    be over before the first checkpoint.

ALPHA_LR therefore has to be separable from LR -- that is the whole point of
the knob, not a convenience.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
pytest.importorskip("torch")

from tests.test_one_critic import _cfg  # noqa: E402
from torch_rl.gated_recurrent_sac import GatedRecurrentSAC  # noqa: E402


def _rates(agent):
    return {
        "critic": agent.opt_critic.param_groups[0]["lr"],
        "actor": agent.opt_actor.param_groups[0]["lr"],
        "temp_g": agent.opt_temp_g.param_groups[0]["lr"],
        "temp_e": agent.opt_temp_e.param_groups[0]["lr"],
    }


def test_one_rate_still_drives_everything_by_default():
    r = _rates(GatedRecurrentSAC(_cfg(lr=3e-4)))
    assert r == {"critic": 3e-4, "actor": 3e-4, "temp_g": 3e-4, "temp_e": 3e-4}


def test_alpha_lr_moves_only_the_temperatures():
    r = _rates(GatedRecurrentSAC(_cfg(lr=1e-4, alpha_lr=1e-5)))
    assert r["critic"] == 1e-4 and r["actor"] == 1e-4
    assert r["temp_g"] == 1e-5 and r["temp_e"] == 1e-5


def test_zero_alpha_lr_means_follow_lr():
    """0 is 'unset', so every run before this knob existed is unchanged."""
    r = _rates(GatedRecurrentSAC(_cfg(lr=2e-4, alpha_lr=0.0)))
    assert r["temp_g"] == 2e-4 and r["temp_e"] == 2e-4


def test_a_lower_temperature_rate_slows_the_collapse():
    """End to end: the same updates, two temperature rates. Entropy is far
    above the target here, so alpha falls in both -- it must fall LESS with the
    lower rate, which is what keeps exploration alive under high replay."""
    from tests.test_one_critic import _chunk

    def run(alpha_lr):
        agent = GatedRecurrentSAC(_cfg(lr=1e-4, alpha_lr=alpha_lr,
                                       target_entropy=-10.0))
        start = float(agent.log_alpha_g.detach())
        for i in range(25):
            agent.update(_chunk(agent.cfg, [1, 1], seed=i))
        return start - float(agent.log_alpha_g.detach())

    fast = run(1e-3)
    slow = run(1e-5)
    assert fast > 0 and slow >= 0
    assert slow < fast / 5, (slow, fast)


def test_the_trainer_reads_both_from_the_environment():
    src = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    assert 'env_float("LR", 3e-4)' in src
    assert 'env_float("ALPHA_LR", 0.0)' in src
    assert "alpha_lr=" in src
