"""Faster learner-C updates with the same numbers (user 2026-11-12: make training faster). Measured on the RTX
4070 at the real sizes: 41.6 ms per update as it was, 30.8 with batched heads, 22.7 with the target critic's
Polyak average done as grouped (foreach) tensor ops as well.
  * Polyak: the same float operations in the same order (x(1 - tau), then + (tau x p)) -> bit-identical;
  * batched heads: a switch (LoopSettings.batched_heads, FRL_BATCHED_HEADS), which a resumed run may change."""
import copy
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
os.environ.setdefault("MONO_CHW", "1")
os.environ.setdefault("SPEED_OBS", "1")
torch = pytest.importorskip("torch")

from torch_rl.frl import carla as fc  # noqa: E402
from torch_rl.frl.learner import SACLearner  # noqa: E402


def _learner():
    return SACLearner(action_dim=2, embedding_dim=12, state_dim=2, latent_dim=8, hidden_dims=(16, 16),
                      num_qs=4, num_min_qs=2, critic_layer_norm=True, seed=0)


def test_the_polyak_average_is_bit_identical_to_the_per_parameter_loop():
    learner = _learner()
    with torch.no_grad():
        for p in learner.critic.parameters():
            p.add_(torch.randn_like(p))                  # the critic moved away from its target
    ref = copy.deepcopy(learner.target_critic)
    with torch.no_grad():
        for p, pt in zip(learner.critic.parameters(), ref.parameters()):
            pt.mul_(1.0 - learner.tau).add_(learner.tau * p)        # the original rule
    learner._polyak()
    for t0, t1 in zip(ref.parameters(), learner.target_critic.parameters()):
        assert torch.equal(t0, t1)


def test_settings_switch_the_batched_heads_and_a_resume_may_change_it():
    assert fc.LoopSettings.from_env({"FRL_BATCHED_HEADS": "1"}).batched_heads == 1
    assert fc.LoopSettings.from_env({}).batched_heads == 0
    assert "batched_heads" in fc.RESUME_MAY_DIFFER
    learner = fc.make_learner(12, 2, 2, "cpu", batched_heads=True)
    assert learner.critic.q.batched and learner.target_critic.q.batched
    assert not fc.make_learner(12, 2, 2, "cpu").critic.q.batched
