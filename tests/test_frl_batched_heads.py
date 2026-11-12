"""The critic's 10 heads computed in ONE batched pass (user 2026-11-12: make training faster; the updates are
the bottleneck, ~32-38 ms each, mostly launching 10 heads' small kernels one head at a time). The weights stay
where they are -- one nn.Sequential per head, the same state dict, the same optimizer state, the same init --
so checkpoints and a resume need no conversion; only the forward stacks each layer's per-head weights and
runs all heads with one matmul (each head keeps its own LayerNorm affine). It must compute what the loop
computes: outputs, gradients, and whole learner updates."""
import copy
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
torch = pytest.importorskip("torch")

from torch_rl.frl.learner import SACLearner  # noqa: E402
from torch_rl.frl.networks import EnsembleQ  # noqa: E402


def _pair():
    torch.manual_seed(0)
    loop = EnsembleQ(12, 2, num_qs=10, hidden_dims=(16, 16), use_layer_norm=True)
    fast = copy.deepcopy(loop)
    fast.batched = True
    return loop, fast


@pytest.mark.parametrize("heads", [None, [3, 7]])
def test_batched_heads_compute_what_the_loop_computes(heads):
    loop, fast = _pair()
    g = torch.Generator().manual_seed(1)
    obs, act = torch.randn(32, 12, generator=g), torch.rand(32, 2, generator=g) * 2 - 1
    h = None if heads is None else torch.tensor(heads)
    q0 = loop(obs, act, heads=h)
    calls = []
    for head in fast.heads:                          # the batched pass must not run a head module
        head.register_forward_hook(lambda m, i, o: calls.append(1))
    q1 = fast(obs, act, heads=h)
    assert calls == []
    assert q1.shape == q0.shape == (10 if heads is None else 2, 32)
    torch.testing.assert_close(q1, q0, rtol=1e-5, atol=1e-5)
    (q0 ** 2).sum().backward()
    (q1 ** 2).sum().backward()
    for (n0, p0), (n1, p1) in zip(loop.named_parameters(), fast.named_parameters()):
        assert n0 == n1
        if p0.grad is None:
            assert p1.grad is None or float(p1.grad.abs().max()) == 0.0, n0
        else:
            torch.testing.assert_close(p1.grad, p0.grad, rtol=1e-4, atol=1e-5)


def test_a_batched_learner_keeps_the_state_dict_and_updates_like_the_loop():
    def make(batched):
        torch.manual_seed(0)
        return SACLearner(action_dim=2, embedding_dim=12, state_dim=2, latent_dim=8,
                          hidden_dims=(16, 16), num_qs=10, num_min_qs=2, critic_layer_norm=True,
                          backup_entropy=False, init_temperature=0.1, seed=0,
                          batched_heads=batched)
    a, b = make(False), make(True)
    assert list(a.critic.state_dict()) == list(b.critic.state_dict())
    for t0, t1 in zip(a.critic.state_dict().values(), b.critic.state_dict().values()):
        assert torch.equal(t0, t1)                       # the same init, the same layout
    g = torch.Generator().manual_seed(2)
    obs = {"embeddings": torch.randn(64, 12, generator=g), "states": torch.randn(64, 2, generator=g)}
    batch = {"observations": obs, "next_observations": copy.deepcopy(obs),
             "actions": torch.rand(64, 2, generator=g) * 2 - 1, "rewards": torch.randn(64, generator=g),
             "masks": torch.ones(64)}
    torch.manual_seed(5)
    a.update(copy.deepcopy(batch), utd_ratio=4)
    torch.manual_seed(5)
    b.update(copy.deepcopy(batch), utd_ratio=4)
    for (n0, t0), (n1, t1) in zip(a.critic.state_dict().items(), b.critic.state_dict().items()):
        torch.testing.assert_close(t1, t0, rtol=1e-4, atol=1e-5, msg=n0)
    for t0, t1 in zip(a.actor.state_dict().values(), b.actor.state_dict().values()):
        torch.testing.assert_close(t1, t0, rtol=1e-4, atol=1e-5)
    assert float(b.log_temp) == pytest.approx(float(a.log_temp), rel=1e-5)
