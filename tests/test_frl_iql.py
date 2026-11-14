"""IQL for learner C, ported from FastRLAP's own implementation (user 2026-11-14: "ensure you borrow
IQL from fastrlap's and use the new proven learner"). Source read on 2026-11-14:
kylestach/fastrlap-release, jaxrl5/jaxrl5/agents/iql/iql_learner.py.

These tests pin the FIVE places FastRLAP's IQL differs from the textbook version, because each one is
a silent difference in a port -- the code would train either way and just be a different algorithm:
  1. expectile weighting   where(diff > 0, tau, 1 - tau) * diff**2        (diff = q - v)
  2. the V target is the MIN over TARGET-Q heads, and takes no gradient
  3. target_value is a HARD COPY of value after each V step, not a Polyak average
  4. the critic loss SUMS over the Q heads and averages over the batch
  5. the AWR weight is exp((q - v) * temperature), CLIPPED AT 100, with q from the TARGET critic at
     the dataset action and v the MEAN over value heads; the objective is a weighted log-likelihood
and the update order is V -> ACTOR -> Q (the actor sees the new V and the old target critic).

Deviation recorded on purpose: FastRLAP's actor is a Normal with std fixed at 0.1 and no squashing.
Ours is learner C's TanhNormalActor, because the SAC stage that follows needs a squashed stochastic
actor and the point of the pretraining is to hand it over.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
torch = pytest.importorskip("torch")

from torch_rl.frl.iql import IQLLearner, expectile_loss  # noqa: E402


def _batch(b=16, d=12, a=2, dev="cpu"):
    g = torch.Generator().manual_seed(0)
    return {
        "observations": torch.randn(b, d, generator=g),
        "actions": torch.rand(b, a, generator=g) * 1.6 - 0.8,
        "rewards": torch.randn(b, generator=g),
        "masks": (torch.rand(b, generator=g) > 0.2).float(),
        "next_observations": torch.randn(b, d, generator=g),
    }


def _learner(**kw):
    kw.setdefault("num_qs", 4)
    return IQLLearner(obs_dim=12, action_dim=2, device="cpu", seed=0, **kw)


def test_expectile_loss_is_fastrlaps_asymmetric_square():
    diff = torch.tensor([2.0, -2.0])
    out = expectile_loss(diff, 0.8)
    assert out[0].item() == pytest.approx(0.8 * 4.0)      # q > v  -> tau
    assert out[1].item() == pytest.approx(0.2 * 4.0)      # q < v  -> 1 - tau


def test_the_value_step_regresses_to_the_min_over_target_q_and_leaves_the_critic_alone():
    lr = _learner()
    b = _batch()
    before = [p.detach().clone() for p in lr.critic.parameters()]
    lr.update_v(b)
    for p, p0 in zip(lr.critic.parameters(), before):
        assert torch.equal(p, p0), "update_v must not move the critic"
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in lr.critic.parameters())


def test_target_value_is_a_hard_copy_after_the_value_step():
    lr = _learner()
    lr.update_v(_batch())
    for p, tp in zip(lr.value.parameters(), lr.target_value.parameters()):
        assert torch.equal(p, tp), "FastRLAP hard-copies target_value; no tau averaging"


def test_the_critic_loss_sums_over_heads_rather_than_averaging():
    lr = _learner(num_qs=4)
    b = _batch()
    info = lr.update_q(b)
    # recompute the loss from the recorded per-head residual: summing 4 heads gives 4x the mean
    assert info["critic_loss"] == pytest.approx(info["critic_loss_per_head"] * 4, rel=1e-5)


def test_the_awr_weight_is_clipped_at_one_hundred():
    """A big temperature alone does NOT trigger the clip: at initialisation q - v is near zero and
    often negative, so exp(A * temp) comes out small (measured 0.069 on the first attempt at this
    test). The clip binds only on a large POSITIVE advantage -- with FastRLAP's temperature of 0.1,
    once q - v >= 46. So the advantage is forced here."""
    lr = _learner(temperature=1.0)
    lr.value.forward = lambda obs: torch.full((1, obs.shape[0]), -1000.0)   # v = -1000 -> A huge
    info = lr.update_actor(_batch())
    assert info["awr_weight_max"] == pytest.approx(100.0)
    assert info["awr_weight_mean"] == pytest.approx(100.0)


def test_the_update_order_is_value_actor_critic():
    lr = _learner()
    seen = []
    for name in ("update_v", "update_actor", "update_q"):
        real = getattr(lr, name)
        setattr(lr, name, (lambda n, f: (lambda b: (seen.append(n), f(b))[1]))(name, real))
    lr.update(_batch())
    assert seen == ["update_v", "update_actor", "update_q"]


def test_a_full_update_moves_value_actor_and_critic_and_polyaks_the_target_critic():
    lr = _learner(tau=0.5)
    b = _batch()
    v0 = [p.detach().clone() for p in lr.value.parameters()]
    a0 = [p.detach().clone() for p in lr.actor.parameters()]
    c0 = [p.detach().clone() for p in lr.critic.parameters()]
    t0 = [p.detach().clone() for p in lr.target_critic.parameters()]
    info = lr.update(b)
    assert any(not torch.equal(p, q) for p, q in zip(lr.value.parameters(), v0))
    assert any(not torch.equal(p, q) for p, q in zip(lr.actor.parameters(), a0))
    assert any(not torch.equal(p, q) for p, q in zip(lr.critic.parameters(), c0))
    assert any(not torch.equal(p, q) for p, q in zip(lr.target_critic.parameters(), t0))
    for k in ("value_loss", "actor_loss", "critic_loss", "q_mean", "v_mean", "adv_mean"):
        assert k in info and torch.isfinite(torch.tensor(info[k]))


def test_terminal_rows_drop_the_bootstrap():
    """masks = 1 - terminal, FastRLAP's convention: target_q = r + gamma * mask * next_v."""
    lr = _learner()
    b = _batch(b=4)
    b["masks"] = torch.zeros(4)
    with torch.no_grad():
        tq = lr.td_target(b)
    assert torch.allclose(tq, b["rewards"])
