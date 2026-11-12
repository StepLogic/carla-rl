"""REDQ target operator: the user's "SAC not TQC, 10 critics" decision, pinned.

User 2026-10-12, after being shown the alternatives: 10 critics per head, target =
min over a RANDOM SUBSET OF 2 of the 10. Not TQC's truncated mean (which the peer's
existing loss already computes, and which would have been a one-line config change
at n_quantiles=1), and not the full min over 10 -- both the peer and this side
expect a 10-way min at gamma 0.99 to idle, the way our drop=2 TQC critic already
idles at short horizons.

WHAT THESE TESTS ARE FOR. The operator is three lines of arithmetic; what is easy
to get wrong is everything around them, and each test below names a mistake that
would survive a training run while quietly changing the algorithm:
  * sampling the subset PER SAMPLE instead of per update (that is not REDQ; it
    averages the pessimism away over a batch and the ensemble stops mattering)
  * regressing each critic onto its OWN min instead of the shared target (then the
    critics never disagree, which is the one thing the ensemble is there for)
  * clamping before the entropy term or before the bootstrap (changes what the
    guard guards)
  * letting a terminal row bootstrap (masks=0 must give exactly the reward)
  * drawing the subset with replacement (a "min over 2" that is sometimes a min
    over 1, at random, for no stated reason)
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from torch_rl import redq_loss  # noqa: E402


def _q(values):
    """(n_critics, batch, 1) scalar critic output, as QuantileCritics gives at
    n_quantiles=1."""
    t = torch.tensor(values, dtype=torch.float32)
    return t[..., None] if t.dim() == 2 else t


# --------------------------------------------------------------- the subset --
def test_the_subset_is_distinct_indices():
    """Drawn WITHOUT replacement. With replacement, a 'min over 2' silently
    becomes a min over 1 on 1/N of updates."""
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        idx = redq_loss.sample_subset(10, 2, generator=g)
        assert idx.numel() == 2
        assert idx[0] != idx[1]
        assert bool(((idx >= 0) & (idx < 10)).all())


def test_the_subset_covers_every_critic_over_many_draws():
    """A biased draw would train a few heads and leave the rest as dead weight."""
    g = torch.Generator().manual_seed(0)
    seen = set()
    for _ in range(400):
        seen.update(redq_loss.sample_subset(10, 2, generator=g).tolist())
    assert seen == set(range(10))


def test_asking_for_more_critics_than_exist_is_refused():
    with pytest.raises(ValueError, match="subset"):
        redq_loss.sample_subset(5, 7)


# ----------------------------------------------------------- the target maths --
def test_the_target_is_the_min_over_the_SUBSET_not_over_all_critics():
    """THE POINT OF REDQ. Critic 2 is the lowest of the five, but it is not in
    the subset, so it must not set the target."""
    next_q = _q([[10.0], [20.0], [1.0], [30.0], [40.0]])      # 5 critics, batch 1
    target = redq_loss.redq_target(
        next_q, rewards=torch.zeros(1), masks=torch.ones(1), discount=1.0,
        alpha=0.0, next_log_probs=torch.zeros(1),
        subset_idx=torch.tensor([0, 1]))
    assert target.item() == pytest.approx(10.0)               # min(10, 20), not 1


def test_one_subset_is_used_for_the_WHOLE_batch():
    """Per-sample subsets are a different algorithm: each row would get its own
    pessimism and the batch average would wash it out. Same two critics for every
    row means the whole batch is evaluated under one draw."""
    # critic 0 is lowest on row 0, critic 1 is lowest on row 1.
    next_q = _q([[0.0, 9.0], [9.0, 1.0], [-5.0, -5.0]])
    target = redq_loss.redq_target(
        next_q, rewards=torch.zeros(2), masks=torch.ones(2), discount=1.0,
        alpha=0.0, next_log_probs=torch.zeros(2),
        subset_idx=torch.tensor([0, 1]))
    # Row 0 takes critic 0's 0.0, row 1 takes critic 1's 1.0 -- an elementwise min
    # over the SAME subset, never critic 2's -5.0.
    assert target.tolist() == pytest.approx([0.0, 1.0])


def test_the_full_subset_is_the_full_min():
    """subset = n_critics must reduce exactly to the pessimistic baseline, so the
    operator spans the whole family and the chosen setting is a point in it."""
    next_q = _q([[10.0], [20.0], [1.0], [30.0]])
    target = redq_loss.redq_target(
        next_q, rewards=torch.zeros(1), masks=torch.ones(1), discount=1.0,
        alpha=0.0, next_log_probs=torch.zeros(1),
        subset_idx=torch.arange(4))
    assert target.item() == pytest.approx(1.0)


def test_the_entropy_term_is_subtracted_before_the_discount():
    """r + gamma * mask * (minQ - alpha * logp), the SAC/TQC ordering. Subtracting
    alpha*logp outside the discount would scale the entropy bonus wrongly at every
    gamma but 1.0 -- invisible in a gamma=1.0 run, wrong in this one (0.99)."""
    next_q = _q([[5.0]])
    t = redq_loss.redq_target(
        next_q, rewards=torch.tensor([1.0]), masks=torch.ones(1), discount=0.5,
        alpha=2.0, next_log_probs=torch.tensor([3.0]),
        subset_idx=torch.tensor([0]))
    assert t.item() == pytest.approx(1.0 + 0.5 * (5.0 - 2.0 * 3.0))


def test_a_terminal_row_backs_up_exactly_the_reward():
    """masks=0 must kill the bootstrap completely. A leak here is the classic
    silent value inflation at episode ends, and our episodes end on faults."""
    next_q = _q([[1000.0], [1000.0]])
    t = redq_loss.redq_target(
        next_q, rewards=torch.tensor([-7.0]), masks=torch.zeros(1), discount=0.99,
        alpha=5.0, next_log_probs=torch.tensor([2.0]),
        subset_idx=torch.tensor([0, 1]))
    assert t.item() == pytest.approx(-7.0)


def test_alpha_may_be_a_device_tensor():
    """The peer's measured compile trap: changing Python floats passed into a
    compiled update retrace it every call, so alpha, Q-scales and warm-up weights
    live as device tensors. The operator must take either."""
    next_q = _q([[5.0]])
    args = dict(rewards=torch.zeros(1), masks=torch.ones(1), discount=1.0,
                next_log_probs=torch.ones(1), subset_idx=torch.tensor([0]))
    a = redq_loss.redq_target(next_q, alpha=0.3, **args)
    b = redq_loss.redq_target(next_q, alpha=torch.tensor(0.3), **args)
    assert a.item() == pytest.approx(b.item())


def test_scalar_critics_may_arrive_with_or_without_the_quantile_axis():
    """QuantileCritics at n_quantiles=1 returns (n, B, 1); a plain scalar head
    returns (n, B). Both must mean the same thing rather than broadcasting into
    an (n, B, B) surprise."""
    args = dict(rewards=torch.zeros(2), masks=torch.ones(2), discount=1.0,
                alpha=0.0, next_log_probs=torch.zeros(2),
                subset_idx=torch.tensor([0, 1]))
    with_axis = redq_loss.redq_target(_q([[1.0, 2.0], [3.0, 4.0]]), **args)
    without = redq_loss.redq_target(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), **args)
    assert with_axis.shape == (2,) and without.shape == (2,)
    assert with_axis.tolist() == pytest.approx(without.tolist())


# ------------------------------------------------------------ the value guard --
def test_the_clamp_is_applied_AFTER_the_bootstrap_and_the_entropy_term():
    """It is a divergence guard on the backed-up value, not a reward clip. Our
    gamma=1.0 goal critic needed exactly this; the explore critic diverged
    20 -> 11,801 without it."""
    next_q = _q([[1e6]])
    t = redq_loss.redq_target(
        next_q, rewards=torch.tensor([0.0]), masks=torch.ones(1), discount=0.99,
        alpha=0.0, next_log_probs=torch.zeros(1), subset_idx=torch.tensor([0]),
        value_floor=-100.0, value_ceiling=100.0)
    assert t.item() == pytest.approx(100.0)


def test_the_floor_holds_against_a_diverging_critic():
    next_q = _q([[-1e6]])
    t = redq_loss.redq_target(
        next_q, rewards=torch.tensor([0.0]), masks=torch.ones(1), discount=0.99,
        alpha=0.0, next_log_probs=torch.zeros(1), subset_idx=torch.tensor([0]),
        value_floor=-100.0, value_ceiling=100.0)
    assert t.item() == pytest.approx(-100.0)


def test_a_ceiling_below_the_floor_is_refused():
    """THE PEER'S TRAP, MADE LOUD. value_ceiling defaulted to -1 in their older
    config; with a reward that can be positive, every target pins at the ceiling
    and the run looks alive while learning nothing. An inverted pair cannot be
    anything but a mistake, so it raises instead of silently clamping."""
    with pytest.raises(ValueError, match="value_ceiling"):
        redq_loss.redq_target(
            _q([[0.0]]), rewards=torch.zeros(1), masks=torch.ones(1), discount=1.0,
            alpha=0.0, next_log_probs=torch.zeros(1), subset_idx=torch.tensor([0]),
            value_floor=0.0, value_ceiling=-1.0)


def test_no_bounds_means_no_clamping():
    """Every arm that passes neither bound must be arithmetically untouched."""
    next_q = _q([[12345.0]])
    t = redq_loss.redq_target(
        next_q, rewards=torch.zeros(1), masks=torch.ones(1), discount=1.0,
        alpha=0.0, next_log_probs=torch.zeros(1), subset_idx=torch.tensor([0]))
    assert t.item() == pytest.approx(12345.0)


# --------------------------------------------------------------- the critic loss --
def test_EVERY_critic_regresses_onto_the_SHARED_target():
    """REDQ updates all N critics toward one target. Giving each critic its own
    min would make the heads collapse together, and the ensemble spread -- which
    is the only reason to hold 10 of them -- would go to zero."""
    current = _q([[0.0], [10.0]])            # critic 0 is 0, critic 1 is 10
    target = torch.tensor([10.0])
    per = redq_loss.critic_loss_per_sample(current, target)
    # mean over critics of (0-10)^2 and (10-10)^2 = (100 + 0)/2
    assert per.shape == (1,)
    assert per.item() == pytest.approx(50.0)


def test_the_loss_is_zero_only_when_every_critic_matches():
    current = _q([[3.0], [3.0], [3.0]])
    assert redq_loss.critic_loss_per_sample(current, torch.tensor([3.0])).item() \
        == pytest.approx(0.0)


def test_the_per_sample_loss_keeps_one_number_per_row():
    """The learner weights rows by a per-mode mask, so the reduction over critics
    must happen without touching the batch axis."""
    current = _q([[0.0, 0.0], [2.0, 4.0]])
    per = redq_loss.critic_loss_per_sample(current, torch.tensor([0.0, 0.0]))
    assert per.shape == (2,)
    assert per.tolist() == pytest.approx([(0.0 + 4.0) / 2, (0.0 + 16.0) / 2])


def test_the_target_never_carries_gradient():
    """A target that backpropagates into the target net is the bootstrapping bug
    that looks like fast early learning and then diverges."""
    next_q = _q([[5.0]]).requires_grad_(True)
    t = redq_loss.redq_target(
        next_q, rewards=torch.zeros(1), masks=torch.ones(1), discount=0.99,
        alpha=0.0, next_log_probs=torch.zeros(1), subset_idx=torch.tensor([0]))
    assert not t.requires_grad


def test_the_critic_loss_does_carry_gradient_to_every_head():
    """The flip side: all 10 heads must receive gradient from the shared target,
    not just the two that were sampled."""
    current = _q([[0.0], [0.0], [0.0]]).requires_grad_(True)
    redq_loss.critic_loss_per_sample(current, torch.tensor([1.0])).mean().backward()
    grad = current.grad.abs().sum(dim=(1, 2))
    assert bool((grad > 0).all()), "a critic head got no gradient"
