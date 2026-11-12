"""DroQ-style critic regularisation (user 2026-11-01: "i want REDQ and DroQ
style setup for critics").

WHAT WE ALREADY HAD: REDQ. The run banner says it -- 10 critics, target = min
over a random subset of 2. ("ONE CRITIC" in this codebase means one critic
network shared by both reward modes, with the mode bit as an input; it is not
one Q-function.)

WHAT WAS MISSING: DroQ's regularisation. QuantileCritics was a plain ReLU MLP,
and dropout + LayerNorm on every hidden layer is the part that makes a high
replay ratio stable -- the configuration this arm moves to with UPDATE_EVERY=1
(160 replayed samples per new transition, against v30's 8).

Order is Linear -> Dropout -> LayerNorm -> ReLU, as in the DroQ paper. Both are
per-critic: the ensemble is stacked parameters, so the norm's affine terms carry
a critic axis too, or all 10 heads would share one normalisation and stop being
independent. Off by default, so every earlier run is unchanged.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")

from torch_rl.nets import QuantileCritics  # noqa: E402

S, A, N, Q = 6, 2, 4, 3


def _c(**kw):
    return QuantileCritics(S, A, n_critics=N, n_quantiles=Q, hidden=(8, 8), **kw)


def _in(b=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, S, generator=g), torch.randn(b, A, generator=g))


def test_plain_critic_is_unchanged():
    c = _c()
    s, a = _in()
    assert c(s, a).shape == (N, 5, Q)
    assert not any("ln" in n for n, _ in c.named_parameters())


def test_layernorm_adds_per_critic_affine_terms():
    c = _c(layer_norm=True)
    lns = [p for n, p in c.named_parameters() if "ln_w" in n]
    assert len(lns) == 2                      # one per HIDDEN layer, not the head
    for p in lns:
        assert p.shape[0] == N                # a critic axis, not shared
    s, a = _in()
    assert c(s, a).shape == (N, 5, Q)


def test_layernorm_actually_normalises():
    """With the affine terms at their init (weight 1, bias 0) a hidden layer's
    pre-ReLU output has unit scale per critic per row, whatever the input
    scale -- that is what stops a drifting critic from running away."""
    c = _c(layer_norm=True)
    s, a = _in()
    small = c(s * 0.01, a * 0.01)
    big = c(s * 100.0, a * 100.0)
    # not equal (inputs differ) but both bounded, which a plain MLP is not
    assert big.abs().max() < 50.0 and small.abs().max() < 50.0
    plain = _c()
    assert plain(s * 100.0, a * 100.0).abs().max() > big.abs().max()


def test_dropout_is_on_in_training_and_off_in_eval():
    c = _c(dropout=0.5)
    s, a = _in()
    c.train()
    a1, a2 = c(s, a), c(s, a)
    assert not torch.allclose(a1, a2)         # stochastic while training
    c.eval()
    assert torch.allclose(c(s, a), c(s, a))   # deterministic for acting


def test_zero_dropout_is_deterministic_in_training():
    c = _c(dropout=0.0)
    c.train()
    s, a = _in()
    assert torch.allclose(c(s, a), c(s, a))


def test_single_forward_matches_the_ensemble_in_eval():
    """single_forward is the acting path; it must agree with the batched one, or
    the policy acts on a different critic from the one being trained."""
    c = _c(layer_norm=True, dropout=0.1)
    c.eval()
    s, a = _in()
    full = c(s, a)
    for i in range(N):
        assert torch.allclose(c.single_forward(i, s, a), full[i], atol=1e-5)


def test_the_config_and_trainer_carry_both():
    from tests.test_one_critic import _cfg
    cfg = _cfg(critic_dropout=0.01, critic_layer_norm=True)
    assert cfg.critic_dropout == 0.01 and cfg.critic_layer_norm
    src = open(os.path.join(ROOT, "train_carla_sac.py")).read()
    assert 'env_float("CRITIC_DROPOUT", 0.0)' in src
    assert 'CRITIC_LAYERNORM' in src
