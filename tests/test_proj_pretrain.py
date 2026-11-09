"""The critic's projection head pretrained by imitation, then frozen (user 2026-11-09:
"pretrain projection layer with imitation learning using demo data like c0 before -
freeze and start town05_town01 SAC train").

torch_rl.proj_pretrain.pretrain_critic_projection trains ONE head -- Linear -> P ->
LayerNorm -> tanh, exactly QuantileCritics.project -- on the critic's own input
[h | state cols], with an actor-sized decoder predicting the expert's label action
under c0's cloning loss (the pre-tanh Gaussian NLL of bc_trainer.bc_loss). The trunk
stays frozen. The best validation head is kept, copied into every critic of the
goal ensemble and its target, and frozen there: SAC then trains the critic MLPs on
top of it, never the head.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")

from tests.test_one_critic import S0, STATE, _cfg, _chunk  # noqa: E402
from torch_rl.gated_recurrent_sac import GatedRecurrentSAC  # noqa: E402
from torch_rl.proj_pretrain import ProjectionHead, pretrain_critic_projection  # noqa: E402

SRC = open(os.path.join(ROOT, "train_carla_sac.py")).read()
P = 6


def _agent():
    torch.manual_seed(0)
    return GatedRecurrentSAC(_cfg(critic_proj_dim=P, critic_layer_norm=True))


def _labelled(cfg, n, seed):
    """Chunks whose label action is a fixed function of state column 0, which the
    goal critic's view (goal_cols) includes: what a head reading [h | state] can learn."""
    ch = _chunk(cfg, [1] * n, seed=seed)
    K = int(cfg.burn_in)
    st = ch["obs"][:, K:-1, S0:S0 + STATE]
    ch["actions"] = torch.stack([torch.tanh(4.0 * (st[..., 0] - 0.5)),
                                 torch.tanh(-4.0 * (st[..., 0] - 0.5))], dim=-1)
    return ch


def test_the_head_computes_what_a_critic_projection_computes():
    a = _agent()
    head = ProjectionHead(a.critics_g.proj_w.shape[1], P)
    x = torch.randn(9, a.critics_g.proj_w.shape[1])
    with torch.no_grad():
        a.critics_g.proj_w.copy_(head.w.detach()[None].expand_as(a.critics_g.proj_w))
        a.critics_g.proj_b.copy_(head.b.detach()[None].expand_as(a.critics_g.proj_b))
        a.critics_g.proj_ln_w.copy_(head.ln_w.detach()[None].expand_as(a.critics_g.proj_ln_w))
        a.critics_g.proj_ln_b.copy_(head.ln_b.detach()[None].expand_as(a.critics_g.proj_ln_b))
        assert torch.allclose(a.critics_g.project(x)[3], head(x), atol=1e-6)


def test_pretraining_learns_the_labels_and_keeps_the_best_head():
    a = _agent()
    g = iter(range(10_000))
    stats = pretrain_critic_projection(
        a, lambda: _labelled(a.cfg, 8, seed=next(g)), _labelled(a.cfg, 16, seed=99_999),
        updates=400, eval_every=50, log=lambda m: None)
    assert stats["best_val_mae"] < 0.5 * stats["init_val_mae"]
    # early stopping on the validation NLL (the cloning objective), trained heads only
    assert stats["best_val_nll"] == min(h["val_nll"] for h in stats["history"])
    assert stats["final_val_nll"] == pytest.approx(stats["best_val_nll"], abs=1e-5)
    assert stats["best_step"] > 0
    assert {"actor_val_nll", "actor_val_mae", "best_step"} <= set(stats)


def test_the_best_head_and_decoder_are_restored_together():
    # Validation labels are the training labels reversed: validation gets worse as
    # training goes on, so the best trained head is an early one, and the final
    # numbers must be exactly its eval's (head AND decoder restored, not only the head).
    a = _agent()
    val = _labelled(a.cfg, 16, seed=99_999)
    val["actions"] = -val["actions"]
    g = iter(range(10_000))
    stats = pretrain_critic_projection(a, lambda: _labelled(a.cfg, 8, seed=next(g)), val,
                                       updates=200, eval_every=50, log=lambda m: None)
    at_best = next(h for h in stats["history"] if h["step"] == stats["best_step"])
    assert stats["best_step"] > 0
    assert stats["final_val_nll"] == pytest.approx(at_best["val_nll"], abs=1e-5)
    assert stats["final_val_mae"] == pytest.approx(at_best["val_mae"], abs=1e-5)


def test_validation_takes_several_chunks():
    a = _agent()
    g = iter(range(10_000))
    stats = pretrain_critic_projection(
        a, lambda: _labelled(a.cfg, 8, seed=next(g)),
        [_labelled(a.cfg, 8, seed=70_000), _labelled(a.cfg, 8, seed=70_001)],
        updates=20, eval_every=10, log=lambda m: None)
    assert stats["val_rows"] == 2 * 8 * 6


def test_selection_is_on_validation_nll_and_never_the_untrained_head():
    # The untrained head can score well on action error (a decoder predicting ~0
    # is close to most labels) -- it must never be the one kept; among trained heads
    # the lowest validation NLL wins (early stopping on the cloning objective).
    import torch_rl.proj_pretrain as pp
    # (nll, mae) of the validation calls in order: c0's actor (reference), the
    # untrained head (best on both), then the evals at updates 10, 20, 30
    calls = iter([(9.0, 0.90), (0.1, 0.10), (1.0, 0.50), (0.5, 0.60), (0.8, 0.30)])
    a = _agent()
    real_val_errors = pp._nll_mae

    def fake_nll_mae(mean, log_std, act, v, clip=0.995):
        nll, mae = real_val_errors(mean, log_std, act, v, clip)
        if mean.shape[0] == 16 * 6:              # the validation chunk's rows
            try:
                n, m = next(calls)
                return torch.tensor(n), torch.tensor(m)
            except StopIteration:
                pass
        return nll, mae
    pp._nll_mae = fake_nll_mae
    try:
        g = iter(range(10_000))
        stats = pretrain_critic_projection(a, lambda: _labelled(a.cfg, 8, seed=next(g)),
                                           _labelled(a.cfg, 16, seed=99_999),
                                           updates=30, eval_every=10, log=lambda m: None)
    finally:
        pp._nll_mae = real_val_errors
    assert stats["best_step"] == 20 and stats["best_val_nll"] == pytest.approx(0.5)


def test_every_critic_and_its_target_get_the_head_frozen():
    a = _agent()
    pretrain_critic_projection(a, lambda: _labelled(a.cfg, 4, seed=1),
                               _labelled(a.cfg, 4, seed=2), updates=20, eval_every=10,
                               log=lambda m: None)
    for C in (a.critics_g, a.critics_g_targ):
        for name in ("proj_w", "proj_b", "proj_ln_w", "proj_ln_b"):
            t = getattr(C, name)
            assert not t.requires_grad, name
            assert all(torch.equal(t[0], t[i]) for i in range(1, t.shape[0])), name
    for name in ("proj_w", "proj_b", "proj_ln_w", "proj_ln_b"):
        assert torch.equal(getattr(a.critics_g, name), getattr(a.critics_g_targ, name))


def test_sac_updates_train_the_critic_mlps_but_never_the_frozen_head():
    a = _agent()
    pretrain_critic_projection(a, lambda: _labelled(a.cfg, 4, seed=3),
                               _labelled(a.cfg, 4, seed=4), updates=20, eval_every=10,
                               log=lambda m: None)
    head = {n: getattr(a.critics_g, n).detach().clone()
            for n in ("proj_w", "proj_b", "proj_ln_w", "proj_ln_b")}
    mlp0 = a.critics_g.weights[0].detach().clone()
    a.set_warmup(True)
    a.update(_chunk(a.cfg, [1, 0, 1, 0], seed=5))          # the warm-up's critic update
    a.set_warmup(False)
    for s in range(3):
        a.update(_chunk(a.cfg, [1, 0, 1, 0], seed=6 + s))  # full SAC updates
    for n, v in head.items():
        assert torch.equal(v, getattr(a.critics_g, n)), n
        # the soft update keeps the target's head (0.995 x + 0.005 x: rounding only)
        assert torch.allclose(v, getattr(a.critics_g_targ, n), rtol=0, atol=1e-6), n
    assert not torch.equal(mlp0, a.critics_g.weights[0])


def test_it_refuses_a_critic_without_a_head():
    a = GatedRecurrentSAC(_cfg(critic_layer_norm=True))
    with pytest.raises(ValueError, match="CRITIC_PROJ_DIM"):
        pretrain_critic_projection(a, lambda: _labelled(a.cfg, 4, seed=1),
                                   _labelled(a.cfg, 4, seed=2), updates=1, log=lambda m: None)


def test_the_trainer_reads_the_settings_and_fails_fast():
    from train_carla_sac import proj_pretrain_settings
    assert proj_pretrain_settings({}) is None
    s = proj_pretrain_settings({"CRITIC_PROJ_PRETRAIN_UPDATES": "10000",
                                "CRITIC_PROJ_DIM": "50", "IL_PRELOAD_ROOT": "/d"})
    assert s == {"updates": 10000, "chunks": 32, "eval_every": 500, "freeze": True}
    with pytest.raises(SystemExit, match="CRITIC_PROJ_DIM"):
        proj_pretrain_settings({"CRITIC_PROJ_PRETRAIN_UPDATES": "10", "IL_PRELOAD_ROOT": "/d"})
    with pytest.raises(SystemExit, match="IL_PRELOAD_ROOT"):
        proj_pretrain_settings({"CRITIC_PROJ_PRETRAIN_UPDATES": "10", "CRITIC_PROJ_DIM": "50"})
    main_src = SRC[SRC.index("\ndef main():"):]
    i = main_src.index("proj_pretrain_settings()")
    assert i < main_src.index("DebitTinyEncoder(")              # checked before CARLA starts
    assert "proj_pretrain=_proj_pretrain)" in main_src          # and handed to the preparation
    from torch_rl import bc_trainer
    import inspect
    assert "proj_pretrain" in inspect.signature(bc_trainer.il_finetune_prepare).parameters
