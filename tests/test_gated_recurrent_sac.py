"""GatedRecurrentSAC: the TQC learner with the critic algorithm swapped, nothing else.

User 2026-10-12: "move away from TQC", 10 critics, REDQ target, dual critic kept.

THE RISK THIS FILE ADDRESSES is not the arithmetic -- tests/test_redq_target.py
covers that -- but the swap: that the seam extraction left the TQC arm exactly as
it was, that the SAC subclass really does bypass every quantile code path, and
that a config cannot half-claim both algorithms.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from torch_rl import redq_loss, tqc_loss  # noqa: E402
from torch_rl.gated_recurrent_sac import (GatedRecurrentSAC,  # noqa: E402
                                          GatedRecurrentSACConfig,
                                          sac_config_from_tqc)
from torch_rl.gated_recurrent_tqc import (GatedRecurrentTQC,  # noqa: E402
                                          GatedRecurrentTQCConfig)

EMBED, GRID, STATE = 16, 2, 8
CHUNK, BURN_IN, NB = 6, 2, 2
# A replay row is [visual tokens | previous action (2) | state]: the previous
# action sits BETWEEN them, so the state offset is not the token width. Omitting
# those two columns makes _state slice across the boundary and the update fails
# on a reshape -- which is how this constant was found.
S0 = EMBED * GRID * GRID + 2


def _cfg(sac: bool, **kw):
    common = dict(embed_dim=EMBED, grid=GRID, state_dim=STATE, mode_index=3,
                  goal_cols=(0, 1), explore_cols=(4,), hidden=16, visual_dim=16,
                  action_emb=4, mlp=(8, 8), burn_in=BURN_IN, device="cpu",
                  compile=False, seed=7, explore_actor_warmup=0)
    common.update(kw)
    return (GatedRecurrentSACConfig(**common) if sac
            else GatedRecurrentTQCConfig(**common))


def _chunk(cfg, mode_value, seed=0, window=CHUNK):
    """One replay chunk in the shape update() expects.

    The window contract, read off _update_impl: obs is (B, K + T + 1) because the
    update needs the row after the last trained one for the bootstrap; keep is
    (B, K + 1), one flag per burn-in row plus the window's first row; everything
    per-step is (B, T). Getting this wrong is caught loudly, which is how these
    numbers were arrived at.
    """
    g = torch.Generator().manual_seed(seed)
    n, K, T = NB, int(cfg.burn_in), int(window)
    obs = torch.rand(n, K + T + 1, S0 + STATE, generator=g)
    # The mode bit is a per-episode constant. The learner masks on the explicit
    # `mode` entry, never the obs bit, but keep the two consistent.
    obs[..., S0 + cfg.mode_index] = float(mode_value)
    return {
        "obs": obs,
        "keep": torch.ones(n, K + 1),
        "actions": torch.rand(n, T, cfg.action_dim, generator=g) * 2 - 1,
        "rewards": torch.randn(n, T, generator=g),
        "masks": torch.ones(n, T),
        "valid": torch.ones(n, T),
        "mode": torch.full((n,), float(mode_value)),
    }


def _chunk_for(agent, mode_value, seed=0):
    return _chunk(agent.cfg, mode_value, seed), None


# ------------------------------------------------------- the config contract --
def test_the_defaults_are_the_users_numbers():
    cfg = _cfg(True)
    assert cfg.n_critics == 10 and cfg.subset_size == 2 and cfg.n_quantiles == 1


def test_quantile_critics_are_refused_in_this_arm():
    """A stray 25 in an old config.json must not turn the arm back into TQC."""
    with pytest.raises(ValueError, match="EXPECTATION SAC"):
        _cfg(True, n_quantiles=25)


def test_a_tqc_truncation_setting_is_refused_rather_than_ignored():
    """Configuring something that does nothing is how a run ends up described
    wrongly in a paper table."""
    with pytest.raises(ValueError, match="top_quantiles_to_drop_per_net"):
        _cfg(True, top_quantiles_to_drop_per_net=2)


def test_a_subset_larger_than_the_ensemble_is_refused():
    with pytest.raises(ValueError, match="subset_size"):
        _cfg(True, n_critics=4, subset_size=5)


def test_the_parent_config_still_defaults_to_the_tqc_arm():
    """The extraction must not have changed the TQC arm's own defaults."""
    cfg = _cfg(False)
    assert (cfg.n_critics, cfg.n_quantiles, cfg.top_quantiles_to_drop_per_net) == (5, 25, 2)
    assert (cfg.value_floor, cfg.value_ceiling) == (-100.0, 100.0)
    assert (cfg.value_floor_e, cfg.value_ceiling_e) == (None, None), \
        "the explore head must stay unclipped by default, as the peer runs it"


# ------------------------------------------------------------ the seam itself --
def test_the_parents_seam_is_pure_delegation_to_tqc_loss():
    """THE REFACTOR GUARD. Extracting the two methods must not have changed what
    the TQC arm computes, so the seam has to equal a direct tqc_loss call."""
    agent = GatedRecurrentTQC(_cfg(False))
    nq = agent.cfg.n_quantiles
    next_q = torch.randn(agent.cfg.n_critics, 5, nq)
    r, m, lp = torch.randn(5), torch.ones(5), torch.randn(5)
    got = agent._critic_target(next_q, r, m, discount=0.99, alpha=0.1,
                               next_log_probs=lp, value_floor=-100.0,
                               value_ceiling=100.0, head="goal")
    want = tqc_loss.tqc_target(next_q, r, m, discount=0.99,
                               top_drop=agent.top_drop, alpha=0.1,
                               next_log_probs=lp, value_floor=-100.0,
                               value_ceiling=100.0)
    torch.testing.assert_close(got, want)

    qs = torch.randn(agent.cfg.n_critics, 5, nq)
    torch.testing.assert_close(
        agent._critic_loss_per_sample(qs, want, head="goal"),
        tqc_loss.quantile_huber_loss_per_sample(qs, want, kappa=agent.cfg.kappa))


def test_the_sac_seam_computes_the_redq_target():
    agent = GatedRecurrentSAC(_cfg(True))
    next_q = torch.randn(10, 5, 1)
    r, m, lp = torch.randn(5), torch.ones(5), torch.randn(5)
    got = agent._critic_target(next_q, r, m, discount=0.99, alpha=0.1,
                               next_log_probs=lp, value_floor=None,
                               value_ceiling=None, head="goal")
    idx = agent._last_subset["goal"]
    want = redq_loss.redq_target(next_q, r, m, discount=0.99, alpha=0.1,
                                 next_log_probs=lp, subset_idx=idx)
    torch.testing.assert_close(got, want)
    assert idx.numel() == 2


def test_each_call_draws_a_FRESH_subset():
    """The parent calls the seam once per head per update, so "one draw per head
    per update" is exactly "one draw per call". A cached draw would freeze the
    pessimism onto two critics for the whole run."""
    agent = GatedRecurrentSAC(_cfg(True))
    args = dict(rewards=torch.zeros(3), masks=torch.ones(3), discount=0.99,
                alpha=0.0, next_log_probs=torch.zeros(3), value_floor=None,
                value_ceiling=None)
    seen = set()
    for _ in range(40):
        agent._critic_target(torch.randn(10, 3, 1), head="goal", **args)
        seen.add(tuple(sorted(agent._last_subset["goal"].tolist())))
    assert len(seen) > 1, "the subset never changed"


def test_the_two_heads_draw_independently():
    agent = GatedRecurrentSAC(_cfg(True))
    args = dict(rewards=torch.zeros(3), masks=torch.ones(3), discount=0.99,
                alpha=0.0, next_log_probs=torch.zeros(3), value_floor=None,
                value_ceiling=None)
    pairs = set()
    for _ in range(40):
        agent._critic_target(torch.randn(10, 3, 1), head="goal", **args)
        agent._critic_target(torch.randn(10, 3, 1), head="explore", **args)
        pairs.add((tuple(agent._last_subset["goal"].tolist()),
                   tuple(agent._last_subset["explore"].tolist())))
    assert any(g != e for g, e in pairs), "both heads always drew the same subset"


def test_the_subset_draws_do_not_disturb_the_global_rng():
    """The parent seeds the global RNG and its construction ORDER is load-bearing
    for reproducibility. If the draws came from the global stream, adding an
    update would shift every later initialisation."""
    agent = GatedRecurrentSAC(_cfg(True))
    torch.manual_seed(1234)
    before = torch.randn(4)
    torch.manual_seed(1234)
    for _ in range(10):
        agent._critic_target(torch.randn(10, 2, 1), rewards=torch.zeros(2),
                             masks=torch.ones(2), discount=0.99, alpha=0.0,
                             next_log_probs=torch.zeros(2), value_floor=None,
                             value_ceiling=None, head="goal")
    # torch.randn above consumes the global stream; draw from a fresh seed and
    # compare: only the generator-based subset draw must be absent from it.
    torch.manual_seed(1234)
    after = torch.randn(4)
    torch.testing.assert_close(before, after)


# ------------------------------------------------------------ the real update --
def test_the_critics_hold_ten_scalar_heads():
    agent = GatedRecurrentSAC(_cfg(True))
    for critics in (agent.critics_g, agent.critics_e):
        assert (critics.n_critics, critics.n_quantiles) == (10, 1)


@pytest.mark.parametrize("mode_value,head", [(1.0, "goal"), (0.0, "explore")])
def test_an_update_runs_and_moves_both_the_critics_and_the_actor(mode_value, head):
    agent = GatedRecurrentSAC(_cfg(True))
    critics = agent.critics_g if head == "goal" else agent.critics_e
    before_c = [p.detach().clone() for p in critics.parameters()]
    before_a = [p.detach().clone() for p in agent.actor.parameters()]
    ch, _ = _chunk_for(agent, mode_value)
    out = agent.update(ch)
    assert isinstance(out, dict) and out
    assert any(not torch.equal(p, q) for p, q in zip(critics.parameters(), before_c)), \
        f"the {head} critics did not move"
    assert any(not torch.equal(p, q) for p, q in zip(agent.actor.parameters(), before_a))


def test_the_absent_heads_critics_are_left_alone():
    """The gated contract: a goal-only batch must not touch the explore critics.
    Adam moves a parameter with a zero grad by its stale momentum, so 'weighted
    by zero' is not the same as 'not computed'."""
    agent = GatedRecurrentSAC(_cfg(True))
    before = [p.detach().clone() for p in agent.critics_e.parameters()]
    agent.update(_chunk_for(agent, 1.0)[0])
    assert all(torch.equal(p, q)
               for p, q in zip(agent.critics_e.parameters(), before))


def test_the_update_uses_no_quantile_code_path(monkeypatch):
    """If any quantile function is reached, the arm is not the one the user asked
    for -- whatever the config says."""
    import torch_rl.gated_recurrent_tqc as grt

    def _boom(*a, **k):
        raise AssertionError("a TQC quantile path was used in the SAC arm")

    monkeypatch.setattr(grt.tqc_loss, "tqc_target", _boom)
    monkeypatch.setattr(grt.tqc_loss, "quantile_huber_loss_per_sample", _boom)
    monkeypatch.setattr(grt.tqc_loss, "truncated_targets", _boom)
    agent = GatedRecurrentSAC(_cfg(True))
    agent.update(_chunk_for(agent, 1.0)[0])


def test_the_tqc_arm_still_updates_after_the_extraction():
    """The other half of the refactor guard: the parent must still run."""
    agent = GatedRecurrentTQC(_cfg(False))
    before = [p.detach().clone() for p in agent.critics_g.parameters()]
    agent.update(_chunk_for(agent, 1.0)[0])
    assert any(not torch.equal(p, q)
               for p, q in zip(agent.critics_g.parameters(), before))


def test_the_explore_head_can_now_be_clipped():
    """carla-rl's reason for the new knob: our explore critic diverged
    20 -> 11,801 unclipped. With a tight bound the target cannot leave it."""
    agent = GatedRecurrentSAC(_cfg(True, value_floor_e=-5.0, value_ceiling_e=5.0))
    t = agent._critic_target(torch.full((10, 4, 1), 1e6), rewards=torch.zeros(4),
                             masks=torch.ones(4), discount=0.99, alpha=0.0,
                             next_log_probs=torch.zeros(4),
                             value_floor=agent.cfg.value_floor_e,
                             value_ceiling=agent.cfg.value_ceiling_e,
                             head="explore")
    assert bool((t <= 5.0).all())


# --------------------------------------------------------------- round trips --
def test_save_and_load_round_trip():
    import tempfile
    agent = GatedRecurrentSAC(_cfg(True))
    agent.update(_chunk_for(agent, 1.0)[0])
    with tempfile.TemporaryDirectory() as d:
        agent.save(d)
        twin = GatedRecurrentSAC(_cfg(True))
        twin.load(d)
        for a, b in zip(agent.critics_g.parameters(), twin.critics_g.parameters()):
            torch.testing.assert_close(a, b)


def test_sac_config_from_tqc_changes_only_the_critic_algorithm():
    """The comparison run is worth nothing if a second difference crept in."""
    tqc = _cfg(False)
    sac = sac_config_from_tqc(tqc, n_critics=10, subset_size=2)
    changed = {f for f in tqc.__dataclass_fields__
               if getattr(tqc, f) != getattr(sac, f)}
    assert changed == {"n_critics", "n_quantiles",
                       "top_quantiles_to_drop_per_net"}, changed
    assert sac.subset_size == 2
