"""One critic, two reward modes.

User 2026-10-13: "I want a single critic with two different reward modes", and,
asked about the consequence, "The mode bit should be observed by critic". This
replaces the dual critic decided on 2026-10-12; the two modes now differ only in
what the ENV pays -- GatedReward already pays a 50 arrival bonus in goal mode and
10 at 100 m in explore -- and the critic tells them apart from the mode column in
its own view.

Answered with the questions: ONE temperature (not one per mode), and the replay
keeps its fixed 50/50 goal/explore mix per batch, so the shared critic sees both
reward modes in every batch even though goal episodes are currently much longer.

WHAT EACH TEST PROTECTS. A single critic that cannot see the mode does not fail
-- it learns the AVERAGE of two different returns, and the run reads as merely
mediocre. That is the failure these tests exist for, along with the bookkeeping
around it: a second ensemble still collecting Adam momentum, an explore
temperature still stepping, or a banner still claiming two heads.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")

from torch_rl.gated_recurrent_sac import (GatedRecurrentSAC,  # noqa: E402
                                          GatedRecurrentSACConfig)

EMBED, GRID, STATE = 16, 2, 8
MODE_IDX = STATE - 1
S0 = EMBED * GRID * GRID + 2
NB, CHUNK = 2, 6


def _cfg(one=True, **kw):
    base = dict(embed_dim=EMBED, grid=GRID, state_dim=STATE, mode_index=MODE_IDX,
                goal_cols=(0, MODE_IDX) if one else (0, 1), explore_cols=(4,),
                hidden=16, visual_dim=16, action_emb=4, mlp=(8, 8), burn_in=2,
                device="cpu", compile=False, seed=5, explore_actor_warmup=0,
                one_critic=one)
    base.update(kw)
    return GatedRecurrentSACConfig(**base)


def _chunk(cfg, modes, window=CHUNK, seed=0):
    g = torch.Generator().manual_seed(seed)
    n, K, T = len(modes), int(cfg.burn_in), int(window)
    obs = torch.rand(n, K + T + 1, S0 + STATE, generator=g)
    for i, m in enumerate(modes):
        obs[i, :, S0 + cfg.mode_index] = float(m)
    return {"obs": obs, "keep": torch.ones(n, K + 1),
            "actions": torch.rand(n, T, 2, generator=g) * 2 - 1,
            "rewards": torch.randn(n, T, generator=g),
            "masks": torch.ones(n, T), "valid": torch.ones(n, T),
            "mode": torch.tensor([float(m) for m in modes])}


# ----------------------------------------------------------- the mode bit --
def test_the_single_critic_REQUIRES_the_mode_bit_in_its_view():
    """Without it the critic predicts two different returns from otherwise
    identical inputs and quietly learns their average."""
    with pytest.raises(ValueError, match="mode bit"):
        _cfg(one=True, goal_cols=(0, 1))


def test_the_valid_single_critic_config_is_accepted():
    cfg = _cfg(one=True)
    assert cfg.mode_index in cfg.goal_cols


def test_two_critics_still_FORBID_the_mode_bit():
    """The old rule is right for the old arm: each critic sees only its own
    mode's rows, so there the bit is a constant -- a bias, not a feature."""
    with pytest.raises(ValueError, match="mode bit"):
        _cfg(one=False, goal_cols=(0, MODE_IDX))


def test_the_critic_input_width_includes_the_mode_column():
    agent = GatedRecurrentSAC(_cfg(one=True))
    # The critic reads [hidden | the state columns in its view].
    assert agent.critics_g.weights[0].shape[1] == agent.cfg.hidden + 2 + 2


# ------------------------------------------------------- both modes train it --
def test_EVERY_row_trains_the_single_critic_whatever_mode_it_came_from():
    """With two critics a goal-only batch left the explore critic untouched.
    Here one critic must move on an explore-only batch too, or half the data is
    being discarded."""
    for modes in ([1.0, 1.0], [0.0, 0.0], [1.0, 0.0]):
        agent = GatedRecurrentSAC(_cfg(one=True))
        before = [p.detach().clone() for p in agent.critics_g.parameters()]
        out = agent.update(_chunk(agent.cfg, modes))
        assert out.get("skipped", 0.0) == 0.0, modes
        assert any(not torch.equal(p, q)
                   for p, q in zip(agent.critics_g.parameters(), before)), modes


def test_the_two_modes_give_DIFFERENT_values_for_the_same_state():
    """The point of the mode bit: one critic, two reward modes, so the same
    observation under the two modes must be allowed to differ in value. Equal
    outputs here would mean the column is not reaching the critic."""
    agent = GatedRecurrentSAC(_cfg(one=True))
    # Train briefly so the critic is not at its (mode-blind) initialisation.
    for i in range(6):
        agent.update(_chunk(agent.cfg, [1.0, 0.0], seed=i))
    h = torch.randn(4, agent.cfg.hidden)
    st = torch.randn(4, STATE)
    a = torch.rand(4, 2)
    st_goal, st_expl = st.clone(), st.clone()
    st_goal[:, MODE_IDX] = 1.0
    st_expl[:, MODE_IDX] = 0.0
    q_goal = agent._q(agent.critics_g, h, st_goal, agent.goal_cols, a)
    q_expl = agent._q(agent.critics_g, h, st_expl, agent.goal_cols, a)
    assert not torch.allclose(q_goal, q_expl), \
        "the critic's value does not depend on the mode: the bit is not an input"


# --------------------------------------------------------- the bookkeeping --
def test_the_second_ensemble_never_moves():
    """It is still constructed, so the checkpoint format and the construction
    RNG order are identical either way -- but nothing may train it, and no Adam
    momentum may drift it."""
    agent = GatedRecurrentSAC(_cfg(one=True))
    before = [p.detach().clone() for p in agent.critics_e.parameters()]
    for i in range(3):
        agent.update(_chunk(agent.cfg, [1.0, 0.0], seed=i))
    assert all(torch.equal(p, q)
               for p, q in zip(agent.critics_e.parameters(), before))


def test_its_target_never_moves_either():
    agent = GatedRecurrentSAC(_cfg(one=True))
    before = [p.detach().clone() for p in agent.critics_e_targ.parameters()]
    agent.update(_chunk(agent.cfg, [1.0, 0.0]))
    assert all(torch.equal(p, q)
               for p, q in zip(agent.critics_e_targ.parameters(), before))


def test_the_second_ensemble_is_not_in_the_optimiser():
    agent = GatedRecurrentSAC(_cfg(one=True))
    in_opt = {id(p) for grp in agent.opt_critic.param_groups for p in grp["params"]}
    assert not any(id(p) in in_opt for p in agent.critics_e.parameters())
    assert all(id(p) in in_opt for p in agent.critics_g.parameters())


def test_only_one_temperature_moves():
    """User's answer: one temperature, not one per mode. The explore
    temperature must stay exactly at its initial value."""
    agent = GatedRecurrentSAC(_cfg(one=True))
    a_e0 = float(agent.log_alpha_e.detach())
    for i in range(3):
        agent.update(_chunk(agent.cfg, [1.0, 0.0], seed=i))
    assert float(agent.log_alpha_e.detach()) == a_e0
    assert float(agent.log_alpha_g.detach()) != a_e0


def test_the_explore_critic_loss_is_reported_as_zero():
    """Half the metrics are meaningless in this route; they must read zero
    rather than a stale number from a previous configuration."""
    agent = GatedRecurrentSAC(_cfg(one=True))
    out = agent.update(_chunk(agent.cfg, [1.0, 0.0]))
    assert out["critic_loss_e"] == 0.0
    assert out["q_e"] == 0.0


def test_the_banner_does_not_claim_two_heads():
    """A log is read back months later; it must not say 'critics per head' when
    there is one head."""
    one = GatedRecurrentSAC(_cfg(one=True)).describe()
    assert "ONE CRITIC" in one and "per head" not in one
    assert "mode bit is a critic INPUT" in one
    two = GatedRecurrentSAC(_cfg(one=False)).describe()
    assert "per head" in two


def test_the_two_critic_arm_is_unchanged_by_the_new_flag():
    """one_critic defaults off, and the old route must behave exactly as before:
    a goal-only batch leaves the explore critic alone."""
    agent = GatedRecurrentSAC(_cfg(one=False))
    assert agent.cfg.one_critic is False
    before = [p.detach().clone() for p in agent.critics_e.parameters()]
    agent.update(_chunk(agent.cfg, [1.0, 1.0]))
    assert all(torch.equal(p, q)
               for p, q in zip(agent.critics_e.parameters(), before))


def test_the_replay_keeps_the_fixed_fifty_fifty_mix():
    """User's answer: keep it. Goal episodes are much longer than explore ones,
    so an unstratified draw would let them crowd the shared critic's batches."""
    from torch_rl.episode_buffer import DeviceEpisodeBuffer
    buf = DeviceEpisodeBuffer(S0 + STATE, 2, 4096, "cpu", seed=0,
                              goal_batch_frac=0.5)
    rng = np.random.default_rng(0)
    for ep, mode in enumerate([1, 0] * 6):
        n = 60 if mode == 1 else 20            # goal episodes much longer
        for t in range(n):
            o = rng.normal(size=S0 + STATE).astype(np.float32)
            buf.add(o, rng.uniform(-1, 1, 2), 0.0, t == n - 1, False, o, mode)
        buf.end_episode()
    got = buf.sample_chunks(8, 12, 2)["mode"]
    assert float(got.sum()) == 4.0, f"expected 4 goal chunks of 8, got {got}"
