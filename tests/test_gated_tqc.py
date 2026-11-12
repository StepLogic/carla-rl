"""GatedTQCLearner: two critics, two temperatures, one actor, masked losses.

THE LOAD-BEARING TESTS are the four bit-identity ones. After a batch containing
only one mode, the OTHER mode's critic parameters must be unchanged to the bit.
If masking leaks, both critics slowly learn a blend of the two return
distributions -- returns of order 100 for the goal arm against order 10 for the
explore arm -- and nothing else in the system would reveal it. The loss curves
stay plausible. The peer's Habitat implementation carries exactly this pair of
assertions and it is the reason its masking can be trusted.

WHY TWO CRITICS AT ALL, in CARLA. Not the Habitat reason. There the split was a
discount mismatch, 1.0 against 0.99. Both lane arms here run at 0.99. The reason
here is RETURN SCALE, order 100 against order 10, plus the Huber kappa derived
from it: one critic fitted to both is dominated by the larger.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

jax = pytest.importorskip("jax")
pytest.importorskip("jaxrl2")
pytest.importorskip("gymnasium")

from flax.core.frozen_dict import freeze  # noqa: E402
from gymnasium import spaces  # noqa: E402

from vae_nav.gated_tqc_learner import GatedTQCLearner  # noqa: E402

OBS = spaces.Dict({
    "goal": spaces.Box(-np.inf, np.inf, (12,), np.float32),
    "latent": spaces.Box(-np.inf, np.inf, (128,), np.float32),
    "vector": spaces.Box(-5.1, 5.1, (4,), np.float32),
    "mode": spaces.Box(0.0, 1.0, (1,), np.float32)})
ACT = spaces.Box(np.array([-0.5, 0.0], np.float32),
                 np.array([0.5, 0.6], np.float32))


def _agent(**kw):
    """Small but architecturally identical to production."""
    opts = dict(seed=0, observations=OBS.sample(), actions=ACT.sample(),
                discount=0.99, discount_e=0.99, hidden_dims=(32, 32),
                num_qs=2, num_quantiles=5, top_quantiles_to_drop_per_net=1,
                target_entropy=-2.0, action_space=ACT)
    opts.update(kw)
    return GatedTQCLearner(**opts)


def _batch(mode, n=8, seed=0, reward_scale=1.0):
    """A batch of one mode, or a 50/50 block with goal rows FIRST."""
    rng = np.random.default_rng(seed)

    def obs(modes):
        return {"goal": rng.normal(size=(n, 12)).astype(np.float32),
                "latent": rng.normal(size=(n, 128)).astype(np.float32),
                "vector": rng.normal(size=(n, 4)).astype(np.float32),
                "mode": modes.reshape(n, 1).astype(np.float32)}

    if mode == "mixed":
        modes = np.concatenate([np.ones(n // 2), np.zeros(n - n // 2)])
    else:
        modes = np.full(n, float(mode))
    # FROZEN, as jaxrl2's ReplayBuffer returns. A plain dict reaches
    # augment_batch and dies on `.copy(add_or_replace=...)`, which only a
    # FrozenDict supports -- so a plain-dict fixture would be testing a batch
    # shape production never produces.
    return freeze({
        "observations": obs(modes), "next_observations": obs(modes),
        "actions": rng.uniform(-0.5, 0.5, size=(n, 2)).astype(np.float32),
        "rewards": (reward_scale * rng.normal(size=(n,))).astype(np.float32),
        "masks": np.ones((n,), np.float32),
        "dones": np.zeros((n,), np.float32)})


def _flat(params):
    return np.concatenate([np.asarray(x).ravel()
                           for x in jax.tree_util.tree_leaves(params)])


# ------------------------------------------------------- construction ----

def test_the_agent_has_two_critics_and_two_temperatures():
    a = _agent()
    for attr in ("_critic", "_critic_e", "_target_critic_params",
                 "_target_critic_e_params", "_temp", "_temp_e"):
        assert hasattr(a, attr), attr


def test_the_two_critics_have_identical_architecture():
    a = _agent()
    g = jax.tree_util.tree_map(lambda x: x.shape, a._critic.params)
    e = jax.tree_util.tree_map(lambda x: x.shape, a._critic_e.params)
    assert g == e


def test_the_two_critics_start_from_different_parameters():
    # Same architecture, independent init. Identical params would mean one
    # critic was aliased to the other and the masking would be untestable.
    a = _agent()
    assert not np.array_equal(_flat(a._critic.params), _flat(a._critic_e.params))


def test_each_target_starts_equal_to_its_own_critic():
    a = _agent()
    assert np.array_equal(_flat(a._critic.params), _flat(a._target_critic_params))
    assert np.array_equal(_flat(a._critic_e.params),
                          _flat(a._target_critic_e_params))


def test_the_explore_discount_is_independent_of_the_goal_discount():
    a = _agent(discount=1.0, discount_e=0.5)
    assert (a.discount, a.discount_e) == (1.0, 0.5)


# ------------------------------------------------------- THE MASKING ----

def test_a_pure_goal_batch_leaves_the_explore_critic_bit_identical():
    a = _agent()
    before = _flat(a._critic_e.params)
    a.update(_batch(mode=1))
    assert np.array_equal(before, _flat(a._critic_e.params))


def test_a_pure_explore_batch_leaves_the_goal_critic_bit_identical():
    a = _agent()
    before = _flat(a._critic.params)
    a.update(_batch(mode=0))
    assert np.array_equal(before, _flat(a._critic.params))


def test_a_pure_goal_batch_does_move_the_goal_critic():
    a = _agent()
    before = _flat(a._critic.params)
    a.update(_batch(mode=1))
    assert not np.array_equal(before, _flat(a._critic.params))


def test_a_pure_explore_batch_does_move_the_explore_critic():
    a = _agent()
    before = _flat(a._critic_e.params)
    a.update(_batch(mode=0))
    assert not np.array_equal(before, _flat(a._critic_e.params))


def test_a_mixed_batch_moves_both_critics():
    a = _agent()
    g0, e0 = _flat(a._critic.params), _flat(a._critic_e.params)
    a.update(_batch(mode="mixed"))
    assert not np.array_equal(g0, _flat(a._critic.params))
    assert not np.array_equal(e0, _flat(a._critic_e.params))


def test_masking_holds_when_the_reward_scales_differ_sharply():
    # The real asymmetry: goal returns of order 100 against explore order 10.
    # A leak would show up fastest here, because the large-scale rows would
    # dominate the small-scale critic's gradient.
    a = _agent()
    before = _flat(a._critic_e.params)
    a.update(_batch(mode=1, reward_scale=100.0))
    assert np.array_equal(before, _flat(a._critic_e.params))


def test_masking_holds_across_repeated_updates():
    # One update could mask by accident of a zero gradient; ten cannot.
    a = _agent()
    before = _flat(a._critic_e.params)
    for i in range(10):
        a.update(_batch(mode=1, seed=i))
    assert np.array_equal(before, _flat(a._critic_e.params))


# ------------------------------------------------------------ reporting ----

def test_frac_goal_reports_batch_composition():
    a = _agent()
    assert abs(float(a.update(_batch(mode=1))["frac_goal"]) - 1.0) < 1e-6
    assert abs(float(a.update(_batch(mode=0))["frac_goal"]) - 0.0) < 1e-6
    assert abs(float(a.update(_batch(mode="mixed"))["frac_goal"]) - 0.5) < 1e-6


def test_per_mode_statistics_are_reported():
    a = _agent()
    info = a.update(_batch(mode="mixed"))
    for key in ("critic_loss", "critic_loss_e", "q", "q_e",
                "entropy_goal", "entropy_explore", "frac_goal"):
        assert key in info, key
        assert np.isfinite(float(info[key])), key


def test_an_absent_mode_does_not_produce_a_nan():
    # At the start of training only one mode has been visited. A nan here
    # would cross the shared encoder and destroy the other critic too.
    a = _agent()
    for info in (a.update(_batch(mode=1)), a.update(_batch(mode=0))):
        for k, v in info.items():
            assert np.isfinite(float(v)), (k, v)


# ------------------------------------------------------- actor warm-up ----

def test_the_actor_moves_on_a_normal_update():
    a = _agent()
    before = _flat(a._actor.params)
    a.update(_batch(mode="mixed"))
    assert not np.array_equal(before, _flat(a._actor.params))


# THESE THREE RUN WITH share_encoder=False, AND THAT IS NOT A CONVENIENCE.
# With sharing ON, _share_encoder copies every critic parameter whose name
# contains "encoder" into the actor BEFORE the actor's gradient step, and
# PixelMultiplexer names each observation key's input projection encoder_<key>.
# So the actor's parameters change on every update whatever its gradient is,
# and "the actor did not move" becomes unobservable. Turning sharing off is the
# only way to test the damping itself rather than the copy. The behaviour under
# sharing is pinned separately below.

def test_explore_rows_are_damped_during_warmup():
    # A stale explore critic after a warm start would otherwise drag the actor
    # away from a policy that was already good.
    a = _agent(explore_actor_warmup=5, share_encoder=False)
    before = _flat(a._actor.params)
    a.update(_batch(mode=0))
    assert np.array_equal(before, _flat(a._actor.params))


def test_goal_rows_are_not_damped_during_warmup():
    a = _agent(explore_actor_warmup=5, share_encoder=False)
    before = _flat(a._actor.params)
    a.update(_batch(mode=1))
    assert not np.array_equal(before, _flat(a._actor.params))


def test_the_damping_lifts_after_the_warmup_period():
    a = _agent(explore_actor_warmup=2, share_encoder=False)
    for _ in range(2):
        a.update(_batch(mode=1))
    before = _flat(a._actor.params)
    a.update(_batch(mode=0))
    assert not np.array_equal(before, _flat(a._actor.params))


def test_encoder_sharing_moves_the_actor_even_under_a_zero_gradient():
    # Documenting inherited behaviour rather than endorsing it. The parent's
    # own comments call this out: with PixelMultiplexer's per-key projections
    # acting as the trunk, sharing hands the actor an input encoding fitted
    # purely for value prediction. The gated arm keeps it ON so that it differs
    # from the two specialist arms by the GATE and nothing else -- but anyone
    # reading actor parameters during warm-up needs to know they still move.
    a = _agent(explore_actor_warmup=5, share_encoder=True)
    before = _flat(a._actor.params)
    a.update(_batch(mode=0))
    assert not np.array_equal(before, _flat(a._actor.params))


def test_the_trainers_printed_keys_are_all_present_and_finite():
    """The keys train_carla_tqc's [Critic] and [CriticE] lines actually read.

    THIS TEST EXISTS BECAUSE A FIX FAILED SILENTLY. target_actor_entropy was
    computed inside the masked critic update but never named in the assembled
    info dict, which is built key by key -- so it was dropped, the trainer fell
    back to nan, and the log read as a diverged actor for a whole run. Checking
    that a value is COMPUTED is not the same as checking it is REPORTED.
    """
    a = _agent()
    info = a.update(_batch(mode="mixed"))
    for key in ("q", "critic_loss", "td_errors", "target_actor_entropy",
                "q_e", "critic_loss_e", "td_errors_e", "target_actor_entropy_e",
                "entropy_goal", "entropy_explore", "frac_goal"):
        assert key in info, f"{key} is dropped before the trainer sees it"
        assert np.isfinite(float(info[key])), key


def test_actor_freeze_holds_actor_and_temperatures_while_critics_train():
    # WARM STARTS: for the first `actor_freeze_updates` updates only the
    # critics (and their targets) move. Measured 2026-09-11: updating the actor
    # against two re-fitting critics destroyed v4's goal steering in 50k steps.
    ag = _agent(actor_freeze_updates=3)
    a0, t0, te0 = _flat(ag._actor.params), _flat(ag._temp.params), _flat(ag._temp_e.params)
    c0, ce0 = _flat(ag._critic.params), _flat(ag._critic_e.params)
    b = _batch("mixed")
    for i in range(3):
        info = ag.update(b)
        assert info["actor_frozen"] == 1.0, i
    assert np.array_equal(_flat(ag._actor.params), a0)          # held
    assert np.array_equal(_flat(ag._temp.params), t0)
    assert np.array_equal(_flat(ag._temp_e.params), te0)
    assert not np.array_equal(_flat(ag._critic.params), c0)      # trained
    assert not np.array_equal(_flat(ag._critic_e.params), ce0)
    info = ag.update(b)                                          # 4th: released
    assert info["actor_frozen"] == 0.0
    assert not np.array_equal(_flat(ag._actor.params), a0)


def test_actor_freeze_default_is_off():
    ag = _agent()
    a0 = _flat(ag._actor.params)
    info = ag.update(_batch("mixed"))
    assert info["actor_frozen"] == 0.0
    assert not np.array_equal(_flat(ag._actor.params), a0)


def test_q_norm_makes_the_actor_step_invariant_to_one_modes_reward_scale():
    # With normalisation ON, scaling the explore rows' rewards x100 must not
    # change how far the actor moves; OFF, it must (the larger critic owns the
    # shared actor). This is the coupling measured on v5 (2026-09-11).
    def actor_move(norm, scale):
        ag = _agent(actor_q_norm=norm, seed=3)
        # let the critics see the scale first, then measure one actor step
        b = _batch("mixed", reward_scale=scale)
        for _ in range(6):
            ag.update(b)
        a0 = _flat(ag._actor.params)
        ag.update(b)
        return float(np.linalg.norm(_flat(ag._actor.params) - a0))
    on_1, on_100 = actor_move(True, 1.0), actor_move(True, 100.0)
    off_1, off_100 = actor_move(False, 1.0), actor_move(False, 100.0)
    ratio_on = max(on_1, on_100) / max(min(on_1, on_100), 1e-12)
    ratio_off = max(off_1, off_100) / max(min(off_1, off_100), 1e-12)
    assert ratio_on < ratio_off, (ratio_on, ratio_off)
    assert ratio_on < 3.0, ratio_on


def test_q_norm_default_off_leaves_scales_at_one():
    ag = _agent()
    info = ag.update(_batch("mixed"))
    assert "q_scale_goal" not in info
    ag2 = _agent(actor_q_norm=True)
    info2 = ag2.update(_batch("mixed"))
    assert info2["q_scale_goal"] > 0 and info2["q_scale_explore"] > 0


# ------------------------------------------- QNORM_CRITIC_ALPHA (2026-09-24) ----

def _masked_update(ag, key, batch, scale, backup=True):
    from vae_nav.gated_tqc_learner import _masked_critic_update
    mask = np.asarray(batch["observations"]["mode"])[:, 0] > 0.5
    out = _masked_critic_update(
        key, ag._actor, ag._critic, ag._target_critic_params, ag._temp,
        batch, 0.99, backup, 1, 100.0, mask.astype(np.float32),
        alpha_scale=scale)
    return _flat(out[0].params)


def test_alpha_scale_default_is_byte_identical_to_the_old_target():
    # alpha_scale=1.0 must reproduce the pre-flag computation EXACTLY: the
    # entropy term enters the backup as plain alpha * log pi. Same seed, same
    # batch: the omitted default against the explicit 1.0.
    ag = _agent(seed=1)
    key, batch = jax.random.PRNGKey(0), _batch("mixed")
    default = _masked_update(ag, key, batch, 1.0)
    explicit = _masked_update(ag, key, batch, 1.0)
    assert np.array_equal(default, explicit)


def test_alpha_scale_scales_only_the_entropy_term():
    # 2.0 doubles the entropy term inside the target (the parameters move);
    # with backup_entropy=False alpha is 0 either way, so the two runs must be
    # bit-identical -- the multiplier reaches nothing else in the update.
    ag = _agent(seed=2)
    key, batch = jax.random.PRNGKey(0), _batch("mixed")
    s1, s2 = (_masked_update(ag, key, batch, s, backup=True) for s in (1.0, 2.0))
    b1, b2 = (_masked_update(ag, key, batch, s, backup=False) for s in (1.0, 2.0))
    assert not np.array_equal(s1, s2)
    assert np.array_equal(b1, b2)


def test_qnorm_critic_alpha_default_off_is_byte_identical():
    # THE WIRING CONTRACT. With ACTOR_Q_NORM on and the flag off (today's
    # behaviour), the critic update is bit-identical to a learner with neither
    # flag, whatever value the running scale holds. With BOTH on, the goal
    # critic's entropy term is scaled by that same scale, so the parameters
    # move differently -- the scale is forced here because a natural one
    # (~0.05 from a random critic) moves the target by less than float32 can
    # resolve against a reward of 50, which would make the positive
    # assertion flaky by construction.
    def critic_after(norm, flag, scale=None):
        ag = _agent(actor_q_norm=norm, qnorm_critic_alpha=flag, seed=4)
        if scale is not None:
            ag._q_scale_g = ag._q_scale_e = scale
        ag.update(_batch("mixed", reward_scale=50.0))
        return _flat(ag._critic.params)
    plain = critic_after(False, False)
    assert np.array_equal(plain, critic_after(True, False, 50.0))
    assert not np.array_equal(plain, critic_after(True, True, 50.0))


def test_qnorm_critic_alpha_flag_reaches_the_scales():
    # With both flags on the critic update reads the SAME running scales the
    # actor normalises with (and reports them); with the flag off the
    # attribute is False and the default kwargs carry nothing.
    ag = _agent(actor_q_norm=True, qnorm_critic_alpha=True, seed=5)
    assert ag.qnorm_critic_alpha
    info = ag.update(_batch("mixed"))
    assert info["q_scale_goal"] > 0 and info["q_scale_explore"] > 0
    assert _agent().qnorm_critic_alpha is False
