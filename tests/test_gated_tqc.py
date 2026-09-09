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
