"""The mode-split buffer feeding the gated learner, for real.

WHAT THIS COVERS THAT THE UNIT TESTS DO NOT. test_gated_tqc.py builds its
batches by hand; test_mode_split_buffer.py checks composition without a
learner. Neither proves the two fit together, and the seam between them is
where the failures would be quiet ones: the buffer strips its routing key on
insert, so the learner recovers the mode from the OBSERVATION, and if those two
conventions ever disagreed each critic would silently train on the wrong rows.

No simulator and no GPU: conftest.py pins JAX to CPU, and everything here is a
buffer and two small networks.
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

from gymnasium import spaces  # noqa: E402

from vae_nav.gated_reward import EXPLORE, GOAL  # noqa: E402
from vae_nav.gated_tqc_learner import GatedTQCLearner  # noqa: E402
from vae_nav.mode_split_buffer import ModeSplitBuffer  # noqa: E402

# The real gated layout: 12 + 128 + 4 + 1 = 145.
OBS = spaces.Dict({
    "goal": spaces.Box(-np.inf, np.inf, (12,), np.float32),
    "latent": spaces.Box(-np.inf, np.inf, (128,), np.float32),
    "vector": spaces.Box(-5.1, 5.1, (4,), np.float32),
    "mode": spaces.Box(0.0, 1.0, (1,), np.float32)})
ACT = spaces.Box(np.array([-0.5, 0.0], np.float32),
                 np.array([0.5, 0.6], np.float32))


def _obs(mode, rng):
    # The goal block is ZEROED in explore mode, as VAEGoalWrapper does.
    goal = (rng.normal(size=12).astype(np.float32) if mode == GOAL
            else np.zeros(12, np.float32))
    return {"goal": goal,
            "latent": rng.normal(size=128).astype(np.float32),
            "vector": rng.normal(size=4).astype(np.float32),
            "mode": np.array([float(mode)], np.float32)}


def _fill(buf, n_goal, n_explore, seed=0, goal_reward=100.0,
          explore_reward=10.0):
    """Insert with the real per-mode reward SCALES, order 100 against order 10."""
    rng = np.random.default_rng(seed)
    for mode, count, scale in ((GOAL, n_goal, goal_reward),
                               (EXPLORE, n_explore, explore_reward)):
        for _ in range(count):
            o = _obs(mode, rng)
            buf.insert(dict(observations=o,
                            actions=rng.uniform(-0.5, 0.5, 2).astype(np.float32),
                            rewards=np.float32(scale * rng.normal()),
                            masks=np.float32(1.0), dones=np.float32(0.0),
                            next_observations=_obs(mode, rng),
                            mode=mode),
                       mode=mode)
    return buf


def _agent(**kw):
    opts = dict(seed=0, observations=OBS.sample(), actions=ACT.sample(),
                discount=0.99, discount_e=0.99, hidden_dims=(32, 32),
                num_qs=2, num_quantiles=5, top_quantiles_to_drop_per_net=1,
                target_entropy=-2.0, action_space=ACT)
    opts.update(kw)
    return GatedTQCLearner(**opts)


def _flat(params):
    return np.concatenate([np.asarray(x).ravel()
                           for x in jax.tree_util.tree_leaves(params)])


def test_a_real_buffer_batch_drives_a_real_update():
    buf = _fill(ModeSplitBuffer(OBS, ACT, 1000), 200, 200)
    a = _agent()
    g0, e0 = _flat(a._critic.params), _flat(a._critic_e.params)
    info = a.update(buf.sample(16))
    assert not np.array_equal(g0, _flat(a._critic.params))
    assert not np.array_equal(e0, _flat(a._critic_e.params))
    for k, v in info.items():
        assert np.isfinite(float(v)), (k, v)


def test_the_learner_reads_the_mode_the_buffer_routed_by():
    # THE SEAM. The buffer routes on its `mode` argument and strips the key;
    # the learner reads observations["mode"]. If those disagreed, frac_goal
    # would not match the composition the buffer was asked for.
    buf = _fill(ModeSplitBuffer(OBS, ACT, 1000, goal_frac=0.25), 200, 200)
    a = _agent()
    assert abs(float(a.update(buf.sample(16))["frac_goal"]) - 0.25) < 1e-6


def test_the_goal_critic_is_untouched_by_an_explore_only_buffer():
    # End to end, with the real reward-scale asymmetry in the rows.
    buf = _fill(ModeSplitBuffer(OBS, ACT, 1000), 0, 200)
    a = _agent()
    before = _flat(a._critic.params)
    a.update(buf.sample(16))
    assert np.array_equal(before, _flat(a._critic.params))


def test_the_explore_critic_is_untouched_by_a_goal_only_buffer():
    buf = _fill(ModeSplitBuffer(OBS, ACT, 1000), 200, 0)
    a = _agent()
    before = _flat(a._critic_e.params)
    a.update(buf.sample(16))
    assert np.array_equal(before, _flat(a._critic_e.params))


def test_composition_holds_against_a_ninety_ten_insert_skew():
    # CARLA's real skew: the goal cap is 2000 steps against 500, so goal rows
    # dominate. The batch must still be half and half.
    buf = _fill(ModeSplitBuffer(OBS, ACT, 2000), 900, 100)
    a = _agent()
    assert abs(float(a.update(buf.sample(32))["frac_goal"]) - 0.5) < 1e-6


def test_repeated_updates_stay_finite_under_asymmetric_reward_scales():
    # The failure this guards: the larger-scale mode driving the shared encoder
    # to a magnitude the smaller-scale critic cannot fit, ending in nan.
    buf = _fill(ModeSplitBuffer(OBS, ACT, 2000), 400, 400,
                goal_reward=100.0, explore_reward=1.0)
    a = _agent()
    for _ in range(15):
        info = a.update(buf.sample(16))
    for k, v in info.items():
        assert np.isfinite(float(v)), (k, v)


def test_both_q_estimates_are_reported_separately():
    # A single blended q would look healthy while hiding a starved critic.
    buf = _fill(ModeSplitBuffer(OBS, ACT, 2000), 400, 400)
    a = _agent()
    for _ in range(10):
        info = a.update(buf.sample(16))
    assert float(info["q"]) != float(info["q_e"])
