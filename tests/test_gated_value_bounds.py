"""Per-critic value bounds: the goal critic is clipped, the explore critic is not.

WHY BOUNDS EXIST AT ALL. Under the SORB minimum-time reward the goal critic runs
at gamma = 1.0, where V(s) = -c + V(s') has no bounded fixed point on a state the
agent never escapes, and a truncated episode bootstraps rather than terminating.
The goal return is bounded in [-step_cost * max_steps, 0] BY CONSTRUCTION, so a
target outside that window is unrealisable rather than merely unlikely and
clipping adds information rather than bias.

WHY THEY MUST BE PER CRITIC, which is the whole point of this file. The obvious
route is the existing TQC_TARGET_MIN/TQC_TARGET_MAX in critic_updater.py. Those
are read at module import and baked into every trace, so they would clip the
EXPLORE critic too -- and the explore critic's targets are POSITIVE. A [-1000, 0]
window would flatten every one of them to zero, leaving a critic that has learnt
nothing while its loss curve looks entirely healthy. That is the failure this
file exists to make impossible.
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
    opts = dict(seed=0, observations=OBS.sample(), actions=ACT.sample(),
                discount=1.0, discount_e=0.99, hidden_dims=(32, 32),
                num_qs=2, num_quantiles=5, top_quantiles_to_drop_per_net=1,
                target_entropy=-2.0, action_space=ACT,
                value_floor=-10.0, value_ceiling=0.0)
    opts.update(kw)
    return GatedTQCLearner(**opts)


def _batch(mode, n=8, seed=0, reward=1.0):
    rng = np.random.default_rng(seed)

    def obs():
        return {"goal": rng.normal(size=(n, 12)).astype(np.float32),
                "latent": rng.normal(size=(n, 128)).astype(np.float32),
                "vector": rng.normal(size=(n, 4)).astype(np.float32),
                "mode": np.full((n, 1), float(mode), np.float32)}
    return freeze({
        "observations": obs(), "next_observations": obs(),
        "actions": rng.uniform(-0.5, 0.5, size=(n, 2)).astype(np.float32),
        "rewards": np.full((n,), float(reward), np.float32),
        "masks": np.ones((n,), np.float32),
        "dones": np.zeros((n,), np.float32)})


def test_the_bounds_reach_the_learner():
    a = _agent()
    assert (a.value_floor, a.value_ceiling) == (-10.0, 0.0)


def test_bounds_default_to_none():
    a = _agent(value_floor=None, value_ceiling=None, discount=0.99)
    assert a.value_floor is None and a.value_ceiling is None


def test_an_undiscounted_goal_critic_trains_without_diverging():
    # gamma=1.0 with a floor: the goal q must stay inside the window rather
    # than walking off, which is what the bound is for.
    a = _agent(discount=1.0)
    for i in range(20):
        info = a.update(_batch(mode=1, seed=i, reward=-1.0))
    q = float(info["q"])
    assert np.isfinite(q)
    assert -10.5 <= q <= 0.5, q


def test_the_explore_critic_is_not_clipped_by_the_goal_window():
    # THE LOAD-BEARING TEST. Explore rewards are POSITIVE. If the goal critic's
    # [-10, 0] window reached the explore critic, its targets would all be
    # clipped to zero and its q would collapse toward zero while its loss curve
    # stayed healthy. It must be free to rise above the goal ceiling.
    a = _agent()
    for i in range(25):
        info = a.update(_batch(mode=0, seed=i, reward=10.0))
    q_e = float(info["q_e"])
    assert np.isfinite(q_e)
    assert q_e > 0.5, (
        f"explore q={q_e}: it has been dragged to the goal critic's ceiling, "
        "which means the clip is not per critic")


def test_both_critics_train_together_under_opposite_signs():
    # The real configuration: goal rewards negative and bounded, explore
    # rewards positive and unbounded, in the same batch.
    a = _agent()
    rng = np.random.default_rng(0)
    n = 8

    def obs(modes):
        return {"goal": rng.normal(size=(n, 12)).astype(np.float32),
                "latent": rng.normal(size=(n, 128)).astype(np.float32),
                "vector": rng.normal(size=(n, 4)).astype(np.float32),
                "mode": modes.reshape(n, 1).astype(np.float32)}

    modes = np.concatenate([np.ones(4), np.zeros(4)])
    rewards = np.concatenate([np.full(4, -1.0), np.full(4, 10.0)]).astype(np.float32)
    batch = freeze({
        "observations": obs(modes), "next_observations": obs(modes),
        "actions": rng.uniform(-0.5, 0.5, size=(n, 2)).astype(np.float32),
        "rewards": rewards, "masks": np.ones((n,), np.float32),
        "dones": np.zeros((n,), np.float32)})
    for _ in range(20):
        info = a.update(batch)
    assert float(info["q"]) <= 0.5, "goal q rose above its ceiling"
    assert float(info["q_e"]) > 0.5, "explore q was clipped by the goal window"


def test_the_banner_reports_the_clip_and_that_explore_is_unclipped(capsys):
    _agent()
    out = capsys.readouterr().out
    assert "goal critic target clip" in out
    assert "UNCLIPPED" in out


def test_an_undiscounted_critic_without_a_floor_is_warned_about(capsys):
    _agent(discount=1.0, value_floor=None, value_ceiling=None)
    assert "no value floor" in capsys.readouterr().out.lower()
