"""Uncertainty scores for the uncertainty maps: epistemic = critic disagreement, aleatoric = return spread,
both read from the EXPLORE critic with the mode bit set to explore."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uncertainty_map import aleatoric, epistemic, explore_quantiles, with_explore_mode  # noqa: E402


def test_epistemic_is_spread_of_critic_means():
    # 2 critics, 1 state, 3 quantiles: critic means 1 and 3 -> std 1
    q = np.array([[[0.0, 1.0, 2.0]], [[2.0, 3.0, 4.0]]])
    assert np.allclose(epistemic(q), [1.0])


def test_aleatoric_is_mean_quantile_spread_and_ignores_critic_offset():
    # both critics have the same quantile spread (std sqrt(2/3)); only their means differ
    q = np.array([[[0.0, 1.0, 2.0]], [[10.0, 11.0, 12.0]]])
    assert np.allclose(aleatoric(q), [np.sqrt(2.0 / 3.0)])
    assert np.allclose(epistemic(np.array([[[5.0, 5.0, 5.0]], [[5.0, 5.0, 5.0]]])), [0.0])


def test_mode_bit_is_set_to_explore_without_touching_the_input():
    obs = {"latent": np.ones((2, 3)), "mode": np.ones((2, 1))}
    out = with_explore_mode(obs)
    assert np.all(out["mode"] == 0.0) and np.all(obs["mode"] == 1.0)
    assert out["latent"] is obs["latent"]


class _Critic:
    def __init__(self, value):
        self.params = {}
        self.value = value

    def apply_fn(self, variables, observations, actions):
        return np.full((5, len(actions), 25), self.value)


class _Agent:
    def __init__(self):
        self._critic = _Critic(-1.0)      # goal critic
        self._critic_e = _Critic(7.0)     # explore critic
        self.seen_modes = None

    def eval_actions(self, observations):
        self.seen_modes = np.asarray(observations["mode"]).copy()
        return np.zeros((len(observations["mode"]), 2))


def test_quantiles_come_from_the_explore_critic_at_the_explore_mode_action():
    agent = _Agent()
    q = explore_quantiles(agent, {"latent": np.ones((4, 3)), "mode": np.ones((4, 1))})
    assert q.shape == (5, 4, 25) and np.all(q == 7.0)
    assert np.all(agent.seen_modes == 0.0)
