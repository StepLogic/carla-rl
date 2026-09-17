"""The oracle coin must not be the mode coin.

GatedReward seeds its mode draw with the env seed (SEED for env 0) and the
trainer's oracle coin was seeded with SEED too -- identical uniforms, one per
episode, so u < EXPERT_PROB (0.4) implied u < EXPLORE_PROB (0.5): the oracle
only ever drove explore episodes. The trainer now seeds the coin from
[SEED, 0x0E5]; this pins that the two streams disagree.
"""
import numpy as np

from vae_nav.expert_oracle import should_use_expert


def test_expert_coin_stream_differs_from_mode_stream():
    seed = 42
    mode_rng = np.random.default_rng(seed)               # what GatedReward does
    coin_rng = np.random.default_rng([seed, 0x0E5])      # what the trainer does
    explore = [mode_rng.random() < 0.5 for _ in range(200)]
    armed = [should_use_expert(coin_rng, 0.4) for _ in range(200)]
    armed_goal = sum(a and not e for a, e in zip(armed, explore))
    assert armed_goal > 0, "oracle never lands on a goal-mode episode"


def test_lockstep_streams_would_reproduce_the_bug():
    seed = 42
    mode_rng, coin_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    explore = [mode_rng.random() < 0.5 for _ in range(200)]
    armed = [should_use_expert(coin_rng, 0.4) for _ in range(200)]
    assert all(e for a, e in zip(armed, explore) if a)
