"""The mode bit as its own observation key, and the goal block it masks.

WHY A SEPARATE KEY RATHER THAN A FIFTH SLOT IN `vector`. PixelMultiplexer gives
every non-image key its own Dense -> LayerNorm -> tanh projection before
concatenating. Appending the bit to `vector` would push a single 0/1 value
through a LayerNorm shared with steer, throttle and speed, where it would be
partly normalised away. Its own key gives it its own projection, and it leaves
the 4-vector layout untouched.

WHAT THIS FILE PROTECTS, beyond the layout. OBS_WITH_GOAL=0 has an existing,
load-bearing meaning: omit the 12-d goal block, 144 -> 132. run_lanefollow_nogoal.sh
depends on it and every carla_lanefollow_nogoal checkpoint is 132-d because of
it. The gate must NOT overload that flag. Gating needs the goal PRESENT and
zeroed, because one actor cannot accept two input shapes.
"""
import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

pytest.importorskip("gymnasium")

from gymnasium import spaces  # noqa: E402


def _reload(**env):
    """Module-level flags are read at import, so a layout change needs a reload.

    Same pattern the explorer layout tests use. Restores nothing on purpose:
    every test sets every flag it depends on.
    """
    for k, v in env.items():
        os.environ[k] = v
    import vae_nav.vae_goal_wrapper as m
    return importlib.reload(m)


GATED = dict(OBS_WITH_MODE="1", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="1")


# ---------------------------------------------------------------- flags ----

def test_mode_key_is_off_by_default():
    m = _reload(OBS_WITH_MODE="0", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="1")
    assert m.WITH_MODE is False


def test_mode_key_is_on_when_requested():
    m = _reload(**GATED)
    assert m.WITH_MODE is True
    assert m.MODE_DIM == 1


def test_mode_encoding_matches_the_reward_wrapper():
    # GOAL=1, EXPLORE=0. Inverting this silently swaps which critic trains on
    # which mode, and nothing downstream would raise.
    m = _reload(**GATED)
    from vae_nav import gated_reward
    assert (m.GOAL, m.EXPLORE) == (gated_reward.GOAL, gated_reward.EXPLORE)


# ------------------------------------------------------------ coherence ----

def test_gating_requires_the_goal_block():
    m = _reload(OBS_WITH_MODE="1", OBS_WITH_GOAL="0", OBS_WITH_VECTOR="1")
    with pytest.raises(ValueError, match="requires the goal block"):
        m.check_gated_coherence()


def test_gating_requires_the_vector_block():
    # BOTH gated rewards are speed-gated, so a policy that cannot observe its
    # own speed sees the gate as an unobservable coin flip.
    m = _reload(OBS_WITH_MODE="1", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="0")
    with pytest.raises(ValueError, match="requires the vector block"):
        m.check_gated_coherence()


def test_a_coherent_gated_configuration_is_accepted():
    m = _reload(**GATED)
    m.check_gated_coherence()


def test_coherence_check_is_inert_when_not_gating():
    # It must not fire on the two existing arms, which set neither flag.
    m = _reload(OBS_WITH_MODE="0", OBS_WITH_GOAL="0", OBS_WITH_VECTOR="0")
    m.check_gated_coherence()


# ----------------------------------------------------------- the masking ----

def test_explore_mode_zeroes_the_goal_block():
    m = _reload(**GATED)
    block = np.arange(m.GOAL_DIM, dtype=np.float32) + 1.0
    assert np.all(m.apply_mode(block.copy(), m.EXPLORE) == 0.0)


def test_goal_mode_leaves_the_goal_block_untouched():
    m = _reload(**GATED)
    block = np.arange(m.GOAL_DIM, dtype=np.float32) + 1.0
    assert np.array_equal(m.apply_mode(block.copy(), m.GOAL), block)


def test_masking_preserves_shape_and_dtype():
    # ZEROED, NOT OMITTED: one actor cannot take two input shapes.
    m = _reload(**GATED)
    block = np.ones(m.GOAL_DIM, np.float32)
    out = m.apply_mode(block, m.EXPLORE)
    assert out.shape == (m.GOAL_DIM,)
    assert out.dtype == np.float32


# -------------------------------------------------- the observation space ----

class _StubSource:
    """Stands in for LatentSource so the space can be built without a VAE."""
    name = "stub"
    latent_dim = 128

    def latent(self, rgb):
        return np.zeros(self.latent_dim, np.float32)

    def identity(self):
        return {"backend": "stub", "latent_dim": self.latent_dim}


import gymnasium as gym  # noqa: E402


class _StubEnv(gym.Env):
    """Real gymnasium.Env: the Wrapper base asserts on the type."""
    metadata = {}

    def __init__(self):
        self.observation_space = spaces.Dict({})
        self.action_space = spaces.Box(
            np.array([-0.5, 0.0], np.float32), np.array([0.5, 0.6], np.float32))

    def reset(self, **kw):
        return {}, {}

    def step(self, a):
        return {}, 0.0, False, False, {}


def _wrapper(module):
    return module.VAEGoalWrapper(_StubEnv(), source=_StubSource())


def test_gated_space_has_all_four_keys_at_the_pinned_widths():
    # These widths are also pinned by tests/test_mode_split_buffer.py. If the
    # two disagree, the buffer and the wrapper are building different agents.
    m = _reload(**GATED)
    space = _wrapper(m).observation_space
    assert set(space.spaces) == {"goal", "latent", "vector", "mode"}
    assert space["goal"].shape == (12,)
    assert space["latent"].shape == (128,)
    assert space["vector"].shape == (4,)
    assert space["mode"].shape == (1,)


def test_gated_observation_is_145_wide():
    m = _reload(**GATED)
    space = _wrapper(m).observation_space
    assert sum(int(np.prod(s.shape)) for s in space.spaces.values()) == 145


def test_goal_free_arm_is_still_132_wide():
    # THE REGRESSION GUARD. run_lanefollow_nogoal.sh and every
    # carla_lanefollow_nogoal checkpoint depend on OBS_WITH_GOAL=0 meaning
    # exactly "omit the 12-d block". Adding the mode key must not change it.
    m = _reload(OBS_WITH_MODE="0", OBS_WITH_GOAL="0", OBS_WITH_VECTOR="1")
    space = _wrapper(m).observation_space
    assert set(space.spaces) == {"latent", "vector"}
    assert sum(int(np.prod(s.shape)) for s in space.spaces.values()) == 132


def test_goal_conditioned_arm_is_still_144_wide():
    m = _reload(OBS_WITH_MODE="0", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="1")
    space = _wrapper(m).observation_space
    assert sum(int(np.prod(s.shape)) for s in space.spaces.values()) == 144


def test_identity_records_whether_the_run_was_gated():
    # Merged into the checkpoint so evaluation cannot silently mismatch a gated
    # policy with a plain one.
    m = _reload(**GATED)
    assert _wrapper(m).identity().get("with_mode") is True


def test_the_banner_lists_the_mode_key_and_the_total_width(capsys):
    # CAUGHT IN A LIVE RUN. The space was right but the banner printed only
    # three keys, so a gated run read on screen as an ordinary
    # goal-conditioned one. The banner is the single place an operator
    # confirms the layout, so it must name every key in the space.
    m = _reload(**GATED)
    _wrapper(m)
    out = capsys.readouterr().out
    assert "'mode': (1,)" in out, out
    assert "145-d" in out, out
    assert "GATED" in out, out


def test_the_banner_is_unchanged_for_the_goal_free_arm(capsys):
    m = _reload(OBS_WITH_MODE="0", OBS_WITH_GOAL="0", OBS_WITH_VECTOR="1")
    _wrapper(m)
    out = capsys.readouterr().out
    assert "'mode'" not in out
    assert "132-d" in out, out
    assert "GOAL-FREE" in out


def test_the_banner_is_unchanged_for_the_goal_conditioned_arm(capsys):
    m = _reload(OBS_WITH_MODE="0", OBS_WITH_GOAL="1", OBS_WITH_VECTOR="1")
    _wrapper(m)
    out = capsys.readouterr().out
    assert "'mode'" not in out
    assert "144-d" in out, out
    assert "GATED" not in out
