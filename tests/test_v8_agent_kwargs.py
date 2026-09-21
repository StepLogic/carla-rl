"""The v8 conv-compressor agent kwargs and the augmentation override.

WHAT THIS CLOSES. Tasks 1+2 gave the wrapper a (7, 7, 768) `pixels` obs key;
D2 turns it into a TRAINABLE conv compressor purely through TQCLearner's
existing kwargs (encoder="d4pg", the cnn_* lists, latent_dim=512) and turns
OFF the augmentation that TQCLearner hardcodes on whenever encoder is not
None -- `self.augument = not encoder is None`. Cropping a semantic ViT token
grid drops tokens; it is wrong here (design decision, spec D2).

THE PIN. The default dict must be byte-identical to today's: the new keys
appear ONLY when AGENT_ENCODER is set to something other than "none", so
every existing run's kwargs are untouched and a test fails if that changes.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

pytest.importorskip("jax")
pytest.importorskip("jaxrl2")

import train_carla_tqc  # noqa: E402

# The launcher's v8 exports (Task 5). CNN_FEATURES carries spaces on purpose:
# the split must strip them.
V8_ENV = {
    "AGENT_ENCODER": "d4pg",
    "CNN_FEATURES": "64, 128, 512",
    "CNN_FILTERS": "3,3,1",
    "CNN_STRIDES": "2,2,1",
    "AGENT_LATENT_DIM": "512",
}

NEW_KEYS = ("encoder", "cnn_features", "cnn_filters", "cnn_strides",
            "latent_dim")

AUG_KEYS = ("AGENT_ENCODER", "CNN_FEATURES", "CNN_FILTERS", "CNN_STRIDES",
            "AGENT_LATENT_DIM", "AGENT_AUGMENT_PIXELS")


@pytest.fixture
def clean_env(monkeypatch):
    for k in AUG_KEYS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


# -- agent_kwargs ---------------------------------------------------------

def test_defaults_reproduce_todays_dict_exactly(clean_env):
    """No new keys when AGENT_ENCODER is unset -- the default dict is pinned."""
    kw = train_carla_tqc.agent_kwargs()
    for k in NEW_KEYS:
        assert k not in kw, f"{k} leaked into the default kwargs"


def test_v8_env_produces_the_compressor_kwargs(clean_env):
    for k, v in V8_ENV.items():
        clean_env.setenv(k, v)
    kw = train_carla_tqc.agent_kwargs()
    assert kw["encoder"] == "d4pg"
    assert kw["cnn_features"] == (64, 128, 512)
    assert kw["cnn_filters"] == (3, 3, 1)
    assert kw["cnn_strides"] == (2, 2, 1)
    assert kw["latent_dim"] == 512


def test_encoder_d4pg_without_the_lists_uses_tqc_defaults(clean_env):
    """CNN_FEATURES/FILTERS/STRIDES and AGENT_LATENT_DIM default to today's
    TQCLearner defaults, so AGENT_ENCODER=d4pg alone is a valid config."""
    clean_env.setenv("AGENT_ENCODER", "d4pg")
    kw = train_carla_tqc.agent_kwargs()
    assert kw["cnn_features"] == (32, 32, 32, 32)
    assert kw["cnn_filters"] == (3, 3, 3, 3)
    assert kw["cnn_strides"] == (2, 1, 1, 1)
    assert kw["latent_dim"] == 50


def test_unknown_encoder_is_rejected(clean_env):
    clean_env.setenv("AGENT_ENCODER", "resnetv8")
    with pytest.raises(SystemExit):
        train_carla_tqc.agent_kwargs()


# -- the augmentation override --------------------------------------------

class _StubLearner:
    """Stands in for a built learner: jaxrl2 hardcodes augument=True."""

    def __init__(self):
        self.augument = True


def test_override_fires_on_d4pg_with_flag_zero(clean_env):
    clean_env.setenv("AGENT_ENCODER", "d4pg")
    clean_env.setenv("AGENT_AUGMENT_PIXELS", "0")
    learner = _StubLearner()
    train_carla_tqc.apply_augment_override(learner)
    assert learner.augument is False


def test_override_is_inert_when_the_flag_is_unset(clean_env):
    """AGENT_AUGMENT_PIXELS unset = today's behaviour (the old hardcode)."""
    clean_env.setenv("AGENT_ENCODER", "d4pg")
    learner = _StubLearner()
    train_carla_tqc.apply_augment_override(learner)
    assert learner.augument is True


def test_override_ignores_encoder_none(clean_env):
    """With no conv encoder there is no token grid to protect; also, the old
    hardcode sets augument True even for encoder="none", and we do not change
    behaviour for the existing arms."""
    clean_env.setenv("AGENT_AUGMENT_PIXELS", "0")
    learner = _StubLearner()
    train_carla_tqc.apply_augment_override(learner)
    assert learner.augument is True


# -- the agent-boundary pixels view ----------------------------------------

def test_agent_view_expands_pixels_only(clean_env):
    """The trailing-singleton view: pixels (7,7,768) -> (7,7,768,1), every
    other key untouched, idempotent, and inert without the encoder flag."""
    clean_env.setenv("AGENT_ENCODER", "d4pg")
    obs = {"goal": np.zeros((4,), np.float32),
           "pixels": np.zeros((7, 7, 768), np.float32),
           "mode": np.zeros((1,), np.float32),
           "state": np.zeros((9,), np.float32)}
    v = train_carla_tqc.agent_view(obs)
    assert v["pixels"].shape == (7, 7, 768, 1)
    assert v["goal"] is obs["goal"]
    assert v["mode"] is obs["mode"]
    assert v["state"] is obs["state"]
    # Idempotent: the (already viewed) batch must not grow a second axis.
    assert train_carla_tqc.agent_view(v)["pixels"].shape == (7, 7, 768, 1)


def test_agent_view_expands_a_sampled_batch(clean_env):
    """The update batch: pixels expand under both observations and
    next_observations, in a frozen (non-dict) mapping too."""
    from flax.core import frozen_dict

    clean_env.setenv("AGENT_ENCODER", "d4pg")
    batch = frozen_dict.FrozenDict(
        observations=frozen_dict.FrozenDict(
            goal=np.zeros((2, 4), np.float32),
            pixels=np.zeros((2, 7, 7, 768), np.float32)),
        next_observations=frozen_dict.FrozenDict(
            pixels=np.zeros((2, 7, 7, 768), np.float32)),
        actions=np.zeros((2, 2), np.float32))
    v = train_carla_tqc.agent_view(batch)
    assert v["observations"]["pixels"].shape == (2, 7, 7, 768, 1)
    assert v["next_observations"]["pixels"].shape == (2, 7, 7, 768, 1)
    assert v["actions"].shape == (2, 2)


def test_agent_view_is_inert_without_the_encoder(clean_env):
    obs = {"pixels": np.zeros((7, 7, 768), np.float32)}
    v = train_carla_tqc.agent_view(obs)
    assert v["pixels"].shape == (7, 7, 768)
    assert v is obs


def test_the_real_construction_pins_the_hardcode_then_the_override(clean_env):
    """ONE real learner construction (init on CPU costs tens of seconds),
    through the production build_agent with the v8 kwargs.

    Built with AGENT_AUGMENT_PIXELS unset it must come up with augument True
    -- jaxrl2's hardcode, pinned so a silent upstream flip cannot go
    unnoticed. The production override (the exact function build_agent wraps
    every construction site with) then turns it off. The construction also
    proves the (7,7,768) grid survives the real D4PGEncoder via agent_view,
    and that the compressor params actually exist in the critic.
    """
    from gymnasium import spaces
    from jaxrl2.agents import TQCLearner

    clean_env.setenv("AGENT_ENCODER", "d4pg")
    for k in ("CNN_FEATURES", "CNN_FILTERS", "CNN_STRIDES", "AGENT_LATENT_DIM"):
        clean_env.setenv(k, V8_ENV[k])
    obs_space = spaces.Dict({
        "goal": spaces.Box(-np.inf, np.inf, (4,), np.float32),
        "pixels": spaces.Box(-np.inf, np.inf, (7, 7, 768), np.float32),
        "mode": spaces.Box(0.0, 1.0, (1,), np.float32),
        "state": spaces.Box(-np.inf, np.inf, (9,), np.float32)})
    act = spaces.Box(np.array([0.0, 0.0], np.float32),
                     np.array([1.0, 1.0], np.float32))
    agent = train_carla_tqc.build_agent(obs_space, act, 0)
    assert type(agent) is TQCLearner
    assert agent.augument is True  # the old hardcode, pinned
    clean_env.setenv("AGENT_AUGMENT_PIXELS", "0")
    train_carla_tqc.apply_augment_override(agent)
    assert agent.augument is False
    # The compressor exists: the critic carries the pixels conv encoder, and
    # its trunk width is the design's 512.
    flat = str(agent._critic.params)
    assert "encoder_pixels" in flat


def test_build_agent_applies_the_override_at_construction(clean_env):
    """The brief's pin, end to end: AGENT_AUGMENT_PIXELS=0 -> a learner built
    by build_agent comes up with .augument False, with no post-hoc call.

    (A second real construction, budgeted deliberately: measured ~2 s on
    CPU, not the 10-30 s worst case the plan assumed.)
    """
    from gymnasium import spaces

    clean_env.setenv("AGENT_ENCODER", "d4pg")
    for k in ("CNN_FEATURES", "CNN_FILTERS", "CNN_STRIDES", "AGENT_LATENT_DIM"):
        clean_env.setenv(k, V8_ENV[k])
    clean_env.setenv("AGENT_AUGMENT_PIXELS", "0")
    obs_space = spaces.Dict({
        "goal": spaces.Box(-np.inf, np.inf, (4,), np.float32),
        "pixels": spaces.Box(-np.inf, np.inf, (7, 7, 768), np.float32),
        "mode": spaces.Box(0.0, 1.0, (1,), np.float32),
        "state": spaces.Box(-np.inf, np.inf, (9,), np.float32)})
    act = spaces.Box(np.array([0.0, 0.0], np.float32),
                     np.array([1.0, 1.0], np.float32))
    agent = train_carla_tqc.build_agent(obs_space, act, 0)
    assert agent.augument is False