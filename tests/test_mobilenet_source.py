"""MobileNetV3-Small features[:4] as the perception backend.

Replaces the frozen VAE. The tests that matter here are the ones guarding
things that fail SILENTLY: a hardcoded feature width, a missing ImageNet
normalisation, a backend swap that no longer trips assert_matches, and a
replay-buffer blowup.
"""
import os

import numpy as np
import pytest

from vae_nav.latent_source import LatentSource, from_env
from vae_nav.mobilenet_source import (IMAGENET_MEAN, IMAGENET_STD,
                                      MobileNetSource)


@pytest.fixture(scope="module")
def src():
    return MobileNetSource(image_size=120, pool=4)


def rgb(seed=0, size=120):
    return np.random.default_rng(seed).integers(
        0, 256, (size, size, 3), dtype=np.uint8)


def test_latent_shape_matches_declared_dim(src):
    v = src.latent(rgb())
    assert v.shape == (src.latent_dim,)
    assert v.dtype == np.float32


def test_pooled_dim_is_24_channels_by_4_by_4(src):
    assert src.latent_dim == 24 * 4 * 4 == 384


def test_dim_is_derived_not_hardcoded():
    """A literal 2904 was only ever correct at 84x84.

    Changing the image size must change the width, not raise thousands of
    steps into training.
    """
    small = MobileNetSource(image_size=84, pool=0)
    big = MobileNetSource(image_size=120, pool=0)
    assert small.latent_dim == 24 * 11 * 11 == 2904
    assert big.latent_dim == 24 * 15 * 15 == 5400
    assert small.latent_dim != big.latent_dim


def test_pooling_is_what_makes_the_replay_buffer_affordable():
    """MEASURED: the raw flatten is 43 GB of replay at 1M capacity.

    obs + next_obs, float32. The VAE's 128-d was 1.0 GB. This repo has already
    filled a root filesystem once.
    """
    flat = MobileNetSource(image_size=120, pool=0).latent_dim
    pooled = MobileNetSource(image_size=120, pool=4).latent_dim

    def gb(dim):
        return dim * 4 * 1_000_000 * 2 / 1e9

    assert gb(flat) > 40, "the flatten really is that big"
    assert gb(pooled) < 4, "pooling has to bring it under a few GB"
    assert pooled < flat / 10


def test_imagenet_normalisation_is_applied(src):
    """Pretrained weights expect ImageNet statistics.

    Feeding [0,1] instead degrades the features silently rather than raising,
    so this checks the prepared tensor really is standardised.
    """
    x = src._prepare(np.full((120, 120, 3), 128, np.uint8)).cpu().numpy()[0]
    expected = ((128 / 255.0) - IMAGENET_MEAN) / IMAGENET_STD
    for c in range(3):
        assert x[c].mean() == pytest.approx(expected[c], abs=0.05), (
            "channel not standardised with ImageNet mean/std")


def test_normalisation_actually_shifts_the_input():
    """Guards against a no-op normalisation constant."""
    assert not np.allclose(IMAGENET_MEAN, 0.0)
    assert not np.allclose(IMAGENET_STD, 1.0)


def test_encoder_is_frozen(src):
    assert not any(p.requires_grad for p in src.features.parameters())
    assert not src.features.training, "must be in eval mode (BN/dropout)"


def test_deterministic_for_the_same_image(src):
    a, b = src.latent(rgb(1)), src.latent(rgb(1))
    np.testing.assert_array_equal(a, b)


def test_different_images_give_different_latents(src):
    a, b = src.latent(rgb(1)), src.latent(rgb(2))
    assert not np.allclose(a, b), "encoder is not discriminating at all"


def test_identity_names_the_backend_distinctly(src):
    ident = src.identity()
    assert "mobilenet" in ident["latent_backend"]
    assert ident["latent_dim"] == src.latent_dim
    assert "vae" not in ident["latent_backend"]


def test_assert_matches_refuses_a_cross_backend_evaluation(src):
    """THE GUARD THAT MATTERS.

    The two backends differ in width AND meaning. Evaluating a VAE-trained
    policy against MobileNet features would compare a run against a different
    observation distribution and report it as a result.
    """
    vae_identity = {"latent_backend": "lane_obstacle_vae"}
    with pytest.raises(SystemExit):
        LatentSource.assert_matches(vae_identity, src)
    LatentSource.assert_matches({"latent_backend": src.name}, src)


def test_from_env_selects_the_backend(monkeypatch):
    monkeypatch.setenv("PERCEPTION", "mobilenet")
    assert isinstance(from_env(), MobileNetSource)
    monkeypatch.setenv("PERCEPTION", "vae")
    monkeypatch.setenv("VAE_CKPT", "")
    assert isinstance(from_env(), LatentSource)


def test_unknown_backend_fails_loudly(monkeypatch):
    """A silent fallback would make a backend ablation compare a run to itself."""
    monkeypatch.setenv("PERCEPTION", "resnet50")
    with pytest.raises(SystemExit):
        from_env()
