"""PERCEPTION=segvae backend (2026-09-28)."""
import os
import pickle

import numpy as np
import pytest

jax = pytest.importorskip("jax")
from vae_nav import seg_vae as sv  # noqa: E402
from vae_nav.segvae_source import SegVaeSource  # noqa: E402


def test_latent_matches_the_model_encoder(tmp_path):
    m = sv.SegVAE(latent_dim=16, dec_start=30)
    x = np.random.default_rng(0).integers(0, 255, (120, 120, 3)).astype(np.uint8)
    p = m.init(jax.random.PRNGKey(0), jax.numpy.zeros((1, 120, 120, 3)), jax.random.PRNGKey(1))
    ck = tmp_path / "run" / "seg_vae.pkl"
    os.makedirs(ck.parent)
    pickle.dump({"params": jax.device_get(p), "latent_dim": 16, "dec_start": 30, "step": 7}, open(ck, "wb"))
    src = SegVaeSource(str(ck))
    z = src.latent(x)
    want = np.asarray(m.apply(p, jax.numpy.asarray(x[None].astype(np.float32) / 255.0),
                              method=sv.SegVAE.encode))[0]
    assert z.shape == (16,) and z.dtype == np.float32
    assert np.allclose(z, want, atol=1e-5)
    assert src.name == "segvae_run_s7" and src.identity()["latent_dim"] == 16


def test_from_env_selects_segvae(tmp_path, monkeypatch):
    from vae_nav import latent_source
    monkeypatch.setenv("PERCEPTION", "segvae")
    monkeypatch.setenv("SEGVAE_CKPT", str(tmp_path / "missing.pkl"))
    with pytest.raises(SystemExit, match="SEGVAE_CKPT"):
        latent_source.from_env()
