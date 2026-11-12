"""The VC-1 encoder source. Heavy tests skip unless vc_models is importable."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

vc = pytest.importorskip("vc_models")


def test_identity_names_the_backend_and_dim():
    from vae_nav.vc1_source import VcSource
    src = VcSource(device="cpu")
    assert src.name == "vc1_vitb"
    assert src.latent_dim == 768
    # The trainer's town-leak check reads this unconditionally on every
    # backend (train_carla_tqc.py); a backend without it crashes at startup.
    assert src.trained_on_towns == []
    ident = src.identity()
    assert ident["latent_backend"] == "vc1_vitb"
    assert ident["latent_dim"] == 768


@pytest.mark.slow
def test_latent_output_shape_and_dtype():
    from vae_nav.vc1_source import VcSource
    src = VcSource(device="cpu")
    rgb = np.zeros((120, 160, 3), np.uint8)
    z = src.latent(rgb)
    assert z.shape == (768,) and z.dtype == np.float32
    assert np.isfinite(z).all()


def test_from_env_refuses_unknown_backends(monkeypatch):
    monkeypatch.setenv("PERCEPTION", "nonsense")
    from vae_nav.latent_source import from_env
    with pytest.raises(SystemExit):
        from_env()