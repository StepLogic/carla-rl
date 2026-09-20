"""VC1_READOUT: the grid readout is pure math on patch tokens; the real-model
probe (scripts/vc1_grid_probe.py) verifies the extraction against the live
ViT -- these tests only guard the wiring."""
import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

def test_pool_grid_shape_and_values():
    from vae_nav.vc1_source import pool_grid
    tokens = np.arange(196 * 768, dtype=np.float32).reshape(196, 768)
    g = pool_grid(tokens, pool=2)
    assert g.shape == (37632,)
    grid = g.reshape(7, 7, 768)
    # 2x2 average over the token grid: cell (0,0) averages tokens (0,0),(0,1),(1,0),(1,1)
    t = tokens.reshape(14, 14, 768)
    np.testing.assert_allclose(grid[0, 0], t[:2, :2].reshape(-1, 768).mean(0))

def test_readout_env_selects_grid(monkeypatch):
    import importlib
    import vae_nav.vc1_source as m
    monkeypatch.setenv("VC1_READOUT", "grid2")
    importlib.reload(m)
    assert m.READOUT == "grid2"
    src = m.VcSource.__new__(m.VcSource)   # no torch load; attribute wiring only
    assert m.VC1_GRID_DIM == 37632
    monkeypatch.delenv("VC1_READOUT")
    importlib.reload(m)
    assert m.READOUT == "cls"

def test_grid_backend_name_and_identity(monkeypatch):
    import importlib, vae_nav.vc1_source as m
    monkeypatch.setenv("VC1_READOUT", "grid2"); importlib.reload(m)
    monkeypatch.setattr(m.VcSource, "_load_model", lambda self: None)
    src = m.VcSource.__new__(m.VcSource)
    src.name = m.backend_name()   # the name/identity functions read the env
    src.latent_dim = m.VC1_GRID_DIM
    assert src.name == "vc1_vitb_grid2"
    monkeypatch.delenv("VC1_READOUT"); importlib.reload(m)