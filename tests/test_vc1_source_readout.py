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


# --- pyramid readout (2026-09-21): global mean + four 14x14 quadrant means --
def test_pool_pyramid_shape_dtype_and_block_order():
    from vae_nav.vc1_source import pool_pyramid
    # Quadrant-constant tokens (tl=1, tr=2, bl=3, br=4): every pooled block
    # must equal its constant and the global mean must be 2.5. This pins the
    # BLOCK ORDER [global, tl, tr, bl, br] -- a reordered concatenate would
    # silently scramble the layout the critic learns.
    t = np.empty((14, 14, 768), np.float32)
    t[:7, :7], t[:7, 7:] = 1.0, 2.0
    t[7:, :7], t[7:, 7:] = 3.0, 4.0
    p = pool_pyramid(t.reshape(196, 768))
    assert p.shape == (3840,) and p.dtype == np.float32
    np.testing.assert_allclose(p[0], 2.5)          # global = (1+2+3+4)/4
    assert p[768] == 1.0 and p[2 * 768] == 2.0
    assert p[3 * 768] == 3.0 and p[4 * 768] == 4.0
    # every slot of each block is the same constant (constant input)
    for k, v in enumerate([2.5, 1.0, 2.0, 3.0, 4.0]):
        assert (p[k * 768:(k + 1) * 768] == v).all()


def test_pool_pyramid_global_equals_quadrant_mean():
    from vae_nav.vc1_source import pool_pyramid
    tokens = np.arange(196 * 768, dtype=np.float32).reshape(196, 768)
    p = pool_pyramid(tokens)
    # linearity: the global mean must equal the mean of the four equal-size
    # quadrant means -- a cheap global sanity check on the quadrant split.
    np.testing.assert_allclose(p[:768],
                               p[768:].reshape(4, 768).mean(0), rtol=1e-5)


def test_readout_env_selects_pyramid(monkeypatch):
    import importlib, vae_nav.vc1_source as m
    monkeypatch.setenv("VC1_READOUT", "pyramid"); importlib.reload(m)
    assert m.READOUT == "pyramid"
    assert m.VC1_PYRAMID_DIM == 3840
    monkeypatch.setattr(m.VcSource, "_load_model", lambda self: (None, None))
    src = m.VcSource(device="cpu")
    assert src.latent_dim == 3840
    assert src.name == "vc1_vitb_pyramid"
    ident = src.identity()
    assert ident["latent_backend"] == "vc1_vitb_pyramid"
    assert ident["readout"] == "pyramid" and ident["latent_dim"] == 3840
    monkeypatch.delenv("VC1_READOUT"); importlib.reload(m)
    assert m.READOUT == "cls"


def test_readout_refuses_unknown_value():
    import subprocess
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys, os; sys.path.insert(0, 'src'); "
         "os.environ['VC1_READOUT'] = 'mean9'; import vae_nav.vc1_source"],
        capture_output=True, text=True)
    assert r.returncode != 0 and "cls|grid2|pyramid" in (r.stderr + r.stdout)