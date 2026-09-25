"""VC1_MODEL selects the backbone; every readout width follows from it.

DEFAULT IS ViT-B 768 and must stay byte-identical: the running experiments,
every checkpoint on disk and the 3854-d observation all assume it.

The point of these tests is the POOLING RESHAPES. They used to hardcode 768,
so a wider backbone would not have raised -- it would have reshaped a
(196, 1024) token array against a 768 stride and silently scrambled the
features. A wrong-but-plausible observation is the worst failure mode here,
so the reshape width is pinned for both backbones.
"""
import importlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def _load(model=None, readout="pyramid"):
    if model is None:
        os.environ.pop("VC1_MODEL", None)
    else:
        os.environ["VC1_MODEL"] = model
    os.environ["VC1_READOUT"] = readout
    import vae_nav.vc1_source as v
    return importlib.reload(v)


@pytest.fixture(autouse=True)
def _restore_env():
    keep = {k: os.environ.get(k) for k in ("VC1_MODEL", "VC1_READOUT")}
    yield
    for k, val in keep.items():
        if val is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = val
    importlib.reload(importlib.import_module("vae_nav.vc1_source"))


def test_default_is_vitb_768():
    """No VC1_MODEL set: the shipped backbone, unchanged."""
    v = _load(None)
    assert v.VC1_MODEL_NAME == "vc1_vitb"
    assert v.VC1_EMBED_DIM == 768
    assert v.VC1_PYRAMID_DIM == 3840          # the live 3854-d observation
    assert v.VC1_GRID_DIM == 37632


def test_vitl_widths():
    v = _load("vc1_vitl")
    assert v.VC1_EMBED_DIM == 1024
    assert v.VC1_PYRAMID_DIM == 5120
    assert v.VC1_GRID_DIM == 50176


def test_unknown_backbone_refuses():
    with pytest.raises(SystemExit):
        _load("vc1_vith")


@pytest.mark.parametrize("model,d", [("vc1_vitb", 768), ("vc1_vitl", 1024)])
def test_pyramid_pooling_is_width_correct(model, d):
    """[global, tl, tr, bl, br] means of the 14x14 grid, at the right width.

    Built so each quadrant has a distinct constant value: if the reshape used
    the wrong stride the quadrant means would blend and these exact equalities
    would fail.
    """
    v = _load(model)
    t = np.zeros((14, 14, d), np.float32)
    t[:7, :7] = 1.0
    t[:7, 7:] = 2.0
    t[7:, :7] = 3.0
    t[7:, 7:] = 4.0
    out = v.pool_pyramid(t.reshape(196, d))
    assert out.shape == (5 * d,)
    blocks = out.reshape(5, d)
    assert np.allclose(blocks[0], 2.5)                       # global mean
    for i, want in enumerate((1.0, 2.0, 3.0, 4.0), start=1):
        assert np.allclose(blocks[i], want), f"quadrant {i} blended"


@pytest.mark.parametrize("model,d", [("vc1_vitb", 768), ("vc1_vitl", 1024)])
def test_grid_pooling_is_width_correct(model, d):
    v = _load(model)
    t = np.arange(14 * 14 * d, dtype=np.float32).reshape(14, 14, d)
    out = v.pool_grid(t.reshape(196, d))
    assert out.shape == (7 * 7 * d,)
    # cell (0,0) is the mean of the 2x2 patch block at the top-left
    assert np.allclose(out.reshape(7, 7, d)[0, 0],
                       t[:2, :2].reshape(4, d).mean(axis=0))


def test_backend_name_carries_model_and_readout():
    """The name reaches the checkpoint's identity record, so a ViT-L policy
    can never be loaded as a ViT-B one by accident."""
    assert _load("vc1_vitb", "pyramid").backend_name() == "vc1_vitb_pyramid"
    assert _load("vc1_vitl", "pyramid").backend_name() == "vc1_vitl_pyramid"
    assert _load("vc1_vitb", "cls").backend_name() == "vc1_vitb"
