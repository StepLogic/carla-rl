"""The ResNet's frame layout, and photometric augmentation for it (2026-11-05).

LAYOUT. The learner reads the row's pixel block as (3, s, s) channels-first
(gated_recurrent_tqc: "the /255 RGB frame, (3, s, s) C-order ... _embed's
reshape yields (N, 3, s, s) pixels with no permute"), but mono_row stored the
frame (s, s, 3) channels-LAST. The ResNet therefore saw three horizontal BANDS
of the picture (top / middle / bottom third) as its "colour channels", with
RGB interleaved along each row -- measured: a top-half-white frame gives plane
means 1.0 / 0.5 / 0.0, a pure-red frame gives 1,0,0,1,0,0... in every plane.
Consistent between training and driving, so every run still learned, but the
convolutions lost the image's spatial structure and the pretrained DEBiT trunk
loaded into the encoder saw nothing like the images it was trained on.
MONO_CHW=1 stores the frame channels-first; off by default so every earlier
checkpoint keeps the layout it was trained on.

AUGMENTATION (user 2026-11-05: "data augmentation for the resnet part to help
with Town03"): Town03's faults sit in a dark lamp-lit tunnel and on glare-
washed roads, lighting the training towns never show. Brightness, contrast and
per-colour gain on the cloning batches' frames, one draw per chunk (a stretch
of driving under one lighting), never touching mode, speed or binoc.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
torch = pytest.importorskip("torch")

from torch_rl import pixel_rows as pxr  # noqa: E402

P, S, D = pxr.MONO_PIX, pxr.MONO_IMG, pxr.BINOC_DIM


def _planes(row):
    """What the learner hands the ResNet: the pixel block reshaped (3, s, s)."""
    return np.asarray(row[:P]).reshape(3, S, S)


@pytest.fixture
def chw_on(monkeypatch):
    monkeypatch.setattr(pxr, "MONO_CHW", True)


def test_with_mono_chw_the_resnet_sees_real_colour_channels(chw_on):
    red = np.zeros((S, S, 3), np.uint8)
    red[..., 0] = 255
    p = _planes(pxr.mono_row(red, 1.0, np.zeros(D, np.float32)))
    assert np.all(p[0] == 1.0) and np.all(p[1] == 0.0) and np.all(p[2] == 0.0)


def test_with_mono_chw_the_resnet_sees_the_picture_upright(chw_on):
    top = np.zeros((S, S, 3), np.uint8)
    top[: S // 2] = 255
    p = _planes(pxr.mono_row(top, 1.0, np.zeros(D, np.float32)))
    for c in range(3):
        assert np.all(p[c, : S // 2] == 1.0) and np.all(p[c, S // 2:] == 0.0)


def test_the_old_layout_is_kept_by_default():
    """Default off: a pure-red frame still interleaves, as every checkpoint saw it."""
    assert not pxr.MONO_CHW
    red = np.zeros((S, S, 3), np.uint8)
    red[..., 0] = 255
    p = _planes(pxr.mono_row(red, 1.0, np.zeros(D, np.float32)))
    assert p[0].ravel()[:6].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def _chunk(B=4, L=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.rand(B, L, P + 1 + D, generator=g) * 0.6 + 0.2
    return {"obs": obs}


def test_augmentation_changes_only_the_frame_and_stays_in_range():
    from torch_rl.bc_trainer import augment_frames
    obs = _chunk()["obs"]
    out = augment_frames(obs, torch.Generator().manual_seed(1))
    assert out.shape == obs.shape
    assert torch.equal(out[..., P:], obs[..., P:])           # mode / speed / binoc untouched
    assert not torch.allclose(out[..., :P], obs[..., :P])
    assert float(out[..., :P].min()) >= 0.0 and float(out[..., :P].max()) <= 1.0
    assert torch.equal(obs, _chunk()["obs"])                 # input not modified in place


def test_one_lighting_draw_per_chunk():
    """Brightness only: every frame of a chunk is scaled by the same factor."""
    from torch_rl.bc_trainer import augment_frames
    obs = _chunk()["obs"]
    # +-20% on pixels in [0.2, 0.8] never clips, so the ratio is the draw itself
    out = augment_frames(obs, torch.Generator().manual_seed(2), bright=0.2, contrast=0.0,
                         color=0.0)
    ratio = out[..., :P] / obs[..., :P]
    for b in range(obs.shape[0]):
        assert torch.allclose(ratio[b], ratio[b, 0, 0].expand_as(ratio[b]), atol=1e-5)
    assert not torch.allclose(ratio[0, 0, 0], ratio[1, 0, 0])


@pytest.mark.parametrize("chw", [False, True])
def test_colour_gain_hits_the_real_colour_channel(monkeypatch, chw):
    """Per-colour gain must scale RED where red really is, in either layout."""
    from torch_rl.bc_trainer import augment_frames
    monkeypatch.setattr(pxr, "MONO_CHW", chw)
    red = np.zeros((S, S, 3), np.uint8)
    red[..., 0] = 128
    row = torch.as_tensor(pxr.mono_row(red, 1.0, np.zeros(D, np.float32)))
    obs = row.reshape(1, 1, -1).repeat(2, 3, 1)
    out = augment_frames(obs, torch.Generator().manual_seed(3), bright=0.0, contrast=0.0,
                         color=0.3)
    pix = out[..., :P].reshape(2, 3, S, S, 3) if not chw else out[..., :P].reshape(2, 3, 3, S, S)
    green_blue = pix[..., 1:] if not chw else pix[:, :, 1:]
    assert float(green_blue.abs().max()) == 0.0              # still no green or blue
    red_now = pix[..., 0] if not chw else pix[:, :, 0]
    assert not torch.allclose(red_now, torch.full_like(red_now, 128 / 255))


@pytest.mark.parametrize("on,expect", [("1", 3), ("", 0)])
def test_bc_aug_augments_every_training_batch_and_never_validation(tmp_path, monkeypatch, on, expect):
    import torch_rl.bc_trainer as bct
    from tests.test_bc_trainer import _bc_env, _shards, _small_pixel_agent
    calls = []
    real = bct.augment_frames
    monkeypatch.setattr(bct, "augment_frames",
                        lambda obs, **kw: calls.append(kw) or real(obs, **kw))
    _shards(tmp_path / "data")
    _bc_env(monkeypatch, tmp_path / "data", BC_AUG=on)       # 3 updates, validation at the end
    bct.run_bc_train(str(tmp_path / "out"), make_agent=_small_pixel_agent)
    assert len(calls) == expect
    if expect:
        assert {"bright", "contrast", "color"} <= set(calls[0])
