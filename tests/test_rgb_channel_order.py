"""ENV_RGB_CHANNEL_FIX: obs_img was BGR because two flips cancelled.

THE DEFECT, verified 2026-09-26 by pushing a known buffer through the real code
rather than by reading it:
  1. rlib_integration sensors/sensor.py BaseCamera.parse(): BGRA -> drop alpha
     -> [:, :, ::-1]  ->  RGB.  Correct, and LOAD-BEARING: vae_nav/lane_mask.py
     reads CARLA's semantic tag out of channel 0 only because of this reversal.
  2. rlib_integration helper.py post_process_image(): cv2.COLOR_BGR2RGB on the
     already-RGB array  ->  back to BGR.  Wrong.
So every PERCEPTION=vc1 run, the VAE arms and the DINOv2 bank were fed
channel-swapped frames under ImageNet normalisation.

The fix is applied at the point of use in train_env_config_v4_image_goal, not
upstream, because neither flip can be deleted without breaking something else.
These tests pin the composition end to end -- the thing a code-reading check
cannot establish -- and pin that the escape hatch really is byte-identical, since
every checkpoint on disk was trained on the swapped frames.
"""
import importlib
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))
sys.path.insert(0, os.path.dirname(_HERE))

cv2 = pytest.importorskip("cv2")


def _post_process():
    from rlib_integration.helper import post_process_image
    return post_process_image


def _parse_like(raw_bgra):
    """BaseCamera.parse(), sensors/sensor.py:103-106, verbatim."""
    a = raw_bgra[:, :, :3]
    return np.ascontiguousarray(a[:, :, ::-1])


def _blue_bgra(h=32, w=32):
    """A pure-BLUE frame in CARLA's NATIVE buffer layout, which is BGRA."""
    raw = np.zeros((h, w, 4), np.uint8)
    raw[:, :, 0] = 255        # B
    raw[:, :, 3] = 255        # A
    return raw


def test_parse_alone_produces_rgb():
    """Step 1 is correct on its own: blue ends up in channel 2."""
    out = _parse_like(_blue_bgra())
    assert int(np.argmax(out[0, 0])) == 2, out[0, 0].tolist()


def test_the_two_flips_cancel_and_that_is_the_bug():
    """THE DEFECT ITSELF. Blue enters channel 2 and leaves in channel 0."""
    img = _post_process()(_parse_like(_blue_bgra()), normalized=False,
                          crop=False, grayscale=False, image_size=16)
    assert int(np.argmax(img[0, 0])) == 0, (
        "post_process_image no longer re-swaps; the fix below is now a THIRD "
        "flip and must be removed")


def _fix_with(flag, monkeypatch):
    monkeypatch.setenv("ENV_RGB_CHANNEL_FIX", flag)
    import configs.train_env_config_v4_image_goal as cfg
    importlib.reload(cfg)
    return cfg


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    yield
    monkeypatch.delenv("ENV_RGB_CHANNEL_FIX", raising=False)


def test_fix_on_restores_true_rgb(monkeypatch):
    cfg = _fix_with("1", monkeypatch)
    swapped = _post_process()(_parse_like(_blue_bgra()), normalized=False,
                              crop=False, grayscale=False, image_size=16)
    fixed = cfg._fix_channel_order(swapped)
    assert int(np.argmax(fixed[0, 0])) == 2, "blue must sit in channel 2"
    assert fixed.dtype == swapped.dtype
    assert fixed.flags["C_CONTIGUOUS"], "downstream torch/cv2 want contiguous"


def test_fix_off_is_byte_identical(monkeypatch):
    """The escape hatch must not perturb ONE byte: it is what every recorded
    checkpoint is evaluated under."""
    cfg = _fix_with("0", monkeypatch)
    img = _post_process()(_parse_like(_blue_bgra()), normalized=False,
                          crop=False, grayscale=False, image_size=16)
    out = cfg._fix_channel_order(img)
    assert np.array_equal(out, img)
    assert int(np.argmax(out[0, 0])) == 0, "off must leave the frame BGR"


def test_default_is_fix_on(monkeypatch):
    """User 2026-09-26 asked for the confirmed issues to be FIXED, so the
    default corrects the order; the old path is opt-in."""
    monkeypatch.delenv("ENV_RGB_CHANNEL_FIX", raising=False)
    import configs.train_env_config_v4_image_goal as cfg
    importlib.reload(cfg)
    assert cfg.RGB_CHANNEL_FIX is True


def test_fix_is_an_involution(monkeypatch):
    """Applying it twice returns the original, so a double application is
    detectable rather than silently producing a third order."""
    cfg = _fix_with("1", monkeypatch)
    img = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    assert np.array_equal(cfg._fix_channel_order(cfg._fix_channel_order(img)), img)


@pytest.mark.parametrize("bad", [None,
                                 np.zeros((8, 8), np.uint8),
                                 np.zeros((8, 8, 1), np.uint8)])
def test_non_three_channel_input_is_untouched(monkeypatch, bad):
    """Greyscale (grayscale=True) and odd shapes pass through: guessing what to
    swap would be worse than leaving them."""
    cfg = _fix_with("1", monkeypatch)
    out = cfg._fix_channel_order(bad)
    if bad is None:
        assert out is None
    else:
        assert np.array_equal(out, bad)


def test_green_is_unaffected_by_the_swap():
    """A sanity check on the whole argument: R<->B leaves GREEN in place, which
    is why vegetation could never have revealed the bug and the sky had to."""
    raw = np.zeros((8, 8, 4), np.uint8)
    raw[:, :, 1] = 255        # G in BGRA
    img = _post_process()(_parse_like(raw), normalized=False, crop=False,
                          grayscale=False, image_size=8)
    assert int(np.argmax(img[0, 0])) == 1
