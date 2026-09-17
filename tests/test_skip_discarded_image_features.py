"""The env's two MobileNet passes feed nothing, so skipping them must be exact.

MEASURED 2026-09-11: torch.cuda.is_available() is False on this box (the
installed build wants a newer driver), so STBL3Experiment.get_img_obs and
get_goal_img_obs each ran a MobileNetV3 forward pass on the CPU for EVERY
simulation step -- 17.3 of 32 cores, against 3.5 for the simulator itself,
while the GPU sat at 0% and 701 MiB of 12282.

Both results are then DISCARDED. VAEGoalWrapper._observation builds the gated
observation from info["rgb"] (set in core_state.py from experiment.obs_img),
the goal fields, raw_obs["vector"] and info["mode"]. It never reads the
"state" or "goal" keys these two methods produce.

ENV_SKIP_IMAGE_FEATURES=1 therefore returns a correctly-shaped zero vector
instead, exactly as the pre-existing mask_goal path already does. These tests
pin the two properties that make that safe:
  1. the shape and dtype the observation space declares are unchanged, and
  2. obs_img -- the frame the VAE actually encodes -- is byte-identical.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")


def _experiment(monkeypatch, skip):
    """A bare STBL3Experiment with only what the two image methods touch."""
    monkeypatch.setenv("ENV_IMAGE_SIZE", "120")
    monkeypatch.setenv("ENV_SKIP_IMAGE_FEATURES", "1" if skip else "0")
    from src.configs.train_env_config_v4_image_goal import STBL3Experiment
    exp = STBL3Experiment.__new__(STBL3Experiment)   # no CARLA, no config
    exp.image_size = 120
    exp.mask_goal = False
    exp.obs_img = None
    exp.prev_image_0 = exp.prev_image_1 = exp.prev_image_2 = None
    exp._feat_dim_cache = None
    exp.device = torch.device("cpu")
    from torchvision import models, transforms as T
    exp.model = models.mobilenet_v3_small(weights=None).features[:4].eval()
    exp.image_transform = T.Compose([
        T.ToPILImage(), T.Resize((120, 120)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    exp.skip_image_features = (
        os.environ.get("ENV_SKIP_IMAGE_FEATURES", "0") == "1")
    return exp


def _sensor_data(seed=0):
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 255, (348, 748, 3), dtype=np.uint8)
    return {"rgb": (0, frame), "goal": (0, [frame.copy()])}


def test_skipping_keeps_the_declared_shape_and_dtype(monkeypatch):
    on = _experiment(monkeypatch, skip=True)
    off = _experiment(monkeypatch, skip=False)
    a = on.get_img_obs(_sensor_data(), core=None)
    b = off.get_img_obs(_sensor_data(), core=None)
    assert a.shape == b.shape, "the observation space would no longer match"
    assert a.dtype == b.dtype
    assert np.all(a == 0.0), "the skipped path must be an explicit zero vector"
    assert on._feature_dim() == off._feature_dim() == a.shape[0]


def test_skipping_leaves_the_vae_frame_byte_identical(monkeypatch):
    """obs_img is what becomes info['rgb'] and feeds the frozen VAE."""
    on = _experiment(monkeypatch, skip=True)
    off = _experiment(monkeypatch, skip=False)
    on.get_img_obs(_sensor_data(7), core=None)
    off.get_img_obs(_sensor_data(7), core=None)
    assert on.obs_img is not None and off.obs_img is not None
    np.testing.assert_array_equal(on.obs_img, off.obs_img)
    # the frame-stack side effects must survive too
    for a, b in ((on.prev_image_0, off.prev_image_0),
                 (on.prev_image_1, off.prev_image_1),
                 (on.prev_image_2, off.prev_image_2)):
        np.testing.assert_array_equal(a, b)


def test_goal_image_skip_matches_the_existing_mask_goal_shape(monkeypatch):
    on = _experiment(monkeypatch, skip=True)
    off = _experiment(monkeypatch, skip=False)
    a = on.get_goal_img_obs(_sensor_data(3), core=None)
    b = off.get_goal_img_obs(_sensor_data(3), core=None)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert np.all(a == 0.0)
