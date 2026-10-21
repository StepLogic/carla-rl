"""GOAL_CAM_MATCH: the goal camera becomes the car's camera at the goal.

User 2026-10-21, "match the cameras". The goal image came from a 112x112, 90 deg
sensor 2.0 m above the road; the view from the 748x348, 110 deg hero camera
1.9 m ahead / 1.7 m up. HER's hindsight goals are view frames, and DEBiT reads
same-camera pairs. These tests need no simulator.
"""
import importlib
import math
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from torch_rl import goal_camera as gc  # noqa: E402


def test_the_mount_is_read_from_the_hero_camera_string():
    assert gc.hero_mount("1.9, 0.0, 1.7, 0.0, -15.0, 0.0") == (1.9, 1.7, -15.0)


@pytest.mark.parametrize("yaw", [0.0, 90.0, -135.0])
def test_the_camera_sits_ahead_along_the_lane_and_up(yaw):
    x, y, z = gc.matched_location((10.0, 20.0, 0.5), yaw, 1.9, 1.7)
    assert math.hypot(x - 10.0, y - 20.0) == pytest.approx(1.9)
    assert math.degrees(math.atan2(y - 20.0, x - 10.0)) == pytest.approx(
        yaw if yaw > -180 else yaw + 360)
    assert z == pytest.approx(0.5 + 1.7)


def _config(flag):
    os.environ["GOAL_CAM_MATCH"] = flag
    try:
        import src.configs.train_env_config_v4_image_goal as m
        importlib.reload(m)
        return m.config["env_config"]["experiment"]["hero"]["sensors"]
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"env config not importable here: {e}")
    finally:
        os.environ.pop("GOAL_CAM_MATCH", None)


def test_matched_goal_sensor_has_the_hero_cameras_intrinsics():
    s = _config("1")
    assert s["goal"]["image_size_x"] == s["rgb"]["image_size_x"]
    assert s["goal"]["image_size_y"] == s["rgb"]["image_size_y"]
    assert s["goal"]["fov"] == s["rgb"]["fov"]


def test_flag_off_keeps_the_original_goal_sensor():
    s = _config("0")
    assert s["goal"]["image_size_x"] == 112 and "fov" not in s["goal"]


def test_the_flag_is_off_by_default():
    os.environ.pop("GOAL_CAM_MATCH", None)
    importlib.reload(gc)
    assert gc.ENABLED is False
