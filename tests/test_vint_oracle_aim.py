"""VINT_ORACLE_AIM=1 (PRIVILEGED diagnostic, development set only, never a result):
ViNT's waypoints are replaced by the true lane-centre aim points before the
controller, so a drift that remains is the controller's, not ViNT's.
oracle_aim_points(lane_priv) builds them from the env's lane state (offset LEFT
positive, heading error = hero yaw - lane yaw in CARLA's clockwise yaw), straight
lane ahead, in ViNT's frame (x forward, y left)."""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from torch_rl.vint_driver import oracle_aim_points, signed_angle  # noqa: E402


def _lp(off_left=0.0, herr_cw_deg=0.0):
    lp = np.zeros(13)
    h = math.radians(herr_cw_deg)
    lp[0], lp[2], lp[3], lp[12] = off_left, math.sin(h), math.cos(h), 1.0
    return lp


def test_centred_and_aligned_points_straight_ahead():
    w = oracle_aim_points(_lp(), spacing=1.0, n=5)
    assert w == pytest.approx(np.stack([np.arange(1, 6), np.zeros(5)], 1))


def test_a_car_left_of_centre_aims_right():
    w = oracle_aim_points(_lp(off_left=1.0), spacing=1.0, n=5)
    assert np.all(w[:, 1] == pytest.approx(-1.0))
    assert signed_angle(w[2]) > 0                    # positive angle = steer right


def test_a_car_turned_right_of_the_lane_aims_left():
    """CARLA yaw is clockwise: hero yaw > lane yaw = the car points right of the lane."""
    w = oracle_aim_points(_lp(herr_cw_deg=10.0), spacing=1.0, n=5)
    assert np.all(w[:, 1] > 0)                        # aim left
    assert signed_angle(w[4]) == pytest.approx(math.radians(-10.0), abs=1e-6)


def test_no_lane_found_falls_back_to_none():
    lp = _lp()
    lp[12] = 0.0
    assert oracle_aim_points(lp) is None
