"""Lane labels for recorded episodes (plan 2026-11-09, Step 0: "is it the encoder?").

A shard stores the car's position AFTER each step, not its yaw or lane, so the
probe's labels come from positions plus the town's lane geometry. These tests pin
the geometry on synthetic lanes, without CARLA: a fake lane source stands in for
the OpenDRIVE map.
"""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from torch_rl import lane_labels as ll  # noqa: E402


class StraightLanes:
    """A lane along a line through the origin at `yaw_deg`."""

    def __init__(self, yaw_deg=0.0, width=3.5, junction=False):
        self.yaw, self.width, self.junction = float(yaw_deg), float(width), bool(junction)

    def query(self, P, ahead_m=10.0):
        P = np.asarray(P, float)
        u = np.array([math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))])
        centre = np.outer(P @ u, u)
        n = len(P)
        return {"centre": centre, "yaw": np.full(n, self.yaw), "width": np.full(n, self.width),
                "junction": np.full(n, self.junction), "yaw_ahead": np.zeros(n)}


class ArcLanes:
    """A lane on the circle of radius R about the origin, driven counter-clockwise
    in CARLA's (x, y) frame, i.e. yaw = angle + 90 degrees."""

    def __init__(self, R=20.0):
        self.R = float(R)

    def query(self, P, ahead_m=10.0):
        P = np.asarray(P, float)
        ang = np.arctan2(P[:, 1], P[:, 0])
        centre = self.R * np.stack([np.cos(ang), np.sin(ang)], -1)
        yaw = np.degrees(ang) + 90.0
        n = len(P)
        return {"centre": centre, "yaw": ll.wrap_deg(yaw), "width": np.full(n, 3.5),
                "junction": np.zeros(n, bool),
                "yaw_ahead": np.full(n, math.degrees(ahead_m / self.R))}


def test_frame_positions_put_frame_k_at_pos_k_minus_1_and_the_reset_frame_at_pos_0():
    pos = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], np.float32)
    P = ll.frame_positions(pos)
    assert P.shape == (4, 2)
    np.testing.assert_allclose(P[:, 0], [1.0, 1.0, 2.0, 3.0])


def test_wrap_deg_maps_into_minus_180_to_180():
    np.testing.assert_allclose(ll.wrap_deg([190.0, -190.0, 360.0, 180.0, -180.0]),
                               [-170.0, 170.0, 0.0, -180.0, -180.0])


def test_motion_heading_uses_carla_yaw_convention_and_is_nan_at_rest():
    P = np.array([[0, 0], [1, 1], [2, 2], [2, 2], [2, 2], [2, 2]], float)
    h = ll.motion_heading(P, min_step_m=0.15)
    assert h[1] == pytest.approx(45.0)
    assert math.isnan(h[4])                       # centred span 0 m: at rest
    P2 = np.array([[0, 0], [0, -1], [0, -2]], float)     # driving along -y
    assert ll.motion_heading(P2)[1] == pytest.approx(-90.0)


def test_lane_offset_is_positive_to_the_lanes_right():
    # CARLA: x forward, y right; right = (-sin yaw, cos yaw)
    assert ll.lane_offset([[5.0, 0.5]], [[5.0, 0.0]], [0.0])[0] == pytest.approx(0.5)
    assert ll.lane_offset([[-0.5, 3.0]], [[0.0, 3.0]], [90.0])[0] == pytest.approx(0.5)
    assert ll.lane_offset([[5.0, -0.7]], [[5.0, 0.0]], [0.0])[0] == pytest.approx(-0.7)


def test_straight_lane_labels_offset_heading_error_and_heading_change():
    # driving at 10 degrees to a lane along x, 0.3 m right of it at the start
    t = np.arange(12, dtype=float)
    d = np.radians(10.0)
    path = np.stack([t * 0.4 * math.cos(d), 0.3 + t * 0.4 * math.sin(d)], -1)
    lab = ll.episode_labels(path[1:], StraightLanes(0.0))      # pos[t] = after step t
    k = 5
    assert lab["heading_err"][k] == pytest.approx(10.0, abs=1e-6)
    assert lab["offset"][k] == pytest.approx(path[k][1], abs=1e-9)   # frame k = pos[k-1] = path[k]
    assert lab["dheading"][k] == pytest.approx(0.0, abs=1e-6)  # straight motion
    assert not lab["junction"].any()
    assert lab["valid"][2:-1].all() and not lab["valid"][0]


def test_arc_lane_heading_error_is_zero_and_heading_change_is_the_step_angle():
    R, step = 20.0, 0.4
    ang = np.arange(15) * step / R
    path = R * np.stack([np.cos(ang), np.sin(ang)], -1)        # counter-clockwise on the lane
    lab = ll.episode_labels(path[1:], ArcLanes(R))
    for k in range(3, 12):
        assert lab["heading_err"][k] == pytest.approx(0.0, abs=1e-6)
        assert lab["offset"][k] == pytest.approx(0.0, abs=1e-6)
        assert lab["dheading"][k] == pytest.approx(math.degrees(step / R), abs=1e-6)
    assert lab["curv"][5] == pytest.approx(math.degrees(10.0 / R))


def test_junction_frames_are_flagged_from_the_lane_source():
    lab = ll.episode_labels(np.array([[0.4 * i, 0.0] for i in range(1, 8)]),
                            StraightLanes(0.0, junction=True))
    assert lab["junction"].all()


def test_window_noise_is_the_mean_executed_minus_label_steering_over_k_steps():
    exe = np.array([[0.3, 0.0], [0.1, 0.0], [-0.2, 0.0], [0.0, 0.0]], np.float32)
    lab = np.array([[0.1, 0.0], [0.1, 0.0], [0.0, 0.0], [0.0, 0.0]], np.float32)
    n1 = ll.window_noise(exe, lab, 1)
    n2 = ll.window_noise(exe, lab, 2)
    np.testing.assert_allclose(n1, [0.2, 0.0, -0.2, 0.0], atol=1e-6)
    np.testing.assert_allclose(n2[:3], [0.1, -0.1, -0.1], atol=1e-6)
    assert math.isnan(n2[3])                       # the window runs past the episode
