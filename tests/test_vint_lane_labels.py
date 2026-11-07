"""Lane-centre waypoint labels for ViNT (src/vae_nav/vint_lane_labels.py).

ViNT's waypoints are used as AIM points (the driver steers at one of them), so
the label for frame t is the lane centre ahead of the car -- where the expert's
own controller aims -- not the noisy path the recorded car drove: the point at
arc length s_t + j * spacing along the lane centre of the car's route (j = 1..5),
with the lane's heading. Frame 0 of each label is the car itself (ViNT's local
frame origin). Training labels only; the policy still sees images."""
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vae_nav.vint_lane_labels import lane_centre_labels, point_at, project  # noqa: E402

SP = 1.1346


def _line(n=200, step=0.5):
    """A straight lane centre along +x (right-handed), y = 0."""
    return np.stack([np.arange(n) * step, np.zeros(n)], 1)


def test_point_at_interpolates_position_and_heading():
    poly = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    p, h = point_at(poly, 12.5)
    assert p == pytest.approx([10.0, 2.5]) and h == pytest.approx(math.pi / 2)


def test_project_finds_arc_length_and_lateral_gap():
    s, gap = project(_line(), np.array([3.2, -0.7]))
    assert s == pytest.approx(3.2) and gap == pytest.approx(0.7)


def test_an_offset_car_is_labelled_back_onto_the_centre():
    xy = np.array([[5.0, 1.0], [6.0, 1.0]])            # 1 m left of the lane centre
    yaw = np.zeros(2)
    pos, yw, ok = lane_centre_labels(xy, yaw, _line(), spacing=SP, n=5)
    assert ok.all()
    assert pos[0, 0] == pytest.approx([5.0, 1.0])          # origin = the car
    for j in range(1, 6):
        assert pos[0, j] == pytest.approx([5.0 + j * SP, 0.0])  # on the centre line
        assert yw[0, j] == pytest.approx(0.0)


def test_heading_labels_are_the_lanes_and_unwrapped_against_the_car():
    poly = np.stack([np.arange(100) * -0.5, np.zeros(100)], 1)   # lane runs along -x
    xy = np.array([[-5.0, 0.0]])
    yaw = np.array([-math.pi + 0.05])                               # car heading ~ -pi
    _, yw, _ = lane_centre_labels(xy, yaw, poly, spacing=SP, n=5)
    assert yw[0, 0] == pytest.approx(-math.pi + 0.05)
    assert all(abs(yw[0, j] - yw[0, 0]) < 0.1 for j in range(1, 6))   # no 2*pi jump


def test_a_car_far_from_its_lane_keeps_its_recorded_path():
    xy = np.array([[5.0, 4.0], [6.0, 4.0], [7.0, 4.0], [8.0, 4.0], [9.0, 4.0], [10.0, 4.0],
                   [11.0, 4.0], [12.0, 4.0], [13.0, 4.0], [14.0, 4.0], [15.0, 4.0],
                   [16.0, 4.0], [17.0, 4.0], [18.0, 4.0], [19.0, 4.0], [20.0, 4.0]])
    pos, yw, ok = lane_centre_labels(xy, np.zeros(len(xy)), _line(), spacing=SP, n=5,
                                     max_gap_m=2.5)
    assert not ok[0]
    assert pos[0] == pytest.approx(xy[0:16:3])                 # the original label path
