"""The visitation ring, and the left-handed bearing convention it depends on.

CARLA is left-handed: forward = (cos yaw, sin yaw), right = (-sin yaw, cos yaw).
A textbook right-handed rotation mirrors the bearing. That exact error survived
in this repo's goal block for a whole training run -- a goal 20 m to the right
reported left = +20.0 -- so these tests pin the convention rather than trusting
it.
"""
import numpy as np
import pytest

from vae_nav.visit_ring import (BEARINGS, N_SLOTS, RING_RANGES_M, cell_of,
                                visit_ring)


def test_bearings_match_the_peer_formula():
    """b_k = -pi + k*2pi/16, k = 0..15."""
    assert len(BEARINGS) == 16
    for k, b in enumerate(BEARINGS):
        assert b == pytest.approx(-np.pi + k * 2.0 * np.pi / 16)


def test_six_ranges_half_metre_to_three():
    assert RING_RANGES_M == (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)


def test_cell_of_floors_on_a_half_metre_grid():
    assert cell_of((0.0, 0.0)) == (0, 0)
    assert cell_of((0.4, 0.4)) == (0, 0)
    assert cell_of((0.6, 0.6)) == (1, 1)
    assert cell_of((-0.1, -0.1)) == (-1, -1), "must floor, not truncate toward zero"


def test_empty_visited_set_gives_all_zeros():
    r = visit_ring((0.0, 0.0), 0.0, set())
    assert r.shape == (N_SLOTS,)
    assert np.all(r == 0.0)


def test_fully_visited_neighbourhood_gives_all_ones():
    visited = {(i, j) for i in range(-10, 11) for j in range(-10, 11)}
    r = visit_ring((0.0, 0.0), 0.0, visited)
    assert np.all(r == 1.0)


def test_values_are_fractions_of_six_samples():
    """Six ranges per bearing, so every value is a multiple of 1/6."""
    rng = np.random.default_rng(0)
    visited = {(int(rng.integers(-8, 8)), int(rng.integers(-8, 8)))
               for _ in range(60)}
    r = visit_ring((0.0, 0.0), 0.7, visited)
    for v in r:
        assert (v * 6) == pytest.approx(round(v * 6)), f"{v} is not k/6"
        assert 0.0 <= v <= 1.0


def visited_strip_ahead(distance_m=2.0, yaw=0.0):
    """Mark cells along the world direction the hero at `yaw` is facing."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    return {cell_of((d * cy, d * sy)) for d in np.arange(0.25, distance_m, 0.1)}


@pytest.mark.parametrize("yaw", [0.0, np.pi / 2, np.pi, -np.pi / 2, 0.7])
def test_a_visited_strip_ahead_lights_the_forward_slot(yaw):
    """THE CONVENTION TEST.

    Bearing 0 in the robot frame is straight ahead. If cells directly in front
    are visited, the slot at bearing 0 must be the one that lights up -- at
    EVERY yaw. A mirrored or transposed rotation passes at yaw=0 and fails as
    soon as the hero turns, which is exactly how the goal-block bug hid.
    """
    ring = visit_ring((0.0, 0.0), yaw, visited_strip_ahead(yaw=yaw))
    fwd = int(np.argmin(np.abs(BEARINGS)))          # bearing closest to 0
    assert ring[fwd] > 0.0, "forward slot did not see cells straight ahead"
    assert ring[fwd] == ring.max(), "some other bearing beat straight-ahead"


def test_a_strip_behind_lights_the_rear_slot_not_the_front():
    """Distinguishes a 180-degree slot rotation, which a symmetric test cannot."""
    behind = {cell_of((-d, 0.0)) for d in np.arange(0.25, 2.0, 0.1)}
    ring = visit_ring((0.0, 0.0), 0.0, behind)
    fwd = int(np.argmin(np.abs(BEARINGS)))
    rear = int(np.argmin(np.abs(np.abs(BEARINGS) - np.pi)))
    assert ring[rear] > 0.0
    assert ring[fwd] == 0.0, "cells behind the hero lit the forward slot"


def test_left_and_right_are_not_mirrored():
    """THE BUG THIS REPO ACTUALLY SHIPPED, in ring form.

    With the hero at yaw=0 facing +x, CARLA's left is +y (left = -right, and
    right = (-sin yaw, cos yaw) = (0, 1) at yaw=0, so left = (0, -1)).
    Marking cells on one side must light the slot on that side only.
    """
    left_dir = np.array([np.sin(0.0), -np.cos(0.0)])      # (0, -1) at yaw=0
    left_cells = {cell_of(left_dir * d) for d in np.arange(0.25, 2.0, 0.1)}
    ring = visit_ring((0.0, 0.0), 0.0, left_cells)

    left_slot = int(np.argmin(np.abs(BEARINGS - np.pi / 2)))
    right_slot = int(np.argmin(np.abs(BEARINGS + np.pi / 2)))
    assert ring[left_slot] > 0.0, "left-hand cells did not light the left slot"
    assert ring[right_slot] == 0.0, (
        "left-hand cells lit the RIGHT slot -- the bearing is mirrored, which "
        "is the left/right sign error this repo already paid for once")


def test_ring_is_translation_equivariant():
    """Same local geometry at a different place gives the same ring."""
    a = visit_ring((0.0, 0.0), 0.3, visited_strip_ahead(yaw=0.3))
    shift = np.array([37.0, -21.5])
    shifted = {cell_of(shift + np.array([d * np.cos(0.3), d * np.sin(0.3)]))
               for d in np.arange(0.25, 2.0, 0.1)}
    b = visit_ring(shift, 0.3, shifted)
    np.testing.assert_allclose(a, b)
