"""The decision+coverage selection rule: causal triggers, the entropy floor, and the cooldown."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build_decision_coverage_map import select_nodes  # noqa: E402


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _same_view(n, dim=4):
    """n identical unit latents: the coverage trigger never fires."""
    return _unit(np.tile(np.array([1.0, 0.0, 0.0, 0.0])[:dim], (n, 1)))


def test_first_sample_is_always_a_node_by_coverage():
    picked = select_nodes(np.full(10, 2.0), _same_view(10), w=50, cooldown=0)
    assert picked[0] == (0, "coverage")


def test_entropy_spike_fires_only_after_the_window_fills():
    U = np.full(60, 2.0); U[10] = 9.0; U[55] = 9.0        # one spike before the window fills, one after
    picked = select_nodes(U, _same_view(60), w=50, tau_ent=2.0, cooldown=0)
    idx = [t for t, _ in picked]
    assert 10 not in idx                                   # |B| < W, no decision trigger yet
    assert (55, "decision") in picked


def test_u_min_floor_blocks_a_relative_spike_in_a_low_stretch():
    U = np.full(60, 1.0); U[55] = 3.0                      # a big robust-z, but still a low absolute entropy
    fires = select_nodes(U, _same_view(60), w=50, tau_ent=2.0, cooldown=0, u_min=0.0)
    blocked = select_nodes(U, _same_view(60), w=50, tau_ent=2.0, cooldown=0, u_min=5.0)
    assert (55, "decision") in fires
    assert 55 not in [t for t, _ in blocked]


def test_coverage_fires_when_the_view_stops_matching_the_last_node():
    Z = _same_view(6)
    Z[3] = _unit([0.0, 1.0, 0.0, 0.0])                     # orthogonal to the last node's view
    picked = select_nodes(np.full(6, 2.0), Z, w=50, tau_loc=0.80, cooldown=0)
    assert (3, "coverage") in picked


def test_cooldown_suppresses_a_node_that_would_otherwise_fire():
    Z = _same_view(8)
    Z[1] = _unit([0.0, 1.0, 0.0, 0.0])                     # would fire coverage at t=1
    assert (1, "coverage") in select_nodes(np.full(8, 2.0), Z, w=50, cooldown=0)
    assert 1 not in [t for t, _ in select_nodes(np.full(8, 2.0), Z, w=50, cooldown=3)]


def test_both_is_recorded_when_the_two_triggers_coincide():
    U = np.full(60, 2.0); U[55] = 9.0
    Z = _same_view(60); Z[55] = _unit([0.0, 1.0, 0.0, 0.0])
    picked = select_nodes(U, Z, w=50, tau_ent=2.0, tau_loc=0.80, cooldown=0)
    assert (55, "both") in picked
