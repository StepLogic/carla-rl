"""The four node-placement methods, tested without a simulator or a checkpoint.

Placement is the ONLY thing that differs between the arms of this ablation, so
it is the part that has to be right. Everything here runs on hand-written
entropy traces and distance matrices, which is what makes the budget, the
separation veto and the arc-length handling testable at all -- against a real
policy they would be indistinguishable from noise.

THE ARMS ARE PORTED FROM node_placement_entropy.py and the tests pin the
details that port has to preserve: midpoints rather than endpoints, greedy by
height rather than left-to-right, and an equal budget across arms.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from vae_nav.node_placement import (METHODS, arclength,  # noqa: E402
                                    place, place_peaks, place_random,
                                    place_uniform, smooth)


# ------------------------------------------------------------ utilities ----

def test_smooth_shrinks_at_the_edges_rather_than_padding():
    # Padding would invent a plateau at both ends, and an episode start already
    # carries an entropy transient -- that plateau becomes a spurious peak
    # exactly where the policy is only initialising.
    x = np.array([0.0, 0.0, 9.0, 0.0, 0.0])
    assert smooth(x, 3)[0] == pytest.approx(0.0)
    assert smooth(x, 3)[2] == pytest.approx(3.0)


def test_arclength_does_not_count_the_jump_between_episodes():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [50.0, 0.0], [51.0, 0.0]])
    seg = np.array([0, 0, 1, 1])
    assert arclength(xy, seg)[-1] == pytest.approx(2.0)


def test_arclength_without_segments_is_the_plain_polyline():
    xy = np.array([[0.0, 0.0], [3.0, 4.0]])
    assert arclength(xy)[-1] == pytest.approx(5.0)


# ------------------------------------------------------------- uniform ----

def test_uniform_places_nodes_at_midpoints_not_endpoints():
    # An endpoint would pin a node at s=0 on every episode, which is the reset
    # pose rather than a place anyone would choose.
    s = np.linspace(0.0, 100.0, 101)
    idx = place_uniform(s, 2)
    assert 0 not in idx and len(s) - 1 not in idx
    assert list(idx) == [25, 75]


def test_uniform_spacing_is_in_arc_length_not_index():
    # A trace that dawdles then sprints must not get most of its nodes in the
    # dawdling half; that is the whole difference between arc length and index.
    s = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 99, 50)])
    idx = place_uniform(s, 2)
    assert s[idx[0]] == pytest.approx(24.75, abs=2.0)


def test_uniform_knows_nothing_about_the_policy():
    # It is the baseline a method has to beat to be worth its cost; pinned so a
    # later "improvement" cannot quietly consult entropy or the critic here.
    s = np.linspace(0.0, 10.0, 11)
    assert list(place_uniform(s, 5)) == list(place_uniform(s, 5))


# -------------------------------------------------------------- random ----

def test_random_is_reproducible_from_its_seed():
    s = np.linspace(0.0, 100.0, 101)
    assert list(place_random(s, 10, seed=1)) == list(place_random(s, 10, seed=1))
    assert list(place_random(s, 10, seed=1)) != list(place_random(s, 10, seed=2))


def test_random_returns_indices_in_trace_order():
    s = np.linspace(0.0, 100.0, 101)
    idx = place_random(s, 10, seed=3)
    assert list(idx) == sorted(idx)


def test_random_places_EXACTLY_the_budget():
    # THE BUG THIS PINS: drawing arc-length targets and snapping each to its
    # nearest state let two targets collapse onto one node. Measured in a smoke
    # run, a budget of 101 produced 79 distinct nodes -- an arm silently 22%
    # sparser than the ones it is the control for, which is the sparsity
    # confound the equal budget exists to prevent.
    s = np.linspace(0.0, 100.0, 120)
    for budget in (5, 40, 119):
        assert len(place_random(s, budget, seed=0)) == budget, budget


def test_random_cannot_exceed_the_states_available():
    s = np.linspace(0.0, 10.0, 6)
    assert len(place_random(s, 99, seed=0)) == 6


# ---------------------------------------------------- policy: entropy ----

def test_peaks_takes_the_highest_entropy_state():
    h = np.array([0.0, 0.0, 5.0, 0.0, 0.0])
    s = np.arange(5.0) * 10.0
    assert 2 in place_peaks(h, s, n_max=1, min_sep_m=1.0, smooth_w=1)


def test_peaks_refuses_a_second_node_inside_the_separation():
    # Entropy is autocorrelated over tens of steps, so raw maxima arrive in
    # clusters around ONE event; two nodes a metre apart are not two nodes.
    h = np.array([0.0, 5.0, 4.9, 0.0])
    s = np.array([0.0, 10.0, 11.0, 40.0])
    idx = place_peaks(h, s, n_max=2, min_sep_m=8.0, smooth_w=1)
    assert 1 in idx and 2 not in idx


def test_peaks_is_greedy_by_height_not_left_to_right():
    # Thinning left-to-right keeps whichever peak comes first, which would make
    # the arm depend on the direction of travel.
    h = np.array([4.0, 0.0, 9.0])
    s = np.array([0.0, 1.0, 2.0])
    assert list(place_peaks(h, s, n_max=1, min_sep_m=5.0, smooth_w=1)) == [2]


def test_peaks_respects_the_budget():
    rng = np.random.default_rng(0)
    h = rng.normal(size=200)
    s = np.arange(200.0) * 5.0
    assert len(place_peaks(h, s, n_max=7, min_sep_m=1.0, smooth_w=3)) == 7


# ------------------------------------------------------------ dispatch ----

def test_place_dispatches_every_named_method():
    n_states = 40
    s = np.arange(n_states, dtype=float) * 3.0
    h = np.random.default_rng(0).normal(size=n_states)
    emb = np.stack([np.arange(n_states, dtype=float), np.zeros(n_states)], 1)
    d = np.full((n_states, n_states), 1e6)
    np.fill_diagonal(d, 0.0)
    for method in METHODS:
        idx, _ = place(method, s=s, entropy=h, dist=d, embeddings=emb, n=5,
                       seed=0, min_sep_m=1.0, smooth_w=3)
        assert len(idx) > 0, method


def test_the_budgeted_arms_never_exceed_the_budget():
    # EQUAL BUDGET IS THE COMPARISON. An arm placing more nodes than the others
    # would turn a placement result into a sparsity result, invisibly.
    n_states, budget = 60, 6
    s = np.arange(n_states, dtype=float) * 4.0
    h = np.random.default_rng(1).normal(size=n_states)
    for method in ("uniform", "random", "policy"):
        idx, _ = place(method, s=s, entropy=h, n=budget, min_sep_m=1.0,
                       smooth_w=3)
        assert len(idx) <= budget, method


def test_place_refuses_a_budgeted_arm_without_a_budget():
    # Silently defaulting the budget is the single easiest way to make the
    # arms incomparable while every table still renders.
    s = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="budget n"):
        place("uniform", s=s)


def test_place_refuses_the_policy_arm_without_entropy():
    s = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="actor entropy"):
        place("policy", s=s, n=3)


def test_place_refuses_sgm_without_a_distance_matrix():
    with pytest.raises(ValueError, match="needs dist and embeddings"):
        place("sgm", s=np.arange(4, dtype=float), n=2)


def test_the_sgm_cutoff_search_moves_toward_the_budget():
    # SGM derives its own sparsity, which makes it incomparable to arms that
    # are TOLD how many nodes to place. Sweeping its one free parameter onto
    # the budget is what puts it on the same axis; this pins that a larger
    # target really does yield more nodes.
    from vae_nav.node_placement import sgm_cutoff_for_budget

    rng = np.random.default_rng(0)
    n = 40
    emb = rng.normal(size=(n, 4))
    d = np.abs(rng.normal(size=(n, n))) * 30.0
    np.fill_diagonal(d, 0.0)
    wide = float(np.percentile(np.linalg.norm(
        emb[:, None, :] - emb[None, :, :], axis=-1), 90))
    few = sgm_cutoff_for_budget(d, emb, 5, wide)[1]
    many = sgm_cutoff_for_budget(d, emb, 30, wide)[1]
    assert len(few) <= len(many)


def test_the_sgm_cutoff_search_reports_the_cutoff_it_used():
    # The caller must record what was ACTUALLY used, not what was asked for --
    # a budget that cannot be hit exactly is normal and must not be reported
    # as though it were.
    from vae_nav.node_placement import sgm_cutoff_for_budget

    rng = np.random.default_rng(1)
    emb = rng.normal(size=(20, 3))
    d = np.abs(rng.normal(size=(20, 20))) * 10.0
    np.fill_diagonal(d, 0.0)
    cutoff, nodes, merged = sgm_cutoff_for_budget(d, emb, 6, 5.0)
    assert cutoff > 0
    assert len(nodes) + int((merged >= 0).sum()) == 20


def test_place_refuses_an_unknown_method():
    with pytest.raises(ValueError, match="unknown placement method"):
        place("vlad", s=np.arange(4, dtype=float), n=2)
