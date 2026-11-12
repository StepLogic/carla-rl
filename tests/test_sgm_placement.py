"""SGM node placement: the merge test, on synthetic distance matrices.

NO JAX, NO CHECKPOINT, NO SIMULATOR. The algorithm takes a distance matrix, so
the part worth testing can be tested exactly. Querying the critic is a separate
function and is covered where it is used.

WHAT THESE PROTECT. SGM's sparsity is emergent, not budgeted: nodes appear only
where no existing node is two-way consistent. That makes the failure modes quiet
-- a too-loose cutoff merges distinct places into one node and the graph plans
through walls, while a too-tight one degenerates to keeping everything and SGM
stops being sparse at all. Neither raises.

THE TWO-WAY REQUIREMENT IS THE SUBTLE PART. Two states can agree perfectly on
what they can reach while differing entirely on what can reach them -- a one-way
street is exactly that. Merging those is invisible at construction and shows up
much later as plans that cannot be executed.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from vae_nav.sgm_placement import (embedding_neighbours,  # noqa: E402
                                   sgm_select, summarise, two_way_consistency)


def _chain(n, spacing=10.0):
    """States in a line: dist[i,j] = |i-j| * spacing, symmetric."""
    idx = np.arange(n)
    return np.abs(idx[:, None] - idx[None, :]).astype(float) * spacing


def _embeddings(n, spread=0.0):
    """Embeddings all within any sensible cutoff, so the prefilter never binds."""
    return np.zeros((n, 2)) + spread * np.arange(n)[:, None]


# ------------------------------------------------------- the prefilter ----

def test_no_neighbours_when_there_are_no_nodes_yet():
    assert len(embedding_neighbours([], np.zeros(2))) == 0


def test_neighbours_are_returned_nearest_first():
    emb = np.array([[3.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    got = embedding_neighbours(emb, np.zeros(2), cutoff=10.0)
    assert list(got) == [1, 2, 0]


def test_the_cutoff_excludes_distant_embeddings():
    emb = np.array([[0.01, 0.0], [5.0, 0.0]])
    got = embedding_neighbours(emb, np.zeros(2), cutoff=0.05)
    assert list(got) == [0]


# --------------------------------------------------- two-way consistency ----

def test_an_empty_graph_is_infinitely_inconsistent():
    # The first candidate must never merge into nothing.
    assert two_way_consistency(_chain(3), 0, 0, []) == np.inf


def test_identical_states_are_perfectly_consistent():
    d = _chain(4)
    d[1] = d[0]          # state 1 reaches everything exactly as state 0 does
    d[:, 1] = d[:, 0]    # and is reached exactly as state 0 is
    assert two_way_consistency(d, 1, 0, [0, 2, 3]) == pytest.approx(0.0)


def test_distant_states_are_inconsistent():
    d = _chain(4, spacing=10.0)
    assert two_way_consistency(d, 3, 0, [0, 1, 2]) > 5.0


def test_a_one_way_difference_is_caught():
    # THE REASON THE TEST IS TWO-WAY. Outgoing distances agree exactly; only
    # the incoming ones differ, as they would either side of a one-way street.
    # A one-directional check would merge these and the graph would plan
    # through a route that cannot be driven.
    d = _chain(4, spacing=1.0)
    d[2] = d[1]                 # identical outgoing
    d[:, 2] = d[:, 1] + 40.0    # very different incoming
    outgoing_only = np.max(np.abs(d[2, [1, 3]] - d[1, [1, 3]]))
    assert outgoing_only == pytest.approx(0.0)
    assert two_way_consistency(d, 2, 1, [1, 3]) >= 40.0


# ------------------------------------------------------------- selection ----

def test_the_first_state_always_becomes_a_node():
    members, merged = sgm_select(_chain(3), _embeddings(3))
    assert members[0] == 0
    assert merged[0] == -1


def test_duplicated_states_merge_rather_than_adding_nodes():
    # Three exact copies of one place should yield ONE node.
    d = np.zeros((3, 3))
    members, merged = sgm_select(d, _embeddings(3))
    assert list(members) == [0]
    assert list(merged[1:]) == [0, 0]


def test_well_separated_states_all_become_nodes():
    members, _ = sgm_select(_chain(4, spacing=100.0), _embeddings(4))
    assert list(members) == [0, 1, 2, 3]


def test_a_looser_cutoff_yields_fewer_nodes():
    # THE KNOB'S DIRECTION, pinned. Loosening consistency must never increase
    # the node count; that would mean the merge test is inverted.
    d = _chain(8, spacing=3.0)
    tight, _ = sgm_select(d, _embeddings(8), consistency_cutoff=1.0)
    loose, _ = sgm_select(d, _embeddings(8), consistency_cutoff=50.0)
    assert len(loose) <= len(tight)


def test_a_tight_embedding_cutoff_prevents_all_merging():
    # Degenerate but important: if the prefilter never admits a neighbour, SGM
    # keeps everything and is no longer sparse. It should fail this way rather
    # than silently merging on embedding alone.
    d = np.zeros((5, 5))
    emb = np.arange(5, dtype=float)[:, None] * 10.0
    members, _ = sgm_select(d, emb, embedding_cutoff=0.001)
    assert len(members) == 5


def test_every_state_is_either_a_node_or_merged_into_one():
    d = _chain(10, spacing=4.0)
    members, merged = sgm_select(d, _embeddings(10))
    for i in range(10):
        assert (i in members) != (merged[i] >= 0), i
    assert set(merged[merged >= 0]).issubset(set(members.tolist()))


def test_order_changes_which_states_are_kept():
    # SGM is greedy and sequential, so the result depends on traversal order.
    # Asserted rather than hidden, because a reader comparing node COUNTS
    # across methods needs to know this is not an order-invariant algorithm.
    d = _chain(6, spacing=3.0)
    fwd, _ = sgm_select(d, _embeddings(6), consistency_cutoff=7.0)
    rev, _ = sgm_select(d, _embeddings(6), consistency_cutoff=7.0,
                        order=list(reversed(range(6))))
    assert set(fwd.tolist()) != set(rev.tolist())


def test_an_empty_input_is_handled():
    members, merged = sgm_select(np.zeros((0, 0)), [])
    assert len(members) == 0 and len(merged) == 0


# ------------------------------------------------------------- reporting ----

def test_density_is_reported_not_just_count():
    # Arms are compared over trajectories of different lengths, so a raw count
    # is not comparable between them.
    members, merged = sgm_select(np.zeros((4, 4)), _embeddings(4))
    out = summarise(members, merged, path_length_m=200.0)
    assert out["n_nodes"] == 1
    assert out["n_merged"] == 3
    assert out["nodes_per_100m"] == pytest.approx(0.5)


def test_density_survives_a_zero_length_path():
    members, merged = sgm_select(np.zeros((1, 1)), _embeddings(1))
    assert np.isfinite(summarise(members, merged, 0.0)["nodes_per_100m"])
