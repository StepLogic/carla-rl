"""The release-faithful SGM frame, on hand-built distance matrices."""
import numpy as np

from vae_nav.sgm_frame import (sgm_faithful, learned_edges, directed_reachability,
                               undirected_components, edge_separation_stats)


def _line(n, step_m=1.0):
    """n points on a line, critic distance = |i-j| steps, plus a reverse
    penalty so the matrix is asymmetric like a one-way road."""
    xy = np.stack([np.arange(n) * step_m, np.zeros(n)], 1)
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    d = np.abs(i - j).astype(float)
    d[j < i] *= 3.0               # going back is three times as far
    return xy, d


def test_no_gate_means_every_point_is_a_node():
    xy, d = _line(6)
    kept, edges = sgm_faithful(d, xy, consistency_cutoff=100, edge_cutoff=2.5, gate_m=0.5)
    assert list(kept) == list(range(6))
    # forward edges within 2.5 steps: i->i+1, i->i+2 ; backward within 2.5: only i+1->i? 3.0 no
    assert all(w < 2.5 for _a, _b, w in edges)
    assert (0, 1, 1.0) in edges and (0, 2, 2.0) in edges
    assert (1, 0, 3.0) not in edges


def test_gate_and_consistency_merge_duplicates():
    # two copies of the same place: a revisit should merge, not add a node
    xy = np.array([[0, 0], [5, 0], [10, 0], [0.2, 0.0]], float)
    d = np.array([[0, 5, 10, 0.5],
                  [15, 0, 5, 15],
                  [30, 15, 0, 30],
                  [0.5, 5, 10, 0]], float)
    kept, edges = sgm_faithful(d, xy, consistency_cutoff=1.0, edge_cutoff=6.0, gate_m=1.0)
    assert list(kept) == [0, 1, 2]          # point 3 merged into 0
    # a revisit that DISAGREES about its distances must NOT merge
    d2 = d.copy(); d2[3, 1] = 40.0           # from the revisit, node 1 is far
    kept2, _ = sgm_faithful(d2, xy, consistency_cutoff=1.0, edge_cutoff=6.0, gate_m=1.0)
    assert list(kept2) == [0, 1, 2, 3]


def test_a_node_with_no_edges_absorbs_the_sample():
    # release behaviour: inconsistency is 0 when m has no neighbours yet
    xy = np.array([[0, 0], [0.5, 0]], float)
    d = np.array([[0, 100], [100, 0]], float)      # nothing within any cutoff
    kept, _ = sgm_faithful(d, xy, consistency_cutoff=0.01, edge_cutoff=1.0, gate_m=1.0)
    assert list(kept) == [0]


def test_directed_reachability_differs_from_undirected_components():
    # 0->1->2, no way back: undirected says 1 component, directed says 3/6 pairs
    edges = [(0, 1, 1.0), (1, 2, 1.0)]
    assert undirected_components(3, edges) == 1
    assert abs(directed_reachability(3, edges) - 0.5) < 1e-9
    edges += [(2, 0, 1.0)]
    assert abs(directed_reachability(3, edges) - 1.0) < 1e-9


def test_learned_edges_use_the_same_cutoff_as_the_sgm_row():
    xy, d = _line(5)
    kept = [0, 2, 4]
    e = learned_edges(d, kept, edge_cutoff=2.5)
    assert (0, 1, 2.0) in e and (1, 2, 2.0) in e       # local indices
    assert all(w < 2.5 for _a, _b, w in e)
    st = edge_separation_stats(e, xy, kept, horizon_m=1.5)
    assert st["n_edges"] == len(e) and st["frac_beyond_horizon"] == 1.0
