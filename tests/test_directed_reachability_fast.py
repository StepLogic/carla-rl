"""The scipy reachability must equal the exhaustive Python BFS exactly.

MEASURED 2026-09-13 00:37: the Town04 10k-step calibration spent 83 min on
its sweep without finishing, because directed_reachability ran a pure-Python
BFS from every node on graphs of ~1200 nodes and ~1M edges (cutoffs up to
200 steps admit nearly every pair) -- ~1.7e9 interpreted steps per row. The
replacement does the same BFS in compiled code. It is a SPEED change only:
the fraction it returns is what chooses the SGM setting, so it must match to
the last pair.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def _random_edges(n, p, rng):
    m = rng.random((n, n)) < p
    np.fill_diagonal(m, False)
    u, v = np.nonzero(m)
    return [(int(a), int(b), float(rng.random())) for a, b in zip(u, v)]


@pytest.mark.parametrize("n,p,seed", [
    (2, 0.0, 0), (2, 1.0, 1), (5, 0.3, 2), (30, 0.02, 3), (30, 0.1, 4),
    (60, 0.05, 5), (80, 0.5, 6), (120, 0.01, 7), (150, 0.9, 8),
])
def test_fast_equals_exhaustive_bfs(n, p, seed):
    from vae_nav.sgm_frame import directed_reachability, _directed_reachability_bfs
    edges = _random_edges(n, p, np.random.default_rng(seed))
    assert directed_reachability(n, edges) == _directed_reachability_bfs(n, edges)


def test_disconnected_and_one_way_chain():
    from vae_nav.sgm_frame import directed_reachability, _directed_reachability_bfs
    # 0->1->2 one way, plus an isolated pair 3<->4
    edges = [(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0), (4, 3, 1.0)]
    n = 5
    expected = (2 + 1 + 0 + 1 + 1) / (n * (n - 1))
    assert _directed_reachability_bfs(n, edges) == pytest.approx(expected)
    assert directed_reachability(n, edges) == _directed_reachability_bfs(n, edges)


def test_trivial_sizes():
    from vae_nav.sgm_frame import directed_reachability
    assert directed_reachability(0, []) == 1.0
    assert directed_reachability(1, []) == 1.0
    assert directed_reachability(3, []) == 0.0


def test_duplicate_edges_do_not_change_the_answer():
    from vae_nav.sgm_frame import directed_reachability, _directed_reachability_bfs
    edges = [(0, 1, 1.0), (0, 1, 2.0), (1, 2, 1.0), (2, 0, 3.0), (2, 0, 1.0)]
    assert directed_reachability(4, edges) == _directed_reachability_bfs(4, edges)
