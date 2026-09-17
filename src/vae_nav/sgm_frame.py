"""Sparse Graphical Memory, the RELEASE-FAITHFUL frame -- ported from
robot-learning-tools/node_placers.py (sgm_frame_placer, consistency_merge,
always_merge) so the CARLA SGM row and the Habitat SGM row are ONE algorithm.

WHY A SECOND SGM. sgm_placement.py implements the PAPER's test: two-way
consistency of a candidate against EVERY current node, an embedding-space
prefilter, and edges formed afterwards under a horizon. The released code
(SGMSearchPolicy._update_graph + qvalue_consistency) does something narrower
and is what the Habitat tree ported: the candidate is compared with a gated
node m ONLY through m's existing graph edges, and edges are LEARNED DIRECTED
ones built during aggregation under an edge cutoff -- the merge test reads
them. Two repos reporting "SGM" on two different tests would not be one
baseline. This file is the release; sgm_placement.py stays as the labelled
diagnostic `sgm_paper`.

THE FRAME, in the release's order, for each trace point i:
  1. GATE: kept nodes within `gate_m` metres of i in the plane (the release
     gates in embedding space at 0.05; a state observation's embedding is the
     state, so a metric gate is the analogue, as in the Habitat port);
  2. MERGE TEST against a gated node m, over m's CURRENT edges:
         new = [d(p, i) for p in pred(m)] ++ [d(i, s) for s in succ(m)]
         old = [d(p, m) for p in pred(m)] ++ [d(m, s) for s in succ(m)]
     merge iff max|new - old| < consistency_cutoff. A node with no edges yet
     has inconsistency 0 and absorbs the sample -- the release's behaviour;
  3. else ADD i as a node with learned directed edges: k -> i when
     d(k, i) < edge_cutoff, i -> k when d(i, k) < edge_cutoff, weight = d.

Everything except the merge test -- gate, edge cutoff, the distance supplied
-- is shared by the count-matched controls (uniform via always_merge, random
via a seeded random merge), so a difference between rows can only come from
which states are kept. `pair_distance(i, j)` is the critic read as steps,
i -> j, from the cached all-pairs matrix.

DIRECTED REACHABILITY is reported alongside the undirected component count,
because routing is over a DIRECTED graph and a union-find over (u, v) ignoring
direction can flatter it: measured in the Habitat tree, cutoffs at which the
undirected count said 4-11 components had directed reachability 0.70-0.86.
"""
from typing import Callable, Dict, List, Tuple

import numpy as np

Edge = Tuple[int, int, float]


def consistency_merge(consistency_cutoff: float) -> Callable:
    """SGM's own test: two-way Q-profile consistency against m's neighbours."""
    def merge(i, m, kept, preds, succs, ctx) -> bool:
        d = ctx["pair_distance"]
        new, old = [], []
        for p, w in preds[m]:
            new.append(float(d(kept[p], i))); old.append(w)
        for s, w in succs[m]:
            new.append(float(d(i, kept[s]))); old.append(w)
        inconsistency = (float(np.max(np.abs(np.asarray(new) - np.asarray(old))))
                         if new else 0.0)
        return inconsistency < consistency_cutoff
    return merge


def always_merge() -> Callable:
    """Uniform in the SGM frame: any kept node inside the gate absorbs the
    sample, so the gate radius IS the spacing."""
    def merge(i, m, kept, preds, succs, ctx) -> bool:
        return True
    return merge


def sgm_frame_placer(merge: Callable, edge_cutoff: float, gate_m: float):
    """Returns place(path_xy, ctx) -> kept trace indices; ctx["sgm_edges"] gets
    the learned directed edges as (a, b, w) over LOCAL indices of `kept`."""
    def place(path: np.ndarray, ctx: Dict) -> List[int]:
        d = ctx["pair_distance"]
        plane = np.asarray(path, np.float64)[:, :2]
        kept: List[int] = []
        preds: List[List] = []
        succs: List[List] = []
        for i in range(len(plane)):
            if kept:
                K = np.asarray(kept)
                gate = np.linalg.norm(plane[K] - plane[i], axis=1) < gate_m
                merged = False
                for m in np.nonzero(gate)[0]:
                    if merge(i, int(m), kept, preds, succs, ctx):
                        merged = True
                        break
                if merged:
                    continue
            new_idx = len(kept)
            preds.append([]); succs.append([])
            for k in range(new_idx):
                w_in = float(d(kept[k], i))
                if w_in < edge_cutoff:
                    succs[k].append((new_idx, w_in)); preds[new_idx].append((k, w_in))
                w_out = float(d(i, kept[k]))
                if w_out < edge_cutoff:
                    succs[new_idx].append((k, w_out)); preds[k].append((new_idx, w_out))
            kept.append(i)
        ctx["sgm_edges"] = [(a, b, float(w)) for a in range(len(kept))
                            for b, w in succs[a]]
        return kept
    return place


def sgm_faithful(dist: np.ndarray, path_xy: np.ndarray, consistency_cutoff: float,
                 edge_cutoff: float, gate_m: float = 1.0):
    """Convenience: (kept trace indices, learned edges over local indices)."""
    ctx = {"pair_distance": lambda i, j: dist[i, j]}
    kept = sgm_frame_placer(consistency_merge(consistency_cutoff),
                            edge_cutoff=edge_cutoff, gate_m=gate_m)(path_xy, ctx)
    return np.asarray(kept, dtype=int), ctx["sgm_edges"]


def learned_edges(dist: np.ndarray, kept, edge_cutoff: float) -> List[Edge]:
    """Directed edges among `kept` under the SAME cutoff as the SGM row, for
    the arms that choose nodes some other way (uniform, random, madent). Same
    rule, same weights, so the rows differ in placement only."""
    kept = np.asarray(kept, dtype=int)
    sub = dist[np.ix_(kept, kept)]
    out = []
    for a in range(len(kept)):
        for b in range(len(kept)):
            if a != b and sub[a, b] < edge_cutoff:
                out.append((a, b, float(sub[a, b])))
    return out


def undirected_components(n: int, edges) -> int:
    """Union-find ignoring direction. Labelled as such wherever printed."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for u, v, _w in edges:
        parent[find(u)] = find(v)
    return len({find(i) for i in range(n)})


def directed_reachability(n: int, edges) -> float:
    """Fraction of ORDERED node pairs (u != v) with a directed path u -> v,
    by exhaustive BFS from every node. 1.0 = every node reaches every other.

    SAME BFS, COMPILED. The pure-Python version (kept below as
    _directed_reachability_bfs, and pinned equal to this one by
    tests/test_directed_reachability_fast.py) took 83+ min on ONE calibration
    trace of 1200 nodes / ~1M edges on 2026-09-13, because wide edge cutoffs
    admit nearly every pair and it ran once per sweep row. Speed only: the
    count of reachable ordered pairs is identical.
    """
    if n < 2:
        return 1.0
    if not edges:
        return 0.0
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import breadth_first_order
    e = np.asarray([(u, v) for u, v, *_ in edges], dtype=np.int64)
    g = csr_matrix((np.ones(len(e), dtype=np.int8), (e[:, 0], e[:, 1])), shape=(n, n))
    g.sum_duplicates()
    total = 0
    for s in range(n):
        total += len(breadth_first_order(g, s, directed=True,
                                         return_predecessors=False)) - 1
    return total / (n * (n - 1))


def _directed_reachability_bfs(n: int, edges) -> float:
    """The original pure-Python exhaustive BFS; the reference the fast
    version is tested against."""
    if n < 2:
        return 1.0
    adj = [[] for _ in range(n)]
    for u, v, _w in edges:
        adj[u].append(v)
    total = 0
    for s in range(n):
        seen = np.zeros(n, dtype=bool)
        seen[s] = True
        stack = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        total += int(seen.sum()) - 1
    return total / (n * (n - 1))


def edge_separation_stats(edges, path_xy: np.ndarray, kept, horizon_m: float):
    """Median / p90 true separation of admitted edges, and the share beyond
    the trained horizon -- the honesty line the caption must carry."""
    kept = np.asarray(kept, dtype=int)
    if not edges:
        return {"n_edges": 0}
    P = np.asarray(path_xy, np.float64)[:, :2][kept]
    sep = np.array([np.linalg.norm(P[a] - P[b]) for a, b, _w in edges])
    return {"n_edges": int(len(sep)),
            "sep_median_m": float(np.median(sep)),
            "sep_p90_m": float(np.percentile(sep, 90)),
            "frac_beyond_horizon": float(np.mean(sep > horizon_m))}
