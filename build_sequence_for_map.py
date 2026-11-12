"""Build the sequence edges for ONE driving map, without sequence_link_map.py's hard-coded map table.

sequence_link_map.py is the production edge builder, but it only knows the maps in its MAPS dict, at paths under
results/, and its main() imports carla (for the lane figure). Its two functions that matter are pure, so this reuses
them verbatim -- load_nodes_and_assign for the sample -> node assignment and sequence_edges for the walk -- and only
replaces the plumbing around them. Edges are therefore built by exactly the same rule as every other map in the pack.

Reachability is recomputed here rather than imported from vae_nav.sgm_frame (which would pull in the src tree); the
BFS below reproduces the logged directed/undirected values of the existing maps exactly.
"""
import argparse, json, os, pickle, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sequence_link_map import load_nodes_and_assign, sequence_edges  # noqa: E402


def reachability(n, edges, directed=True):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        if not directed:
            adj[v].append(u)
    total = 0
    for s in range(n):
        seen = [False] * n
        seen[s] = True
        stack, cnt = [s], 0
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if not seen[y]:
                    seen[y] = True
                    cnt += 1
                    stack.append(y)
        total += cnt
    return total / (n * (n - 1)) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="driving map JSON")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--name", required=True, help="map name (only steers the assignment branch; not 'junction')")
    ap.add_argument("--out", required=True)
    ap.add_argument("--town", default="Town02")
    a = ap.parse_args()

    tr = pickle.load(open(a.trace, "rb"))
    P = np.asarray(tr["positions"], float)[:, :2]
    yaw = np.asarray(tr["yaws"], float)
    seg = np.asarray(tr["segments"])

    nodes, assign, old_edges, old_reach = load_nodes_and_assign(a.name, a.map, P)
    n = len(nodes)
    edges = sequence_edges(assign, P, yaw, seg)
    el = [(e["from"], e["to"]) for e in edges]
    for k, nd in enumerate(nodes):
        nd["source_id"] = nd.get("id")
        nd["id"] = k

    out = {"schema": "carla-rl sequence map v1: nodes = the source map's nodes (id = position); edges = node changes "
                     "along the trace in order ACROSS episodes, directed, shortest driven span per node pair",
           "town": a.town, "source_map": a.map, "trace": a.trace,
           "assigned_samples": int((assign >= 0).sum()), "trace_samples": len(P),
           "directed_reachability": reachability(n, el, True),
           "undirected_reachability": reachability(n, el, False),
           "nodes": nodes, "edges": edges}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    lens = np.asarray([e["driven_length_m"] for e in edges], float)
    print(f"[seq] {a.name}: {n} nodes, {len(edges)} edges ({sum(e['crosses_episodes'] for e in edges)} across resets) "
          f"| reach directed {out['directed_reachability']:.3f} undirected {out['undirected_reachability']:.3f} "
          f"| driven length median {np.median(lens):.1f} m max {lens.max():.1f} m "
          f"| assigned {out['assigned_samples']}/{len(P)} -> {a.out}")


if __name__ == "__main__":
    main()
