#!/usr/bin/env python3
"""Build the image-goal map arms from exploration traces (vae_nav.image_map).

For each town: the real-time entropy map first (the peer's online MAD, z=1.0,
110 m window), then uniform spacing and random placement MATCHED to its node
count, so the comparison is about where nodes sit and never how many.

    build_image_maps.py --trace-root checkpoints --trace-fmt trace_v30det_{town} \
        --towns Town02 Town03 Town04 Town10HD --out maps/v30det

writes <out>/<placement>/<Town>.json + <Town>_frames.npy and prints, per map,
nodes / edges / nodes per km driven / largest strongly connected component.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from vae_nav import image_map as im  # noqa: E402


def trace_dir(root, fmt, town):
    d = os.path.join(root, fmt.format(town=town))
    if os.path.exists(os.path.join(d, "index.json")):
        return d
    hits = sorted(glob.glob(os.path.join(d, "*", "index.json")))
    if len(hits) != 1:
        raise SystemExit(f"{d}: expected one index.json, found {hits}")
    return os.path.dirname(hits[0])


def largest_scc(g):
    """Size of the largest strongly connected component (Kosaraju)."""
    n = len(g.nodes)
    fwd, rev = [[] for _ in range(n)], [[] for _ in range(n)]
    for a, b in g.edges:
        fwd[a].append(b)
        rev[b].append(a)
    seen, order = [False] * n, []
    for s in range(n):
        if seen[s]:
            continue
        stack = [(s, 0)]
        seen[s] = True
        while stack:
            u, i = stack.pop()
            if i < len(fwd[u]):
                stack.append((u, i + 1))
                v = fwd[u][i]
                if not seen[v]:
                    seen[v] = True
                    stack.append((v, 0))
            else:
                order.append(u)
    comp, best = [-1] * n, 0
    for s in reversed(order):
        if comp[s] >= 0:
            continue
        size, stack = 0, [s]
        comp[s] = s
        while stack:
            u = stack.pop()
            size += 1
            for v in rev[u]:
                if comp[v] < 0:
                    comp[v] = s
                    stack.append(v)
        best = max(best, size)
    return best


def save(out, placement, town, trace, idx, meta, max_turn_deg=None):
    d = os.path.join(out, placement)
    os.makedirs(d, exist_ok=True)
    g = im.build_graph(trace, idx, meta={"placement": placement, "town": town,
                                         "trace": trace["dir"], **meta,
                                         **({"max_turn_deg": max_turn_deg}
                                            if max_turn_deg is not None else {})},
                       max_turn_deg=max_turn_deg)
    g.save(os.path.join(d, f"{town}.json"))
    np.save(os.path.join(d, f"{town}_frames.npy"), im.node_frames(trace, idx))
    km = trace["meta"].get("total_m") or float(np.sum(np.hypot(*np.diff(trace["pos"], axis=0).T)))
    scc = largest_scc(g)
    lens = sorted(g.edges.values())
    row = {"town": town, "placement": placement, "nodes": len(g.nodes),
           "edges": len(g.edges), "nodes_per_km": round(len(g.nodes) / (km / 1000.0), 1),
           "largest_scc": scc, "scc_frac": round(scc / max(len(g.nodes), 1), 3),
           "edge_m_median": round(float(np.median(lens)), 1) if lens else None,
           "edge_m_p95": round(float(np.percentile(lens, 95)), 1) if lens else None, **meta}
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-root", default="checkpoints")
    ap.add_argument("--trace-fmt", default="trace_v30det_{town}")
    ap.add_argument("--towns", nargs="+", default=["Town02", "Town03", "Town04", "Town10HD"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--z", type=float, default=1.0)
    ap.add_argument("--window-m", type=float, default=110.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sgm", action="store_true",
                    help="build only the SGM map from <out>/sgm_pairs_<Town>.npz")
    ap.add_argument("--sgm-gate-m", type=float, default=10.0)
    ap.add_argument("--sgm-cutoff", type=float, default=5.0)
    a = ap.parse_args()
    rows = []
    if a.sgm:
        for town in a.towns:
            tr = im.load_trace(trace_dir(a.trace_root, a.trace_fmt, town))
            z = np.load(os.path.join(a.out, f"sgm_pairs_{town}.npz"))
            keep, st = im.sgm_place(z["cand"], z["pairs"], z["q"], z["xy"], z["yaw"],
                                    gate_m=a.sgm_gate_m, cutoff=a.sgm_cutoff)
            rows.append(save(a.out, "sgm", town, tr, keep, st))
        with open(os.path.join(a.out, "maps_summary_sgm.json"), "w") as fh:
            json.dump(rows, fh, indent=1)
        return
    for town in a.towns:
        tr = im.load_trace(trace_dir(a.trace_root, a.trace_fmt, town))
        ent, st = im.place_entropy(tr, z=a.z, window_m=a.window_m, return_stats=True)
        rows.append(save(a.out, "entropy", town, tr, ent,
                         {"z": a.z, "window_m": a.window_m,
                          "degenerate_window_frac": st["degenerate_window_frac"]}))
        spacing, uni = im.match_uniform(tr, len(ent))
        rows.append(save(a.out, "uniform", town, tr, uni, {"spacing_m": round(spacing, 3)}))
        rnd = im.place_random(tr, len(ent), seed=a.seed)
        rows.append(save(a.out, "random", town, tr, rnd, {"seed": a.seed}))
    with open(os.path.join(a.out, "maps_summary.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
