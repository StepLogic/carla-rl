"""Reduce a trace-edge uniform graph to JUNCTION NODES ONLY.

Input: a render_uniform_frame.py JSON made with --edges trace (nodes + directed
driven edges). Offline, CPU only.

1. Each node is snapped to the town's OpenDRIVE map (carla.Map from the .xodr,
   project_to_road). A node is a junction node if its waypoint is_junction.
   Nodes inside the SAME junction (waypoint.get_junction().id) collapse into one
   junction vertex at their centroid.
2. Edges are CONTRACTED: from every junction node, driven edges are followed
   through non-junction nodes until a junction node is reached; that chain becomes
   one directed edge junction_a -> junction_b carrying its driven length and the
   node path. A chain that never reaches a junction (a crash dead end, or the
   trace's end) is dropped. Self-loops (leaving and re-entering the same junction)
   are dropped.
"""
import argparse, json, os, sys
from collections import defaultdict

import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))
XODR = "/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive/{}.xodr"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-json", default="results/figures/uniform_frame_trace_edges/"
                    "uniform6_placement_v6n_town02_budget10k_laneoff_speed10_s7.json")
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--out", default="results/figures/junction_graph")
    a = ap.parse_args()

    import carla
    import pickle
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from render_placement_maps import lane_polylines, finish
    from vae_nav.sgm_frame import directed_reachability

    g = json.load(open(a.graph_json))
    tr = pickle.load(open(g["trace"], "rb"))
    P = np.asarray(tr["positions"], np.float64)[:, :2]
    seg = np.asarray(tr["segments"])
    idx = [int(i) for i in g["node_trace_index"]]
    xy = np.asarray(g["node_xy"], np.float64)
    local = {ti: n for n, ti in enumerate(idx)}
    succ = defaultdict(list)
    for fa, fb, _s in g["edges"]:
        succ[local[int(fa)]].append(local[int(fb)])

    amap = carla.Map(a.town, open(XODR.format(a.town)).read())
    jid = np.full(len(idx), -1, int)
    for n, (x, y) in enumerate(xy):
        w = amap.get_waypoint(carla.Location(x=float(x), y=float(y), z=0.0), project_to_road=True)
        if w is not None and w.is_junction:
            jid[n] = int(w.get_junction().id)
    junctions = sorted({int(j) for j in jid if j >= 0})
    jindex = {j: k for k, j in enumerate(junctions)}
    jxy = np.array([xy[jid == j].mean(axis=0) for j in junctions]) if junctions else np.zeros((0, 2))
    print(f"[junctions] {int((jid >= 0).sum())} of {len(idx)} nodes lie in {len(junctions)} map junctions")

    def seglen(path):
        return float(sum(np.linalg.norm(xy[path[i]] - xy[path[i + 1]]) for i in range(len(path) - 1)))

    chains = {}
    for start in np.flatnonzero(jid >= 0):
        for nxt in succ[int(start)]:
            path, cur, seen, left, ok = [int(start)], int(nxt), {int(start)}, False, False
            while True:
                path.append(cur)
                if jid[cur] >= 0 and jid[cur] != jid[start]:
                    ok = True                               # reached ANOTHER junction
                    break
                if jid[cur] == jid[start]:
                    if left:
                        break                               # left and came back: a loop, dropped
                    # still inside the start junction (junctions hold 2-3 nodes): keep walking
                else:
                    left = True                             # on open road between junctions
                nexts = [v for v in succ[cur] if v not in seen]
                if not nexts:
                    break                                   # dead end: never reached a junction
                seen.add(cur)
                cur = nexts[0]
            if not ok:
                continue
            key = (jindex[int(jid[start])], jindex[int(jid[path[-1]])])
            L = seglen(path)
            if key not in chains or L < chains[key]["length_m"]:
                chains[key] = {"length_m": L, "node_path": path}
    edges = [(u, v, c["length_m"]) for (u, v), c in sorted(chains.items())]
    reach = directed_reachability(len(junctions), edges) if len(junctions) > 1 else float("nan")
    lens = [c["length_m"] for c in chains.values()]
    print(f"[junction graph] {len(junctions)} junction nodes, {len(edges)} contracted edges, "
          f"median edge {np.median(lens) if lens else float('nan'):.1f} m driven, directed reach {reach:.3f}")

    lines = lane_polylines(a.town)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.add_collection(LineCollection(lines, colors="#c8c8c8", linewidths=2.2, zorder=0))
    ax.add_collection(LineCollection([P[seg == k] for k in np.unique(seg)], colors="#1f77b4",
                                     linewidths=0.7, alpha=0.35, zorder=1, label="Exploration trace"))
    ax.add_collection(LineCollection([xy[c["node_path"]] for c in chains.values()], colors="#d62728",
                                     linewidths=1.6, alpha=0.9, zorder=2))
    ax.plot([], [], color="#d62728", label=f"Contracted edges ({len(edges)}, drawn along the driven path)")
    ax.scatter(xy[jid >= 0, 0], xy[jid >= 0, 1], s=10, c="#ff7f0e", zorder=3, alpha=0.7,
               label=f"uniform nodes inside junctions ({int((jid >= 0).sum())})")
    ax.scatter(jxy[:, 0], jxy[:, 1], s=90, c="#2ca02c", edgecolors="black", linewidths=0.8, zorder=4,
               label=f"Junction nodes ({len(junctions)})")
    base = os.path.basename(a.graph_json).replace(".json", "")
    finish(ax, P, f"Junction-only graph from {base.split('_placement')[0]}[trace edges]\n"
                  f"{len(junctions)} junction nodes, {len(edges)} contracted edges, reach {reach:.3f}")
    L = np.concatenate([np.asarray(l) for l in lines if len(l)]); pad = 20
    ax.set_xlim(L[:, 0].min() - pad, L[:, 0].max() + pad); ax.set_ylim(L[:, 1].max() + pad, L[:, 1].min() - pad)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, framealpha=0.95)
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"junctions_{base}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    json.dump({"source_graph": a.graph_json, "town": a.town,
               "junction_ids": junctions, "junction_xy": jxy.tolist(),
               "node_junction_id": jid.tolist(),
               "edges": [{"from_junction": junctions[u], "to_junction": junctions[v], "driven_length_m": L,
                          "node_path": chains[(u, v)]["node_path"]} for u, v, L in edges],
               "directed_reachability": reach}, open(out.replace(".png", ".json"), "w"), indent=1)
    print(f"[Render] {out} (+ .json)")


if __name__ == "__main__":
    main()
