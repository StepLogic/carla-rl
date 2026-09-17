"""Node+edge graphs drawn with STRAIGHT edges (user 2026-09-16: "show plots").

sequence_link_map.py's own figure draws each edge's stored waypoints, which span only the ~2 m across a node
boundary, so connectivity is invisible. Here every edge is a straight segment between the two node positions, which
is what makes loop closures (an edge between nodes far apart in trace order) visible.
"""
import argparse, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", required=True, help="label=map_name")
    ap.add_argument("--seq-dir", default="results/maps/sequence")
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--out", default="results/figures/graphs.png")
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from render_placement_maps import lane_polylines
    lanes = lane_polylines(a.town)
    L = np.concatenate([np.asarray(l) for l in lanes if len(l)]); pad = 15
    specs = [m.split("=") for m in a.maps]
    fig, axes = plt.subplots(1, len(specs), figsize=(5.4 * len(specs), 5.4), squeeze=False)
    for ax, (label, name) in zip(axes[0], specs):
        d = json.load(open(f"{a.seq_dir}/{name}_sequence.json"))
        xy = np.asarray([[n["x_m"], n["y_m"]] for n in d["nodes"]], float)
        ti = np.asarray([n["trace_index"] for n in d["nodes"]])
        segs, loops = [], []
        for e in d["edges"]:
            u, v = int(e["from"]), int(e["to"])
            (loops if abs(ti[u] - ti[v]) > 20 else segs).append([xy[u], xy[v]])
        ax.add_collection(LineCollection(lanes, colors="#e2e2e2", linewidths=1.6, zorder=0))
        if segs:
            ax.add_collection(LineCollection(segs, colors="#d62728", linewidths=1.1, alpha=0.85, zorder=2))
        if loops:
            ax.add_collection(LineCollection(loops, colors="#1f77b4", linewidths=2.0, alpha=0.95, zorder=3))
        ax.scatter(xy[:, 0], xy[:, 1], s=9, c="black", zorder=4)
        ax.set_title(f"{label}\n{len(xy)} nodes, {len(d['edges'])} edges "
                     f"({len(loops)} joining different passes)", fontsize=10)
        ax.set_xlim(L[:, 0].min() - pad, L[:, 0].max() + pad); ax.set_ylim(L[:, 1].max() + pad, L[:, 1].min() - pad)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.5, 0.02, "red: edge between consecutive passes   blue: edge joining nodes far apart in trace order (loop closure)",
             ha="center", fontsize=9)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=170, bbox_inches="tight"); print("[wrote]", a.out)


if __name__ == "__main__":
    main()
