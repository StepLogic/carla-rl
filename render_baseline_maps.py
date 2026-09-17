"""Node-kind map figures for the two added baselines (evidence figures, no titles).

  map_entropy_nosmooth_high_thin4.png   kept entropy nodes (red), nodes removed by 4 m thinning (grey rings),
                                        sequence edges drawn along the driven waypoints (blue)
  map_hybrid_uniform6_entropy_thin4.png thinned entropy nodes (red), uniform 6 m nodes (white), edges (blue)
  map_entropy_nosmooth_high.png         the unthinned nosmooth-high nodes (red) and edges (blue), same style
Offline, no agent.
"""
import argparse, json, os, sys
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/drive_plan_c/figures")
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from render_placement_maps import lane_polylines
    from render_plan_evidence import place_legend

    os.makedirs(a.out_dir, exist_ok=True)
    lines = lane_polylines("Town02"); L = np.concatenate([np.asarray(l) for l in lines if len(l)]); pad = 15
    orig = json.load(open("results/maps/town02_speed10_s7_nosmooth_high_driving_map.json"))["nodes"]
    oxy = np.array([[d["x_m"], d["y_m"]] for d in orig])

    def frame(ax):
        ax.add_collection(LineCollection(lines, colors="#d6d6d6", linewidths=2.0, zorder=0))
        ax.set_xlim(L[:, 0].min() - pad, L[:, 0].max() + pad); ax.set_ylim(L[:, 1].max() + pad, L[:, 1].min() - pad)
        ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

    def edges_of(name):
        m = json.load(open(f"results/maps/sequence/{name}_sequence.json"))
        xy = np.array([[d["x_m"], d["y_m"]] for d in m["nodes"]])
        polys = [np.vstack([xy[e["from"]], [[w["x_m"], w["y_m"]] for w in e["waypoints"]], xy[e["to"]]]) for e in m["edges"]]
        return m, xy, polys

    m, xy, polys = edges_of("entropy_nosmooth_high_thin4")
    n_kept = len(xy)
    fig, ax = plt.subplots(figsize=(8, 8)); frame(ax)
    ax.add_collection(LineCollection(polys, colors="#1f77b4", linewidths=1.3, alpha=0.7, zorder=1))
    kept = {tuple(np.round(p, 3)) for p in xy}
    rem = np.array([p for p in oxy if tuple(np.round(p, 3)) not in kept])
    ax.scatter(rem[:, 0], rem[:, 1], s=22, facecolors="none", edgecolors="#888888", linewidths=0.8, zorder=2,
               label=f"Removed by thinning ({len(rem)})")
    ax.scatter(xy[:, 0], xy[:, 1], s=60, c="#d62728", edgecolors="black", linewidths=0.8, zorder=3,
               label=f"Kept entropy nodes ({len(xy)})")
    ax.plot([], [], color="#1f77b4", lw=1.8, label=f"Edges ({len(m['edges'])})")
    place_legend(fig, ax, np.concatenate([xy, rem]), np.concatenate([L] + list(polys)), ncol=1, fontsize=9, framealpha=0.95)
    fig.savefig(os.path.join(a.out_dir, "map_entropy_nosmooth_high_thin4.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    # user 2026-09-14: the unthinned nosmooth-high map in the same style as the thinned one
    m0, xy0, polys0 = edges_of("entropy_nosmooth_high")
    fig, ax = plt.subplots(figsize=(8, 8)); frame(ax)
    ax.add_collection(LineCollection(polys0, colors="#1f77b4", linewidths=1.3, alpha=0.7, zorder=1))
    ax.scatter(xy0[:, 0], xy0[:, 1], s=60, c="#d62728", edgecolors="black", linewidths=0.8, zorder=3,
               label=f"Entropy nodes ({len(xy0)})")
    ax.plot([], [], color="#1f77b4", lw=1.8, label=f"Edges ({len(m0['edges'])})")
    place_legend(fig, ax, xy0, np.concatenate([L] + list(polys0)), ncol=1, fontsize=9, framealpha=0.95)
    fig.savefig(os.path.join(a.out_dir, "map_entropy_nosmooth_high.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    m, xy, polys = edges_of("hybrid_uniform6_entropy_thin4")
    kind = np.array([d["kind"] for d in m["nodes"]])
    fig, ax = plt.subplots(figsize=(8, 8)); frame(ax)
    ax.add_collection(LineCollection(polys, colors="#1f77b4", linewidths=1.3, alpha=0.7, zorder=1))
    u, e = kind == "uniform", kind == "entropy"
    ax.scatter(xy[u, 0], xy[u, 1], s=22, c="white", edgecolors="black", linewidths=0.7, zorder=2,
               label=f"Uniform 6 m nodes ({u.sum()})")
    ax.scatter(xy[e, 0], xy[e, 1], s=60, c="#d62728", edgecolors="black", linewidths=0.8, zorder=3,
               label=f"Thinned entropy nodes ({e.sum()})")
    ax.plot([], [], color="#1f77b4", lw=1.8, label=f"Edges ({len(m['edges'])})")
    place_legend(fig, ax, xy, np.concatenate([L] + list(polys)), ncol=1, fontsize=9, framealpha=0.95)
    fig.savefig(os.path.join(a.out_dir, "map_hybrid_uniform6_entropy_thin4.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[maps] {a.out_dir}: thin4 kept {n_kept} / removed {len(rem)}; hybrid uniform {u.sum()} entropy {e.sum()}")


if __name__ == "__main__":
    main()
