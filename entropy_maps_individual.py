"""Individual, untitled entropy plots + driving maps from render_entropy_maps.py JSON.

No agent/GPU: reads the saved per-candidate entropy and the high/low madent node
indices. Writes, per input JSON:
  <prefix>_entropy.png      trace coloured by actor entropy
  <prefix>_high_nodes.png   HIGH-entropy madent nodes on the trace
  <prefix>_low_nodes.png    LOW-entropy madent nodes on the trace
  (no titles; axes, legend and colour bar kept)
  maps/<prefix>_{high,low}_driving_map.json
Driving-map edges: each selected node links to the NEXT selected node reached along
the trace within the same episode (directed, driving direction). The waypoints are
the trace candidates in between. Nodes last in their episode get no outgoing edge.
Revisits are not merged.
"""
import argparse, json, os, pickle, sys
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))


def driving_map(sel, P, yaw, seg, H, z, meta, kind, src):
    sel = sorted(int(i) for i in sel)
    nodes = [{"id": n, "trace_index": ti, "x_m": float(P[ti, 0]), "y_m": float(P[ti, 1]),
              "yaw_rad": float(yaw[ti]), "entropy_nats": float(H[ti]), "z": float(z[ti]),
              "episode": int(seg[ti])} for n, ti in enumerate(sel)]
    edges = []
    for n in range(len(sel) - 1):
        a, b = sel[n], sel[n + 1]
        if seg[a] != seg[b]:
            continue                                       # episode ended between them
        span = list(range(a, b + 1))
        d = float(np.sum(np.linalg.norm(np.diff(P[span], axis=0), axis=1)))
        edges.append({"from": n, "to": n + 1, "directed": True, "driven_length_m": round(d, 2),
                      "n_waypoints": len(span),
                      "waypoints": [{"x_m": float(P[i, 0]), "y_m": float(P[i, 1]), "trace_index": i,
                                     "yaw_rad": float(yaw[i])} for i in span]})
    from vae_nav.sgm_frame import directed_reachability
    reach = directed_reachability(len(nodes), [(e["from"], e["to"], 1.0) for e in edges]) if len(nodes) > 1 else float("nan")
    return {"schema": "carla-rl entropy node map v1: nodes = madent-selected trace candidates; edges = next selected "
                      "node reached along the trace within the same episode (directed), waypoints = trace candidates between",
            "town": meta["town"], "coordinates": "CARLA world metres (left-handed; plot y inverted for top-down)",
            "selection": kind, "source": src, "directed_reachability": reach,
            "nodes": nodes, "edges": edges,
            "note": "Not yet loadable by eval_placement.py; revisits are not merged, so reachability is limited."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--town", default="Town02")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from render_placement_maps import lane_polylines

    d = json.load(open(a.json))
    tr = pickle.load(open(d["trace"], "rb"))
    P = np.asarray(tr["positions"], np.float64)[:, :2]
    yaw = np.asarray(tr["yaws"], np.float64); seg = np.asarray(tr["segments"])
    H = np.asarray(d["entropy_nats"], np.float64)
    sw = int(d["smooth_w"])
    from vae_nav.node_placement import smooth
    hs = smooth(H, sw)
    c = float(np.median(hs)); scale = 1.4826 * float(np.median(np.abs(hs - c)))
    z_hi = (hs - c) / scale
    hi = np.asarray(d["high"]["node_trace_index"], int); lo = np.asarray(d["low"]["node_trace_index"], int)

    lines = lane_polylines(a.town)
    L = np.concatenate([np.asarray(l) for l in lines if len(l)]); pad = 20
    os.makedirs(a.out, exist_ok=True); os.makedirs(os.path.join(a.out, "maps"), exist_ok=True)

    def base_ax(alpha):
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.add_collection(LineCollection(lines, colors="#d0d0d0", linewidths=2.0, zorder=0))
        lc = None
        for k in np.unique(seg):
            i = np.flatnonzero(seg == k)
            if len(i) < 2:
                continue
            lc = LineCollection(np.stack([P[i][:-1], P[i][1:]], 1), cmap="viridis", linewidths=2.2,
                                zorder=1, alpha=alpha)
            lc.set_array(H[i][:-1]); lc.set_clim(H.min(), H.max()); ax.add_collection(lc)
        ax.set_xlim(L[:, 0].min() - pad, L[:, 0].max() + pad); ax.set_ylim(L[:, 1].max() + pad, L[:, 1].min() - pad)
        ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.02); cb.set_label("Actor entropy (nats)")
        return fig, ax

    written = []
    fig, ax = base_ax(1.0)
    p = os.path.join(a.out, f"{a.prefix}_entropy.png"); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); written.append(p)
    for name, nodes, colour, lab in (("high", hi, "#d62728", "High-entropy nodes"), ("low", lo, "#1f77b4", "Low-entropy nodes")):
        fig, ax = base_ax(0.45)
        if len(nodes):
            ax.scatter(P[nodes, 0], P[nodes, 1], s=55, c=colour, edgecolors="black", linewidths=0.7, zorder=3,
                       label=f"{lab} ({len(nodes)}), mean {H[nodes].mean():.3f} nats")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, framealpha=0.95)
        p = os.path.join(a.out, f"{a.prefix}_{name}_nodes.png"); fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); written.append(p)

        z_sel = z_hi if name == "high" else -z_hi
        src = {"entropy_json": a.json, "trace": d["trace"], "checkpoint": d["checkpoint"], "trace_meta": tr["meta"],
               "madent": {"z": d["z"], "min_sep_m": d["min_sep_m"], "smooth_w": sw,
                          "signal": "entropy" if name == "high" else "-entropy", "nms_w": 0}}
        m = driving_map(nodes, P, yaw, seg, H, z_sel, tr["meta"], f"madent {name}-entropy (z>={d['z']:g}, smooth_w={sw})", src)
        p = os.path.join(a.out, "maps", f"{a.prefix}_{name}_driving_map.json"); json.dump(m, open(p, "w"), indent=1); written.append(p)
        print(f"[map] {name}: {len(m['nodes'])} nodes, {len(m['edges'])} edges, "
              f"{sum(e['n_waypoints'] for e in m['edges'])} waypoints, reach {m['directed_reachability']:.3f}")
    for p in written:
        print("[wrote]", p)


if __name__ == "__main__":
    main()
