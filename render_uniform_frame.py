"""uniform[g]: the SGM frame's uniform placement (always_merge) on a trace.

Every sample merges into ANY kept node within gate_m metres, so the gate radius
IS the spacing and revisits collapse. Edges use the one edge rule (critic steps
< edge cutoff, directed). With no calibration for the trace, the cutoff is
--edge-factor x the band median critic distance (20-50 m), the same rule the
Town02 calibration landed on (1.5x). Offline, CPU only.
"""
import argparse, json, os, pickle, sys
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/placement_v6n_town02_budget10k_laneoff_speed6_sample")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--gate-m", type=float, default=2.0)
    ap.add_argument("--edge-factor", type=float, default=1.5)
    ap.add_argument("--edge-sample", type=int, default=3000)
    ap.add_argument("--out", default="results/figures/uniform_frame")
    # critic: the SGM frame's learned edges (critic steps < cutoff).
    # trace:  only the edges the car actually drove -- each sample is assigned to
    #         the node that absorbed it (itself if kept, else the first earlier-
    #         kept node inside the gate, as always_merge does) and consecutive
    #         samples of the same episode link their nodes.
    ap.add_argument("--edges", choices=("critic", "trace"), default="critic")
    a = ap.parse_args()

    import calibrate_critic as cc
    from vae_nav.sgm_frame import sgm_frame_placer, always_merge, directed_reachability
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from render_placement_maps import lane_polylines, finish

    tp = os.path.join(a.run_dir, f"trace_s{a.seed}.pkl")
    tr = pickle.load(open(tp, "rb"))
    P, dist = cc.load(tp)
    seg = np.asarray(tr["segments"])
    t = cc.separation_table(P, dist)
    base = float(np.median([t[b][0] for b in ("20-30", "30-50") if t[b][0] is not None]))
    edge = round(base * a.edge_factor, 1)
    ctx = {"pair_distance": lambda i, j: dist[i, j]}
    kept = np.asarray(sgm_frame_placer(always_merge(), edge_cutoff=edge, gate_m=a.gate_m)(P, ctx), int)
    edges = ctx["sgm_edges"]
    if a.edges == "trace":
        # node that absorbed each sample: itself if kept, else the FIRST node
        # kept before it inside the gate (always_merge's loop order)
        local = {int(k): n for n, k in enumerate(kept)}
        assign = np.full(len(P), -1, int)
        for i in range(len(P)):
            if i in local:
                assign[i] = local[i]
                continue
            prior = [n for n, k in enumerate(kept) if k < i and np.linalg.norm(P[k] - P[i]) < a.gate_m]
            if prior:
                assign[i] = prior[0]
        es = set()
        for i in range(len(P) - 1):
            u, v = assign[i], assign[i + 1]
            if seg[i] == seg[i + 1] and u >= 0 and v >= 0 and u != v:
                es.add((int(u), int(v)))
        edges = [(u, v, float(dist[kept[u], kept[v]])) for u, v in sorted(es)]
    reach = directed_reachability(len(kept), edges)
    N = P[kept]
    sep = np.array([np.linalg.norm(N[u] - N[v]) for u, v, _ in edges]) if edges else np.zeros(0)
    print(f"[uniform{a.gate_m:g}] {len(kept)}/{len(P)} nodes, {len(edges)} edges (cutoff {edge} = "
          f"{a.edge_factor:g} x band median {base:.1f} steps), directed reach {reach:.3f}, "
          f"edge sep median {np.median(sep) if len(sep) else float('nan'):.1f} m")

    lines = lane_polylines(a.town)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.add_collection(LineCollection(lines, colors="#c8c8c8", linewidths=2.2, zorder=0))
    ax.add_collection(LineCollection([P[seg == k] for k in np.unique(seg)], colors="#1f77b4",
                                     linewidths=0.8, alpha=0.5, zorder=1, label="Exploration trace"))
    if edges:
        ei = np.array([(u, v) for u, v, _ in edges], int)
        if a.edges == "trace":
            # few enough to draw all; darker, so the driven connectivity reads
            ax.add_collection(LineCollection(np.stack([N[ei[:, 0]], N[ei[:, 1]]], 1), colors="#d62728",
                                             linewidths=1.0, alpha=0.9, zorder=2))
            ax.plot([], [], color="#d62728", label=f"Trace edges ({len(edges)}, consecutive driven nodes)")
        else:
            pick = np.random.default_rng(0).choice(len(ei), size=min(a.edge_sample, len(ei)), replace=False)
            ax.add_collection(LineCollection(np.stack([N[ei[pick, 0]], N[ei[pick, 1]]], 1), colors="#7f7f7f",
                                             linewidths=0.4, alpha=0.2, zorder=2, rasterized=True))
            ax.plot([], [], color="#7f7f7f", label=f"Critic edges (random {len(pick)} of {len(edges)})")
    ax.scatter(N[:, 0], N[:, 1], s=7, c="#2ca02c", zorder=3, label=f"uniform[{a.gate_m:g} m] nodes ({len(kept)})")
    ck = tr["meta"]["checkpoint"]; ckname = ck.split("/")[-2] + "@" + ck.split("_")[-1].split(".")[0]
    speed = tr["meta"].get("target_speed_kmh") or "20"
    etxt = (f"trace edges only ({len(edges)})" if a.edges == "trace"
            else f"critic edges, cutoff {edge} steps")
    finish(ax, P, f"{ckname}  {a.town}  trace s{a.seed} ({tr['meta'].get('collect_action_mode', 'eval')} actions, "
                  f"{speed} km/h)\nuniform[{a.gate_m:g} m] (SGM frame, always-merge): {len(kept)} nodes, "
                  f"{etxt}, reach {reach:.3f}")
    L = np.concatenate([np.asarray(l) for l in lines if len(l)]); pad = 20
    ax.set_xlim(L[:, 0].min() - pad, L[:, 0].max() + pad); ax.set_ylim(L[:, 1].max() + pad, L[:, 1].min() - pad)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, framealpha=0.95)
    os.makedirs(a.out, exist_ok=True)
    tag = os.path.basename(a.run_dir.rstrip("/"))
    out = os.path.join(a.out, f"uniform{a.gate_m:g}_{tag}_s{a.seed}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    json.dump({"trace": tp, "gate_m": a.gate_m, "edge_cutoff_steps": edge, "band_median_steps": base,
               "node_trace_index": kept.tolist(), "node_xy": N.tolist(),
               "edges": [[int(kept[u]), int(kept[v]), float(w)] for u, v, w in edges],
               "directed_reachability": reach}, open(out.replace(".png", ".json"), "w"))
    print(f"[Render] {out} (+ .json)")


if __name__ == "__main__":
    main()
