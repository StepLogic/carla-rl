"""Figures that were first drawn with inline code, now reproducible (no titles, legends below the axes).

  <drive-out>/success_breakdown.png           successes per map by how they happened (option C JSONL)
  <entropy-out>/madent_{high,low}_smooth9_entropy_trace.png  trace coloured by robust z of smoothed entropy + nodes
  <entropy-out>/madent_{high,low}_smooth9_map.png            the madent nodes with two-way sequence edges
Offline, no agent.
"""
import argparse, json, os, pickle, sys
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))

ENTROPY_JSON = "results/figures/entropy_maps/entropy_madent_hi_lo_placement_v6n_town02_budget10k_laneoff_speed10_s7.json"
LEG_BELOW = dict(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=9, framealpha=0.95)


def success_breakdown(jsonl, out_png, plt):
    R = [json.loads(l) for l in open(jsonl) if l.strip()]
    maps = list(dict.fromkeys(r["map"] for r in R))
    cats = [("Followed multi-hop plan", "#2ca02c"), ("0-hop plan", "#9ecae1"),
            ("Passed goal before first node", "#fdae6b"), ("Explore fallback", "#bcbddc")]

    def kind(r):
        if r["plan_hops"] == 0:
            return 1
        if r["explore_steps"] > 0:
            return 3
        return 2 if r["relocalisations"] == 0 else 0
    counts = np.zeros((len(maps), 4))
    for k, m in enumerate(maps):
        for r in R:
            if r["map"] == m and r["success"]:
                counts[k, kind(r)] += 1
    lab = {"entropy_nosmooth_high_thin4": "entropy\nnosmooth_high\nthinned 4 m",
           "hybrid_uniform6_entropy_thin4": "uniform6 +\nthinned\nentropy high"}
    fig, ax = plt.subplots(figsize=(12, 4.5)); x = np.arange(len(maps)); b = np.zeros(len(maps))
    for c, (l, col) in enumerate(cats):
        ax.bar(x, counts[:, c], 0.6, bottom=b, color=col, label=l); b += counts[:, c]
    ax.set_xticks(x)
    ax.set_xticklabels([lab.get(m, m.replace("_trace", "").replace("entropy_", "entropy\n")) for m in maps], fontsize=8.5)
    ax.set_ylabel("Successful episodes (of 40)"); ax.set_ylim(0, 40)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, framealpha=0.95)
    fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)


def madent_figures(out_dir, plt):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from render_placement_maps import lane_polylines
    from vae_nav.node_placement import smooth
    E = json.load(open(ENTROPY_JSON))
    tr = pickle.load(open(E["trace"], "rb")); P = np.asarray(tr["positions"])[:, :2]; seg = np.asarray(tr["segments"])
    Hs = smooth(np.asarray(E["entropy_nats"]), E["smooth_w"])
    z = (Hs - np.median(Hs)) / (1.4826 * np.median(np.abs(Hs - np.median(Hs))))
    lines = lane_polylines("Town02"); L = np.concatenate([np.asarray(l) for l in lines if len(l)]); pad = 15

    def frame(ax):
        ax.add_collection(LineCollection(lines, colors="#d6d6d6", linewidths=2.0, zorder=0))
        ax.set_xlim(L[:, 0].min() - pad, L[:, 0].max() + pad); ax.set_ylim(L[:, 1].max() + pad, L[:, 1].min() - pad)
        ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

    segs, vals = [], []
    for i in range(len(P) - 1):
        if seg[i] == seg[i + 1]:
            segs.append([P[i], P[i + 1]]); vals.append(z[i])
    lim = max(2.0, float(np.percentile(np.abs(z), 99)))
    for side, key, colour, thr, label in (("high", "high", "#d62728", E["z"], f"z ≥ {E['z']:g}"),
                                          ("low", "low", "#1f4fd6", -E["z"], f"z ≤ −{E['z']:g}")):
        idx = np.asarray(E[key]["node_trace_index"])
        fig, ax = plt.subplots(figsize=(9, 8.6)); frame(ax)
        lc = LineCollection(segs, cmap="coolwarm", norm=Normalize(-lim, lim), linewidths=2.2, zorder=1)
        lc.set_array(np.asarray(vals)); ax.add_collection(lc)
        ax.scatter(P[idx, 0], P[idx, 1], s=34, c=colour, edgecolors="black", linewidths=0.6, zorder=3,
                   label=f"madent {side} nodes, {label} ({len(idx)})")
        cb = fig.colorbar(lc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(f"Robust z of actor entropy (smoothed, window {E['smooth_w']})"); cb.ax.axhline(thr, color="black", lw=1.2)
        ax.legend(**{**LEG_BELOW, "ncol": 1})
        fig.savefig(os.path.join(out_dir, f"madent_{side}_smooth9_entropy_trace.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

        m = json.load(open(f"results/maps/sequence/entropy_smooth9_{side}_sequence.json"))
        xy = np.array([[d["x_m"], d["y_m"]] for d in m["nodes"]])
        polys = [np.vstack([xy[e["from"]], [[w["x_m"], w["y_m"]] for w in e["waypoints"]], xy[e["to"]]]) for e in m["edges"]]
        fig, ax = plt.subplots(figsize=(8, 8.6)); frame(ax)
        ax.add_collection(LineCollection(polys, colors="#1f77b4", linewidths=1.3, alpha=0.7, zorder=1))
        ax.scatter(xy[:, 0], xy[:, 1], s=40, c=colour, edgecolors="black", linewidths=0.7, zorder=3,
                   label=f"madent {side} nodes ({len(xy)})")
        ax.plot([], [], color="#1f77b4", lw=1.8, label=f"Edges ({len(m['edges'])})")
        ax.legend(**LEG_BELOW)
        fig.savefig(os.path.join(out_dir, f"madent_{side}_smooth9_map.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="results/drive_plan_c/sequence_undirected.jsonl")
    ap.add_argument("--drive-out", default="results/drive_plan_c")
    ap.add_argument("--entropy-out", default="results/figures/entropy_maps")
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(a.drive_out, exist_ok=True); os.makedirs(a.entropy_out, exist_ok=True)
    success_breakdown(a.jsonl, os.path.join(a.drive_out, "success_breakdown.png"), plt)
    madent_figures(a.entropy_out, plt)
    print(f"[misc figures] {a.drive_out}/success_breakdown.png; {a.entropy_out}/madent_{{high,low}}_smooth9_{{entropy_trace,map}}.png")


if __name__ == "__main__":
    main()
