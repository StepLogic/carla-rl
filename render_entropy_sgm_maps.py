"""Entropy + SGM node placement against plain SGM at matched node counts (user 2026-09-19 "plot map").

Top row: the entropy seeds (pinned; madent in the code) and the SGM nodes placed around them. Bottom row: the SGM row's own
map at the same budget. Annotated with the coverage gap -- how far a trace sample can be from its nearest
node -- because that is what separates the two rows at the sparse end.
Run from the icra-2027 folder: python3 <worktree>/render_entropy_sgm_maps.py
"""
import json, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))

TRACE = "results/placement_v6n_town02_budget10k_laneoff_speed10/trace_s7.pkl"
MAPS = "results/maps"
SEED = "#c0398b"    # the Entropy + SGM hue in the Pareto figure
FILL = "#4a3aa7"    # the SGM hue
BLOCKS = [("hybrid_sgm_g44_entropy_thin4", "sgm_g24_e55_c55"),
          ("hybrid_sgm_g14_entropy_thin4", "sgm_g10_e55_c55"),
          ("hybrid_sgm_g9_entropy_thin4",  "sgm_g7_e55_c55"),
          ("hybrid_sgm_g2_entropy_thin4",  "sgm_g2_e55_c55")]

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8})


def nodes_of(name):
    d = json.load(open(f"{MAPS}/town02_speed10_s7_{name}_driving_map.json"))["nodes"]
    xy = np.array([[n["x_m"], n["y_m"]] for n in d])
    kind = np.array([n.get("kind", "sgm") for n in d])
    return xy, kind


def coverage(P, xy):
    """Distance from every trace sample to its nearest node -> (median, p90, max)."""
    d = np.linalg.norm(P[:, None] - xy[None], axis=-1).min(1)
    return np.median(d), np.percentile(d, 90), d.max()


def main():
    from render_placement_maps import lane_polylines
    tr = pickle.load(open(TRACE, "rb")); P = np.asarray(tr["positions"], float)[:, :2]
    lanes = [np.asarray(l) for l in lane_polylines("Town02") if len(l)]
    L = np.concatenate(lanes); pad = 15
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 6.4), constrained_layout=True)
    handles = [None, None]
    for j, (ours, theirs) in enumerate(BLOCKS):
        for i, name in enumerate((ours, theirs)):
            ax = axes[i, j]
            for l in lanes:
                ax.plot(l[:, 0], l[:, 1], color="#dedede", lw=1.6, solid_capstyle="round", zorder=1)
            ax.plot(P[:, 0], P[:, 1], color="#b8b8b8", lw=0.5, zorder=2)
            xy, kind = nodes_of(name)
            fillxy, seedxy = xy[kind != "entropy"], xy[kind == "entropy"]
            h1 = ax.scatter(*fillxy.T, s=11, color=FILL, lw=0, zorder=3, label="SGM node")
            h2 = None
            if len(seedxy):
                h2 = ax.scatter(*seedxy.T, s=34, color=SEED, marker="^", lw=0, zorder=4, label="Entropy seed")
            handles[0], handles[1] = h1, h2 or handles[1]
            med, p90, mx = coverage(P, xy)
            sub = f" ({len(seedxy)} seeds + {len(fillxy)} SGM)" if len(seedxy) else ""
            ax.set_title(("Entropy + SGM" if i == 0 else "SGM") + f" — {len(xy)} nodes" + sub, fontsize=9)
            ax.text(0.5, -0.04, f"Coverage gap: median {med:.1f} m, p90 {p90:.1f} m, max {mx:.1f} m",
                    transform=ax.transAxes, fontsize=7, color="#444444", ha="center", va="top")
            ax.set_xlim(L[:, 0].min() - pad, L[:, 0].max() + pad)
            ax.set_ylim(L[:, 1].max() + pad, L[:, 1].min() - pad)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#cccccc")
    fig.suptitle("Town02: the entropy seeds are pinned, then SGM fills around them (top) vs SGM alone (bottom)",
                 fontsize=10)
    fig.legend([h for h in handles if h is not None], ["SGM node", "Entropy seed"],
               loc="outside lower center", ncol=2, frameon=False, fontsize=8.5, handletextpad=0.3)
    for e in ("pdf", "png"):
        fig.savefig(f"figures/entropy_sgm_maps_town02.{e}", **({"dpi": 200} if e == "png" else {}))
    print("-> figures/entropy_sgm_maps_town02.{pdf,png}")


if __name__ == "__main__":
    main()
