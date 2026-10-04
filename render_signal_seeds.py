"""What each uncertainty signal picks as seeds (user 2026-09-20: epistemic vs aleatoric vs policy entropy).

Top row: the 30 seeds each signal selects. COUNT-MATCHED on purpose -- the signals natively return 71 / 93 /
202 nodes, and damage tracks the seed share of a budget, so an unmatched comparison would measure the count.
Same z>=2 MAD filter, same greedy 4 m spatial thinning, capped at the 30 highest-scoring separated samples.
The three sets share 0 of 30 trace samples.
Bottom row: the ~60-node SGM hybrid each seed set produces -- the budget where seeds dominate the map.
Annotated with the offline median localisation error (DINOv2 patch-mean) from localisation_report.py.
Run from the icra-2027 folder.
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
# signal -> (label, colour, seed map, ~60-node SGM hybrid, offline median err under DINOv2 patch-mean)
SIGNALS = [("entropy", "Policy entropy", "#c0398b", "nosmooth_high_thin4", "hybrid_sgm_g44_entropy_thin4", 17.3),
           ("epistemic", "Epistemic", "#2a78d6", "epistemic_thin4_n30", "hybrid_sgm_g44_epistemic_thin4", 10.5),
           ("aleatoric", "Aleatoric", "#eb6834", "aleatoric_thin4_n30", "hybrid_sgm_g44_aleatoric_thin4", 7.2)]

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8})


def nodes_of(name):
    d = json.load(open(f"{MAPS}/town02_speed10_s7_{name}_driving_map.json"))["nodes"]
    return np.array([[n["x_m"], n["y_m"]] for n in d]), np.array([n.get("kind", "seed") for n in d])


def main():
    from render_placement_maps import lane_polylines
    tr = pickle.load(open(TRACE, "rb")); P = np.asarray(tr["positions"], float)[:, :2]
    lanes = [np.asarray(l) for l in lane_polylines("Town02") if len(l)]
    L = np.concatenate(lanes); pad = 15
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.6), constrained_layout=True)
    for j, (_key, label, colour, seedmap, hybrid, err) in enumerate(SIGNALS):
        for i, name in enumerate((seedmap, hybrid)):
            ax = axes[i, j]
            for l in lanes:
                ax.plot(l[:, 0], l[:, 1], color="#e2e2e2", lw=1.6, solid_capstyle="round", zorder=1)
            ax.plot(P[:, 0], P[:, 1], color="#bcbcbc", lw=0.6, zorder=2)
            xy, kind = nodes_of(name)
            if i == 0:
                ax.scatter(*xy.T, s=40, color=colour, marker="^", lw=0, zorder=4)
                ax.set_title(f"{label} — {len(xy)} seeds", fontsize=9.5, color=colour)
            else:
                fill = xy[kind != "entropy"]; seed = xy[kind == "entropy"]
                ax.scatter(*fill.T, s=11, color="#4a3aa7", lw=0, zorder=3)
                ax.scatter(*seed.T, s=34, color=colour, marker="^", lw=0, zorder=4)
                ax.set_title(f"+ SGM — {len(xy)} nodes ({len(seed)} seeds)", fontsize=9)
                ax.text(0.5, -0.03, f"Localisation error {err:.1f} m (median)", transform=ax.transAxes,
                        fontsize=7.5, ha="center", va="top", color="#444444")
            ax.set_xlim(L[:, 0].min() - pad, L[:, 0].max() + pad)
            ax.set_ylim(L[:, 1].max() + pad, L[:, 1].min() - pad)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#cccccc")
    h = [plt.Line2D([], [], ls="", marker="^", ms=7, color=c) for _k, _l, c, _s, _hy, _e in SIGNALS]
    h.append(plt.Line2D([], [], ls="", marker="o", ms=5, color="#4a3aa7"))
    fig.legend(h, [l for _k, l, _c, _s, _hy, _e in SIGNALS] + ["SGM node"], loc="outside lower center",
               ncol=4, frameon=False, fontsize=8.5, handletextpad=0.3)
    fig.suptitle("Town02: three uncertainty signals, count-matched at 30 seeds (0 of 30 shared)", fontsize=10)
    for e in ("pdf", "png"):
        fig.savefig(f"figures/signal_seeds_town02.{e}", **({"dpi": 200} if e == "png" else {}))
    print("-> figures/signal_seeds_town02.{pdf,png}")


if __name__ == "__main__":
    main()
