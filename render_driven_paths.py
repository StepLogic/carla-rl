"""What the localiser swap looks like on the road (user 2026-09-20 "plot dinov2 driven path").

One panel per episode, both localisers on the SAME map (Entropy + SGM, 59 nodes -- the sparsest, where
MobileNet aliases worst) and the SAME episode. Drawn: the map's nodes, the true start and goal, the node
each localiser picked for the start, and the path actually driven, ending at a cross where the episode
ended. The two paths diverge from step one whenever the picked start nodes differ, because the first leg
is driven toward the picked node, not the true start.
Run from the icra-2027 folder.
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))
TRAILS = "results/drive_plan_c/trails"
SEQ = "results/maps/sequence"
LOC = [("mobilenet_trace", "MobileNet", "#6b6b6b"), ("dinov2_bank_patchmean", "DINOv2", "#c0398b")]

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8})


def main():
    from render_placement_maps import lane_polylines
    eps = sorted({os.path.basename(f).split("_short")[1][:2]
                  for f in glob.glob(f"{TRAILS}/*_short*_mobilenet_trace.json")})
    mapname = "hybrid_sgm_g44_entropy_thin4"
    nodes = json.load(open(f"{SEQ}/{mapname}_sequence.json"))["nodes"]
    NXY = np.array([[n["x_m"], n["y_m"]] for n in nodes])
    lanes = [np.asarray(l) for l in lane_polylines("Town02") if len(l)]
    L = np.concatenate(lanes); pad = 12

    fig, axes = plt.subplots(1, len(eps), figsize=(3.4 * len(eps), 3.9), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, eps):
        for l in lanes:
            ax.plot(l[:, 0], l[:, 1], color="#e4e4e4", lw=1.6, solid_capstyle="round", zorder=1)
        ax.scatter(*NXY.T, s=9, color="#4a3aa7", lw=0, zorder=2, alpha=0.75)
        title = []
        for key, label, colour in LOC:
            f = f"{TRAILS}/{mapname}_short{e}_{key}.json"
            if not os.path.exists(f):
                continue
            d = json.load(open(f))
            t = np.asarray(d["trail"], float)
            ax.plot(t[:, 0], t[:, 1], color=colour, lw=2.0, zorder=4, alpha=0.9)
            ax.plot([t[-1, 0]], [t[-1, 1]], marker="x", ms=7, mew=2, color=colour, zorder=5)
            ax.plot([t[0, 0]], [t[0, 1]], marker=".", ms=6, color=colour, zorder=5)
            sn = NXY[d["start_node"]]
            ax.plot([sn[0]], [sn[1]], marker="o", ms=9, mfc="none", mec=colour, mew=1.6, zorder=5)
            err = float(np.linalg.norm(sn - np.asarray(d["start_xy"], float)))
            title.append(f"{label}: picked a node {err:.0f} m from the true start, {d['end']}")
            s, g = np.asarray(d["start_xy"], float), np.asarray(d["goal_xy"], float)
        ax.plot([s[0]], [s[1]], marker="*", ms=13, color="#1baf7a", zorder=6)
        ax.plot([g[0]], [g[1]], marker="*", ms=13, color="#eb6834", zorder=6)
        ax.set_title(f"Episode short#{e}\n" + "\n".join(title), fontsize=8)
        ax.set_xlim(L[:, 0].min() - pad, L[:, 0].max() + pad)
        ax.set_ylim(L[:, 1].max() + pad, L[:, 1].min() - pad)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
    h = ([Line2D([], [], color=c, lw=2.2) for _k, _l, c in LOC]
         + [Line2D([], [], ls="", marker="o", mfc="none", mec="#444444", mew=1.6, ms=9),
            Line2D([], [], ls="", marker="*", ms=12, color="#1baf7a"),
            Line2D([], [], ls="", marker="*", ms=12, color="#eb6834"),
            Line2D([], [], ls="", marker="o", ms=4, color="#4a3aa7")])
    fig.legend(h, [l for _k, l, _c in LOC] + ["Picked start node", "True start", "Goal", "Map node"],
               loc="outside lower center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle("Entropy + SGM, 59 nodes: the same episodes driven under each localiser", fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/driven_paths_town02.{ext}", **({"dpi": 200} if ext == "png" else {}))
    print("-> figures/driven_paths_town02.{pdf,png}")


if __name__ == "__main__":
    main()
