"""The explore trace every Town02 map is built from (user 2026-09-19 "show trace").

Left: the trace coloured by collection segment -- collection restarts at the nearest waypoint after each
crash, so the 31 segments are one continuous sweep, not 31 random respawns.
Right: the same trace under the 40 fixed evaluation episodes' start and goal poses, which is where map
coverage turns into localisation error.
Annotated with lane coverage: the share of Town02 driving lane within 10 m of some trace sample -- the
ceiling on what ANY map built from this trace can localise against.
Run from the icra-2027 folder.
"""
import os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))
TRACE = "results/placement_v6n_town02_budget10k_laneoff_speed10/trace_s7.pkl"
EPISODES = "results/episodes/town02_speed10_s7_episodes_short20_long20_seed0.pkl"

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9})


def main():
    from render_placement_maps import lane_polylines
    tr = pickle.load(open(TRACE, "rb"))
    P = np.asarray(tr["positions"], float)[:, :2]; seg = np.asarray(tr["segments"])
    ep = pickle.load(open(EPISODES, "rb"))
    lanes = [np.asarray(l) for l in lane_polylines("Town02") if len(l)]
    L = np.concatenate(lanes); pad = 15

    # lane coverage: resample every lane polyline at ~2 m and ask how much is near the trace
    pts = []
    for l in lanes:
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(l, axis=0), axis=1))]
        if d[-1] < 1e-6:
            continue
        s = np.arange(0, d[-1], 2.0)
        pts.append(np.c_[np.interp(s, d, l[:, 0]), np.interp(s, d, l[:, 1])])
    Q = np.concatenate(pts)
    near = np.array([np.linalg.norm(P - q, axis=1).min() for q in Q])
    cov = 100 * np.mean(near < 10)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.9), constrained_layout=True)
    for ax in axes:
        for l in lanes:
            ax.plot(l[:, 0], l[:, 1], color="#dedede", lw=1.8, solid_capstyle="round", zorder=1)
        ax.set_xlim(L[:, 0].min() - pad, L[:, 0].max() + pad)
        ax.set_ylim(L[:, 1].max() + pad, L[:, 1].min() - pad)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#cccccc")

    u = np.unique(seg)
    cmap = plt.get_cmap("viridis")
    for k, s in enumerate(u):
        m = seg == s
        axes[0].plot(P[m, 0], P[m, 1], color=cmap(k / max(len(u) - 1, 1)), lw=1.6, zorder=3)
    axes[0].set_title(f"Explore trace — {len(P)} samples, {len(u)} segments, 2410 m", fontsize=10)
    axes[0].text(0.02, 0.02, f"Lane within 10 m of the trace: {cov:.0f}%", transform=axes[0].transAxes,
                 fontsize=8, color="#444444")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, len(u)))
    cb = fig.colorbar(sm, ax=axes[0], fraction=0.046, pad=0.02)
    cb.set_label("Collection segment", fontsize=8); cb.ax.tick_params(labelsize=7)

    axes[1].plot(P[:, 0], P[:, 1], color="#b0b0b0", lw=1.2, zorder=2, label="Trace")
    for name, colour, mk in (("start", "#2a78d6", "o"), ("goal", "#c0398b", "*")):
        xy = np.array([e[f"{name}_xy"] for s in ("short", "long") for e in ep[s]["episodes"]])
        axes[1].scatter(xy[:, 0], xy[:, 1], s=38 if name == "start" else 70, color=colour, marker=mk,
                        lw=0, zorder=4, label=f"Episode {name} ({len(xy)})")
    axes[1].set_title("The 40 evaluation episodes on that trace", fontsize=10)
    axes[1].legend(loc="lower left", fontsize=7.5, frameon=False, handletextpad=0.3)
    for e in ("pdf", "png"):
        fig.savefig(f"figures/trace_town02.{e}", **({"dpi": 200} if e == "png" else {}))
    print(f"-> figures/trace_town02.{{pdf,png}}  (lane coverage within 10 m: {cov:.1f}%)")


if __name__ == "__main__":
    main()
