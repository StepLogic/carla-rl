"""Publication figure, single column: sigma(steer) as a spatial field over Town02.

    python figure_sigma_field_town02.py

The Habitat side bins entropy by 0.5 m cell; this is the CARLA analogue, built
from the SAME stored goal-free traces as the time-course figures (NO new
rollouts): data/entropy_nogoal_500cap.json, 60 episodes, 24,449 steps, each
carrying (x, y, sigma, in_junction).

Form: magnitude over space -> a sequential single-hue ramp (light to dark
blue), one mean sigma(steer) value per occupied 0.5 m cell. Junction sites are
NOT re-derived here: the 23 sites come from the pack's own map
(data/map_nodes_500cap.json, in_junction events merged at 6 m), marked as open
circles -- same convention and same numbers as map_nodes.png. Uncoloured town
area is simply road the policy never drove; the field is coverage-honest by
construction.
"""
import argparse
import json

import numpy as np

STEER_DARK = "#08519c"   # sequential ramp top (single hue: blues)
STEER_LIGHT = "#f7fbff"
JUNC = "#E69F00"
INK, INK2, MUTED = "#000000", "#3d3d3d", "#8a8a8a"
CELL = 4.0               # m. The Habitat analogue bins 0.5 m, but over a ~10 m
                         # apartment; Town02 is ~400 x 330 m and 0.5 m cells
                         # are sub-pixel at column width. Stated in the caption.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="evidence/2026-09-09_entropy_at_junctions/"
                                     "data/entropy_nogoal_500cap.json")
    ap.add_argument("--out", default="results/fig_sigma_field_town02")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    episodes = json.load(open(args.data))["episodes"]
    xs = np.array([s["x"] for e in episodes for s in e["steps"]])
    ys = np.array([s["y"] for e in episodes for s in e["steps"]])
    sg = np.array([s["sigma"][0] for e in episodes for s in e["steps"]])
    inj = np.array([s["in_junction"] for e in episodes for s in e["steps"]])

    # 0.5 m cells, mean sigma(steer) per occupied cell. Coordinates can be
    # slightly negative, so bin indices are shifted positive before packing.
    gx, gy = np.floor(xs / CELL).astype(int), np.floor(ys / CELL).astype(int)
    gxo, gyo = gx - gx.min(), gy - gy.min()
    width = int(gyo.max()) + 1
    key = gxo * width + gyo
    order = np.argsort(key)
    k_s, s_s = key[order], sg[order]
    bounds = np.flatnonzero(np.diff(k_s)) + 1
    cells = np.split(s_s, bounds)
    uk = k_s[np.r_[0, bounds]]
    mean_sig = np.array([c.mean() for c in cells])
    n_obs = np.array([len(c) for c in cells])
    cx = (uk // width + gx.min()) * CELL + CELL / 2
    cy = (uk % width + gy.min()) * CELL + CELL / 2

    jmap = json.load(open("evidence/2026-09-09_entropy_at_junctions/"
                          "data/map_nodes_500cap.json"))
    sites = np.asarray(jmap["junctions"], float)
    print(f"[Fig] {len(uk)} occupied cells, {len(sites)} junction sites "
          f"(pack merge {jmap.get('merge_m')} m), "
          f"sigma range {mean_sig.min():.3f}-{mean_sig.max():.3f}")

    cmap = LinearSegmentedColormap.from_list("sigma_blue",
                                             [STEER_LIGHT, STEER_DARK])
    # The ramp starts at the TYPICAL ROAD value, not at the minimum: most cells
    # sit at the baseline, so a full-range ramp would render the entire town
    # pale and the junction hotspots would be the only thing visible -- which
    # is the point, but without washing out whether the rest is low or just
    # baseline. Cells below the anchor read as "baseline".
    vmin = np.percentile(mean_sig, 30)
    vmax = np.percentile(mean_sig, 99)

    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    rects = [Rectangle((x - CELL / 2, y - CELL / 2), CELL, CELL)
             for x, y in zip(cx, cy)]
    pc = PatchCollection(rects, cmap=cmap, linewidths=0)
    pc.set_array(mean_sig); pc.set_clim(vmin, vmax)
    ax.add_collection(pc)

    ax.scatter(sites[:, 0], sites[:, 1], s=26, facecolors="none",
               edgecolors=JUNC, linewidths=1.0, zorder=3,
               label=f"Junction sites ({len(sites)})")

    ax.set_aspect("equal")
    ax.set_xlabel("World x (m)")
    ax.set_ylabel("World y (m)")
    ax.set_title("Sigma(steer) across Town02", fontsize=7.8, color=INK, pad=3)
    # town interior is unoccupied -- keeps the legend off the junction circles
    ax.legend(frameon=False, loc="center", handlelength=1.0,
              borderpad=0.2, labelspacing=0.2, fontsize=6.8)
    ax.tick_params(colors=INK2)
    for sp in ax.spines.values():
        sp.set_color(MUTED)
    # pad so cells at the data edge are not clipped
    ax.set_xlim(cx.min() - 4, cx.max() + 4)
    ax.set_ylim(cy.min() - 4, cy.max() + 4)

    cb = fig.colorbar(pc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(f"Mean sigma(steer) per {CELL:g} m cell", fontsize=6.8)
    cb.ax.tick_params(labelsize=6.5, colors=INK2)
    cb.outline.set_edgecolor(MUTED)

    fig.tight_layout(pad=0.3)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 600})):
        fig.savefig(args.out + ext, bbox_inches="tight", facecolor="white", **kw)
        print(f"[Fig] -> {args.out}{ext}")


if __name__ == "__main__":
    main()