#!/usr/bin/env python3
"""Draw the placed maps: one row per town, one column per placer.

Node counts are MATCHED by construction, so the only thing these panels can
differ in is WHERE the nodes went -- which is the entire question. Seglinks are
drawn apart from trace edges because they are respawn jumps, not drivable
connections; the Habitat peer measured plans stalling on exactly those.

usage: plot_maps_v43.py OUT.png MAPS1.json [MAPS2.json ...]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_explore_trace import town_waypoints  # noqa: E402

ORDER = ["mad", "temporal", "uniform"]
TITLE = {"mad": "Online MAD (actor entropy)",
         "temporal": "Temporal control (every Nth step)",
         "uniform": "Uniform control (every s metres)"}
NODE_C = "#2a78d6"        # categorical slot 1
SEGLINK_C = "#eb6834"     # slot 2: the edges that are not drivable


def panel(ax, m, town, kind):
    xy = np.asarray(m["xy"], np.float64)
    bg = town_waypoints(town)
    if bg is not None:
        ax.scatter(bg[:, 0], bg[:, 1], s=0.6, c="#dddddd", linewidths=0,
                   zorder=1, label="Drivable road")
    seg_drawn = trace_drawn = False
    for (a, b), kind_e in zip(m["edges"], m["edge_kinds"]):
        p = xy[[a, b]]
        if kind_e == "seglink":
            ax.plot(p[:, 0], p[:, 1], "-", lw=0.8, color=SEGLINK_C, alpha=0.8,
                    zorder=2, label=(None if seg_drawn else "Seglink (respawn)"))
            seg_drawn = True
        else:
            ax.plot(p[:, 0], p[:, 1], "-", lw=0.7, color="#9ec5f4", alpha=0.8,
                    zorder=2, label=(None if trace_drawn else "Trace edge"))
            trace_drawn = True
    ax.scatter(xy[:, 0], xy[:, 1], s=9, c=NODE_C, linewidths=0, zorder=3,
               label="Node")
    ax.set_title(f"{TITLE[kind]}\n{town}: {m['n_nodes']} nodes, "
                 f"{m['n_edges']} edges, {m['n_seglinks']} seglinks",
                 fontsize=9)
    ax.set_xlabel("Map X (m)")
    ax.set_ylabel("Map Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()                 # CARLA's Y grows southward
    ax.legend(loc="upper right", fontsize=6.5, markerscale=1.6, framealpha=0.9)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out, paths = sys.argv[1], sys.argv[2:]
    res = [json.load(open(p)) for p in paths]
    fig, axes = plt.subplots(len(res), len(ORDER), squeeze=False,
                             figsize=(6.4 * len(ORDER), 6.2 * len(res)))
    for row, r in zip(axes, res):
        for ax, kind in zip(row, ORDER):
            panel(ax, r["maps"][kind], r["town"], kind)
    st = res[0]["mad_stats"]
    fig.suptitle(
        "Node placement from explore-mode actor entropy, with matched controls\n"
        f"Online MAD: {st['window_m']:.0f} m window "
        f"({st['window_samples']} samples), warmup {st['warmup_samples']}, "
        f"z={st['z']}, dedupe {st['dedupe_m']} m   "
        "Counts matched across placers", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"-> {out}")

    for r in res:
        print(f"\n{r['town']}")
        for kind in ORDER:
            m = r["maps"][kind]
            print(f"  {kind:9s} {m['n_nodes']:4d} nodes  "
                  f"{m['n_seglinks']:3d} seglinks  median nearest gap "
                  f"{m['median_node_gap_m']:6.2f} m")


if __name__ == "__main__":
    main()
