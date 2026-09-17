"""Entropy vs. time along individual traces, with junctions and failures marked.

    python figure_traces.py

The aligned/peri-event figure shows the AVERAGE transient. It cannot show what a
single drive looks like, and it hides the thing the precision analysis turned up:
entropy spikes at junctions AND in the moments before the policy fails. Both are
visible in a raw trace and neither is visible in an average.

WHAT IS DRAWN, and why each element is there rather than decorative:

  shaded orange   in-junction spans, from the env's own flag, drawn as intervals
                  because a junction IS an interval in time
  markers         the nodes MAD actually placed -- so the reader can see the
                  detector's decisions rather than take the coverage number on
                  trust
  grey band       the first 25 steps, EXCLUDED from detection. Every episode
                  opens with a reset transient of ~1.8 nats decaying to ~0.8;
                  before this exclusion 39% of nodes landed there, at spawn
                  points. Shading it makes the exclusion auditable instead of a
                  silent parameter.
  red line        the episode's terminal event, if it ended in failure rather
                  than the step cap. 11 of the 17 nodes that are NOT at a
                  junction sit within 10 m of one of these.

Episodes are chosen to span the range, not cherry-picked for the effect: the two
with the most junction time, one that ended in failure, and one that ran to the
cap without incident.
"""
import argparse
import json

import numpy as np

import node_placement_entropy as NP

LINE = "#0072B2"
JUNC = "#E69F00"
NODE = "#009E73"
FAIL = "#D55E00"
INK, INK2, MUTED, GRID = "#000000", "#3d3d3d", "#8a8a8a", "#dcdcdc"
WARM, SEP, Z = 25, 8.0, 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/entropy_nogoal.json")
    ap.add_argument("--out", default="results/fig_traces")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    eps = [e for e in json.load(open(args.inp))["episodes"]
           if len(e["steps"]) >= WARM + 60]
    n_junc = lambda e: sum(s["in_junction"] for s in e["steps"])
    failed = [e for e in eps if e["reasons"]]
    clean = [e for e in eps if not e["reasons"]]
    pick, seen = [], set()
    for cand in (sorted(eps, key=lambda e: -n_junc(e))[:2]
                 + sorted(failed, key=lambda e: -n_junc(e))[:1]
                 + sorted(clean, key=lambda e: -len(e["steps"]))[:1]):
        if cand["episode"] not in seen:
            pick.append(cand); seen.add(cand["episode"])
    pick = pick[:args.n]

    fig, axes = plt.subplots(len(pick), 1, figsize=(7.16, 1.35 * len(pick)),
                             sharex=False)
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, pick):
        st = e["steps"]
        h = np.array([s["entropy"] for s in st], float)
        inj = np.array([s["in_junction"] for s in st], bool)
        xy = np.array([[s["x"], s["y"]] for s in st], float)
        s_arc = NP.arclength(xy)
        t = np.arange(len(h))

        ax.axvspan(0, WARM, color=MUTED, alpha=0.20, linewidth=0, zorder=0)
        d = np.diff(np.concatenate([[0], inj.view(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            ax.axvspan(a, min(b, len(t) - 1), color=JUNC, alpha=0.30,
                       linewidth=0, zorder=1)
        ax.plot(t, h, color=LINE, linewidth=0.9, zorder=3)

        idx = [i + WARM for i in NP.mad_spikes(h[WARM:], s_arc[WARM:], SEP, 9, Z)]
        if idx:
            ax.plot(idx, h[idx], "o", color=NODE, markersize=4.2, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.7)
        if e["reasons"]:
            ax.axvline(len(t) - 1, color=FAIL, linewidth=1.1, zorder=5)
            # Verbatim: these are env values (lane_any, off_road), not prose.
            # Title-casing them would display a string the code never emits.
            ax.text(len(t) - 1, ax.get_ylim()[1],
                    " Ended: " + ", ".join(e["reasons"]),
                    fontsize=6.2, color=FAIL, va="top", ha="left")

        ax.set_ylabel("Entropy\n(nats)")
        ax.set_xlim(0, len(t) - 1)
        ax.text(0.996, 0.93, f"Episode {e['episode']}  ·  {int(inj.sum())} junction steps"
                             f"  ·  {len(idx)} nodes  ·  {s_arc[-1]:.0f} m",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.4,
                color=INK2)
        ax.grid(True, color=GRID, linewidth=0.45)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=INK2)
    axes[-1].set_xlabel("Step")

    axes[0].legend(handles=[
        Patch(facecolor=JUNC, alpha=0.30, label="In junction"),
        Patch(facecolor=MUTED, alpha=0.20, label="Excluded (reset transient)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=NODE,
               markersize=5, label="Node placed (MAD $z\\geq3$)"),
        Line2D([0], [0], color=FAIL, lw=1.1, label="Episode failed")],
        frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.02),
        handlelength=1.3, columnspacing=1.1)

    fig.tight_layout(pad=0.3, h_pad=0.55)
    for ext, kw in ((".png", {"dpi": 600}), (".pdf", {})):
        fig.savefig(args.out + ext, bbox_inches="tight", facecolor="white", **kw)
        print(f"[Fig] -> {args.out}{ext}")
    for e in pick:
        st = e["steps"]
        print(f"[Fig] episode {e['episode']:>2}  {len(st):>3} steps  "
              f"{sum(s['in_junction'] for s in st):>3} junction  "
              f"reasons={e['reasons']}")


if __name__ == "__main__":
    main()
