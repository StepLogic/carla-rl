"""Publication figure: entropy is not a useful node-placement criterion.

    python figure_icra_placement.py

Companion to fig_entropy_junctions_icra. That figure establishes that entropy
DOES rise at junctions; this one establishes that the rise is nonetheless
useless for deciding where to put topological-graph nodes.

WHY THE FRONTIER IS THE HONEST PLOT AND A SINGLE BUDGET IS NOT. Comparing
criteria at one node count is a slice through the variable that actually drives
the metric. Two criteria can tie at 5 nodes and separate at 30. Sweeping the
budget and plotting the whole curve asks the real question -- at every budget,
does one criterion buy more coverage per node -- and it exposes the finding that
a single point would hide: above ~20 nodes/episode every criterion, including
RANDOM, is equivalent.

The separation floor shrinks with the budget (min(8 m, half the uniform spacing
at that budget)). A fixed floor would cap the peaks arm at ~path/8 nodes while
uniform kept subdividing, so peaks would appear to plateau for a reason that is
the constraint rather than the criterion.
"""
import argparse
import json

import numpy as np

import node_placement_entropy as NP

PEAKS = "#0072B2"      # Okabe-Ito blue
UNIFORM = "#E69F00"    # Okabe-Ito orange
RANDOM = "#999999"
INK, INK2, MUTED, GRID = "#000000", "#3d3d3d", "#8a8a8a", "#dcdcdc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/entropy_nodes.json")
    ap.add_argument("--out", default="results/fig_node_placement_icra")
    ap.add_argument("--single", action="store_true")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    with open(args.inp) as f:
        eps = json.load(f)["episodes"]
    radii = (5, 10)
    sweep = [0.5, 1, 2, 4, 8, 16, 32, 64]
    arms, out, n_eps = NP.pareto(eps, sweep, 8.0, 9, radii)
    per_ep, _ = NP.analyse(eps, 4.0, 8.0, 9, radii)
    d, (lo, hi) = NP.boot_paired(
        [r["arms"]["uniform"]["median"] for r in per_ep],
        [r["arms"]["peaks"]["median"] for r in per_ep])

    colors = {"peaks": PEAKS, "uniform": UNIFORM, "random": RANDOM}
    labels = {"peaks": "Entropy peaks", "uniform": "Uniform spacing",
              "random": "Random"}
    width = 3.5 if args.single else 7.16
    fig, axes = plt.subplots(1, 2, figsize=(width, 2.35))

    # ---- Panel A: distance-vs-budget frontier -------------------------------
    ax = axes[0]
    for a in arms:
        ax.plot(out[a]["n"], out[a]["median"], color=colors[a], linewidth=1.3,
                marker="o", markersize=3.0, markeredgecolor="white",
                markeredgewidth=0.5, label=labels[a])
    ax.set_xscale("log")
    ax.set_xlabel("Nodes per episode")
    ax.set_ylabel("Median junction$\\rightarrow$node dist. (m)")
    ax.legend(frameon=False, handlelength=1.5, borderpad=0.2,
              labelspacing=0.25)

    # ---- Panel B: coverage-vs-budget ----------------------------------------
    ax = axes[1]
    R = radii[-1]
    for a in arms:
        ax.plot(out[a]["n"], np.array(out[a][f"within_{R}m"]) * 100,
                color=colors[a], linewidth=1.3, marker="o", markersize=3.0,
                markeredgecolor="white", markeredgewidth=0.5, label=labels[a])
    ax.set_xscale("log")
    ax.set_xlabel("Nodes per episode")
    ax.set_ylabel(f"Junctions within {R} m (%)")
    ax.set_ylim(0, 105)
    # The equal-budget comparison, annotated where it actually sits on the curve.
    ax.annotate(f"equal budget:\nuniform $-$ peaks\n$=${d:+.2f} m [{lo:+.2f}, {hi:+.2f}]",
                xy=(4.7, 70.6), xytext=(1.05, 88), fontsize=6.4, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.6,
                                shrinkA=0, shrinkB=2))

    for ax in axes:
        ax.grid(True, color=GRID, linewidth=0.45)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=INK2)

    fig.tight_layout(pad=0.35, w_pad=1.4)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 600})):
        fig.savefig(args.out + ext, bbox_inches="tight", facecolor="white", **kw)
        print(f"[Fig] -> {args.out}{ext}")
    print(f"[Fig] {n_eps} episodes; equal-budget uniform - peaks "
          f"= {d:+.2f} m [{lo:+.2f}, {hi:+.2f}]")


if __name__ == "__main__":
    main()
