"""Publication figure: where entropy peaks DO beat uniform spacing.

    python figure_icra_placement2.py

Replaces the earlier placement figure, which was built on the goal-conditioned
rollout and reported a null. That null was an artefact of the data, not the
method: the goal-conditioned policy was evaluated at 40-113 m goals, far outside
its 5-20 m training rung, and failed most episodes -- 17 usable episodes at 5.8%
junction occupancy. The goal-free policy gives 40 episodes at 10.8%, and the
answer reverses.

TWO PANELS, ONE CLAIM EACH.

(a) The crossover. Coverage within 5 m as the node budget is swept. Entropy
    peaks dominate when nodes are SCARCE and every criterion converges when they
    are plentiful. This is the shape that reconciles our result with the peer's
    four nulls in Habitat: their regime is 86-331 nodes in a 49 m2 scene, where
    the metric saturates and count dominates; ours is 2-9 nodes over hundreds of
    metres, where placement is the whole game.

(b) That the effect is about the POLICY, not the sample. Paired coverage
    differences with 95% CIs for both policies and both detectors, plus the
    goal-free result subsampled to the goal-conditioned n. If the advantage were
    merely more data, the subsampled bar would collapse toward the
    goal-conditioned one. It does not.

WHY COVERAGE AND NOT MEDIAN DISTANCE. Both are reported in the manifest and they
DISAGREE: median distance still favours uniform, because entropy spikes miss
some junctions entirely while uniform is never far from anything. Coverage asks
"is there a node ON this junction", median asks "how far is the nearest node
from the average junction". For a topological graph the first is the operative
question, but the disagreement is real and is stated in the caption rather than
buried.
"""
import argparse
import json

import numpy as np

import node_placement_entropy as NP

PEAKS = "#0072B2"      # Okabe-Ito blue
UNIFORM = "#E69F00"    # Okabe-Ito orange
RANDOM = "#999999"
INK, INK2, MUTED, GRID = "#000000", "#3d3d3d", "#8a8a8a", "#dcdcdc"
RADII = (5, 10)


def paired_cov(eps, det, z, R=5, seed=0, subsample=None, rng=None):
    per, _ = NP.analyse(eps, 4.0, 8.0, 9, RADII, seed=seed, detector=det, z_thresh=z)
    if subsample and len(per) > subsample:
        per = [per[i] for i in rng.choice(len(per), subsample, replace=False)]
    a = [r["arms"]["peaks"][f"within_{R}m"] for r in per]
    b = [r["arms"]["uniform"][f"within_{R}m"] for r in per]
    d, (lo, hi) = NP.boot_paired(a, b)
    return d, lo, hi, len(per), float(np.mean([r["budget"] for r in per]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nogoal", default="results/entropy_nogoal.json")
    ap.add_argument("--goal", default="results/entropy_nodes.json")
    ap.add_argument("--out", default="results/fig_node_placement2_icra")
    ap.add_argument("--single", action="store_true")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    free = json.load(open(args.nogoal))["episodes"]
    goal = json.load(open(args.goal))["episodes"]

    sweep = [0.5, 1, 2, 4, 8, 16, 32, 64]
    arms, out, n_eps = NP.pareto(free, sweep, 8.0, 9, RADII)

    width = 3.5 if args.single else 7.16
    fig, axes = plt.subplots(1, 2, figsize=(width, 2.5),
                             gridspec_kw={"width_ratios": [1, 1.15]})

    # ---- (a) coverage vs budget, the crossover ------------------------------
    ax = axes[0]
    labels = {"peaks": "Entropy peaks", "uniform": "Uniform spacing",
              "random": "Random"}
    colors = {"peaks": PEAKS, "uniform": UNIFORM, "random": RANDOM}
    for a in arms:
        ax.plot(out[a]["n"], np.array(out[a]["within_5m"]) * 100,
                color=colors[a], linewidth=1.3, marker="o", markersize=3.0,
                markeredgecolor="white", markeredgewidth=0.5, label=labels[a])
    ax.set_xscale("log")
    ax.set_xlabel("Nodes per episode")
    ax.set_ylabel("Junctions within 5 m (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="lower right", handlelength=1.5,
              borderpad=0.2, labelspacing=0.25)
    ax.text(0.03, 0.95, "Sparse: placement matters", transform=ax.transAxes,
            fontsize=6.3, color=INK2, va="top")

    # ---- (b) effect sizes ---------------------------------------------------
    ax = axes[1]
    rng = np.random.default_rng(0)
    rows = []
    d, lo, hi, n, b = paired_cov(free, "mad", 3.0)
    rows.append((f"Goal-free, MAD\n($n$={n}, {b:.1f} nodes/ep)", d, lo, hi, PEAKS))
    sub = [paired_cov(free, "mad", 3.0, seed=i, subsample=17, rng=rng)[0]
           for i in range(200)]
    los = [paired_cov(free, "mad", 3.0, seed=i, subsample=17, rng=rng)[1]
           for i in range(60)]
    rows.append((f"Goal-free, MAD\nsubsampled to $n$=17", float(np.mean(sub)),
                 float(np.percentile(sub, 2.5)), float(np.percentile(sub, 97.5)),
                 PEAKS))
    d, lo, hi, n, b = paired_cov(free, "topn", 0)
    rows.append((f"Goal-free, top-$N$\n($n$={n}, {b:.1f} nodes/ep)", d, lo, hi, PEAKS))
    d, lo, hi, n, b = paired_cov(goal, "mad", 3.0)
    rows.append((f"Goal-cond., MAD\n($n$={n}, {b:.1f} nodes/ep)", d, lo, hi, GOALCOND_ := "#D55E00"))

    ys = np.arange(len(rows))[::-1]
    ax.axvline(0, color=MUTED, linewidth=0.7, zorder=0)
    for y, (lab, m, lo, hi, c) in zip(ys, rows):
        ax.plot([lo * 100, hi * 100], [y, y], color=c, linewidth=1.3, zorder=2)
        for x in (lo, hi):
            ax.plot([x * 100] * 2, [y - .11, y + .11], color=c, linewidth=1.0, zorder=2)
        ax.plot([m * 100], [y], "o", color=c, markersize=4.2, zorder=3,
                markeredgecolor="white", markeredgewidth=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.4)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.set_xlabel("Coverage within 5 m,\nentropy peaks $-$ uniform (pp)")
    ax.tick_params(axis="y", length=0)

    for ax in axes:
        ax.grid(True, color=GRID, linewidth=0.45)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=INK2)

    fig.tight_layout(pad=0.35, w_pad=1.1)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 600})):
        fig.savefig(args.out + ext, bbox_inches="tight", facecolor="white", **kw)
        print(f"[Fig] -> {args.out}{ext}")
    for lab, m, lo, hi, _ in rows:
        print(f"[Fig] {lab.splitlines()[0]:<28} {m*100:+6.1f} pp [{lo*100:+6.1f},{hi*100:+6.1f}]")


if __name__ == "__main__":
    main()
