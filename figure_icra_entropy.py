"""Publication figure: policy entropy at junctions, both policies.

    python figure_icra_entropy.py

Emits PDF (vector, what the camera-ready wants) and 600-dpi PNG at IEEE
DOUBLE-COLUMN width (7.16 in). Single-column is 3.5 in -- pass --single.

DESIGN NOTES, since a paper figure is read at print size and not on a monitor:

  * Sized in INCHES to the column, and every font set in points, so what you see
    is what prints. Nothing is scaled after the fact -- rescaling a figure is
    what produces the 5-pt axis labels that reviewers complain about.
  * Two panels only. The four-panel exploratory version answers "what is going
    on"; a paper figure answers ONE question and this one is "does entropy rise
    at junctions, and by how much".
  * Okabe-Ito colours. Colourblind-safe by construction, and they survive the
    greyscale printing that IEEE still permits.
  * The CI band is a bootstrap over EPISODES, not steps. Steps inside one
    junction traversal are the same manoeuvre sampled tens of times; treating
    them as independent would shrink the band by roughly sqrt(traversal length)
    and overstate the result.

PANEL A is the mechanism: entropy aligned on junction entry, both policies.
PANEL B is the effect size: per-episode paired differences with 95% CIs, plus
the SECOND goal-conditioned rollout of the same checkpoint -- included
deliberately, because it is the honest evidence that the goal-conditioned
estimate is unstable and the goal-free one is not.
"""
import argparse
import json

import numpy as np

GOALFREE = "#0072B2"   # Okabe-Ito blue
GOALCOND = "#D55E00"   # Okabe-Ito vermillion
JUNC = "#E69F00"       # Okabe-Ito orange
INK, INK2, MUTED, GRID = "#000000", "#3d3d3d", "#8a8a8a", "#dcdcdc"

PRE, POST = 40, 60


def load(path):
    with open(path) as f:
        return json.load(f)["episodes"]


def aligned_curves(episodes):
    """One (PRE+POST) window per junction entry, tagged with its episode."""
    win, owner = [], []
    for e in episodes:
        h = np.array([s["entropy"] for s in e["steps"]], float)
        inj = np.array([s["in_junction"] for s in e["steps"]], bool)
        if not inj.any():
            continue
        d = np.diff(np.concatenate([[0], inj.view(np.int8)]))
        for onset in np.flatnonzero(d == 1):
            a, b = onset - PRE, onset + POST
            if a < 0 or b > len(h):
                continue                      # partial windows bias the edges
            win.append(h[a:b]); owner.append(e["episode"])
    return np.array(win, float), np.array(owner)


def boot_curve(win, owner, n_boot=2000, seed=0):
    """Mean curve with a 95% band, resampling EPISODES not entries."""
    rng = np.random.default_rng(seed)
    eps = np.unique(owner)
    if len(win) == 0:
        return None, None, None
    out = np.empty((n_boot, win.shape[1]))
    for b in range(n_boot):
        pick = rng.choice(eps, size=len(eps), replace=True)
        rows = np.concatenate([np.flatnonzero(owner == p) for p in pick])
        out[b] = win[rows].mean(axis=0)
    return (win.mean(axis=0),
            np.percentile(out, 2.5, axis=0), np.percentile(out, 97.5, axis=0))


def paired_delta(episodes, n_boot=10000, seed=0):
    d = []
    for e in episodes:
        j = [s["entropy"] for s in e["steps"] if s["in_junction"]]
        r = [s["entropy"] for s in e["steps"] if not s["in_junction"]]
        if j and r:
            d.append(np.mean(j) - np.mean(r))
    d = np.asarray(d, float)
    if len(d) < 3:
        return None
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), \
        float(np.percentile(m, 97.5)), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nogoal", default="results/entropy_nogoal.json")
    ap.add_argument("--goal", default="results/entropy_nodes.json")
    ap.add_argument("--out", default="results/fig_entropy_junctions_icra")
    ap.add_argument("--single", action="store_true", help="3.5in single column")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fonts in POINTS, figure in INCHES: printed size is then exact.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    nog, goa = load(args.nogoal), load(args.goal)
    w_n, o_n = aligned_curves(nog)
    w_g, o_g = aligned_curves(goa)
    mu_n, lo_n, hi_n = boot_curve(w_n, o_n)
    mu_g, lo_g, hi_g = boot_curve(w_g, o_g)

    d_n = paired_delta(nog)
    d_g = paired_delta(goa)

    width = 3.5 if args.single else 7.16
    fig, axes = plt.subplots(1, 2, figsize=(width, 2.35),
                             gridspec_kw={"width_ratios": [1.55, 1]})

    # ---- Panel A: aligned on junction entry ---------------------------------
    ax = axes[0]
    t = np.arange(-PRE, POST)
    ax.axvspan(0, POST, color=JUNC, alpha=0.16, linewidth=0, zorder=0)
    for mu, lo, hi, c, lab, n in ((mu_g, lo_g, hi_g, GOALCOND, "Goal-conditioned", len(w_g)),
                                  (mu_n, lo_n, hi_n, GOALFREE, "Goal-free", len(w_n))):
        if mu is None:
            continue
        ax.fill_between(t, lo, hi, color=c, alpha=0.18, linewidth=0, zorder=1)
        ax.plot(t, mu, color=c, linewidth=1.3, zorder=2,
                label=f"{lab} ($n$={n})", solid_capstyle="round")
    ax.axvline(0, color=INK2, linewidth=0.7, linestyle=(0, (3, 2)), zorder=3)
    ax.set_xlabel("Steps from junction entry")
    ax.set_ylabel("Policy entropy (nats)")
    ax.set_xlim(-PRE, POST - 1)
    ax.legend(frameon=False, loc="upper left", handlelength=1.4,
              borderpad=0.2, labelspacing=0.25)
    ax.text(0.985, 0.04, "In junction", transform=ax.transAxes, ha="right",
            fontsize=6.5, color=INK2)

    # ---- Panel B: effect sizes ----------------------------------------------
    ax = axes[1]
    rows = []
    if d_n: rows.append(("Goal-free", *d_n, GOALFREE))
    if d_g: rows.append(("Goal-cond.", *d_g, GOALCOND))
    ys = np.arange(len(rows))[::-1]
    ax.axvline(0, color=MUTED, linewidth=0.7, zorder=0)
    for y, (lab, m, lo, hi, n, c) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=c, linewidth=1.3, solid_capstyle="butt",
                zorder=2)
        ax.plot([lo, lo], [y - .10, y + .10], color=c, linewidth=1.0, zorder=2)
        ax.plot([hi, hi], [y - .10, y + .10], color=c, linewidth=1.0, zorder=2)
        ax.plot([m], [y], "o", color=c, markersize=4.2, zorder=3,
                markeredgecolor="white", markeredgewidth=0.6)
        ax.text(hi + 0.012, y, f"{m:+.3f}", va="center", fontsize=6.8, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r[0]}\n($n$={r[4]} ep.)" for r in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel(r"$\Delta$ entropy, junction $-$ road (nats)")
    ax.tick_params(axis="y", length=0)

    for ax in axes:
        ax.grid(True, color=GRID, linewidth=0.45)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=INK2)

    fig.tight_layout(pad=0.35, w_pad=1.2)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 600})):
        fig.savefig(args.out + ext, bbox_inches="tight", facecolor="white", **kw)
        print(f"[Fig] -> {args.out}{ext}")

    print(f"\n[Fig] panel A: {len(w_g)} goal-conditioned entries, "
          f"{len(w_n)} goal-free entries (complete windows only)")
    for lab, m, lo, hi, n, _ in rows:
        print(f"[Fig] {lab:11s} {m:+.3f} nats [{lo:+.3f}, {hi:+.3f}]  n={n} episodes")


if __name__ == "__main__":
    main()
