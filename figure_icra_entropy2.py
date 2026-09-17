"""Publication figure: policy entropy at junctions, with the mechanism.

    python figure_icra_entropy2.py

Replaces figure_icra_entropy.py. Three changes, each fixing something the first
version got wrong or hid:

1. PANEL (a) IS BASELINE-SUBTRACTED. The two policies sit at different absolute
   entropies (~0.85 goal-conditioned, ~0.75 goal-free), so plotting raw nats
   forced the reader to subtract by eye to compare transients. Subtracting each
   curve's own pre-entry baseline puts both on one axis where the quantity of
   interest -- the SIZE of the rise -- is read directly.

2. PANEL (b) IS NEW, AND IT IS THE MECHANISM. Entropy is a sum over action
   dimensions, so a rise in H cannot say which control became uncertain. Read in
   ABSOLUTE sigma this looks like nothing (steer 0.17 -> 0.19 against throttle
   0.73 -> 0.73). Read as RELATIVE change it is unambiguous: steering widens
   ~13% at junctions in BOTH policies with the CI clear of zero, while throttle
   does not widen at all. Absolute sigma hid this because throttle's scale is
   four times steering's. That is the "several exits are viable" hypothesis,
   and it is the panel a reviewer will want.

3. The n for each arm is on the figure, not in the caption. The goal-conditioned
   arm has 13 junction entries against 62; its band is wide and should look wide.
"""
import argparse
import json

import numpy as np

GOALFREE = "#0072B2"   # Okabe-Ito blue
GOALCOND = "#D55E00"   # Okabe-Ito vermillion
JUNC = "#E69F00"
INK, INK2, MUTED, GRID = "#000000", "#3d3d3d", "#8a8a8a", "#dcdcdc"
PRE, POST, BASE_HI = 40, 60, -10


def windows(episodes):
    win, owner = [], []
    for e in episodes:
        h = np.array([s["entropy"] for s in e["steps"]], float)
        inj = np.array([s["in_junction"] for s in e["steps"]], bool)
        if not inj.any():
            continue
        d = np.diff(np.concatenate([[0], inj.view(np.int8)]))
        for on in np.flatnonzero(d == 1):
            a, b = on - PRE, on + POST
            if a < 0 or b > len(h):
                continue
            win.append(h[a:b]); owner.append(e["episode"])
    return np.array(win, float), np.array(owner)


def boot_band(win, owner, n_boot=2000, seed=0):
    """Baseline-subtracted mean curve, 95% band resampling EPISODES."""
    if len(win) == 0:
        return None, None, None
    base = win[:, :PRE + BASE_HI].mean(axis=1, keepdims=True)
    w = win - base
    rng = np.random.default_rng(seed)
    eps = np.unique(owner)
    out = np.empty((n_boot, w.shape[1]))
    for b in range(n_boot):
        pick = rng.choice(eps, len(eps), replace=True)
        out[b] = w[np.concatenate([np.flatnonzero(owner == p) for p in pick])].mean(axis=0)
    return w.mean(axis=0), np.percentile(out, 2.5, axis=0), np.percentile(out, 97.5, axis=0)


def paired(episodes, key):
    """Per-episode (junction - road); key='entropy' or a sigma index."""
    d = []
    for e in episodes:
        if key == "entropy":
            j = [s["entropy"] for s in e["steps"] if s["in_junction"]]
            r = [s["entropy"] for s in e["steps"] if not s["in_junction"]]
            if j and r:
                d.append(np.mean(j) - np.mean(r))
        else:
            j = [s["sigma"][key] for s in e["steps"] if s["in_junction"]]
            r = [s["sigma"][key] for s in e["steps"] if not s["in_junction"]]
            if j and r:
                d.append((np.mean(j) - np.mean(r)) / np.mean(r) * 100)
    d = np.asarray(d, float)
    if len(d) < 3:
        return None
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (10000, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nogoal", default="results/entropy_nogoal.json")
    ap.add_argument("--goal", default="results/entropy_nodes.json")
    ap.add_argument("--out", default="results/fig_entropy_junctions2_icra")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    free = json.load(open(args.nogoal))["episodes"]
    goal = json.load(open(args.goal))["episodes"]
    wf, of = windows(free)
    wg, og = windows(goal)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.15),
                             gridspec_kw={"width_ratios": [1.35, 1, 1]})

    # ---- (a) baseline-subtracted transient ----------------------------------
    ax = axes[0]
    t = np.arange(-PRE, POST)
    ax.axvspan(0, POST, color=JUNC, alpha=0.15, linewidth=0, zorder=0)
    ax.axhline(0, color=MUTED, linewidth=0.6, zorder=1)
    for w, o, c, lab in ((wg, og, GOALCOND, "Goal-cond."), (wf, of, GOALFREE, "Goal-free")):
        mu, lo, hi = boot_band(w, o)
        if mu is None:
            continue
        ax.fill_between(t, lo, hi, color=c, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(t, mu, color=c, linewidth=1.3, zorder=3,
                label=f"{lab} ({len(w)} entries)")
    ax.axvline(0, color=INK2, linewidth=0.7, linestyle=(0, (3, 2)), zorder=4)
    ax.set_xlabel("Steps from junction entry")
    ax.set_ylabel(r"$\Delta$ entropy vs. baseline (nats)")
    ax.set_xlim(-PRE, POST - 1)
    ax.legend(frameon=False, loc="upper left", handlelength=1.4,
              borderpad=0.15, labelspacing=0.2)
    ax.set_title("(a) Transient at entry", fontsize=7.6, color=INK, pad=3)

    # ---- (b) which control widens, RELATIVE ---------------------------------
    ax = axes[1]
    dims, x, w = ["Steer", "Throttle"], np.arange(2), 0.34
    for k, (eps, c, lab) in enumerate(((goal, GOALCOND, "Goal-cond."),
                                       (free, GOALFREE, "Goal-free"))):
        vals = [paired(eps, i) for i in range(2)]
        m = [v[0] for v in vals]
        err = np.array([[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]])
        ax.bar(x + (k - 0.5) * (w + 0.02), m, w, color=c, label=lab,
               edgecolor="white", linewidth=0.8, zorder=2)
        ax.errorbar(x + (k - 0.5) * (w + 0.02), m, yerr=err, fmt="none",
                    ecolor=INK2, elinewidth=0.7, capsize=1.8, zorder=3)
    ax.axhline(0, color=MUTED, linewidth=0.6, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(dims)
    ax.set_ylabel(r"$\Delta\sigma$ at junction (%)")
    ax.legend(frameon=False, handlelength=1.2, borderpad=0.15, labelspacing=0.2)
    ax.set_title("(b) Steering, not throttle", fontsize=7.6, color=INK, pad=3)

    # ---- (c) entropy effect size --------------------------------------------
    ax = axes[2]
    rows = []
    for eps, c, lab in ((free, GOALFREE, "Goal-free"), (goal, GOALCOND, "Goal-cond.")):
        v = paired(eps, "entropy")
        if v:
            rows.append((f"{lab}\n($n$={v[3]})", v[0], v[1], v[2], c))
    ys = np.arange(len(rows))[::-1]
    ax.axvline(0, color=MUTED, linewidth=0.7, zorder=0)
    for y, (lab, m, lo, hi, c) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=c, linewidth=1.3, zorder=2)
        for xx in (lo, hi):
            ax.plot([xx, xx], [y - .10, y + .10], color=c, linewidth=1.0, zorder=2)
        ax.plot([m], [y], "o", color=c, markersize=4.2, zorder=3,
                markeredgecolor="white", markeredgewidth=0.6)
        ax.text(hi + 0.012, y, f"{m:+.3f}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.set_xlabel(r"$\Delta$ entropy, junct. $-$ road (nats)")
    ax.tick_params(axis="y", length=0)
    ax.set_title("(c) Effect size", fontsize=7.6, color=INK, pad=3)

    for ax in axes:
        ax.grid(True, color=GRID, linewidth=0.45)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=INK2)

    fig.tight_layout(pad=0.3, w_pad=1.0)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 600})):
        fig.savefig(args.out + ext, bbox_inches="tight", facecolor="white", **kw)
        print(f"[Fig] -> {args.out}{ext}")
    for eps, lab in ((free, "goal-free"), (goal, "goal-cond.")):
        e = paired(eps, "entropy")
        s = paired(eps, 0); th = paired(eps, 1)
        print(f"[Fig] {lab:11s} dH {e[0]:+.3f} [{e[1]:+.3f},{e[2]:+.3f}] n={e[3]}  "
              f"steer {s[0]:+.1f}% [{s[1]:+.1f},{s[2]:+.1f}]  "
              f"throttle {th[0]:+.1f}% [{th[1]:+.1f},{th[2]:+.1f}]")


if __name__ == "__main__":
    main()
