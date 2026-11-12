"""Publication figure, single column: the mechanism, steering not throttle.

    python figure_sigma_junctions_icra.py

Single-panel column-width (3.5 in) version of panel (b) of
evidence/2026-09-09_entropy_at_junctions/figure_icra_entropy2.py, plus the
per-step time course the composite could not fit. DESCRIPTIVE, deliberately:
sigma(steer) widens relative to its own pre-entry baseline while
sigma(throttle) does not. The title does NOT say "uncertainty" -- the
curvature control (analysis_sigma_vs_curvature.py) showed this signal tracks
turning, not decision ambiguity, so the title asserts only what it shows and
the interpretation lives with the control figure.

Data: evidence/2026-09-09_entropy_at_junctions/data/entropy_nogoal_500cap.json
(the 500-cap goal-free build; same file the published figure's numbers come
from -- steer +13.3% [+9.3,+17.3], throttle -1.4% [-1.9,-0.9]). NO new rollouts.

The quantities are the per-dimension standard deviations of the PRE-TANH
Gaussian (dist.distribution.stddev()), which is what the pack stores in
step["sigma"]. NOT per-dimension entropy: for a diagonal Gaussian the two are
monotone-ordered identically (H_i = log sigma_i + const) but the numbers differ,
so the axis is explicit: percent change in sigma.
"""
import argparse
import json

import numpy as np

STEER, THROTTLE = "#0072B2", "#009E73"   # Okabe-Ito blue, bluish green (validated)
JUNC = "#E69F00"
INK, INK2, MUTED, GRID = "#000000", "#3d3d3d", "#8a8a8a", "#dcdcdc"
PRE, POST, BASE_HI = 40, 60, -10   # same window as figure_icra_entropy2.py


def windows(episodes, key_idx):
    """Windows of sigma[key_idx] aligned on every junction-entry step."""
    win, owner = [], []
    for e in episodes:
        sig = np.array([s["sigma"][key_idx] for s in e["steps"]], float)
        inj = np.array([s["in_junction"] for s in e["steps"]], bool)
        if not inj.any():
            continue
        d = np.diff(np.concatenate([[0], inj.view(np.int8)]))
        for on in np.flatnonzero(d == 1):
            a, b = on - PRE, on + POST
            if a < 0 or b > len(sig):
                continue
            win.append(sig[a:b]); owner.append(e["episode"])
    return np.array(win, float), np.array(owner)


def boot_band(win, owner, n_boot=2000, seed=0):
    """RELATIVE-to-baseline mean curve, 95% band resampling EPISODES."""
    base = win[:, :PRE + BASE_HI].mean(axis=1, keepdims=True)
    rel = (win / base - 1.0) * 100.0
    rng = np.random.default_rng(seed)
    eps = np.unique(owner)
    out = np.empty((n_boot, rel.shape[1]))
    for b in range(n_boot):
        pick = rng.choice(eps, len(eps), replace=True)
        out[b] = rel[np.concatenate([np.flatnonzero(owner == p) for p in pick])].mean(axis=0)
    return rel.mean(axis=0), np.percentile(out, 2.5, axis=0), np.percentile(out, 97.5, axis=0)


def paired_pct(episodes, key_idx):
    """Per-episode (junction - road) relative change, %, bootstrap CI."""
    d = []
    for e in episodes:
        sig = np.array([s["sigma"][key_idx] for s in e["steps"]], float)
        inj = np.array([s["in_junction"] for s in e["steps"]], bool)
        if inj.any() and (~inj).any():
            d.append((sig[inj].mean() - sig[~inj].mean()) / sig[~inj].mean() * 100)
    d = np.asarray(d, float)
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (10000, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="evidence/2026-09-09_entropy_at_junctions/"
                                     "data/entropy_nogoal_500cap.json")
    ap.add_argument("--out", default="results/fig_sigma_junctions_icra")
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

    episodes = json.load(open(args.data))["episodes"]

    # median junction duration, so the shading asserts only what the data shows
    runs = []
    for e in episodes:
        inj = np.array([s["in_junction"] for s in e["steps"]], bool)
        d = np.diff(np.concatenate([[0], inj.view(np.int8), [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        runs += list(ends - starts)
    med_dur = float(np.median(runs))
    print(f"[Fig] junction runs n={len(runs)} median={med_dur:g} steps")

    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    t = np.arange(-PRE, POST)
    ax.axvspan(0, min(med_dur, POST), color=JUNC, alpha=0.15, linewidth=0,
               zorder=0)
    ax.axhline(0, color=MUTED, linewidth=0.6, zorder=1)
    for idx, c, dim in ((0, STEER, "Steer"), (1, THROTTLE, "Throttle")):
        win, owner = windows(episodes, idx)
        mu, lo, hi = boot_band(win, owner)
        ax.fill_between(t, lo, hi, color=c, alpha=0.18, linewidth=0, zorder=2)
        v = paired_pct(episodes, idx)
        ax.plot(t, mu, color=c, linewidth=1.6, zorder=3,
                label=f"{dim}  {v[0]:+.1f}% [{v[1]:+.1f}, {v[2]:+.1f}] ($n$={v[3]})")
    ax.axvline(0, color=INK2, linewidth=0.7, linestyle=(0, (3, 2)), zorder=4)
    ax.set_xlabel("Steps From Junction Entry")
    ax.set_ylabel(r"Relative $\Delta\sigma$ vs. Baseline (%)")
    ax.set_xlim(-PRE, POST - 1)
    ax.set_title("Steering Scale, Not Throttle", fontsize=7.8, color=INK, pad=3)
    ax.legend(frameon=False, loc="upper left", handlelength=1.4,
              borderpad=0.15, labelspacing=0.2)
    ax.grid(True, color=GRID, linewidth=0.45)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=INK2)

    fig.tight_layout(pad=0.3)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 600})):
        fig.savefig(args.out + ext, bbox_inches="tight", facecolor="white", **kw)
        print(f"[Fig] -> {args.out}{ext}")
    for idx, dim in ((0, "steer"), (1, "throttle")):
        v = paired_pct(episodes, idx)
        print(f"[Fig] sigma_{dim:8s} {v[0]:+.1f}% [{v[1]:+.1f},{v[2]:+.1f}] n={v[3]}")


if __name__ == "__main__":
    main()