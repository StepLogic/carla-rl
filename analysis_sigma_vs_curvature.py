"""Control: is the junction sigma(steer) effect just turning?

    python3 analysis_sigma_vs_curvature.py

Peer challenge: sigma(steer) may track steering MAGNITUDE / path curvature, not
decision ambiguity -- a junction is also a place where the car turns. All three
proposed controls, from the SAME stored goal-free traces as every other figure
(NO new rollouts): data/entropy_nogoal_500cap.json, 60 episodes, 24,449 steps.

Curvature is computed from the stored trace: speed = |dx,dy| / dt (dt = 0.1 s,
the sim's fixed_delta_seconds), yaw rate = wrapped dyaw / dt,
curvature kappa = |yaw rate| / speed  [1/m].  A 20 m-radius turn is kappa=0.05.
The steering command itself is NOT stored, so |steer| binning is unavailable;
curvature is the physical quantity the peer's control asks for.

Outputs:
  - printed stats (binned curves, curvature-matched delta, bootstrap CIs)
  - results/fig_sigma_vs_curvature.{pdf,png}   (the control figure)
  - results/fig_curv_field_town02.{pdf,png}    (curvature field, same 4 m cells
    as fig_sigma_field_town02 -- if the two maps look the same, sigma is a
    turning map)
"""
import argparse
import json

import numpy as np

STEER, THROTTLE = "#0072B2", "#009E73"
JUNC = "#E69F00"
STEER_DARK, STEER_LIGHT = "#08519c", "#f7fbff"
INK, INK2, MUTED, GRID = "#000000", "#3d3d3d", "#8a8a8a", "#dcdcdc"
DT = 0.1                     # sim fixed_delta_seconds
BINS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, np.inf]   # 1/m
CELL = 4.0


def load_steps(path):
    """Flatten episodes to per-step arrays; attach episode index per step."""
    eps = json.load(open(path))["episodes"]
    rows = []
    for ei, e in enumerate(eps):
        for s in e["steps"]:
            rows.append((ei, s["x"], s["y"], s["yaw"], s["sigma"][0],
                         s["in_junction"]))
    a = np.array(rows, float)
    return eps, a[:, 0].astype(int), a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5] > 0.5


def curvature(x, y, yaw):
    """kappa = |yaw rate| / speed from consecutive samples; NaN where unsafe."""
    vx, vy = np.diff(x), np.diff(y)
    speed = np.hypot(vx, vy) / DT
    dy = np.diff(yaw)
    dy = (dy + np.pi) % (2 * np.pi) - np.pi          # wrap to [-pi, pi]
    yaw_rate = dy / DT
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.abs(yaw_rate) / speed
    # sample i's curvature lives between step i and i+1; kill junk:
    # near-zero speed (numerical blowup) and >180 deg/step (teleport/spawn)
    bad = (speed < 0.5) | (np.abs(dy) > 1.0)
    kappa[bad] = np.nan
    return kappa, speed


def boot_mean(vals, owner, n_boot=2000, seed=0):
    """Episode-bootstrap mean and 95% CI of vals (NaNs dropped)."""
    ok = ~np.isnan(vals)
    v, o = vals[ok], owner[ok]
    eps = np.unique(o)
    rng = np.random.default_rng(seed)
    idx = {e: np.flatnonzero(o == e) for e in eps}
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(eps, len(eps), replace=True)
        draws[b] = v[np.concatenate([idx[e] for e in pick])].mean()
    return v.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="evidence/2026-09-09_entropy_at_junctions/"
                                     "data/entropy_nogoal_500cap.json")
    ap.add_argument("--out", default="results/fig_sigma_vs_curvature")
    ap.add_argument("--field-out", default="results/fig_curv_field_town02")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    eps, eidx, x, y, yaw, sig, inj = load_steps(args.data)
    kap, speed = curvature(x, y, yaw)
    # curvature between i and i+1: attribute to step i; last step of each
    # episode has no next sample -> drop it
    last = np.r_[eidx[1:] != eidx[:-1], True]
    kap_full = np.full(len(x), np.nan)
    kap_full[:-1] = kap
    kap_full[last] = np.nan
    kap = kap_full
    use = ~np.isnan(kap)
    print(f"[Ctl] usable steps {use.sum()}/{len(use)} "
          f"(speed>=0.5 m/s, no teleport); "
          f"kappa p50={np.nanpercentile(kap, 50):.3f} "
          f"p90={np.nanpercentile(kap, 90):.3f} 1/m")

    # ---- 1. binned curves: mean sigma(steer) per curvature bin, in vs out ----
    kb = np.digitize(kap, BINS) - 1
    print("\n[Ctl] mean sigma(steer) per curvature bin (n steps, in/out):")
    print("  bin(1/m)      out            in")
    binned = {}
    for b in range(len(BINS) - 1):
        m = (kb == b) & use
        o, i = m & ~inj, m & inj
        if o.sum() < 30 or i.sum() < 10:
            continue
        lo, hi = BINS[b], BINS[b + 1]
        lbl = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
        binned[lbl] = (sig[o].mean(), o.sum(), sig[i].mean(), i.sum())
        print(f"  {lbl:10s}  {sig[o].mean():.3f} (n={o.sum():5d})   "
              f"{sig[i].mean():.3f} (n={i.sum():4d})")

    # ---- 2. curvature-matched junction - road delta ----
    # within each curvature bin, delta_b = mean_in - mean_out; aggregate with
    # in-junction counts as weights (so the summary matches the junction
    # population's curvature mix, not the road's), bootstrap over EPISODES.
    def matched_delta(sel):
        mask = np.zeros(len(kb), bool)
        mask[sel] = True
        per = []
        for b in range(len(BINS) - 1):
            m = (kb == b) & mask
            o, i = m & ~inj, m & inj
            if o.sum() < 30 or i.sum() < 10:
                continue
            per.append((sig[i].mean(), sig[o].mean(), i.sum()))
        w = np.array([p[2] for p in per], float)
        d = np.array([p[0] - p[1] for p in per])
        return (d * w).sum() / w.sum()

    rng = np.random.default_rng(0)
    uniq = np.unique(eidx[use])
    idx_by_ep = {e: np.flatnonzero((eidx == e) & use) for e in uniq}
    draws = np.empty(10000)
    for k in range(10000):
        pick = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([idx_by_ep[e] for e in pick])
        draws[k] = matched_delta(sel)
    point = matched_delta(np.flatnonzero(use))
    lo, hi = np.percentile(draws, 2.5), np.percentile(draws, 97.5)
    rel = point / sig[use & ~inj].mean() * 100
    print(f"\n[Ctl] curvature-MATCHED delta: {point:+.4f} "
          f"[{lo:+.4f}, {hi:+.4f}]  ({rel:+.1f}% of road mean "
          f"{sig[use & ~inj].mean():.3f})")
    raw = (sig[use & inj].mean() - sig[use & ~inj].mean())
    print(f"[Ctl] UNmatched delta for reference: {raw:+.4f} "
          f"({raw / sig[use & ~inj].mean() * 100:+.1f}%)")

    # ---- 3. the control figure: binned curves + matched delta ----
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ctr = np.array([np.sqrt(BINS[b] * BINS[b + 1]) if np.isfinite(BINS[b + 1])
                    and BINS[b] > 0 else np.sqrt(BINS[b + 1] / 2)
                    for b in range(len(BINS) - 1)])
    for mask, c, lbl, mk in (((kb >= 0) & use & ~inj, STEER, "Off junction", "o"),
                             ((kb >= 0) & use & inj, JUNC, "At junction", "s")):
        mus, cis = [], []
        for b in range(len(BINS) - 1):
            m = (kb == b) & mask
            if m.sum() < 10:
                mus.append(np.nan); cis.append((np.nan, np.nan)); continue
            mu, l, h = boot_mean(sig[m], eidx[m])
            mus.append(mu); cis.append((l, h))
        mus = np.array(mus); cl = np.array([c[0] for c in cis])
        ch = np.array([c[1] for c in cis])
        ax.plot(ctr, mus, color=c, marker=mk, markersize=3.5, linewidth=1.4,
                label=lbl)
        ax.fill_between(ctr, cl, ch, color=c, alpha=0.15, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("Path Curvature $\\kappa$ (1/m)")
    ax.set_ylabel("Mean $\\sigma$(steer)")
    ax.set_title("Sigma(steer) vs. Path Curvature",
                 fontsize=7.8, color=INK, pad=3)
    road_mean = sig[use & ~inj].mean()
    ax.text(0.98, 0.02,
            f"Curvature-matched junction effect: {rel:+.1f}% "
            f"[{lo / road_mean * 100:+.1f}, {hi / road_mean * 100:+.1f}]",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            color=INK2)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, color=GRID, linewidth=0.45, which="both")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=INK2)
    fig.tight_layout(pad=0.3)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 600})):
        fig.savefig(args.out + ext, bbox_inches="tight", facecolor="white", **kw)
        print(f"[Ctl] -> {args.out}{ext}")

    # ---- 4. curvature field, same cells as the sigma field ----
    gxo = np.floor(x[use] / CELL).astype(int)
    gyo = np.floor(y[use] / CELL).astype(int)
    gxo -= gxo.min(); gyo -= gyo.min()
    width = int(gyo.max()) + 1
    key = gxo * width + gyo
    order = np.argsort(key)
    k_s, v_s = key[order], kap[use][order]
    bounds = np.flatnonzero(np.diff(k_s)) + 1
    cells = np.split(v_s, bounds)
    uk = k_s[np.r_[0, bounds]]
    mean_k = np.array([np.nanmean(c) for c in cells])
    cx = (uk // width + np.floor(x[use] / CELL).min()) * CELL + CELL / 2
    cy = (uk % width + np.floor(y[use] / CELL).min()) * CELL + CELL / 2

    jmap = json.load(open("evidence/2026-09-09_entropy_at_junctions/"
                          "data/map_nodes_500cap.json"))
    sites = np.asarray(jmap["junctions"], float)

    cmap = LinearSegmentedColormap.from_list("curv_blue",
                                             [STEER_LIGHT, STEER_DARK])
    vmin, vmax = np.percentile(mean_k, 30), np.percentile(mean_k, 99)
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    rects = [Rectangle((a - CELL / 2, b - CELL / 2), CELL, CELL)
             for a, b in zip(cx, cy)]
    pc = PatchCollection(rects, cmap=cmap, linewidths=0)
    pc.set_array(mean_k); pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.scatter(sites[:, 0], sites[:, 1], s=26, facecolors="none",
               edgecolors=JUNC, linewidths=1.0, zorder=3,
               label=f"Junction sites ({len(sites)})")
    ax.set_aspect("equal")
    ax.set_xlabel("World x (m)")
    ax.set_ylabel("World y (m)")
    ax.set_title("Curvature across Town02", fontsize=7.8, color=INK, pad=3)
    ax.legend(frameon=False, loc="center", handlelength=1.0,
              borderpad=0.2, labelspacing=0.2, fontsize=6.8)
    ax.tick_params(colors=INK2)
    for sp in ax.spines.values():
        sp.set_color(MUTED)
    ax.set_xlim(cx.min() - 4, cx.max() + 4)
    ax.set_ylim(cy.min() - 4, cy.max() + 4)
    cb = fig.colorbar(pc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(f"Mean curvature per {CELL:g} m cell (1/m)", fontsize=6.8)
    cb.ax.tick_params(labelsize=6.5, colors=INK2)
    cb.outline.set_edgecolor(MUTED)
    fig.tight_layout(pad=0.3)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 600})):
        fig.savefig(args.field_out + ext, bbox_inches="tight",
                    facecolor="white", **kw)
        print(f"[Ctl] -> {args.field_out}{ext}")


if __name__ == "__main__":
    main()