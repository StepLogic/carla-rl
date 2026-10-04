#!/usr/bin/env python3
"""Plot the uncertainty telemetry a v43 explore trace recorded (user 2026-10-02).

Two figures, because they answer different questions:

  *_map.png     WHERE the explore critic is unsure: the driven path, every kept
                sample coloured by epistemic uncertainty (spread ACROSS the five
                TQC critics' means). One panel per trace.
  *_signals.png WHAT the three recorded signals look like and whether they
                ANTICIPATE a fault. Row 1 is each signal's distribution per
                town; row 2 is its mean profile over the last metres before a
                segment's fault, against that town's own overall mean. A signal
                that rises only at the end is a usable failure predictor; a flat
                profile is not (cf. the v17 result that goal Q does not).

Signals come from the per-sample `uncertainty` dict written by
collect_explore_trace_v43.py:
  epistemic_e  std across the 5 critics of each critic's own quantile mean
  aleatoric_e  mean across critics of the within-critic quantile std
  ent_total    the policy's true action-space differential entropy, as
               ent_gauss (exact) + ent_rescale (exact) + ent_tanh_mc (MC)
`_e` is the EXPLORE critic, the one that drove these episodes. The goal critic's
`_g` fields are a self-goal query (goal = current pose) and sit near zero by
construction, so they are not plotted -- pairwise goal uncertainty needs an
offline pass over the stored observations.

Samples are the trace's kept samples, ~2.2 m apart, so "metres before the end"
is index x median spacing, measured per trace rather than assumed.

usage: plot_trace_uncertainty.py OUT_PREFIX TRACE.pkl [TRACE2.pkl ...]
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_explore_trace import town_waypoints  # noqa: E402

# Categorical slots 1 and 2 of the validated default palette, in its fixed
# order. Slots 1-3 are the documented all-pairs-safe subset (scatter and
# multi-series line forms), so a two-town figure uses them unchanged.
TOWN_COLOUR = ["#2a78d6", "#eb6834", "#1baf7a"]
# The palette's sequential blue ramp, light -> dark, for magnitude.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
            "#0d366b"]
BLUES = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

# Each panel: (key, axis label, title). Order is fixed across both rows.
SIGNALS = [
    ("epistemic_e", "Epistemic uncertainty (Q units)",
     "Epistemic: disagreement across critics"),
    ("aleatoric_e", "Aleatoric uncertainty (Q units)",
     "Aleatoric: within-critic quantile spread"),
    ("ent_total", "Policy entropy (nats)",
     "Policy entropy: total, action space"),
]
# A fault, as the explore arm reports it. "budget" is the collector stopping,
# not the policy failing, so those segments are excluded from the profiles.
NOT_A_FAULT = {"budget", "cap"}
# Samples of run-up plotted before a fault. 10 x ~2.2 m ~ 22 m, which keeps a
# usable number of segments in both towns (Town04's median segment is short).
LAG = 10


def load(path):
    with open(path, "rb") as fh:
        tr = pickle.load(fh)
    unc = tr.get("uncertainty")
    if not unc:
        raise SystemExit(f"{path}: no 'uncertainty' field -- collected by an "
                         "older collector, so there is nothing to plot here.")
    bad = sum("uncertainty_error" in r for r in unc)
    if bad:
        print(f"WARNING {path}: {bad}/{len(unc)} samples recorded an "
              "uncertainty_error and are dropped from that signal.")
    pos = np.asarray(tr["positions"], np.float64)[:, :2]
    seg = np.asarray(tr["segments"], int)
    gaps = np.linalg.norm(np.diff(pos, axis=0), axis=1)[seg[1:] == seg[:-1]]
    tr = dict(tr)
    tr["_spacing"] = float(np.median(gaps)) if len(gaps) else float("nan")
    tr["_pos"], tr["_seg"] = pos, seg
    return tr


def signal(tr, key):
    """The per-sample values of `key`, NaN where that sample has none."""
    return np.array([float(r[key]) if key in r else np.nan
                     for r in tr["uncertainty"]])


def fault_runs(tr, key, lag=LAG):
    """(n_segments, lag) array of `key` over the last `lag` samples of every
    segment that ended in a fault and is at least `lag` samples long. Column
    -1 is the sample closest to the fault."""
    v, seg, info = signal(tr, key), tr["_seg"], tr.get("segment_info", [])
    out = []
    for s in np.unique(seg):
        if s >= len(info):
            continue
        end = str(info[s].get("end", "?"))
        if end in NOT_A_FAULT:
            continue
        run = v[seg == s]
        if len(run) >= lag and np.isfinite(run[-lag:]).all():
            out.append(run[-lag:])
    return np.array(out) if out else np.zeros((0, lag))


def map_panel(ax, tr, name):
    """The driven path, every kept sample coloured by epistemic uncertainty."""
    pos, town = tr["_pos"], tr.get("meta", {}).get("town", "?")
    v = signal(tr, "epistemic_e")
    ok = np.isfinite(v)
    bg = town_waypoints(town)
    if bg is not None:
        ax.scatter(bg[:, 0], bg[:, 1], s=0.6, c="#dddddd", linewidths=0,
                   zorder=1, label="Drivable road")
    # Clipped to the 5th-95th percentile: the tail is ~5x the median and would
    # otherwise flatten every ordinary sample to the lightest step.
    lo, hi = np.percentile(v[ok], [5, 95])
    sc = ax.scatter(pos[ok, 0], pos[ok, 1], c=v[ok], s=4.5, cmap=BLUES,
                    vmin=lo, vmax=hi, linewidths=0, zorder=3)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Epistemic uncertainty (Q units, 5th-95th pct)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(f"{name}\n{town}: {int(ok.sum())} samples, "
                 f"epistemic median {np.median(v[ok]):.1f}", fontsize=9)
    ax.set_xlabel("Map X (m)")
    ax.set_ylabel("Map Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()                      # CARLA's Y grows southward
    ax.legend(loc="upper right", fontsize=7, markerscale=3, framealpha=0.9)


def dist_panel(ax, traces, key, label, title):
    """One step-outlined histogram per town, on a shared range."""
    vals = [signal(t, key) for t in traces]
    finite = np.concatenate([v[np.isfinite(v)] for v in vals])
    lo, hi = np.percentile(finite, [0.5, 99.5])
    bins = np.linspace(lo, hi, 46)
    for i, (t, v) in enumerate(zip(traces, vals)):
        v = v[np.isfinite(v)]
        c = TOWN_COLOUR[i % len(TOWN_COLOUR)]
        town = t.get("meta", {}).get("town", f"Trace {i}")
        ax.hist(v, bins=bins, histtype="step", lw=1.8, color=c, density=True,
                label=f"{town} (median {np.median(v):.2f})")
        ax.axvline(np.median(v), color=c, lw=1.0, ls=":", alpha=0.9)
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(alpha=0.18, lw=0.6)


def profile_panel(ax, traces, key, label, title, spacing):
    """Mean of `key` over the run-up to a fault, with a 95% CI band, against
    each town's own overall mean."""
    for i, t in enumerate(traces):
        runs = fault_runs(t, key)
        c = TOWN_COLOUR[i % len(TOWN_COLOUR)]
        town = t.get("meta", {}).get("town", f"Trace {i}")
        base = np.nanmean(signal(t, key))
        ax.axhline(base, color=c, lw=1.0, ls=":", alpha=0.9)
        if not len(runs):
            continue
        x = -np.arange(LAG)[::-1] * spacing
        m, se = runs.mean(0), runs.std(0, ddof=1) / np.sqrt(len(runs))
        ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=c, alpha=0.18,
                        linewidth=0)
        ax.plot(x, m, "-o", color=c, lw=2.0, ms=4,
                label=f"{town} ({len(runs)} faults)")
    ax.set_xlabel("Metres before the fault (approx.)")
    ax.set_ylabel(label)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(alpha=0.18, lw=0.6)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    prefix, paths = sys.argv[1], sys.argv[2:]
    traces = [load(p) for p in paths]
    names = [os.path.basename(p).replace(".pkl", "") for p in paths]
    ckpts = sorted({"/".join(str(t.get("meta", {}).get("checkpoint", "?"))
                             .split("/")[-2:]) for t in traces})
    spacing = float(np.nanmedian([t["_spacing"] for t in traces]))

    fig, axes = plt.subplots(1, len(traces), squeeze=False,
                             figsize=(7.6 * len(traces), 7.0))
    for ax, t, n in zip(axes[0], traces, names):
        map_panel(ax, t, n)
    fig.suptitle("Explore-mode epistemic uncertainty along the driven path\n"
                 + ", ".join(ckpts), fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{prefix}_map.png", dpi=150)
    print(f"-> {prefix}_map.png")

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0))
    for j, (key, label, title) in enumerate(SIGNALS):
        dist_panel(axes[0, j], traces, key, label, title)
        profile_panel(axes[1, j], traces, key, label,
                      title.split(":")[0] + ": run-up to a fault", spacing)
    # The profile row is a SELECTED subset -- only segments that survived long
    # enough to have a run-up -- and in Town04 that is a minority of faults, so
    # the selection is stated on the figure rather than left to the caption.
    fig.suptitle("Explore-mode uncertainty and policy entropy: distributions "
                 f"(top) and the last {LAG * spacing:.0f} m before a fault "
                 f"(bottom)\n{', '.join(ckpts)}   "
                 "Dotted line = that town's overall mean; the profiles use "
                 f"only faults that survived {LAG * spacing:.0f} m, so they "
                 "over-represent long segments", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{prefix}_signals.png", dpi=150)
    print(f"-> {prefix}_signals.png")

    # The numbers behind the figure, so a claim about it can be checked.
    for t, n in zip(traces, names):
        town = t.get("meta", {}).get("town", "?")
        print(f"\n{n} ({town}, spacing {t['_spacing']:.2f} m)")
        for key, _, _ in SIGNALS:
            v = signal(t, key)
            v = v[np.isfinite(v)]
            runs = fault_runs(t, key)
            tail = runs[:, -1] if len(runs) else np.array([np.nan])
            far = runs[:, 0] if len(runs) else np.array([np.nan])
            print(f"  {key:12s} all mean {v.mean():8.3f} sd {v.std():7.3f} | "
                  f"{len(runs):3d} faults: {LAG * t['_spacing']:.0f} m out "
                  f"{np.nanmean(far):8.3f} -> last sample "
                  f"{np.nanmean(tail):8.3f}")


if __name__ == "__main__":
    main()
