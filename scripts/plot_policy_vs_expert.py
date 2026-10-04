#!/usr/bin/env python3
"""What the policy WOULD have done where the expert drove (user 2026-10-03).

An oracle trace records both actions at every kept sample: the one the expert
executed, and the one the policy proposed at the same observation. Subtracting
them is the only measurement in this repo that separates "the policy cannot see
the road" from "the policy sees it and chooses differently", because both agents
are answering the identical state -- no distribution shift, no credit
assignment, no reward in the way.

It also reaches states the policy's own trace never contains. v43's explore
reward pays forward progress with no novelty term, so its trace is one ring road
driven 80 times; the expert covers the whole map, and the error is therefore
measured on roads the policy has never driven as well as on ones it has.

Three panels per figure:
  throttle    both agents' distributions on the same states
  steering    the same for steer
  error map   the driven path, coloured by |policy steer - expert steer|, so a
              systematic error at particular geometry is visible as structure
              rather than hidden in a mean

usage: plot_policy_vs_expert.py OUT.png TRACE.pkl [TRACE2.pkl ...]
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

# Validated default palette, categorical slots 1 and 2, and its blue ramp.
EXPERT_C, POLICY_C = "#2a78d6", "#eb6834"
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                 "#184f95", "#0d366b"])
# (index into the action vector, label, title)
CHANNELS = [(1, "Throttle", "Throttle on the same states"),
            (0, "Steering", "Steering on the same states")]


def load(path):
    with open(path, "rb") as fh:
        tr = pickle.load(fh)
    if "policy_actions" not in tr:
        raise SystemExit(f"{path}: no 'policy_actions' -- this is a "
                         "policy-driven trace, so there is nothing to compare. "
                         "Use an oracle trace (collect_oracle_trace_v43.py).")
    tr["_a"] = np.asarray(tr["actions"], np.float64)
    tr["_p"] = np.asarray(tr["policy_actions"], np.float64)
    if len(tr["_a"]) != len(tr["_p"]):
        raise SystemExit(f"{path}: {len(tr['_a'])} expert actions vs "
                         f"{len(tr['_p'])} policy actions")
    return tr


def dist_panel(ax, traces, idx, label, title):
    lo = min(min(t["_a"][:, idx].min(), t["_p"][:, idx].min()) for t in traces)
    hi = max(max(t["_a"][:, idx].max(), t["_p"][:, idx].max()) for t in traces)
    bins = np.linspace(lo, hi, 50)
    for t in traces:
        town = t.get("meta", {}).get("town", "?")
        for arr, c, who in ((t["_a"], EXPERT_C, "Expert"),
                            (t["_p"], POLICY_C, "Policy")):
            v = arr[:, idx]
            ax.hist(v, bins=bins, histtype="step", lw=1.8, color=c,
                    density=True,
                    label=f"{who} {town} (mean {v.mean():.2f})")
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(alpha=0.18, lw=0.6)


def error_map(ax, tr):
    """The path, coloured by how far the policy's steer is from the expert's."""
    pos = np.asarray(tr["positions"], np.float64)[:, :2]
    err = np.abs(tr["_p"][:, 0] - tr["_a"][:, 0])
    town = tr.get("meta", {}).get("town", "?")
    bg = town_waypoints(town)
    if bg is not None:
        ax.scatter(bg[:, 0], bg[:, 1], s=0.6, c="#dddddd", linewidths=0,
                   zorder=1, label="Drivable road")
    hi = float(np.percentile(err, 95))
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=err, s=5, cmap=BLUES, vmin=0,
                    vmax=hi, linewidths=0, zorder=3)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("|Policy steer - expert steer| (0-95th pct)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(f"{town}: steering error, median {np.median(err):.3f}",
                 fontsize=9)
    ax.set_xlabel("Map X (m)")
    ax.set_ylabel("Map Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()                     # CARLA's Y grows southward
    ax.legend(loc="upper right", fontsize=7, markerscale=3, framealpha=0.9)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out, paths = sys.argv[1], sys.argv[2:]
    traces = [load(p) for p in paths]
    ckpts = sorted({"/".join(str(t.get("meta", {}).get("checkpoint", "?"))
                             .split("/")[-2:]) for t in traces})

    ncol = 2 + len(traces)
    fig, axes = plt.subplots(1, ncol, figsize=(5.6 * ncol, 5.6))
    for ax, (idx, label, title) in zip(axes, CHANNELS):
        dist_panel(ax, traces, idx, label, title)
    for ax, t in zip(axes[2:], traces):
        error_map(ax, t)
    fig.suptitle("What the policy would have done where the EXPERT drove "
                 f"-- identical states, both actions recorded\n{', '.join(ckpts)}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"-> {out}")

    for t, p in zip(traces, paths):
        town = t.get("meta", {}).get("town", "?")
        a, pa = t["_a"], t["_p"]
        print(f"\n{os.path.basename(p)} ({town}, {len(a)} samples)")
        for idx, label, _ in CHANNELS:
            print(f"  {label:9s} expert {a[:, idx].mean():6.3f}  "
                  f"policy {pa[:, idx].mean():6.3f}  "
                  f"|diff| mean {np.abs(pa[:, idx] - a[:, idx]).mean():6.3f} "
                  f"p90 {np.percentile(np.abs(pa[:, idx] - a[:, idx]), 90):6.3f}")
        print(f"  Throttle: policy above 0.9 on "
              f"{np.mean(pa[:, 1] > 0.9):.0%} of states, expert on "
              f"{np.mean(a[:, 1] > 0.9):.0%}")


if __name__ == "__main__":
    main()
