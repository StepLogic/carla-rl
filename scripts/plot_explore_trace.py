#!/usr/bin/env python3
"""Plot the path an explore-mode trace actually drove (user 2026-09-30).

One panel per trace: the town's drivable waypoints in grey underneath, the
driven path on top as one line per segment, each segment's start marked and its
END coloured by cause. A trace is a CHAIN of segments -- the collector restarts
each one at the nearest drivable waypoint to where the last stopped -- so the
segments are drawn separately; joining them would draw straight lines across
town through the teleports.

Positions are the trace's kept samples, thinned to --candidate-spacing (2 m by
default), so the line is a 2 m-resolution polyline of the real path, not every
step. Adjacent kept samples more than JUMP_M apart are cut, which hides both the
segment teleports and any thinning gap over a long straight.

The town background comes from the OpenDRIVE file via carla.Map, with no
simulator running, and is skipped if carla is unimportable.

usage: plot_explore_trace.py OUT.png TRACE.pkl [TRACE2.pkl ...]
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

XODR = ("/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/"
        "OpenDrive/{town}.xodr")
# A gap this large between consecutive kept samples is not driving.
JUMP_M = 12.0
# End causes the explore arm reports (explore_reward / gated_reward), plus the
# collector's own cap. Colours are fixed so panels are comparable.
CAUSE_COLOUR = {"crossing": "#d62728", "off_road": "#ff7f0e",
                "collision": "#9467bd", "cap": "#2ca02c",
                "budget": "#2ca02c",   # the collector's sample budget, not a fault
                "reached": "#1f77b4", "wedged": "#8c564b"}


def town_waypoints(town):
    """(x, y) of the town's drivable waypoints, or None without carla."""
    try:
        sys.path.append("/home/robotlab/Apps/CARLA_0.9.15/PythonAPI/carla")
        import carla
    except Exception:
        return None
    path = XODR.format(town=town)
    if not os.path.exists(path):
        return None
    m = carla.Map(town, open(path).read())
    w = m.generate_waypoints(4.0)
    return np.array([[p.transform.location.x, p.transform.location.y]
                     for p in w])


def segment_runs(pos, seg):
    """Yield (segment_id, contiguous (x, y) run) with teleport jumps cut."""
    for s in np.unique(seg):
        p = pos[seg == s]
        if len(p) < 2:
            continue
        gaps = np.r_[0, np.linalg.norm(np.diff(p, axis=0), axis=1)]
        for run in np.split(p, np.flatnonzero(gaps > JUMP_M)):
            if len(run) >= 2:
                yield int(s), run


def panel(ax, trace, name):
    pos = np.asarray(trace["positions"], np.float64)[:, :2]
    seg = np.asarray(trace["segments"], int)
    town = trace.get("meta", {}).get("town", "?")
    info = trace.get("segment_info", []) or []

    bg = town_waypoints(town)
    if bg is not None:
        ax.scatter(bg[:, 0], bg[:, 1], s=0.6, c="#cccccc", linewidths=0,
                   label="Drivable road", zorder=1)

    path_m = 0.0
    for _, run in segment_runs(pos, seg):
        ax.plot(run[:, 0], run[:, 1], "-", lw=1.1, color="#1f3b73",
                alpha=0.85, zorder=3)
        path_m += float(np.linalg.norm(np.diff(run, axis=0), axis=1).sum())

    # Segment ends, coloured by cause. segment_info is per segment, in order.
    seen = set()
    for s in np.unique(seg):
        p = pos[seg == s]
        if not len(p):
            continue
        cause = "?"
        if s < len(info):
            d = info[s]
            cause = str((d.get("end") if isinstance(d, dict) else None) or "?")
        col = CAUSE_COLOUR.get(cause, "#333333")
        ax.scatter(*p[-1], s=26, c=col, marker="X", zorder=5,
                   edgecolors="white", linewidths=0.4,
                   label=(f"End: {cause}" if cause not in seen else None))
        seen.add(cause)
        ax.scatter(*p[0], s=12, c="black", marker="o", zorder=4,
                   label=("Segment start" if "start" not in seen else None))
        seen.add("start")

    n_seg = len(np.unique(seg))
    ax.set_title(f"{name}\n{town}: {len(pos)} samples, {n_seg} segments, "
                 f"{path_m / 1000:.2f} km driven")
    ax.set_xlabel("Map X (m)")
    ax.set_ylabel("Map Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.invert_yaxis()          # CARLA's Y grows southward
    ax.legend(loc="upper right", fontsize=7, markerscale=1.6, framealpha=0.9)
    return dict(town=town, samples=len(pos), segments=n_seg, km=path_m / 1000)


def main():
    out, traces = sys.argv[1], sys.argv[2:]
    if not traces:
        raise SystemExit(__doc__)
    fig, axes = plt.subplots(1, len(traces),
                             figsize=(7.2 * len(traces), 7.0), squeeze=False)
    # The run is READ FROM THE ARTEFACT, never hardcoded: these traces are
    # compared across runs, and a title naming the wrong policy is the kind of
    # mislabelling that outlives the figure.
    ckpts = set()
    for ax, t in zip(axes[0], traces):
        with open(t, "rb") as fh:
            tr = pickle.load(fh)
        meta = tr.get("meta", {})
        ck = str(meta.get("checkpoint", "?"))
        ckpts.add("/".join(ck.split("/")[-2:]))
        stats = panel(ax, tr, os.path.basename(t).replace(".pkl", ""))
        print(f"{t}: town={stats['town']} samples={stats['samples']} "
              f"segments={stats['segments']} driven={stats['km']:.2f} km "
              f"ckpt={ck}")
    # Wrapped and sized to the figure: the run name is long enough that a
    # single-line suptitle is clipped at both ends on a one-panel figure.
    fig.suptitle("Explore-mode driven path\n" + ", ".join(sorted(ckpts)),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
