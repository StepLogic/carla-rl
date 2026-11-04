#!/usr/bin/env python3
"""Metres between failures from explore traces (imitation-first plan, Task 8).

    python3 -I scripts/trace_failure_stats.py checkpoints/trace_v30det_Town02/trace_step_300000 ...

Each argument is a trace folder (its index.json, as EVAL_TRACE=1 writes it).
One row per folder: metres driven, faults, mean metres between failures, the
median and longest fault-ended run, and the mean speed -- a lane-keeping
figure means little without the speed it was driven at.

A failure is a crossing, off-road or collision end. Any other end (the cap,
the step budget) is not one, but its metres count.
"""
import argparse
import json
import os
import sys

import numpy as np

FAULTS = ("crossing", "off_road", "collision")
DT_S = 0.05                                     # one env step at 20 Hz


def failure_stats(segments):
    """segments: [{"metres", "end"}, ...] -> {"metres", "faults",
    "mean_between", "median_run", "longest"}; the runs are over fault-ended
    segments, and mean_between is inf when nothing failed."""
    metres = float(sum(float(s["metres"]) for s in segments))
    runs = [float(s["metres"]) for s in segments if s["end"] in FAULTS]
    return {"metres": metres, "faults": len(runs),
            "mean_between": metres / len(runs) if runs else float("inf"),
            "median_run": float(np.median(runs)) if runs else float("nan"),
            "longest": max(runs) if runs else float("nan")}


def mean_speed_kmh(segments, dt=DT_S):
    """Distance over time across the segments (needs each segment's steps)."""
    steps = sum(int(s["steps"]) for s in segments)
    return float(sum(float(s["metres"]) for s in segments)) / (steps * dt) * 3.6 if steps else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    a = ap.parse_args(argv)
    print(f"{'trace':<56} {'town':<9} {'metres':>8} {'faults':>6} {'m/fail':>7} "
          f"{'median':>7} {'longest':>8} {'km/h':>5}")
    for d in a.traces:
        with open(os.path.join(d, "index.json")) as fh:
            idx = json.load(fh)
        segs = idx["segments"]
        r = failure_stats(segs)
        kmh = mean_speed_kmh(segs) if segs and all("steps" in s for s in segs) else float("nan")
        print(f"{os.path.normpath(d)[-56:]:<56} {idx.get('town', '?'):<9} {r['metres']:>8.0f} "
              f"{r['faults']:>6} {r['mean_between']:>7.1f} {r['median_run']:>7.1f} "
              f"{r['longest']:>8.1f} {kmh:>5.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
