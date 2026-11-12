#!/usr/bin/env python3
"""Does a map trace cover the benchmark BEFORE maps are built from it?

The Habitat peer's check (2026-11-06), in its lane-world form: >= 97% of the
benchmark's starts with a SAME-DIRECTION trace frame within --radius m, and
goals with any trace frame within --radius m (the goal's heading is not
recorded). Starts and goals come from an existing benchmark run's eval.json,
which drove the same fixed seeds (default: c0's mapless true-goal arm).

    check_trace_coverage.py --trace checkpoints/trace_expert/trace_expert_Town02 --town Town02
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from vae_nav import image_map as im  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--town", required=True)
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--evals", default="checkpoints/bench_c0_true_{town}_*/eval_step_15000/eval.json")
    a = ap.parse_args()
    tr = im.load_trace(a.trace)
    pos, yaw = tr["pos"], tr["yaw"]
    tree = cKDTree(pos)
    starts, syaw, goals = [], [], []
    for f in sorted(glob.glob(a.evals.format(town=a.town))):
        for x in json.load(open(f))["episodes_detail"]:
            starts.append(x["start_xy"]); syaw.append(x["start_yaw"]); goals.append(x["goal_xy"])
    starts, syaw, goals = np.asarray(starts), np.asarray(syaw), np.asarray(goals)
    if np.nanmax(np.abs(syaw)) > 2 * np.pi + 1e-6:
        syaw = np.radians(syaw)
    s_ok = np.array([bool(len(ix)) and bool(im._aligned(yaw[ix], y, np.radians(90)).any())
                     for ix, y in zip(tree.query_ball_point(starts, a.radius), syaw)])
    g_ok = np.array([bool(len(ix)) for ix in tree.query_ball_point(goals, a.radius)])
    both = s_ok & g_ok
    out = {"town": a.town, "trace": a.trace, "episodes": int(len(starts)),
           "radius_m": a.radius, "trace_km": round(float(np.sum(np.hypot(*np.diff(pos, axis=0).T))) / 1000, 2),
           "starts_same_direction": round(float(s_ok.mean()), 3),
           "goals_near": round(float(g_ok.mean()), 3), "both": round(float(both.mean()), 3),
           "passes_97": bool(both.mean() >= 0.97)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
