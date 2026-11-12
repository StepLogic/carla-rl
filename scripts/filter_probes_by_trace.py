#!/usr/bin/env python3
"""Keep probe episodes whose start AND goal lie within --radius-m of the expert tour
(plan 2026-11-09, A2). Town05's tour (EXPERT_TOUR_LARGEST_SCC=1) left out 32 lanes
(6 km) that cannot be reached both ways without a lane change; validation episodes
must not start or end there.

usage: filter_probes_by_trace.py TRACE_DIR OUT.json PROBE.json [PROBE.json ...] [--radius-m 5]
"""
import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "src")]


def on_trace(probes, trace_xy, radius_m=5.0, cell_m=None):
    """The probes whose start_xy and goal_xy are both within radius_m of a trace point."""
    T = np.asarray(trace_xy, float)[:, :2]
    cell = float(cell_m or radius_m)
    grid = {}
    for i, (x, y) in enumerate(T):
        grid.setdefault((int(np.floor(x / cell)), int(np.floor(y / cell))), []).append(i)

    def near(p):
        if p is None:
            return False
        x, y = float(p[0]), float(p[1])
        cx, cy = int(np.floor(x / cell)), int(np.floor(y / cell))
        idx = [i for dx in (-1, 0, 1) for dy in (-1, 0, 1) for i in grid.get((cx + dx, cy + dy), ())]
        return bool(idx) and float(np.min(np.hypot(T[idx, 0] - x, T[idx, 1] - y))) <= radius_m

    return [p for p in probes if near(p.get("start_xy")) and near(p.get("goal_xy"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("out")
    ap.add_argument("probes", nargs="+")
    ap.add_argument("--radius-m", type=float, default=5.0)
    a = ap.parse_args()
    from vae_nav import image_map as im
    tr = im.load_trace(a.trace)
    probes = [p for f in a.probes for p in json.load(open(f))]
    kept = on_trace(probes, tr["pos"], a.radius_m)
    json.dump(kept, open(a.out, "w"), indent=1)
    print(f"{len(kept)} of {len(probes)} probes start and end within {a.radius_m:g} m of the tour "
          f"-> {a.out}")


if __name__ == "__main__":
    main()
