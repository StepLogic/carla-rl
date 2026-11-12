#!/usr/bin/env python3
"""A4 (plan 2026-11-09): why does a long route fail?

For every FAILED episode of an eval run with EVAL_TRAJ=1 (<eval>/eval.json +
<eval>/traj/ep*.json): its end reason (crossing / off_road / collision / clock / ...)
and three flags from the per-step trajectory:
  wrong_branch    left the planner's shortest road route by > 5 m for >= 20 steps,
                  starting in or within 30 m after a junction
  stall           below 2 km/h for >= 50 consecutive steps
  passed_subgoal  (map arms) came within 3 m of the current subgoal's node and had
                  not switched 20 steps later
Lane faults (crossing / off_road) are placed: "junction" (in a junction, or within
15 m after one) or "elsewhere". Junction flags come from the town's OpenDRIVE map
(torch_rl.lane_labels.OpenDriveLanes); privileged, diagnostic only.

Gate A (plan): lane faults + wrong branches >= 50% of failures -> the driver (B1);
passed-subgoal / stall dominating while chains complete -> switching (mapping
session); wrong branches outnumbering lane faults in the map arm -> also B2.

usage: python scripts/route_failure_taxonomy.py --town Town01 [--map MAP.json] EVAL_DIR ...
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "src")]

OFF_M, PERSIST, AFTER_JUNCTION_M = 5.0, 20, 30.0
STALL_KMH, STALL_STEPS = 2.0, 50
NEAR_M, SWITCH_WINDOW = 3.0, 20
FAULT_AFTER_M = 15.0
LANE_FAULTS = ("crossing", "off_road")


def dist_to_polyline(P, L):
    """(N, 2) points -> (N,) distance to the polyline L (M, 2)."""
    P = np.asarray(P, float)
    L = np.asarray(L, float)
    if len(L) == 1:
        return np.hypot(*(P - L[0]).T)
    A, B = L[:-1], L[1:]
    AB = B - A
    den = np.maximum((AB ** 2).sum(-1), 1e-12)
    t = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(-1) / den[None], 0.0, 1.0)
    proj = A[None] + t[..., None] * AB[None]
    return np.sqrt(((P[:, None, :] - proj) ** 2).sum(-1)).min(1)


def _runs(mask):
    """(start, length) of each run of True."""
    out, s = [], None
    for i, m in enumerate(np.asarray(mask, bool)):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append((s, i - s))
            s = None
    if s is not None:
        out.append((s, len(mask) - s))
    return out


def stalled(speed_kmh, below_kmh=STALL_KMH, min_steps=STALL_STEPS):
    return any(n >= min_steps for _, n in _runs(np.asarray(speed_kmh, float) < below_kmh))


def _cum_len(P):
    P = np.asarray(P, float)
    return np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))] if len(P) > 1 else np.zeros(len(P))


def wrong_branch(P, route, junction, off_m=OFF_M, persist=PERSIST, after_m=AFTER_JUNCTION_M):
    """A lasting departure (> off_m for >= persist steps) from the road route that
    starts in a junction or within after_m metres after one."""
    if route is None or len(route) == 0 or len(P) == 0:
        return False
    d = dist_to_polyline(P, route)
    cum = _cum_len(P)
    jidx = np.flatnonzero(np.asarray(junction, bool))
    for s, n in _runs(d > off_m):
        if n < persist:
            continue
        before = jidx[jidx <= s]
        if len(before) and cum[s] - cum[before[-1]] <= after_m:
            return True
    return False


def passed_subgoal(P, cur, node_xy, near_m=NEAR_M, window=SWITCH_WINDOW):
    """The car came within near_m of its current subgoal's node and still had that
    subgoal `window` steps after the first close step."""
    P = np.asarray(P, float)
    cur = np.asarray(cur)
    seen = set()
    for t in range(len(P)):
        n = int(cur[t])
        if n < 0 or n in seen or n not in node_xy:
            continue
        if np.hypot(*(P[t] - np.asarray(node_xy[n], float))) <= near_m:
            seen.add(n)
            end = min(len(cur), t + window + 1)
            if end - t > window and np.all(cur[t:end] == n):
                return True
    return False


def fault_place(P, junction, after_m=FAULT_AFTER_M):
    """'junction' if the episode ended in a junction or within after_m metres after
    one, else 'elsewhere'."""
    junction = np.asarray(junction, bool)
    if len(junction) == 0:
        return "elsewhere"
    if junction[-1]:
        return "junction"
    jidx = np.flatnonzero(junction)
    if len(jidx) == 0:
        return "elsewhere"
    cum = _cum_len(P)
    return "junction" if cum[-1] - cum[jidx[-1]] <= after_m else "elsewhere"


def classify(ep, traj, junction=None, node_xy=None):
    """None for a reached episode; else {end, place, wrong_branch, stall, passed_subgoal}."""
    if ep.get("reached") or ep.get("end") == "reached":
        return None
    steps = np.asarray(traj["steps"], float) if traj and traj.get("steps") else np.zeros((0, 6))
    P = steps[:, 1:3] if len(steps) else np.zeros((0, 2))
    speed = steps[:, 4] if len(steps) else np.zeros(0)
    cur = steps[:, 5].astype(int) if len(steps) else np.zeros(0, int)
    junction = np.zeros(len(P), bool) if junction is None else np.asarray(junction, bool)
    route = np.asarray(traj["road_route"], float) if traj and traj.get("road_route") else None
    end = str(ep.get("end"))
    return {"end": end,
            "place": fault_place(P, junction) if end in LANE_FAULTS else None,
            "wrong_branch": bool(wrong_branch(P, route, junction)),
            "stall": bool(stalled(speed)),
            "passed_subgoal": bool(node_xy is not None and passed_subgoal(P, cur, node_xy))}


def summarise(rows):
    n = len(rows)
    lane = sum(1 for r in rows if r["end"] in LANE_FAULTS)
    out = {"failures": n,
           "ends": {e: sum(1 for r in rows if r["end"] == e) for e in sorted({r["end"] for r in rows})},
           "lane_fault_junction": sum(1 for r in rows if r["place"] == "junction"),
           "lane_fault_elsewhere": sum(1 for r in rows if r["place"] == "elsewhere"),
           "wrong_branch": sum(r["wrong_branch"] for r in rows),
           "stall": sum(r["stall"] for r in rows),
           "passed_subgoal": sum(r["passed_subgoal"] for r in rows),
           "lane_fault_or_wrong_branch": sum(1 for r in rows
                                             if r["end"] in LANE_FAULTS or r["wrong_branch"])}
    out["lane_fault_share"] = round(lane / n, 3) if n else None
    out["driver_share"] = round(out["lane_fault_or_wrong_branch"] / n, 3) if n else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evals", nargs="+")
    ap.add_argument("--town", required=True)
    ap.add_argument("--map", default=None, help="map json (node xy) for the passed-subgoal flag")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from torch_rl.lane_labels import OpenDriveLanes
    lanes = OpenDriveLanes(a.town)
    node_xy = None
    if a.map:
        nodes = json.load(open(a.map))["nodes"]
        node_xy = {i: (float(n["x"]), float(n["y"])) if "x" in n else tuple(n["pos"][:2])
                   for i, n in enumerate(nodes)}
    report = {}
    for d in a.evals:
        ev = json.load(open(os.path.join(d, "eval.json")))
        trajs = {}
        for f in glob.glob(os.path.join(d, "traj", "ep*.json")):
            t = json.load(open(f))
            trajs[t["seed"]] = t
        rows = []
        for ep in ev["episodes_detail"]:
            tr = trajs.get(ep.get("seed"))
            if ep.get("reached"):
                continue
            junction = None
            if tr and tr.get("steps"):
                P = np.asarray(tr["steps"], float)[:, 1:3]
                junction = lanes.query(P)["junction"]
            r = classify(ep, tr, junction, node_xy)
            if r is not None:
                rows.append(r)
        report[d] = {"episodes": len(ev["episodes_detail"]),
                     "reached": sum(1 for e in ev["episodes_detail"] if e.get("reached")),
                     "with_traj": len(trajs), **summarise(rows)}
        print(json.dumps({d: report[d]}), flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
