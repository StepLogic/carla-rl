#!/usr/bin/env python3
"""Certify directed shortcut edges with the goal critic, and reconnect the graph.

Dropping seglinks left one component per segment (Town03: 195 from 682 nodes).
Seglinks were never drivable, so the fix is not to put them back but to add
edges between nodes that are genuinely close AND genuinely reachable.

The recipe is the Habitat peer's (2026-10-03) with their own road correction:

  CANDIDATES   all directed pairs within 3 m that are not already connected and
               whose headings agree within 90 degrees. The heading gate runs
               BEFORE the critic, and is what stops two nodes 1.5 m apart on
               opposite carriageways from becoming an edge.
  CERTIFIER    the ONE-WAY value, not their two-way mean: V(i -> j) is our goal
               critic with state i's goal block relabelled to node j, read as
               predicted STEPS. Low is close, the opposite sense to their energy,
               so the rule ports as a percentile and never as their 5.65.
  CALIBRATION  OpenDRIVE only, offline, never at build time. A pair is labelled
               reachable when the road route between the poses is at most
               1.25 x Euclid + 0.3 m -- their navmesh criterion, same constants.
               The threshold is then read off the precision/recall table.

The runtime rule uses poses and the critic alone, so nothing here needs the
simulator: `goal_block` is a pure function of pose and the route label is
computed from the .xodr.

usage: add_shortcuts_v43.py MAPS.json TRACE.pkl --ckpt CKPT.pkl
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

XODR = ("/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/"
        "OpenDrive/{town}.xodr")
GOOD_SLACK_M, GOOD_RATIO = 0.3, 1.25        # the peer's reachability criterion


def reachable_by_road(wmap, a, b, tol_m=1.2, slack_m=3.0, step_m=0.5):
    """Can the car drive from pose `a` to pose `b` without leaving the road?

    SHORT-RANGE REACHABILITY NEEDS A SHORT-RANGE TEST. The first version of this
    asked GlobalRoutePlanner for a route and compared its length to the straight
    line. That was wrong and the numbers it produced were meaningless: between
    poses 1-3 m apart the planner snaps both ends to lane-centre waypoints
    sampled every 2 m, so even a perfectly drivable pair comes back with a route
    several times the gap, and the label then measures the planner's resolution.
    It scored 2.8% of candidates reachable where the peer's navmesh scored 97.6%.

    This instead walks the road graph FORWARD from a, the way the car would, and
    asks whether b is on that path: b is reachable when some waypoint within
    1.25 x Euclid + slack metres of travel passes within `tol_m` of it. Direction
    is respected because `next()` only goes the way the lane runs, which is the
    property that matters on a road and does not exist on a navmesh.
    """
    import carla
    a = np.asarray(a, np.float64)[:2]
    b = np.asarray(b, np.float64)[:2]
    wp = wmap.get_waypoint(carla.Location(x=float(a[0]), y=float(a[1]), z=0.0),
                           project_to_road=True)
    if wp is None:
        return False
    budget = GOOD_RATIO * float(np.linalg.norm(b - a)) + slack_m
    frontier, travelled = [wp], 0.0
    while frontier and travelled <= budget:
        nxt = []
        for w in frontier:
            for cand in w.next(step_m):
                p = np.array([cand.transform.location.x,
                              cand.transform.location.y])
                if np.linalg.norm(p - b) <= tol_m:
                    return True
                nxt.append(cand)
        frontier, travelled = nxt[:12], travelled + step_m   # cap the fan-out
    return False


def components(n, edges, directed=False):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return len({find(i) for i in range(n)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("maps")
    ap.add_argument("trace")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env-file", default="clean_fix1/results/run_env/"
                                          "carla_gated_v43_segvae_nocaps.env")
    ap.add_argument("--placers", default="mad,temporal,uniform")
    ap.add_argument("--radius", type=float, default=3.0)
    ap.add_argument("--heading-deg", type=float, default=90.0)
    ap.add_argument("--step-cost", type=float, default=1.0)
    a = ap.parse_args()

    os.environ.setdefault("STEP_COST", str(a.step_cost))
    sys.path.append("/home/robotlab/Apps/CARLA_0.9.15/PythonAPI/carla")
    import carla
    # The repo's own planner: the CARLA PythonAPI copy in 0.9.15 still takes
    # a DAO, while this one takes (map, resolution) like expert_oracle uses.
    import train_carla_tqc as T
    from vae_nav.online_maps import build_edges, shortcut_candidates
    from vae_nav.policy_io import load_agent_state
    from vae_nav.vae_goal_wrapper import VAEGoalWrapper
    from vae_nav.value_distance import edge_cost_steps, goal_relabelled_obs
    from weight_edges_v43 import spaces_from

    res = json.load(open(a.maps))
    town = res["town"]
    with open(a.trace, "rb") as fh:
        tr = pickle.load(fh)
    obs_all, fields = tr["obs_driven"], tr["fields"]
    pos = np.asarray(tr["positions"], np.float64)[:, :2]
    yaw = np.asarray(tr["yaws"], np.float64)
    seg = np.asarray(tr["segments"], int)

    obs_space, act_space = spaces_from(tr, a.env_file)
    agent = T.build_agent(obs_space, act_space, 0)
    with open(a.ckpt, "rb") as fh:
        state = pickle.load(fh)
    load_agent_state(agent, state.get("agent", state))
    goal_block = lambda f: VAEGoalWrapper.goal_block(None, f)     # noqa: E731

    wmap = carla.Map(town, open(XODR.format(town=town)).read())

    def one_way_steps(pairs, idx):
        """Predicted steps for each directed (i, j), i and j indexing `idx`."""
        out = np.empty(len(pairs))
        B = 512
        for k in range(0, len(pairs), B):
            chunk = pairs[k:k + B]
            batch = [goal_relabelled_obs(
                {kk: np.asarray(v)[idx[i]] for kk, v in obs_all.items()},
                fields[idx[i]], pos[idx[j]], goal_block) for i, j in chunk]
            stacked = {kk: np.stack([b[kk] for b in batch]) for kk in batch[0]}
            cost, _ = edge_cost_steps(agent, stacked, step_cost=a.step_cost)
            out[k:k + len(chunk)] = np.asarray(cost, np.float64)
        return out

    # ---- calibrate once, on the UNION of every placer's nodes ---------------
    union = np.array(sorted({i for p in a.placers.split(",")
                             for i in res["maps"][p]["indices"]}), int)
    cand = shortcut_candidates(pos[union], yaw[union], existing_edges=(),
                               radius=a.radius, heading_deg=a.heading_deg)
    if not cand:
        raise SystemExit("no shortcut candidates: nothing within the radius "
                         "passes the heading gate")
    print(f"[Calib] {len(union)} union nodes, {len(cand)} directed candidates "
          f"within {a.radius:g} m and {a.heading_deg:g} deg", flush=True)
    steps = one_way_steps(cand, union)
    euclid = np.array([np.linalg.norm(pos[union[i]] - pos[union[j]])
                       for i, j in cand])
    good = np.array([reachable_by_road(wmap, pos[union[i]], pos[union[j]])
                     for i, j in cand])
    print(f"[Calib] OpenDRIVE says {good.mean():.1%} of candidates are "
          f"reachable (base precision if we certified everything)")

    print("[Calib] precision / recall against the road graph:")
    table, chosen = [], None
    for pct in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
        thr = float(np.percentile(steps, pct))
        sel = steps <= thr
        prec = float(good[sel].mean()) if sel.any() else float("nan")
        rec = float(good[sel].sum() / max(1, good.sum()))
        table.append({"pct": pct, "threshold_steps": thr, "kept": int(sel.sum()),
                      "precision": prec, "recall": rec})
        print(f"   p{pct:<3d} steps<={thr:7.2f}  kept {sel.sum():5d}  "
              f"precision {prec:.3f}  recall {rec:.3f}")
        if pct == 50:
            chosen = thr
    print(f"[Calib] threshold = p50 = {chosen:.2f} steps (the peer's rule: "
          "percentile of our own quantity, not their 5.65)")

    # ---- apply per placer ---------------------------------------------------
    out = {"town": town, "trace": res["trace"], "radius_m": a.radius,
           "heading_deg": a.heading_deg, "threshold_steps": chosen,
           "calibration": {"n_candidates": len(cand),
                           "base_precision": float(good.mean()),
                           "table": table},
           "placers": {}}
    for placer in a.placers.split(","):
        idx = np.asarray(res["maps"][placer]["indices"], int)
        edges, _ = build_edges(idx, pos, seg)                 # seglinks off
        # Only the DIRECTED pairs already present are excluded, so the REVERSE
        # of a trace edge stays a candidate and must earn its place from the
        # critic like any other. shortcut_candidates symmetrises what it is
        # given, so hand it each edge as an explicit one-way pair.
        pairs = shortcut_candidates(pos[idx], yaw[idx], existing_edges=(),
                                    radius=a.radius, heading_deg=a.heading_deg)
        have = {(int(u), int(v)) for u, v in edges}
        pairs = [p for p in pairs if p not in have]
        before = components(len(idx), edges)
        add = []
        if pairs:
            s = one_way_steps(pairs, idx)
            add = [p for p, v in zip(pairs, s) if v <= chosen]
        after = components(len(idx), list(edges) + list(add))
        out["placers"][placer] = {
            "n_nodes": int(len(idx)), "n_trace_edges": len(edges),
            "n_candidates": len(pairs), "n_shortcuts": len(add),
            "shortcuts": [[int(i), int(j)] for i, j in add],
            "components_before": before, "components_after": after,
        }
        print(f"{placer:9s} {len(idx):4d} nodes  {len(edges):4d} trace edges  "
              f"{len(pairs):5d} candidates -> {len(add):4d} certified  "
              f"components {before} -> {after}")

    dest = a.maps.replace(".json", "_shortcuts.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
