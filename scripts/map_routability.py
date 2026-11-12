#!/usr/bin/env python3
"""Offline routability of the benchmark episodes on each image map: how many
episodes get a start node, a goal node and a route, by placement x town x bin,
for several localisation radii. Uses only poses (no success rates), so it can
set the localisation radius without looking at outcomes.

    map_routability.py --maps maps/v30det --probes 'checkpoints/bench_pose_*/eval_step_300000/probe.json'
"""
import argparse
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from vae_nav import image_map as im  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--episodes", default="logs/bench_long_episodes.json")
    ap.add_argument("--radii", nargs="+", type=float, default=[8, 12, 16, 25])
    ap.add_argument("--placements", nargs="+", default=["entropy", "uniform", "random", "sgm"])
    a = ap.parse_args()
    bins = {(e["town"], e["seed"]): e["bin"] for e in json.load(open(a.episodes))}
    eps = []
    for f in sorted(glob.glob(a.probes)):
        for p in json.load(open(f)):
            if (p["town"], p["seed"]) in bins and p.get("start_xy"):
                eps.append({**p, "bin": bins[(p["town"], p["seed"])]})
    print(f"{len(eps)} episodes with poses")
    for pl in a.placements:
        for r in a.radii:
            tot = collections.Counter()
            why = collections.Counter()
            per = collections.defaultdict(lambda: [0, 0])
            for town in sorted({e["town"] for e in eps}):
                f = os.path.join(a.maps, pl, f"{town}.json")
                if not os.path.exists(f):
                    continue
                g = im.Graph.load(f)
                for e in (x for x in eps if x["town"] == town):
                    s = g.nearest(e["start_xy"], e["start_yaw"], radius=r)
                    gn = g.goal_node(s, e["goal_xy"], radius=r) if s is not None else None
                    ok = g.path(s, gn) is not None
                    w = "ok" if ok else ("no start" if s is None else "no goal" if gn is None else "no route")
                    why[w] += 1
                    per[(town, e["bin"])][0] += ok
                    per[(town, e["bin"])][1] += 1
                    tot[ok] += 1
            n = sum(tot.values())
            if not n:
                continue
            cells = " ".join(f"{t[-4:]}/{b[0]}={k / m:.2f}" for (t, b), (k, m) in sorted(per.items()))
            print(f"{pl:8s} r={r:4.0f}  routable {tot[True] / n:.3f}  {dict(why)}  | {cells}")


if __name__ == "__main__":
    main()
