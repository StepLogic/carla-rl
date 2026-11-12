#!/usr/bin/env python3
"""Add the junction-turn DIRECTIONS to an episode file (junction bands, 2026-11-12).

For every row, the start and goal come from its probe row (same town and seed); the
route is rebuilt offline from the town's OpenDRIVE file with carla.Map and the project's
GlobalRoutePlanner at 2 m -- the planner and resolution the probe used live -- and
vae_nav.route_turns.turn_runs reads its turns. The rebuilt turn COUNT must equal the
count the probe stored for that seed; if any row disagrees, nothing is written.

Adds to each row: turn_dirs (e.g. "LR"), contains_left (bool), mixed_runs.

    annotate_turn_directions.py EPISODES.json OUT.json --probe P1.json [P2.json ...] \
        --xodr-dir /path/to/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "dependencies", "rlib_integration")):
    if p not in sys.path:
        sys.path.insert(0, p)

ap = argparse.ArgumentParser()
ap.add_argument("episodes")
ap.add_argument("out")
ap.add_argument("--probe", nargs="+", required=True)
ap.add_argument("--xodr-dir", required=True)
a = ap.parse_args()

import carla  # noqa: E402
from src.global_route_planner import GlobalRoutePlanner  # noqa: E402
from vae_nav.route_turns import turn_runs  # noqa: E402

probes = {}
for f in a.probe:
    for r in json.load(open(f)):
        probes[(r["town"], r["seed"])] = r
eps = json.load(open(a.episodes))

planners = {}
bad = []
for e in eps:
    town = e["town"]
    if town not in planners:
        xodr = open(os.path.join(a.xodr_dir, f"{town}.xodr")).read()
        planners[town] = GlobalRoutePlanner(carla.Map(town, xodr), 2.0)
    p = probes.get((town, e["seed"]))
    if p is None:
        bad.append(f"{town} seed {e['seed']}: no probe row")
        continue
    s, g = p["start_xy"], p["goal_xy"]
    route = planners[town].trace_route(carla.Location(float(s[0]), float(s[1]), 0.0),
                                       carla.Location(float(g[0]), float(g[1]), 0.0))
    t = turn_runs(opt for _, opt in route)
    if t["turns"] != e["turns"]:
        bad.append(f"{town} seed {e['seed']}: rebuilt {t['turns']} turns ({t['seq']}), "
                   f"probe stored {e['turns']}")
        continue
    e["turn_dirs"], e["contains_left"], e["mixed_runs"] = t["seq"], "L" in t["seq"], t["mixed_runs"]

if bad:
    print(f"VALIDATION FAILED on {len(bad)} of {len(eps)} rows; nothing written:")
    for b in bad[:20]:
        print("  " + b)
    sys.exit(1)
json.dump(eps, open(a.out, "w"), indent=1)
n_left = sum(e["contains_left"] for e in eps)
print(f"{a.out}: {len(eps)} rows, rebuilt turn count equals the probe's on all; "
      f"contains a left turn {n_left}/{len(eps)}; mixed runs {sum(e['mixed_runs'] for e in eps)}")
