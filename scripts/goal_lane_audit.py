#!/usr/bin/env python3
"""Are goals on the start lane, and does the expert's route change lanes?

Offline, CPU only: builds carla.Map from the OpenDRIVE file (no simulator),
then samples goals EXACTLY as CarlaCore.reset_hero does (pick_origin, then
origin.next(d) -> furthest candidate, 24 tries, then 24 unconstrained tries,
then a random waypoint anywhere). For each goal it records which branch
produced it and asks CARLA's GlobalRoutePlanner -- the planner BasicAgent's
trace_route uses, at its default 2.0 m resolution -- for the route, counting
CHANGELANELEFT/RIGHT options.
usage: goal_lane_audit.py TOWN N
"""
import os
import random
import sys
from collections import Counter

sys.path.append("/home/robotlab/Apps/CARLA_0.9.15/PythonAPI/carla")
sys.path.insert(0, "/home/robotlab/scratch/carla-rl/dependencies/rlib_integration")
import carla  # noqa: E402
from agents.navigation.new_route_planner import GlobalRoutePlanner  # noqa: E402  (what basic_agent.py imports)
from agents.navigation.local_planner import RoadOption  # noqa: E402
from rlib_integration.carla_core import pick_origin  # noqa: E402

town, N = sys.argv[1], int(sys.argv[2])
xodr = open(f"/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr").read()
m = carla.Map(town, xodr)
allw = m.generate_waypoints(2)
wps = [w for w in allw if not w.is_junction]
jns = [w for w in allw if w.is_junction]
lo, hi, step = 5.0, 200.0, 2.5
band = [lo + i * step for i in range(int(round((hi - lo) / step)) + 1)]
grp = GlobalRoutePlanner(m, 2.0)
rng = random.Random(0)
random.seed(0)
kinds, branches, lc_routes, lc_by_kind = Counter(), Counter(), 0, Counter()
same_road_lane = 0
for i in range(N):
    origin, maxf, kind = pick_origin(wps, jns, band, 0.5, 0.3, rng=rng)
    b = band if maxf is None else ([d for d in band if d <= maxf] or [maxf * 0.5])
    dest, how = None, None
    for _ in range(24):
        c = origin.next(rng.choice(b))
        if c:
            dest, how = c[-1], "next"
            break
    if dest is None:
        for _ in range(24):
            c = origin.next(rng.choice(band))
            if c:
                dest, how = c[-1], "next_unbanded"
                break
    if dest is None:
        dest, how = rng.choice(wps), "RANDOM_WAYPOINT"
    kinds[kind] += 1
    branches[how] += 1
    route = grp.trace_route(origin.transform.location, dest.transform.location)
    opts = [o for _, o in route]
    if any(o in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT) for o in opts):
        lc_routes += 1
        lc_by_kind[kind] += 1
print(f"{town}: N={N} origins={dict(kinds)}")
print(f"  goal branch: {dict(branches)}")
print(f"  expert (CARLA GRP) routes with a lane change: {lc_routes}/{N} by origin kind {dict(lc_by_kind)}")
