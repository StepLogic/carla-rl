"""Lane-DIRECTION coverage of a map trace: a drivable 2 m waypoint (outside junctions)
counts as covered when some trace sample lies within 2.5 m of it AND heads within 45 deg
of the lane's own direction. Also per lane (road, section, lane id): covered if >= 80% of
its waypoints are. Comparable between the expert tour and a policy drive.

usage: lane_direction_coverage.py TOWN TRACE_DIR [TRACE_DIR ...]
"""
import json
import os
import sys

import carla
import numpy as np
from scipy.spatial import cKDTree

town = sys.argv[1]
xodr = f"/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr"
cmap = carla.Map(town, open(xodr).read())
wps = [w for w in cmap.generate_waypoints(2.0) if not w.is_junction]
wxy = np.array([[w.transform.location.x, w.transform.location.y] for w in wps])
wyaw = np.radians([w.transform.rotation.yaw for w in wps])
lane = [(w.road_id, w.section_id, w.lane_id) for w in wps]

for d in sys.argv[2:]:
    idx = json.load(open(os.path.join(d, "index.json")))
    pos, yaw = [], []
    for s in sorted(idx["segments"], key=lambda s: s["seg"]):
        z = np.load(os.path.join(d, s["file"]))
        pos.append(z["pos"].reshape(-1, 2))
        yaw.append(z["yaw"].reshape(-1))
    pos, yaw = np.concatenate(pos), np.concatenate(yaw)
    if np.nanmax(np.abs(yaw)) > 2 * np.pi + 0.1:      # stored in degrees
        yaw = np.radians(yaw)
    tree = cKDTree(pos)
    covered = np.zeros(len(wps), bool)
    near_any = np.zeros(len(wps), bool)
    for i, nb in enumerate(tree.query_ball_point(wxy, r=2.5)):
        if nb:
            near_any[i] = True
            dyaw = np.abs((yaw[nb] - wyaw[i] + np.pi) % (2 * np.pi) - np.pi)
            covered[i] = bool(np.any(dyaw < np.radians(45)))
    by_lane = {}
    for c, l in zip(covered, lane):
        by_lane.setdefault(l, []).append(c)
    lanes_cov = sum(np.mean(v) >= 0.8 for v in by_lane.values())
    print(f"{os.path.basename(os.path.dirname(d)) or d}: waypoints near the trace (any direction) "
          f"{near_any.mean():.1%}; driven in the lane's own direction {covered.mean():.1%} "
          f"({covered.sum()}/{len(wps)}); lanes >= 80% covered in their direction "
          f"{lanes_cov}/{len(by_lane)}")
