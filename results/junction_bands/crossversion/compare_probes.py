"""Cross-version check: the same Town02 seeds probed on the cluster (CARLA 0.9.16) and
locally (0.9.15) -- start and goal poses, route length, turns and junctions per seed."""
import json
import math
import sys

local_path, cluster_path = sys.argv[1], sys.argv[2]
loc = json.load(open(local_path))
clu = {r["seed"]: r for r in json.load(open(cluster_path))}
worst = {"start": 0.0, "goal": 0.0, "route": 0.0}
same_turns = same_junc = 0
for r in loc:
    c = clu[r["seed"]]
    worst["start"] = max(worst["start"], math.dist(r["start_xy"], c["start_xy"]))
    worst["goal"] = max(worst["goal"], math.dist(r["goal_xy"], c["goal_xy"]))
    worst["route"] = max(worst["route"], abs((r["route_m"] or 0) - (c["route_m"] or 0)))
    same_turns += r["turns"] == c["turns"]
    same_junc += r["junctions"] == c["junctions"]
print(f"{len(loc)} seeds | worst start diff {worst['start']:.3f} m, worst goal diff "
      f"{worst['goal']:.3f} m, worst route_m diff {worst['route']:.3f} m | turns equal "
      f"{same_turns}/{len(loc)}, junctions equal {same_junc}/{len(loc)}")
