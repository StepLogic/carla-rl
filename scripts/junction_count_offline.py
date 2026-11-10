#!/usr/bin/env python3
"""Key nodes inside a junction, offline from the saved OpenDRIVE (no server, no GPU).
carla.Map(name, xodr) + get_waypoint(..., project_to_road=True, Driving).is_junction,
the same test as scripts/junction_node_count.py."""
import json, sys
import carla
ROOT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/maps_expert_jx_img_z25"
out = {}
for T in ("Town02", "Town03"):
    cmap = carla.Map(T, open(f"logs/junctions_{T}.xodr").read())
    def frac(xy):
        n = 0
        for x, y in xy:
            wp = cmap.get_waypoint(carla.Location(x=float(x), y=float(y), z=0.0),
                                   project_to_road=True, lane_type=carla.LaneType.Driving)
            n += bool(wp is not None and wp.is_junction)
        return n
    for pl in ("entropy", "uniform", "sgm_matched"):
        N = json.load(open(f"{ROOT}/{pl}/{T}.json"))["nodes"]
        K = [n["pos"][:2] for n in N if n.get("kind", "key") == "key"]
        k = frac(K)
        out[(T, pl)] = (k, len(K))
        print(f"{T} {pl:12s} {k}/{len(K)} ({k/len(K):.1%})", flush=True)
json.dump({f"{t}|{p}": v for (t, p), v in out.items()}, open("logs/junction_counts_offline.json", "w"))
