#!/usr/bin/env python3
"""How many map KEY nodes lie in a junction, entropy vs uniform placement,
using CARLA's own junction test: map.get_waypoint(node position,
project_to_road=True, Driving).is_junction. The base rate is the same test over
the trace's candidate frames (every 0.5 m). Also saves the town's OpenDRIVE
(<out>_<Town>.xodr) for later offline use.

    junction_node_count.py --town Town02 --maps checkpoints/maps_expert_jx_img \
        --trace checkpoints/trace_expert_jx/trace_expert_Town02 --out logs/junctions
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from torch_rl.carla_server import CarlaServer  # noqa: E402
from vae_nav import image_map as im  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", required=True); ap.add_argument("--maps", required=True)
    ap.add_argument("--trace", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8800)
    a = ap.parse_args()
    import carla
    srv = CarlaServer(a.port, quality="Low", log_dir=os.path.dirname(os.path.abspath(a.out)))
    srv.start(); srv.wait_ready()
    try:
        c = carla.Client("localhost", a.port); c.set_timeout(120.0)
        world = c.load_world(a.town); cmap = world.get_map()
        open(f"{a.out}_{a.town}.xodr", "w").write(cmap.to_opendrive())

        def in_junction(xy):
            out = []
            for x, y in np.asarray(xy, float):
                wp = cmap.get_waypoint(carla.Location(x=float(x), y=float(y), z=0.0),
                                       project_to_road=True, lane_type=carla.LaneType.Driving)
                out.append(bool(wp is not None and wp.is_junction))
            return np.array(out)
        tr = im.load_trace(a.trace)
        S = im.place_uniform(tr, 0.5)
        base = in_junction(tr["pos"][S])
        res = {"town": a.town, "candidates": int(len(S)), "candidates_in_junction": round(float(base.mean()), 4)}
        print(f"[Junction] {a.town}: {base.mean():.1%} of the trace's candidate frames are in a junction", flush=True)
        for pl in ("entropy", "uniform", "random"):
            p = os.path.join(a.maps, pl, f"{a.town}.json")
            if not os.path.exists(p):
                continue
            m = json.load(open(p))
            P = np.array([n["pos"] for n in m["nodes"]]); kind = np.array([n.get("kind", "key") for n in m["nodes"]])
            j = in_junction(P)
            for kk in ("key", "image_gap"):
                sel = kind == kk
                if sel.any():
                    res[f"{pl}_{kk}"] = {"nodes": int(sel.sum()), "in_junction": int(j[sel].sum()),
                                         "share": round(float(j[sel].mean()), 4)}
                    print(f"[Junction] {a.town} {pl:8s} {kk:9s}: {int(j[sel].sum())} of {int(sel.sum())} in a "
                          f"junction ({j[sel].mean():.1%}; {j[sel].mean() / max(base.mean(), 1e-9):.1f}x the trace's share)",
                          flush=True)
        json.dump(res, open(f"{a.out}_{a.town}.json", "w"), indent=1)
    finally:
        srv.stop()


if __name__ == "__main__":
    main()
