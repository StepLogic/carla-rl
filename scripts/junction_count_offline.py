#!/usr/bin/env python3
"""Key nodes inside a junction, offline from the saved OpenDRIVE (no server, no GPU):
carla.Map(name, xodr) + get_waypoint(..., project_to_road=True, Driving).is_junction,
the same test as scripts/junction_node_count.py.

Reports the trace base rate (candidates at --cand-m), the placements stored in the
map root, and RANDOM reproduced here with the builder's rule (im.place_random on the
candidates, seed 0), plus the matched uniform spacing from im.match_uniform.

    junction_count_offline.py checkpoints/maps_expert_jx_img_z25
"""
import glob, json, os, sys
import numpy as np
import carla
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from vae_nav import image_map as im  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/maps_expert_jx_img_z25"
TRACE = "checkpoints/trace_expert_jx/trace_expert_{T}"
CAND_M, SEED = 0.5, 0
out = {}
for T in ("Town02", "Town03"):
    cmap = carla.Map(T, open(f"logs/junctions_{T}.xodr").read())

    def n_junction(xy):
        n = 0
        for x, y in np.asarray(xy, float):
            wp = cmap.get_waypoint(carla.Location(x=float(x), y=float(y), z=0.0),
                                   project_to_road=True, lane_type=carla.LaneType.Driving)
            n += bool(wp is not None and wp.is_junction)
        return n

    pos, yaw = [], []
    for f in sorted(glob.glob(TRACE.format(T=T) + "/seg_*.npz")):
        with np.load(f) as z:
            pos.append(z["pos"]); yaw.append(z["yaw"])
    tr = {"pos": np.concatenate(pos), "yaw": np.concatenate(yaw)}
    S = im.place_uniform(tr, CAND_M)
    cand = {"pos": tr["pos"][S], "yaw": tr["yaw"][S]}
    base = n_junction(cand["pos"])
    print(f"{T}: {len(tr['pos'])} frames -> {len(S)} candidates at {CAND_M} m | "
          f"trace base rate {base}/{len(S)} ({base/len(S):.1%})", flush=True)
    out[f"{T}|Trace"] = (base, len(S))

    n_keys = None
    for pl in ("entropy", "uniform", "sgm_matched"):
        N = json.load(open(f"{ROOT}/{pl}/{T}.json"))["nodes"]
        K = [n["pos"][:2] for n in N if n.get("kind", "key") == "key"]
        if pl == "entropy":
            n_keys = len(K)
        k = n_junction(K); out[f"{T}|{pl}"] = (k, len(K))
        print(f"  {pl:12s} {k}/{len(K)} ({k/len(K):.1%})", flush=True)

    sp, uni = im.match_uniform(cand, n_keys)
    rnd = im.place_random(cand, n_keys, seed=SEED)
    k = n_junction(cand["pos"][np.asarray(rnd, int)])
    out[f"{T}|random"] = (k, len(rnd))
    print(f"  {'random':12s} {k}/{len(rnd)} ({k/len(rnd):.1%}) | matched uniform spacing "
          f"{sp:.2f} m ({len(uni)} nodes)", flush=True)
    out[f"{T}|spacing"] = (round(float(sp), 2), len(uni))
json.dump(out, open("logs/junction_counts_offline.json", "w"), indent=1)
