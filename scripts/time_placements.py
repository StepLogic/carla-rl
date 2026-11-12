#!/usr/bin/env python3
"""Wall-clock of each KEY-NODE PLACEMENT on the same inputs (user 2026-11-09:
"compare the computation time for entropy map and other maps"). Shared steps
(trace, AnyLoc descriptors, gap nodes, edges) are timed once and reported apart.

    time_placements.py --trace checkpoints/trace_expert_jx/trace_expert_Town02 --town Town02 \
        --pairs checkpoints/maps_expert_jx_img/sgm_pairs_Town02.npz
"""
import argparse, os, sys, time, importlib.util, json
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src")); sys.path.insert(0, os.path.dirname(HERE))
from vae_nav import image_map as im, online_maps  # noqa: E402
from torch_rl.expert_trace import drive_ids  # noqa: E402
spec = importlib.util.spec_from_file_location("anyloc", os.path.join(HERE, "anyloc_similarity.py"))
anyloc = importlib.util.module_from_spec(spec); spec.loader.exec_module(anyloc)

ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True); ap.add_argument("--town", required=True); ap.add_argument("--pairs", required=True)
ap.add_argument("--repeats", type=int, default=3)
a = ap.parse_args()
dev = "cuda"
T = {}
def tick(name, fn, reps=1):
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter(); out = fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    T[name] = min(ts); return out

tr = im.load_trace(a.trace); index = json.load(open(os.path.join(a.trace, "index.json")))
drive = drive_ids(tr, index)
S = im.place_uniform(tr, 0.5)
pos, yaw, ent, drv = tr["pos"][S], tr["yaw"][S], tr["entropy"][S], drive[S]
frames = im.node_frames(tr, S)
rng = np.random.default_rng(0)

def descriptors():
    enc = anyloc.ValueFacet(dev)
    vf = rng.choice(len(frames), size=min(2000, len(frames)), replace=False)
    pool = torch.cat([enc.patches(frames[vf[i:i + 128]]).reshape(-1, 768) for i in range(0, len(vf), 128)])
    pool = pool[torch.randperm(len(pool), generator=torch.Generator().manual_seed(0))[:200000].to(pool.device)]
    C = anyloc.spherical_kmeans(pool, 32, 30, 0)
    return torch.cat([anyloc.vlad(enc.patches(frames[i:i + 128]), C).half() for i in range(0, len(frames), 128)])
D = tick("SHARED anyloc descriptors (all maps)", descriptors)
dedupe = 0.653 if a.town == "Town02" else 0.677

def entropy_keys():
    mad, _ = online_maps.online_mad(ent, pos, drv, window_m=15.0, z=1.0)
    keys = []
    Dm = D[np.asarray(mad)].float()
    for r, k in enumerate(mad):
        if not keys or float((Dm[r:r + 1] @ D[np.asarray(keys)].float().T).max()) < dedupe:
            keys.append(int(k))
    return keys
keys = tick("entropy: online MAD + image dedupe", entropy_keys, a.repeats)
n = len(keys); view = {"pos": pos, "yaw": yaw}
tick("uniform: spacing search to n", lambda: im.match_uniform(view, n), a.repeats)
z = np.load(a.pairs)
tick("SGM: cutoff search to n (pairs given)",
     lambda: im.sgm_match(z["cand"], z["pairs"], z["q"], z["xy"], z["yaw"], count=n, gate_m=None, iters=30), a.repeats)
print(json.dumps({"town": a.town, "candidates": len(S), "frames": len(tr["pos"]), "entropy_keys": n,
                  "sgm_pairs": int(len(z["pairs"])), "seconds": {k: round(v, 3) for k, v in T.items()}}, indent=1))
