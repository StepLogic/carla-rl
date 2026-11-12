#!/usr/bin/env python3
"""AnyLoc descriptors for one evaluation map, for pose-free navigation
(vae_nav.image_nav; EVAL_MAP_LOCALISE=image).

Writes <map dir>/<Town>_anyloc.npz:
  C    (32, 768)      the map's own vocabulary, rebuilt exactly as
                      build_image_sim_maps builds it (candidates every 0.5 m,
                      <= 2,000 seeded random frames, 200k patches, seed 0)
  D    (N, 24576)     fp16 VLAD of every node's stored image, in node order
  thr  float          the arrival threshold: this map's 2% far-false point
                      (meta far_false_2pct), the procedure Habitat used for 0.54

    build_map_descriptors.py --map checkpoints/maps_expert_jx_img/uniform/Town02.json \
        --trace checkpoints/trace_expert_jx/trace_expert_Town02
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from vae_nav import anyloc  # noqa: E402
from vae_nav import image_map as im  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--trace", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = json.load(open(a.map))
    thr = m["meta"].get("far_false_2pct")
    if thr is None:
        raise SystemExit(f"{a.map}: no far_false_2pct in meta -- build it with build_image_sim_maps")
    tr = im.load_trace(a.trace)
    S = im.place_uniform(tr, 0.5)
    rng = np.random.default_rng(0)
    vf = rng.choice(len(S), size=min(2000, len(S)), replace=False)
    enc = anyloc.ValueFacet(dev)
    fr = im.node_frames(tr, S[vf])
    pool = torch.cat([enc.patches(fr[i:i + 128]).reshape(-1, 768) for i in range(0, len(fr), 128)])
    pool = pool[torch.randperm(len(pool), generator=torch.Generator().manual_seed(0))[:200000].to(pool.device)]
    C = anyloc.spherical_kmeans(pool, 32, 30, 0)
    frames = np.load(a.map.replace(".json", "_frames.npy"))
    if len(frames) != len(m["nodes"]):
        raise SystemExit(f"{len(frames)} frames for {len(m['nodes'])} nodes")
    D = anyloc.describe(enc, C, frames).half().cpu().numpy()
    out = a.map.replace(".json", "_anyloc.npz")
    np.savez(out, C=C.cpu().numpy(), D=D, thr=float(thr), trace=a.trace, map=a.map)
    print(f"[MapDesc] {a.map}: {D.shape[0]} nodes x {D.shape[1]}, thr {thr} -> {out}", flush=True)


if __name__ == "__main__":
    main()
