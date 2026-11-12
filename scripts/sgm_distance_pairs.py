#!/usr/bin/env python3
"""The SGM pair read with the ViNT-style distance function
(scripts/train_vint_distance.py), in place of v30's goal critic.

Exactly run_sgm_pairs' candidate rule: candidates = the trace thinned at 0.8 m
(directional), every heading-aligned (<= 90 deg) ordered pair within 25 m,
including (a, a). q = the predicted expected temporal distance (in strided
frames). Writes <out>/sgm_pairs_<Town>.npz for build_image_sim_maps
--placements sgm_matched.

    sgm_distance_pairs.py --trace checkpoints/trace_expert_jx/trace_expert_Town02 \
        --town Town02 --model checkpoints/sgm_distance --out checkpoints/maps_expert_jx_img
"""
import argparse
import json
import os
import sys
import importlib.util

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from vae_nav import image_map as im  # noqa: E402
_spec = importlib.util.spec_from_file_location("tvd", os.path.join(HERE, "train_vint_distance.py"))
tvd = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(tvd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True); ap.add_argument("--town", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cand-m", type=float, default=0.8); ap.add_argument("--pair-m", type=float, default=25.0)
    a = ap.parse_args()
    dev = "cuda"
    meta = json.load(open(os.path.join(a.model, "meta.json")))
    head = tvd.DistanceHead(768, meta["max_cls"] + 2).to(dev)
    head.load_state_dict(torch.load(os.path.join(a.model, "head.pt"), map_location=dev)); head.eval()
    tr = im.load_trace(a.trace)
    cand = im.place_uniform(tr, a.cand_m)
    xy, yaw = tr["pos"][cand], tr["yaw"][cand]
    frames = im.node_frames(tr, cand)
    feats = tvd.embed(tvd.dino(dev), frames, dev).float().to(dev)
    pairs = []
    for i in range(len(cand)):
        d = np.hypot(*(xy - xy[i]).T)
        ok = (d <= a.pair_m) & im._aligned(yaw, yaw[i], np.radians(90.0))
        pairs.extend((i, int(j)) for j in np.flatnonzero(ok))
    pairs = np.asarray(pairs, int)
    q = np.empty(len(pairs), np.float32)
    with torch.no_grad():
        for k in range(0, len(pairs), 65536):
            p = pairs[k:k + 65536]
            lg = head(feats[p[:, 0]], feats[p[:, 1]])
            q[k:k + 65536] = tvd.expected_distance(lg, meta["max_cls"], meta["far_value"]).cpu().numpy()
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"sgm_pairs_{a.town}.npz")
    np.savez(out, cand=cand, pairs=pairs, q=q, xy=xy, yaw=yaw, cand_m=a.cand_m, pair_m=a.pair_m,
             trace=a.trace, distance_model=a.model)
    self_d = q[pairs[:, 0] == pairs[:, 1]]
    print(f"[SGMDist] {a.town}: {len(cand)} candidates, {len(pairs)} pairs; predicted distance "
          f"median {np.median(q):.2f} frames, self-pairs median {np.median(self_d):.2f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
