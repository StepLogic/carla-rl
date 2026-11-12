#!/usr/bin/env python3
"""ViNT's topological map from an expert-tour trace (torch_rl.vint_topomap).

    build_topomap.py --trace checkpoints/trace_expert_jx/trace_expert_Town02 \
        --ckpt checkpoints/vint_carla_expert/epoch_1 \
        --out checkpoints/vint_topomap/epoch_1/Town02_topomap.npz

The trace must be ONE continuous drive (every segment continues the last, none
jumped): its drive order is the map's backbone. Frames go through the eval's own
resize (pxr.to_frame to MONO_IMG) before ViNT's crop, exactly as eval frames do.
Writes the .npz (node steps, odometry, 85x64 context crops, edges) and a .json
summary next to it.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))


def load_trace(trace_dir):
    """-> frames (T, H, W, 3) uint8, xy (T, 2), yaw (T,) radians, index."""
    idx = json.load(open(os.path.join(trace_dir, "index.json")))
    segs = sorted(idx["segments"], key=lambda s: s["seg"])
    if len({s["drive"] for s in segs}) != 1 or any(s["jumped"] for s in segs) \
            or not all(s["continues"] for s in segs[1:]):
        raise SystemExit(f"{trace_dir}: not one continuous drive")
    f, p, y = [], [], []
    for s in segs:
        z = np.load(os.path.join(trace_dir, s["file"]))
        f.append(z["frame"])
        p.append(np.asarray(z["pos"], np.float64)[:, :2])
        y.append(np.asarray(z["yaw"], np.float64))
    yaw = np.concatenate(y)
    if np.nanmax(np.abs(yaw)) > 2 * math.pi + 1e-6:          # degrees in some traces
        yaw = np.radians(yaw)
    return np.concatenate(f), np.concatenate(p), yaw, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--ckpt", required=True, help="ViNT checkpoint dir (model.pth + vint.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=10, help="steps per node (create_topomap --dt 1.0)")
    ap.add_argument("--radius-m", type=float, default=8.0)
    ap.add_argument("--min-gap", type=int, default=3)
    ap.add_argument("--q", type=float, default=0.95)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from torch_rl import pixel_rows as pxr
    from torch_rl.vint_driver import load_vint_policy
    from torch_rl.vint_topomap import build_topomap
    frames, xy, yaw, idx = load_trace(a.trace)
    print(f"[Topomap] {a.trace}: {len(frames)} steps, frames {frames.shape[1:]}, "
          f"{idx.get('tour_m')} m tour", flush=True)
    policy, meta = load_vint_policy(a.ckpt, device=a.device)
    m = build_topomap(frames, xy, yaw, policy, every=a.every,
                      context_size=int(meta["context_size"]), stride=int(meta["stride"]),
                      radius_m=a.radius_m, min_gap=a.min_gap, q=a.q,
                      to_frame=lambda f: pxr.to_frame(f, pxr.MONO_IMG).astype(np.uint8),
                      log=lambda s: print(s, flush=True))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, node_step=m["node_step"], node_xy=m["node_xy"], node_yaw=m["node_yaw"],
             ctx85=m["ctx85"], edges=m["edges"])
    summary = {**m["summary"], "trace": a.trace, "ckpt": a.ckpt, "town": idx.get("town"),
               "image_size": meta["image_size"], "context_size": meta["context_size"],
               "stride": meta["stride"], "odometry": "revisit candidates only (disclosed)"}
    with open(os.path.splitext(a.out)[0] + ".json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"[Topomap] wrote {a.out}: {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
