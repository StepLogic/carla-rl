#!/usr/bin/env python3
"""Single-trace EVALUATION maps (user 2026-11-06: "CAN YOU CREATE MAP FROM A
SINGLE TRACE?" -> "I am talking about evaluations") from the expert's closed
coverage-tour trace (EXPERT_TRACE, slurm/expert_trace.env): one continuous
drive per town, a sample every 0.1 s step (~0.4 m at 15 km/h).

At that spacing the Habitat peer's rules apply at their own scale
(habitat-map-spec), so they are used as specified:
  entropy      online MAD (online_maps.online_mad) on the clone's entropy along
               the drive: z = 1.0, a 15 m window of TRAVEL, warm-up 30 samples,
               dedupe 0.8 m against ANY placed node, first qualifying sample
               kept (no local-max, no refractory), buffer reset per DRIVE
               (a drive is one uninterrupted run of the car; the chunk files
               of one drive are one segment), guarantee on.
  uniform      image_map.match_uniform: the directional spacing whose node
               count matches entropy's (the benchmark's existing control).
  random       image_map.place_random, seeded, matched count (existing control).
  sgm_matched  (--sgm) image_map.sgm_match with NO spatial gate and no heading
               test over the critic read <out>/sgm_pairs_<Town>.npz
               (train_carla_sac.run_sgm_pairs: candidates thinned at 0.8 m),
               cutoff bisected to entropy's count.
EDGES: image_map.build_graph, the benchmark's rule -- successive node VISITS
along the drive (3 m, heading-aligned), metres driven. On one continuous trace
that is the peer's "consecutive kept nodes" wherever the tour does not revisit,
and where it revisits a road the visited nodes are joined instead of one
consecutive-node edge spanning the revisit (the wormhole the peer measured).
No seglinks: a single drive has no respawn to link.

Output is maps_v30det's format: <out>/<placement>/<Town>.json + _frames.npy.

    build_single_trace_maps.py --trace-root checkpoints/trace_expert --towns Town02 Town03 \
        --out checkpoints/maps_single
    build_single_trace_maps.py --sgm --trace-root ... --towns ... --out checkpoints/maps_single
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from build_image_maps import save  # noqa: E402
from torch_rl.expert_trace import drive_ids  # noqa: E402
from vae_nav import image_map as im  # noqa: E402
from vae_nav import online_maps  # noqa: E402


def load(root, town):
    d = os.path.join(root, f"trace_expert_{town}")
    with open(os.path.join(d, "index.json")) as fh:
        index = json.load(fh)
    if not index.get("done"):
        print(f"WARNING {town}: the trace is INCOMPLETE ({index['segments'][-1].get('end')})",
              flush=True)
    tr = im.load_trace(d)
    tr["drive"] = drive_ids(tr, index)
    return tr, index


def trace_stats(tr, index):
    step = np.hypot(*np.diff(tr["pos"], axis=0).T)
    same = tr["drive"][1:] == tr["drive"][:-1]
    return {"samples": int(len(tr["pos"])), "drives": int(len(set(tr["drive"].tolist()))),
            "metres": round(float(step[same].sum()), 1),
            "spacing_m_median": round(float(np.median(step[same])), 3),
            "tour_m": index.get("tour_m"), "lanes": index.get("lanes"),
            "done": bool(index.get("done")), "faults_steps": index.get("faults")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-root", default="checkpoints/trace_expert")
    ap.add_argument("--towns", nargs="+", default=["Town02", "Town03"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--z", type=float, default=1.0)
    ap.add_argument("--window-m", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sgm", action="store_true",
                    help="build only sgm_matched from <out>/sgm_pairs_<Town>.npz")
    a = ap.parse_args()
    rows = []
    for town in a.towns:
        tr, index = load(a.trace_root, town)
        ts = trace_stats(tr, index)
        print(f"\n{town}: {json.dumps(ts)}", flush=True)
        ent, st = online_maps.online_mad(tr["entropy"], tr["pos"], tr["drive"],
                                         window_m=a.window_m, z=a.z)
        n = len(ent)
        print(f"  online MAD: {n} nodes, window {st['window_samples']} samples = "
              f"{st['window_m']:.1f} m at {st['sample_spacing_m']:.3f} m spacing, "
              f"degenerate windows {st['degenerate_window_frac']:.1%}", flush=True)
        if a.sgm:
            z = np.load(os.path.join(a.out, f"sgm_pairs_{town}.npz"))
            cut, keep, sst = im.sgm_match(z["cand"], z["pairs"], z["q"], z["xy"], z["yaw"],
                                          count=n, gate_m=None, iters=30)
            rows.append(save(a.out, "sgm_matched", town, tr, keep,
                             {**sst, "cand_m": float(z["cand_m"]), "pair_m": float(z["pair_m"])}))
            continue
        rows.append(save(a.out, "entropy", town, tr, ent,
                         {"z": a.z, "window_m": a.window_m,
                          "window_samples": st["window_samples"],
                          "degenerate_window_frac": st["degenerate_window_frac"],
                          "signal": "entropy (clone c0, GRU carried along the drive)",
                          "segments": "drives", "trace": ts}))
        spacing, uni = im.match_uniform(tr, n)
        rows.append(save(a.out, "uniform", town, tr, uni, {"spacing_m": round(spacing, 3)}))
        rnd = im.place_random(tr, n, seed=a.seed)
        rows.append(save(a.out, "random", town, tr, rnd, {"seed": a.seed}))
    name = "maps_summary_sgm.json" if a.sgm else "maps_summary.json"
    with open(os.path.join(a.out, name), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
