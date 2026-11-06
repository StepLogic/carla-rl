#!/usr/bin/env python3
"""Single-trace EVALUATION maps (user 2026-11-06: "CAN YOU CREATE MAP FROM A
SINGLE TRACE?" -> "I am talking about evaluations" -> "NO drive with new
policy") from ONE per-step trace per town in run_trace's format: c0's own
deterministic explore trace with every env step kept (slurm/il_trace_step*.env;
~0.4 m apart, faults restart it at the nearest drivable waypoint), or the
expert coverage tour (EXPERT_TRACE, kept but no longer the plan).

At that spacing the Habitat peer's rules apply at their own scale
(habitat-map-spec), so they are used as specified:
  entropy      online MAD (online_maps.online_mad) on the driving policy's
               entropy along the trace: z = 1.0, a 15 m window of TRAVEL,
               warm-up 30 samples, dedupe 0.8 m against ANY placed node, first
               qualifying sample kept (no local-max, no refractory), buffer
               reset per segment (their carry_buffer=0; an expert trace's chunk
               files of one drive count as one segment), guarantee on.
  uniform      image_map.match_uniform: the directional spacing whose node
               count matches entropy's (the benchmark's existing control).
  random       image_map.place_random, seeded, matched count (existing control).
  sgm_matched  (--sgm) image_map.sgm_match with NO spatial gate and no heading
               test over the critic read <out>/sgm_pairs_<Town>.npz
               (train_carla_sac.run_sgm_pairs: candidates thinned at 0.8 m),
               cutoff bisected to entropy's count.
EDGES: image_map.build_graph, the benchmark's rule -- successive node VISITS
along the trace (3 m, heading-aligned), metres driven, carried across a
restart only when it continues the drive (not a stuck/stall jump; within 8 m
and aligned). Within a segment that is the peer's "consecutive kept nodes"
wherever the trace does not revisit; where it revisits a road, the visited
nodes are joined instead of one long consecutive-node edge (the peer's
wormhole). There are no seglinks across jumps.

Output is maps_v30det's format: <out>/<placement>/<Town>.json + _frames.npy.

    build_single_trace_maps.py --trace-fmt 'checkpoints/trace1step_il_bc_c0_s15000_{town}/trace_step_15000' \
        --towns Town02 Town03 --out checkpoints/maps_c0step
    build_single_trace_maps.py --sgm --trace-fmt ... --towns ... --out checkpoints/maps_c0step
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


def load(trace_fmt, town):
    d = trace_fmt.format(town=town)
    with open(os.path.join(d, "index.json")) as fh:
        index = json.load(fh)
    if "done" in index and not index["done"]:
        print(f"WARNING {town}: the trace is INCOMPLETE ({index['segments'][-1].get('end')})",
              flush=True)
    tr = im.load_trace(d)
    tr["drive"] = drive_ids(tr, index)      # run_trace: a drive is a segment
    # WHO DROVE each sample (the unbroken trace, run_unbroken_trace: 0 = c0,
    # 1 = the expert recovering from a fault). Absent in every other trace: all c0.
    drv = []
    for sg in index["segments"]:
        if not sg.get("kept"):
            continue
        with np.load(os.path.join(d, sg["file"])) as z:
            n = len(z["t"])
            drv.append(np.asarray(z["driver"], int).reshape(n) if "driver" in z.files
                       else np.zeros(n, int))
    tr["driver"] = np.concatenate(drv)
    return tr, index


def c0_only(idx, tr):
    """Placed sample indices with the expert-driven ones removed: a node is a
    frame c0 itself drove through (user 2026-11-06, "Expert recovers": expert
    steps are flagged so placement can skip them). The expert's stretch of the
    drive still joins the graph's edges -- that is the point of recovering."""
    idx = np.asarray(idx, int)
    return idx[tr["driver"][idx] == 0]


def c0_view(tr):
    """(sub-trace of c0-driven samples, their indices in the full trace)."""
    k = np.flatnonzero(tr["driver"] == 0)
    return {"pos": tr["pos"][k], "yaw": tr["yaw"][k]}, k


def trace_stats(tr, index):
    step = np.hypot(*np.diff(tr["pos"], axis=0).T)
    same = tr["drive"][1:] == tr["drive"][:-1]
    segs = index["segments"]
    gaps = [float(np.hypot(*(np.array(b["start_xy"]) - np.array(a["stop_xy"]))))
            for a, b in zip(segs, segs[1:])]
    ends = {}
    for sg in segs:
        ends[str(sg.get("end"))] = ends.get(str(sg.get("end")), 0) + 1
    return {"samples": int(len(tr["pos"])), "segments": int(len(set(tr["drive"].tolist()))),
            "km": round(float(sum(sg["metres"] for sg in segs)) / 1000, 2),
            "spacing_m_median": round(float(np.median(step[same])), 3),
            "ends": ends, "jumped": sum(1 for sg in segs if sg.get("jumped")),
            "restart_gap_m_median": round(float(np.median(gaps)), 2) if gaps else None,
            "restart_gap_m_max": round(float(np.max(gaps)), 2) if gaps else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-fmt", required=True,
                    help="trace directory with {town}, e.g. checkpoints/trace1step_il_bc_c0_s15000_{town}/trace_step_15000")
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
        tr, index = load(a.trace_fmt, town)
        ts = trace_stats(tr, index)
        print(f"\n{town}: {json.dumps(ts)}", flush=True)
        ent, st = online_maps.online_mad(tr["entropy"], tr["pos"], tr["drive"],
                                         window_m=a.window_m, z=a.z)
        expert_n = int((tr["driver"] == 1).sum())
        if expert_n:
            before = len(ent)
            ent = c0_only(ent, tr)
            print(f"  expert-driven samples: {expert_n} ({expert_n / len(tr['driver']):.1%}); "
                  f"entropy nodes {before} -> {len(ent)} placed on c0's own frames", flush=True)
        n = len(ent)
        print(f"  online MAD: {n} nodes, window {st['window_samples']} samples = "
              f"{st['window_m']:.1f} m at {st['sample_spacing_m']:.3f} m spacing, "
              f"degenerate windows {st['degenerate_window_frac']:.1%}", flush=True)
        if a.sgm:
            z = np.load(os.path.join(a.out, f"sgm_pairs_{town}.npz"))
            cut, keep, sst = im.sgm_match(z["cand"], z["pairs"], z["q"], z["xy"], z["yaw"],
                                          count=n, gate_m=None, iters=30)
            keep = c0_only(keep, tr)
            rows.append(save(a.out, "sgm_matched", town, tr, keep,
                             {**sst, "cand_m": float(z["cand_m"]), "pair_m": float(z["pair_m"])}))
            continue
        rows.append(save(a.out, "entropy", town, tr, ent,
                         {"z": a.z, "window_m": a.window_m,
                          "window_samples": st["window_samples"],
                          "degenerate_window_frac": st["degenerate_window_frac"],
                          "signal": "entropy of the driving policy (act_with_stats)",
                          "trace": ts}))
        view, k = c0_view(tr)
        spacing, uni = im.match_uniform(view, n)
        uni = k[uni]
        rows.append(save(a.out, "uniform", town, tr, uni, {"spacing_m": round(spacing, 3)}))
        rnd = k[im.place_random(view, n, seed=a.seed)]
        rows.append(save(a.out, "random", town, tr, rnd, {"seed": a.seed}))
    name = f"maps_summary{'_sgm' if a.sgm else ''}_{'_'.join(a.towns)}.json"   # per town set
    with open(os.path.join(a.out, name), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
