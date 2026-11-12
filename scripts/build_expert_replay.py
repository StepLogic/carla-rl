#!/usr/bin/env python3
"""Build learner C's expert replay from c0's own demonstrations (torch_rl/frl/expert.py; user 2026-11-10: "can
we use c0 expert data locally with run like fastrlap but rl collects data like now").

Every expert-driven M1 shard of the given towns, minus the steering probe's held-out episodes (taken from the
probe cache's own episode list when one exists), goes through c0's frozen trunk at the run's action repeat k,
with its rewards relabelled to the run's reward (idle charge on, explore junction bonus off) and the no_arrival
cut. The output .npz records k and c0's md5; learner C refuses it for any other k or trunk.

usage: python scripts/build_expert_replay.py --c0 DIR --data IL_DATA_ROOT --towns Town01,Town05 --k 2 \
           --out FILE.npz [--caches DIR] [--idle-penalty 0.5] [--idle-kmh 2.0] [--junction-bonus 10] \
           [--device cuda]
"""
import argparse
import glob
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "src"), os.path.join(REPO, "scripts")]
os.environ.setdefault("MONO_CHW", "1")      # c0's row layout
os.environ.setdefault("SPEED_OBS", "1")

import numpy as np  # noqa: E402


def held_out_names(cache_path):
    """The probe's held-out shard names, from its cache (exactly the probe's episodes)."""
    import offline_priv_critic as opc
    C = dict(np.load(cache_path, allow_pickle=False))
    names = C["e_path"]
    order = np.argsort(names)
    held = set(order[:3].tolist()) | set(order[::10].tolist())
    _, probes = opc.held_out_split(C)               # same rule; asserted below
    n_probe = sum(len(set(C["f_ep"][fr].tolist())) for fr, _ in probes.values())
    assert n_probe == len(held), (n_probe, len(held))
    return {str(names[i]) for i in held}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c0", required=True)
    ap.add_argument("--data", required=True, help="il_data root with <Town>/ep_*.npz")
    ap.add_argument("--towns", default="Town01,Town05")
    ap.add_argument("--k", type=int, required=True, help="the run's ACTION_REPEAT")
    ap.add_argument("--out", required=True)
    ap.add_argument("--caches", default="", help="dir with features_<Town>.npz (probe hold-out)")
    ap.add_argument("--idle-penalty", type=float, default=0.5, help="the run's EXPLORE_IDLE_PENALTY")
    ap.add_argument("--idle-kmh", type=float, default=2.0, help="the run's EXPLORE_IDLE_SPEED_KMH")
    ap.add_argument("--junction-bonus", type=float, default=10.0,
                    help="EXPLORE_JUNCTION_BONUS the shards were RECORDED with (removed)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from torch_rl.frl import expert as fx
    from torch_rl.frl.carla import C0Trunk, params_md5
    trunk = C0Trunk(args.c0, args.device)
    parts, meta = [], {"towns": {}, "k": args.k, "idle_penalty": args.idle_penalty,
                        "idle_kmh": args.idle_kmh, "junction_bonus_removed": args.junction_bonus,
                        "arrival_cut": fx.ARRIVAL_CUT, "data": args.data}
    t0 = time.time()
    for town in args.towns.split(","):
        paths = sorted(glob.glob(os.path.join(args.data, town, "ep_*.npz")))
        names = [os.path.basename(p) for p in paths]
        drivers = [str(np.load(p, allow_pickle=False)["driver"]) for p in paths]
        cache = os.path.join(args.caches, f"features_{town}.npz") if args.caches else ""
        held = held_out_names(cache) if cache and os.path.exists(cache) else None
        keep = set(fx.select_episodes(names, drivers, held_out=held))
        n_tr = 0
        for p, n in zip(paths, names):
            if n not in keep:
                continue
            z = np.load(p, allow_pickle=False)
            tr = fx.episode_transitions(trunk, {key: z[key] for key in z.files}, k=args.k,
                                        idle_penalty=args.idle_penalty, idle_kmh=args.idle_kmh,
                                        junction_bonus=args.junction_bonus)
            if len(tr["rewards"]):
                parts.append(tr)
                n_tr += len(tr["rewards"])
        meta["towns"][town] = {"shards": len(paths), "expert": int(sum(d == "expert" for d in drivers)),
                               "used": len(keep), "probe_held_out": "cache" if held else "rule",
                               "transitions": n_tr}
        print(f"[expert] {town}: {len(paths)} shards, {meta['towns'][town]['expert']} expert, "
              f"{len(keep)} used -> {n_tr} transitions ({time.time() - t0:.0f} s)", flush=True)
    tr = fx.concat(parts)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp.npz"
    fx.save_replay(tmp, tr, k=args.k, c0_md5=params_md5(args.c0), meta=meta)
    os.replace(tmp, args.out)
    r = tr["rewards"]
    print(f"[expert] {len(r)} transitions at k={args.k} -> {args.out} | reward mean {r.mean():.3f} "
          f"min {r.min():.2f} max {r.max():.2f} | terminal masks {int((tr['masks'] == 0).sum())} | "
          f"|steer| mean {np.abs(tr['actions'][:, 0]).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
