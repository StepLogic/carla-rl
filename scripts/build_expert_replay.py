#!/usr/bin/env python3
"""Build learner C's expert replay from c0's own demonstrations (torch_rl/frl/expert.py; user 2026-11-10: "can
we use c0 expert data locally with run like fastrlap but rl collects data like now").

Every expert-driven M1 shard of the given towns, minus the steering probe's held-out episodes (taken from the
probe cache's own episode list when one exists), goes through c0's frozen trunk at the run's action repeat k,
with its rewards relabelled to the run's reward (idle charge on, explore junction bonus off) and the no_arrival
cut. The output .npz records k and c0's md5; learner C refuses it for any other k or trunk.

--control DIR instead of --c0 (user 2026-11-14: "Queue online train for best fixes with SAC with expert pinned
rows") builds the SAME rows through the pretrained control stream (torch_rl.frl.carla.ControlTrunk, a
scripts/pretrain_control.py checkpoint dir): the same episodes, windows, executed actions, rewards and cut --
only the features differ (expert.episode_transitions_control). The .npz then records THAT checkpoint's md5,
which is the trunk the run checks, and the stream's frame stack, decision rate and input size. The stream must
have been pretrained at --k.

--data takes one root or several, comma-separated; the expert filter and the hold-out rule are applied inside
each root (recordings number their episodes independently).

usage: python scripts/build_expert_replay.py (--c0 DIR | --control DIR) --data ROOT[,ROOT...] \
           --towns Town01,Town05 --k 2 --out FILE.npz [--caches DIR] [--idle-penalty 0.5] [--idle-kmh 2.0] \
           [--junction-bonus 10] [--device cuda]
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


def control_args(control_dir):
    """The pretraining arguments a control-stream checkpoint was saved with (scripts/pretrain_control.py)."""
    import torch
    return dict(torch.load(os.path.join(control_dir, "params.pt"), map_location="cpu")["args"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c0", default="", help="c0's checkpoint dir: rows through c0's frozen trunk")
    ap.add_argument("--control", default="",
                    help="a pretrained control-stream checkpoint dir (params.pt): rows through "
                         "carla.ControlTrunk instead of c0's trunk")
    ap.add_argument("--data", required=True,
                    help="il_data root with <Town>/ep_*.npz; several roots comma-separated")
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
    if bool(args.c0) == bool(args.control):
        raise SystemExit("give exactly one of --c0 (c0's trunk) and --control (the control stream)")

    from torch_rl.frl import expert as fx
    from torch_rl.frl.carla import params_md5
    if args.control:
        from torch_rl.frl.carla import ControlTrunk       # here, so the c0 path never needs it
        trunk, trunk_dir = ControlTrunk(args.control, args.device), args.control
        transitions = fx.episode_transitions_control
    else:
        from torch_rl.frl.carla import C0Trunk
        trunk, trunk_dir = C0Trunk(args.c0, args.device), args.c0
        transitions = fx.episode_transitions
    parts, meta = [], {"towns": {}, "k": args.k, "idle_penalty": args.idle_penalty,
                        "idle_kmh": args.idle_kmh, "junction_bonus_removed": args.junction_bonus,
                        "arrival_cut": fx.ARRIVAL_CUT, "data": args.data,
                        "trunk": "control" if args.control else "c0",
                        "trunk_dir": os.path.abspath(trunk_dir)}
    if args.control:
        ca = control_args(args.control)
        meta.update(n_frames=int(ca["n_frames"]), repeat=int(ca.get("repeat", 1)), px=int(ca["px"]))
    roots = [r for r in args.data.split(",") if r]
    t0 = time.time()
    for town in args.towns.split(","):
        cache = os.path.join(args.caches, f"features_{town}.npz") if args.caches else ""
        held = held_out_names(cache) if cache and os.path.exists(cache) else None
        per_root = {}
        for root in roots:                      # selected root by root: episode names repeat across roots
            paths = sorted(glob.glob(os.path.join(root, town, "ep_*.npz")))
            names = [os.path.basename(p) for p in paths]
            drivers = [str(np.load(p, allow_pickle=False)["driver"]) for p in paths]
            keep = set(fx.select_episodes(names, drivers, held_out=held))
            n_tr = 0
            for p, n in zip(paths, names):
                if n not in keep:
                    continue
                z = np.load(p, allow_pickle=False)
                tr = transitions(trunk, {key: z[key] for key in z.files}, k=args.k,
                                 idle_penalty=args.idle_penalty, idle_kmh=args.idle_kmh,
                                 junction_bonus=args.junction_bonus)
                if len(tr["rewards"]):
                    parts.append(tr)
                    n_tr += len(tr["rewards"])
            per_root[root] = {"shards": len(paths), "expert": int(sum(d == "expert" for d in drivers)),
                              "used": len(keep), "transitions": n_tr}
            label = town if len(roots) == 1 else f"{town} {root}"     # one root: the line as it always was
            print(f"[expert] {label}: {len(paths)} shards, {per_root[root]['expert']} expert, "
                  f"{len(keep)} used -> {n_tr} transitions ({time.time() - t0:.0f} s)", flush=True)
        tot = {key: sum(r[key] for r in per_root.values())
               for key in ("shards", "expert", "used", "transitions")}
        meta["towns"][town] = {"shards": tot["shards"], "expert": tot["expert"], "used": tot["used"],
                               "probe_held_out": "cache" if held else "rule",
                               "transitions": tot["transitions"]}
        if len(roots) > 1:
            meta["towns"][town]["roots"] = per_root
    tr = fx.concat(parts)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp.npz"
    fx.save_replay(tmp, tr, k=args.k, c0_md5=params_md5(trunk_dir), meta=meta)
    os.replace(tmp, args.out)
    r = tr["rewards"]
    print(f"[expert] {meta['trunk']} trunk {meta['trunk_dir']} (md5 {params_md5(trunk_dir)})", flush=True)
    print(f"[expert] {len(r)} transitions at k={args.k} -> {args.out} | reward mean {r.mean():.3f} "
          f"min {r.min():.2f} max {r.max():.2f} | terminal masks {int((tr['masks'] == 0).sum())} | "
          f"|steer| mean {np.abs(tr['actions'][:, 0]).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
