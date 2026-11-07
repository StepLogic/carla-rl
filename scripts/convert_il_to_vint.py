#!/usr/bin/env python3
"""c0's expert episodes -> ViNT trajectories (vae_nav.vint_data).

    convert_il_to_vint.py --split checkpoints/il_bc_c0/bc_split.json \
        --out checkpoints/vint_data/carla_expert \
        --splits checkpoints/vint_data/splits/carla_expert --workers 8

ViNT train = c0's train episodes, ViNT test = c0's validation episodes, so the
two models learn from exactly the same drives. Writes <out>/<Town>_<seed>/ per
episode, <splits>/{train,test}/traj_names.txt, and <out>/summary.json with the
counts and the measured metres per frame (ViNT's metric_waypoint_spacing).
An episode folder that is already complete is skipped, so a killed job resumes.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from vae_nav.vint_data import episode_to_traj, write_traj  # noqa: E402


def _complete(folder, n):
    return (os.path.exists(os.path.join(folder, "traj_data.pkl"))
            and all(os.path.exists(os.path.join(folder, f"{k}.jpg")) for k in (0, n - 1)))


def convert_one(job):
    path, out_root, trim, min_frames = job
    z = np.load(path)
    name = f"{z['town']}_{int(z['seed'])}"
    frames, xy, yaw = episode_to_traj(z["frames"], z["pos"], str(z["end"]), trim_frames=trim)
    row = {"name": name, "end": str(z["end"]), "mode": int(z["mode"]),
           "frames_in": int(len(z["frames"])), "frames": int(len(frames))}
    if len(frames) < min_frames:
        row["skipped"] = "short"
        return row
    folder = os.path.join(out_root, name)
    if not _complete(folder, len(frames)):
        write_traj(folder, frames, xy, yaw)
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    row["moving_steps_m"] = d[d > 0.05].round(4).tolist()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="c0's bc_split.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--trim", type=int, default=20,
                    help="frames dropped from the end of a fault-ended episode")
    ap.add_argument("--min-frames", type=int, default=31,
                    help="ViNT needs context*stride + len_pred*stride + 1 frames (5*3+5*3+1)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    split = json.load(open(a.split))
    os.makedirs(a.out, exist_ok=True)
    summary = {"split": a.split, "trim_fault_frames": a.trim, "min_frames": a.min_frames}
    steps = []
    for part, key in (("train", "train"), ("test", "val")):
        jobs = [(p, a.out, a.trim, a.min_frames) for p in split[key]]
        with Pool(a.workers) as pool:
            rows = pool.map(convert_one, jobs, chunksize=4)
        kept = [r for r in rows if "skipped" not in r]
        for r in kept:
            steps += r.pop("moving_steps_m")
        d = os.path.join(a.splits, part)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "traj_names.txt"), "w") as fh:
            fh.write("\n".join(r["name"] for r in kept) + "\n")
        summary[part] = {"episodes_in": len(rows), "episodes": len(kept),
                         "skipped_short": len(rows) - len(kept),
                         "frames": int(sum(r["frames"] for r in kept)),
                         "fault_trimmed": int(sum(r["end"] in ("crossing", "off_road", "collision")
                                                  for r in kept)),
                         "goal_mode": int(sum(r["mode"] == 1 for r in kept))}
        print(part, json.dumps(summary[part]), flush=True)
    steps = np.asarray(steps)
    summary["metric_waypoint_spacing_m"] = round(float(steps.mean()), 4)
    summary["moving_step_median_m"] = round(float(np.median(steps)), 4)
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("metres per moving frame: mean", summary["metric_waypoint_spacing_m"],
          "median", summary["moving_step_median_m"], flush=True)


if __name__ == "__main__":
    main()
