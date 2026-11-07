#!/usr/bin/env python3
"""c0's expert episodes -> ViNT trajectories (vae_nav.vint_data).

    convert_il_to_vint.py --split checkpoints/il_bc_c0/bc_split.json \
        --tar checkpoints/vint_data/carla_expert.tar --workers 16
    convert_il_to_vint.py --split ... --out <data dir> --splits <splits dir>   # small files

ViNT train = c0's train episodes, ViNT test = c0's validation episodes, so the
two models learn from exactly the same drives. Both modes produce the same
layout: carla_expert/<Town>_<seed>/{0..N.jpg, traj_data.pkl},
carla_expert/summary.json (counts, the measured metres per frame = ViNT's
metric_waypoint_spacing) and splits/carla_expert/{train,test}/traj_names.txt.

--tar streams everything into ONE file: CephFS costs ~85 ms per small-file
read, so the ~290k JPEGs of the small-file mode could not be packed afterwards
in useful time (vint_train.sbatch unpacks the tar on node-local disk).
"""
import argparse
import io
import json
import os
import sys
import tarfile
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from vae_nav.vint_data import FAULT_ENDS, encode_traj, episode_to_traj, write_traj  # noqa: E402

DATASET = "carla_expert"


def _complete(folder, n):
    return (os.path.exists(os.path.join(folder, "traj_data.pkl"))
            and all(os.path.exists(os.path.join(folder, f"{k}.jpg")) for k in (0, n - 1)))


def convert_one(job):
    """-> (row, files): files are the encoded episode in --tar mode, else None
    (the episode was written into `out_root`)."""
    path, out_root, trim, min_frames = job
    z = np.load(path)
    name = f"{z['town']}_{int(z['seed'])}"
    frames, xy, yaw = episode_to_traj(z["frames"], z["pos"], str(z["end"]), trim_frames=trim)
    row = {"name": name, "end": str(z["end"]), "mode": int(z["mode"]),
           "frames_in": int(len(z["frames"])), "frames": int(len(frames))}
    if len(frames) < min_frames:
        row["skipped"] = "short"
        return row, None
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    row["moving_steps_m"] = d[d > 0.05].round(4).tolist()
    if out_root is None:
        return row, encode_traj(frames, xy, yaw)
    folder = os.path.join(out_root, name)
    if not _complete(folder, len(frames)):
        write_traj(folder, frames, xy, yaw)
    return row, None


def _add_bytes(tf, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = int(time.time())
    tf.addfile(info, io.BytesIO(data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="c0's bc_split.json")
    ap.add_argument("--tar", help="write one tar with the whole dataset")
    ap.add_argument("--out", help="small-file mode: the data folder")
    ap.add_argument("--splits", help="small-file mode: the splits folder")
    ap.add_argument("--trim", type=int, default=20,
                    help="frames dropped from the end of a fault-ended episode")
    ap.add_argument("--min-frames", type=int, default=31,
                    help="ViNT needs context*stride + len_pred*stride + 1 frames (5*3+5*3+1)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if bool(a.tar) == bool(a.out or a.splits) or (a.out and not a.splits):
        ap.error("give --tar, or both --out and --splits")
    split = json.load(open(a.split))
    tf = None
    if a.tar:
        tmp = a.tar + ".part"
        tf = tarfile.open(tmp, "w")
    else:
        os.makedirs(a.out, exist_ok=True)
    summary = {"split": a.split, "trim_fault_frames": a.trim, "min_frames": a.min_frames}
    steps = []
    with Pool(a.workers) as pool:
        for part, key in (("train", "train"), ("test", "val")):
            jobs = [(p, None if tf else a.out, a.trim, a.min_frames) for p in split[key]]
            kept, n_in = [], 0
            for row, files in pool.imap(convert_one, jobs, chunksize=2):
                n_in += 1
                if "skipped" in row:
                    continue
                steps += row.pop("moving_steps_m")
                kept.append(row)
                if tf is not None:
                    for fname, data in files:
                        _add_bytes(tf, f"{DATASET}/{row['name']}/{fname}", data)
            names = ("\n".join(r["name"] for r in kept) + "\n").encode()
            if tf is not None:
                _add_bytes(tf, f"splits/{DATASET}/{part}/traj_names.txt", names)
            else:
                d = os.path.join(a.splits, part)
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "traj_names.txt"), "wb") as fh:
                    fh.write(names)
            summary[part] = {"episodes_in": n_in, "episodes": len(kept),
                             "skipped_short": n_in - len(kept),
                             "frames": int(sum(r["frames"] for r in kept)),
                             "fault_trimmed": int(sum(r["end"] in FAULT_ENDS for r in kept)),
                             "goal_mode": int(sum(r["mode"] == 1 for r in kept))}
            print(part, json.dumps(summary[part]), flush=True)
    steps = np.asarray(steps)
    summary["metric_waypoint_spacing_m"] = round(float(steps.mean()), 4)
    summary["moving_step_median_m"] = round(float(np.median(steps)), 4)
    blob = json.dumps(summary, indent=1).encode()
    if tf is not None:
        _add_bytes(tf, f"{DATASET}/summary.json", blob)
        tf.close()
        os.replace(a.tar + ".part", a.tar)
    else:
        with open(os.path.join(a.out, "summary.json"), "wb") as fh:
            fh.write(blob)
    print("metres per moving frame: mean", summary["metric_waypoint_spacing_m"],
          "median", summary["moving_step_median_m"], flush=True)


if __name__ == "__main__":
    main()
