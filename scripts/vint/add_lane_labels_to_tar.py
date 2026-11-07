#!/usr/bin/env python3
"""The first ViNT dataset with lane-centre labels added -- nothing else changed.

    add_lane_labels_to_tar.py --tar-in checkpoints/vint_data/carla_expert.tar \
        --labels lane_labels.npz --tar-out checkpoints/vint_data/carla_expert_lc.tar

Streams the tar: every image, split list and summary is copied byte for byte;
each <name>/traj_data.pkl gains cf_position (N, 6, 2) and cf_yaw (N, 6) from
scripts/vint/build_lane_labels.py, cut to the frames the converter kept (fault
tails are trimmed). Refused if a trajectory has no labels or if the labels' own
frame positions do not match the trajectory's (misaligned frames).
"""
import argparse
import io
import os
import pickle
import tarfile

import numpy as np


def _labels(path):
    z = np.load(path)
    starts = np.concatenate([[0], np.cumsum(z["lens"])[:-1]])
    return {str(n): (z["position"][s:s + k], z["yaw"][s:s + k], z["ok"][s:s + k])
            for n, s, k in zip(z["names"], starts, z["lens"])}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar-in", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--tar-out", required=True)
    a = ap.parse_args(argv)
    lab = _labels(a.labels)
    n_traj, n_ok, n_frames = 0, 0, 0
    tmp = a.tar_out + ".part"
    with tarfile.open(a.tar_in) as src, tarfile.open(tmp, "w") as dst:
        for m in src:
            data = src.extractfile(m).read() if m.isfile() else None
            if m.name.endswith("/traj_data.pkl"):
                name = m.name.split("/")[-2]
                if name not in lab:
                    raise SystemExit(f"{name}: no lane-centre labels")
                d = pickle.loads(data)
                n = len(d["position"])
                P, Y, ok = lab[name]
                if len(P) < n:
                    raise SystemExit(f"{name}: {len(P)} label frames for {n} trajectory frames")
                gap = float(np.abs(P[:n, 0] - d["position"]).max())
                if gap > 1e-3:
                    raise SystemExit(f"{name}: label frames misaligned by {gap:.3f} m")
                d["cf_position"] = np.asarray(P[:n], np.float32)
                d["cf_yaw"] = np.asarray(Y[:n], np.float32)
                data = pickle.dumps(d)
                n_traj += 1
                n_ok += int(ok[:n].sum())
                n_frames += n
            elif m.name.endswith("summary.json"):
                s = __import__("json").loads(data or b"{}")
                s["labels"] = ("lane centre of the route (cf_position/cf_yaw, "
                               "scripts/vint/build_lane_labels.py)")
                data = __import__("json").dumps(s, indent=1).encode()
            if data is not None:
                info = tarfile.TarInfo(m.name)
                info.size, info.mtime, info.mode = len(data), m.mtime, m.mode
                dst.addfile(info, io.BytesIO(data))
            else:
                dst.addfile(m)
    os.replace(tmp, a.tar_out)
    print(f"{n_traj} trajectories, {n_frames} frames, lane-centre labels on "
          f"{n_ok / max(n_frames, 1):.3f} -> {a.tar_out}")


if __name__ == "__main__":
    main()
