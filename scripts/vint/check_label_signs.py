#!/usr/bin/env python3
"""Sign check of the ViNT labels on real shards: the expert's steering label at
frame t (CARLA: + = right) against the lateral position, in ViNT's local frame
(+ = left), of the frame `ahead` steps later. A right turn must put the future
path on the right, so the correlation must be strongly NEGATIVE.

usage: check_label_signs.py SPLIT_JSON N_EPISODES [AHEAD]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "src"))
from vae_nav.vint_data import episode_to_traj, local_coords  # noqa: E402

split = json.load(open(sys.argv[1]))
n = int(sys.argv[2])
ahead = int(sys.argv[3]) if len(sys.argv) > 3 else 12
paths = split["train"][:: max(1, len(split["train"]) // n)][:n]
ys, steers, fwd = [], [], []
for p in paths:
    z = np.load(p)
    frames, xy, yaw = episode_to_traj(z["frames"], z["pos"], str(z["end"]), trim_frames=0)
    steer = np.asarray(z["act_label"], np.float64)[:, 0]
    for t in range(0, min(len(xy) - ahead, len(steer))):
        loc = local_coords(xy[t + ahead][None], xy[t], yaw[t])[0]
        if np.hypot(*(xy[t + ahead] - xy[t])) < 1.0:
            continue                                   # stopped: no direction to compare
        ys.append(loc[1])
        fwd.append(loc[0])
        steers.append(steer[t])
ys, steers, fwd = map(np.asarray, (ys, steers, fwd))
r = np.corrcoef(ys, steers)[0, 1]
print(f"{len(paths)} episodes, {len(ys)} frames, {ahead} steps ahead: corr(lateral, steer) = "
      f"{r:+.3f}; ahead x mean {fwd.mean():.2f} m (min {fwd.min():.2f})")
for thr in (0.1, 0.2, 0.3, 0.5):
    turning = np.abs(steers) > thr
    if turning.any():
        agree = np.mean(np.sign(ys[turning]) == -np.sign(steers[turning]))
        print(f"  |steer| > {thr}: {turning.sum()} frames, the path bends the steering's way "
              f"in {agree:.1%}, mean |lateral| {np.abs(ys[turning]).mean():.2f} m")
