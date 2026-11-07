"""Expert episodes (torch_rl.expert_data shards) -> ViNT trajectories.

User 2026-11-07: train ViNT (Shah et al. 2023, visualnav-transformer) on c0's
expert data, fine-tuned from the released weights, and evaluate it like c0.

ViNT reads one folder per trajectory: 0.jpg .. N.jpg and traj_data.pkl with
"position" (N+1, 2) and "yaw" (N+1,), and builds its labels with
to_local_coords: future positions in the current frame, x FORWARD, y LEFT.

Three facts about the shards decide the conversion:
  * T+1 frames, T positions. Frame 0 is the reset view; frame k+1 is the view
    after step k, taken at pos[k] (train_carla_sac.run_record). Frame 0 gets
    pos[0]: the car starts at rest and moves centimetres in its first step.
  * CARLA is left-handed (x east, y SOUTH). Flipping y gives ViNT's convention,
    so a left turn has positive local y, as in the robot datasets the released
    weights learned from.
  * No yaw is stored. The heading is the direction of motion: to the first
    later point at least min_step_m away (the car never reverses). A stopped
    tail keeps the last heading; a car that never moves gets 0.
Fault-ended episodes lose their last frames, so ViNT never learns the drift
into the crossing or off-road that ended them.
"""
import os
import pickle

import numpy as np

FAULT_ENDS = ("crossing", "off_road", "collision")


def frame_positions(pos):
    """(T, 2) positions after each step -> (T+1, 2), one per frame."""
    pos = np.asarray(pos, np.float64)[:, :2]
    return np.concatenate([pos[:1], pos], axis=0)


def to_right_handed(xy):
    """CARLA (x, y south) -> (x, y left)."""
    out = np.array(xy, np.float64, copy=True)
    out[..., 1] *= -1.0
    return out


def headings_from_motion(xy, min_step_m=0.3):
    """Yaw per point (radians, right-handed) from the direction of motion."""
    xy = np.asarray(xy, np.float64)
    n = len(xy)
    yaw = np.full(n, np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            d = xy[j] - xy[i]
            if np.hypot(d[0], d[1]) >= min_step_m:
                yaw[i] = np.arctan2(d[1], d[0])
                break
    valid = np.flatnonzero(~np.isnan(yaw))
    if not len(valid):
        return np.zeros(n)
    yaw[:valid[0]] = yaw[valid[0]]                  # cannot happen (forward scan), kept safe
    for i in range(valid[0] + 1, n):                # stopped tail: keep the last heading
        if np.isnan(yaw[i]):
            yaw[i] = yaw[i - 1]
    return yaw


def local_coords(points, origin, yaw):
    """vint_train.data.data_utils.to_local_coords for 2-D points."""
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return (np.asarray(points, np.float64) - np.asarray(origin, np.float64)).dot(rot)


def episode_to_traj(frames, pos, end, trim_frames=20, min_step_m=0.3):
    """-> (frames, xy, yaw) ready for write_traj. Headings are computed on the
    whole episode before any fault trim, so the last kept frames still have
    headings from real motion."""
    frames = np.asarray(frames)
    xy = to_right_handed(frame_positions(pos))
    if len(frames) != len(xy):
        raise ValueError(f"{len(frames)} frames for {len(xy)} frame positions: a shard "
                         "has exactly one more frame than positions")
    yaw = headings_from_motion(xy, min_step_m=min_step_m)
    keep = len(frames)
    if str(end) in FAULT_ENDS:
        keep = max(0, keep - int(trim_frames))
    return frames[:keep], xy[:keep], yaw[:keep]


def write_traj(folder, frames, xy, yaw, quality=95):
    """0.jpg .. N.jpg and traj_data.pkl, as ViNT's ViNT_Dataset reads them."""
    from PIL import Image
    os.makedirs(folder, exist_ok=True)
    for k, f in enumerate(frames):
        Image.fromarray(np.asarray(f, np.uint8)).save(os.path.join(folder, f"{k}.jpg"),
                                                      quality=int(quality))
    with open(os.path.join(folder, "traj_data.pkl"), "wb") as fh:
        pickle.dump({"position": np.asarray(xy, np.float32),
                     "yaw": np.asarray(yaw, np.float32)}, fh)
