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


def aim_cf_labels(xy, yaw, aim, aim_yaw, aim_ok):
    """DAgger frames (EXPERT_RECORD_AIM, user 2026-11-07 "use DAGGER"): the
    recorder's per-frame aim points (N, n, 2) CARLA x/y and CARLA yaw (N, n,
    radians) -> ViNT's lane-centre label arrays in the trajectory's right-handed
    frame: cf_position (N, n+1, 2) with entry 0 the car itself, cf_yaw (N, n+1)
    unwrapped against the car's heading, cf_ok (N,) = the frame has a label."""
    xy = np.asarray(xy, np.float64)
    yaw = np.asarray(yaw, np.float64)
    a = to_right_handed(np.asarray(aim, np.float64))
    ay = -np.asarray(aim_yaw, np.float64)          # CARLA's clockwise yaw -> right-handed
    d = (ay - yaw[:, None] + np.pi) % (2 * np.pi) - np.pi
    pos = np.concatenate([xy[:, None, :], a], axis=1)
    yw = np.concatenate([yaw[:, None], yaw[:, None] + d], axis=1)
    return pos.astype(np.float32), yw.astype(np.float32), np.asarray(aim_ok, bool).copy()


def encode_traj(frames, xy, yaw, quality=95, extra=None):
    """[(file name, bytes)]: 0.jpg .. N.jpg and traj_data.pkl, as ViNT's
    ViNT_Dataset reads them, encoded in memory (the --tar path never writes a
    small file to the shared filesystem). `extra` arrays (e.g. the cf_* labels)
    join traj_data.pkl."""
    import io
    from PIL import Image
    out = []
    for k, f in enumerate(frames):
        buf = io.BytesIO()
        Image.fromarray(np.asarray(f, np.uint8)).save(buf, format="JPEG", quality=int(quality))
        out.append((f"{k}.jpg", buf.getvalue()))
    data = {"position": np.asarray(xy, np.float32), "yaw": np.asarray(yaw, np.float32)}
    data.update({k: np.asarray(v) for k, v in (extra or {}).items()})
    out.append(("traj_data.pkl", pickle.dumps(data)))
    return out


def write_traj(folder, frames, xy, yaw, quality=95, extra=None):
    """The files of encode_traj, written into `folder`."""
    os.makedirs(folder, exist_ok=True)
    for name, data in encode_traj(frames, xy, yaw, quality=quality, extra=extra):
        with open(os.path.join(folder, name), "wb") as fh:
            fh.write(data)
