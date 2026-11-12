"""Lane labels for recorded episodes, computed offline (plan 2026-11-09, Step 0:
"is it the encoder?").

A shard stores where the car was AFTER each step (pos[t]; see
expert_data.frame_goal_distances), not its yaw or its lane. The per-frame labels a
feature probe needs -- lateral offset from the lane centre, heading error against
the lane, the heading change from one frame to the next -- therefore come from
those positions and the town's lane geometry: OpenDriveLanes reads the town's
OpenDRIVE file into a carla.Map without a server.

DIAGNOSTIC ONLY. These are privileged quantities (they use the map and the true
position); they label probes and the offline privileged-critic control, and are
never an input to the navigator.

Conventions (CARLA): x forward, y right, yaw in degrees = atan2(dy, dx); a lane's
right vector is (-sin yaw, cos yaw), so a positive offset is to the lane's right.
"""
import math
import os

import numpy as np

DEFAULT_XODR_DIR = os.environ.get(
    "XODR_DIR", "/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive")


def wrap_deg(a):
    """Angle(s) in degrees -> [-180, 180)."""
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def frame_positions(pos):
    """(T, 2) positions after each step -> (T + 1, 2), one per frame: frame k >= 1
    was taken at pos[k - 1]; the reset frame 0 is read at pos[0] (the car starts at
    rest, so it has barely moved during step 0)."""
    pos = np.asarray(pos, np.float64).reshape(-1, 2)
    return np.concatenate([pos[:1], pos], axis=0)


def motion_heading(P, min_step_m=0.15):
    """(N, 2) per-frame positions -> (N,) direction of travel in DEGREES, from the
    centred difference P[k+1] - P[k-1] (one-sided at the two ends); NaN where the
    car moved less than `min_step_m` over that span (at rest the direction of
    travel is undefined)."""
    P = np.asarray(P, float)
    n = len(P)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    d = np.empty_like(P)
    d[1:-1] = P[2:] - P[:-2]
    d[0] = P[1] - P[0]
    d[-1] = P[-1] - P[-2]
    ok = np.hypot(d[:, 0], d[:, 1]) >= float(min_step_m)
    out[ok] = np.degrees(np.arctan2(d[ok, 1], d[ok, 0]))
    return out


def lane_offset(P, centre, lane_yaw_deg):
    """Signed lateral offset (m) of points from their lane centres; positive to the
    lane's right."""
    yaw = np.radians(np.asarray(lane_yaw_deg, float))
    right = np.stack([-np.sin(yaw), np.cos(yaw)], -1)
    return np.sum((np.asarray(P, float) - np.asarray(centre, float)) * right, -1)


def episode_labels(pos, lanes, min_step_m=0.15, ahead_m=10.0):
    """Per-frame labels for one episode, from its (T, 2) `pos` and a lane source
    (`lanes.query(P, ahead_m)` -> dict of centre, yaw, width, junction, yaw_ahead).

    Returns arrays of length T + 1:
      offset       signed metres from the lane centre (+ = right)
      heading_err  direction of travel minus lane yaw, degrees
      heading      direction of travel, degrees (NaN at rest)
      dheading     heading change from frame k to frame k + 1, degrees (NaN last)
      curv         lane yaw change over the next `ahead_m` (NaN at a fork)
      junction     the frame's lane point is inside a junction
      width        lane width, metres
      valid        heading defined and k >= 1 (frame 0's position is approximate)
    """
    P = frame_positions(pos)
    q = lanes.query(P, ahead_m=ahead_m)
    psi = motion_heading(P, min_step_m)
    off = lane_offset(P, q["centre"], q["yaw"])
    herr = wrap_deg(psi - np.asarray(q["yaw"], float))
    dpsi = np.full(len(P), np.nan)
    dpsi[:-1] = wrap_deg(psi[1:] - psi[:-1])
    valid = np.isfinite(psi)
    valid[0] = False
    return {"offset": off, "heading_err": herr, "heading": psi, "dheading": dpsi,
            "curv": np.asarray(q["yaw_ahead"], float), "junction": np.asarray(q["junction"], bool),
            "width": np.asarray(q["width"], float), "valid": valid}


def window_noise(act_exec, act_label, k):
    """Executed minus labelled STEERING (the DART noise) averaged over steps
    t .. t + k - 1, per step t; NaN where the window runs past the episode."""
    d = np.asarray(act_exec, float)[:, 0] - np.asarray(act_label, float)[:, 0]
    T = len(d)
    out = np.full(T, np.nan)
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if T >= k:
        c = np.concatenate([[0.0], np.cumsum(d)])
        out[:T - k + 1] = (c[k:] - c[:-k]) / k
    return out


class OpenDriveLanes:
    """Lane queries on a town's OpenDRIVE map, without a CARLA server: carla.Map
    built from the .xodr text. Each point snaps to the nearest DRIVING lane
    (carla.Map.get_waypoint, project_to_road=True)."""

    def __init__(self, town, xodr_dir=DEFAULT_XODR_DIR):
        import carla  # the PythonAPI module only; no server is contacted
        self._carla = carla
        path = os.path.join(xodr_dir, f"{town}.xodr")
        with open(path) as fh:
            self.map = carla.Map(town, fh.read())

    def query(self, P, ahead_m=10.0):
        carla = self._carla
        P = np.asarray(P, float)
        n = len(P)
        centre = np.zeros((n, 2))
        yaw = np.zeros(n)
        width = np.zeros(n)
        junction = np.zeros(n, bool)
        yaw_ahead = np.full(n, np.nan)
        for i, (x, y) in enumerate(P):
            wp = self.map.get_waypoint(carla.Location(x=float(x), y=float(y), z=0.0),
                                       project_to_road=True, lane_type=carla.LaneType.Driving)
            loc, rot = wp.transform.location, wp.transform.rotation
            centre[i] = (loc.x, loc.y)
            yaw[i] = rot.yaw
            width[i] = wp.lane_width
            junction[i] = wp.is_junction
            nxt = wp.next(float(ahead_m))
            if len(nxt) == 1:
                yaw_ahead[i] = float(wrap_deg(nxt[0].transform.rotation.yaw - rot.yaw))
        return {"centre": centre, "yaw": wrap_deg(yaw), "width": width,
                "junction": junction, "yaw_ahead": yaw_ahead}


def _selftest_math():          # pragma: no cover - documentation of the sign convention
    assert math.isclose(float(lane_offset([[0.0, 1.0]], [[0.0, 0.0]], [0.0])[0]), 1.0)
