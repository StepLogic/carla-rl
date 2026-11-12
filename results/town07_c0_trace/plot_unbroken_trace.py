#!/usr/bin/env python3
"""Show a c0 unbroken map trace (run_unbroken_trace output: index.json, seg_*.npz,
map_waypoints.npy): the driven path over the town's drivable waypoints, coloured by
who drove (c0, expert recovery, reversing out), with takeovers and summary numbers;
optionally a video of the drive from the recorded camera frames.

usage: plot_unbroken_trace.py TRACE_DIR OUT.png [--video OUT.mp4 --every 2 --fps 30]
"""
import argparse
import json
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("trace_dir")
ap.add_argument("out_png")
ap.add_argument("--video", default="")
ap.add_argument("--every", type=int, default=2, help="keep every Nth step in the video")
ap.add_argument("--fps", type=int, default=30)
a = ap.parse_args()

idx = json.load(open(os.path.join(a.trace_dir, "index.json")))
segs = sorted(idx["segments"], key=lambda s: s["seg"])
cols = {k: [] for k in ("pos", "driver", "hop", "fault")}
frames_files = []
# an expert coverage trace has no driver/hop/fault columns: the expert drove every step
default_driver = 1 if str(idx.get("driver", "")).startswith("expert") else 0
for s in segs:
    z = np.load(os.path.join(a.trace_dir, s["file"]))
    m = len(z["pos"])
    cols["pos"].append(z["pos"])
    cols["driver"].append(z["driver"] if "driver" in z.files else np.full(m, default_driver))
    cols["hop"].append(z["hop"] if "hop" in z.files else np.full(m, -1))
    cols["fault"].append(z["fault"] if "fault" in z.files else np.zeros(m))
    frames_files.append(s["file"])
pos = np.concatenate(cols["pos"]).reshape(-1, 2)
driver = np.concatenate(cols["driver"]).astype(int)
hop = np.concatenate(cols["hop"]).astype(int)
wp_file = os.path.join(a.trace_dir, "map_waypoints.npy")
if os.path.exists(wp_file):
    wps = np.load(wp_file)[:, :2]
else:                                   # an expert trace saves none: read the OpenDRIVE map
    import carla
    town = idx.get("town", "Town07")
    xodr = (f"/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive/{town}.xodr")
    cmap = carla.Map(town, open(xodr).read())
    wps = np.array([[w.transform.location.x, w.transform.location.y]
                    for w in cmap.generate_waypoints(2.0)])

step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
km = float(step[step < 5.0].sum()) / 1000.0          # ignore any resume jump
n = len(driver)
expert_frac = float(np.mean(driver != 0)) if n else 0.0
takeovers = int(np.sum((driver[1:] != 0) & (driver[:-1] == 0)))
hops = idx.get("hops") or {}
n_hops = int(hops.get("attempted", len({int(h) for h in hop if h >= 0})))
hops_reached = int((hops.get("ends") or {}).get("reached", 0))
by_exit = hops.get("reached_by_exit") or {}
planned = hops.get("planned_exits") or {}
cov = idx.get("coverage_curve") or []          # entries: [metres, coverage fraction, steps]
coverage = float(cov[-1][1]) if cov and isinstance(cov[-1], (list, tuple)) else None
n_drives = len({s.get("drive", 0) for s in segs})

fig, ax = plt.subplots(figsize=(11, 11))
ax.scatter(wps[:, 0], wps[:, 1], s=1.5, c="#c8ccd2", lw=0, label="drivable lanes")
colors = {0: "#1f5fa8", 1: "#e08a1e", 2: "#c0392b"}
names = {0: "c0 driving", 1: "expert recovery", 2: "reversing out"}
if default_driver == 1:
    names[1] = "expert driving"
start = 0
for i in range(1, n + 1):                              # draw runs of one driver
    if i == n or driver[i] != driver[start] or (i < n and step[i - 1] > 5.0):
        seg = pos[start:i + (1 if i < n and step[i - 1] <= 5.0 else 0)]
        ax.plot(seg[:, 0], seg[:, 1], color=colors[driver[start]], lw=1.6 if driver[start] == 0 else 2.4)
        start = i
for d in (0, 1, 2):
    ax.plot([], [], color=colors[d], lw=2.4, label=names[d])
tk = np.flatnonzero((driver[1:] != 0) & (driver[:-1] == 0)) + 1
ax.scatter(pos[tk, 0], pos[tk, 1], marker="x", s=40, c="#c0392b", zorder=5, label="fault -> expert takes over")
ax.scatter(*pos[0], s=90, c="#2e8b57", zorder=6, label="start")
ax.scatter(*pos[-1], s=90, c="#111111", zorder=6, label="end")
ax.set_aspect("equal")
ax.invert_yaxis()                                      # CARLA's top view
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
town = idx.get("town", "")
cov_txt = f"{coverage:.0%}" if isinstance(coverage, (int, float)) else "n/a"
who = "the expert's coverage tour" if default_driver == 1 else "c0"
exits = ", ".join(f"{e} {by_exit.get(e, 0)}/{planned[e]}" for e in ("left", "straight", "right")
                  if e in planned)
ax.set_title(f"{town}: map data driven by {who} | {km:.2f} km, {n} steps, coverage {cov_txt}\n"
             f"expert {expert_frac:.1%} of steps, {takeovers} takeovers"
             + (f" | junction hops reached {hops_reached}/{n_hops} ({exits})" if n_hops else "")
             + f" | {n_drives} drive(s)")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(a.out_png, dpi=110)
print(f"{a.out_png}: {km:.2f} km, {n} steps, coverage {cov_txt}, expert {expert_frac:.1%}, "
      f"{takeovers} takeovers, hops reached {hops_reached}/{n_hops} ({exits}), {n_drives} drive(s)")

if a.video:                                            # the recorded camera frames
    import cv2
    first = np.load(os.path.join(a.trace_dir, frames_files[0]))["frame"][0]
    h, w = first.shape[:2]
    tmp = a.video + ".raw.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    k = 0
    for f in frames_files:
        z = np.load(os.path.join(a.trace_dir, f))
        drv = z["driver"] if "driver" in z.files else np.full(len(z["frame"]), default_driver)
        for fr, dr in zip(z["frame"], drv):
            if k % a.every == 0:
                img = np.ascontiguousarray(fr[..., ::-1]) if fr.ndim == 3 else cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
                cv2.putText(img, names[int(dr)], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1, cv2.LINE_AA)
                vw.write(img)
            k += 1
    vw.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.video], check=True)
    os.remove(tmp)
    print(f"{a.video}: {k // a.every} frames at {a.fps} fps")
