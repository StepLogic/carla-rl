"""Overhead navigation video (user 2026-11-09): camera + current subgoal image on
the left, CARLA top-down on the right with the shortest road route, the planned
subgoal waypoints (entropy map), the current subgoal, the car's track and pose."""
import json, math, sys, cv2, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
traj_f, vid_f, map_f, town, out_f = sys.argv[1:6]
tr = json.load(open(traj_f)); M = json.load(open(map_f))
NXY = np.array([n["pos"][:2] for n in M["nodes"]], float)
m = json.load(open(f"topdown_{town}.json"))
big = np.asarray(Image.open(f"topdown_{town}.png"))[..., :3]
big = np.fliplr(big.transpose(1, 0, 2))                       # same orientation as the paper figures
f = m["W"] / (2 * math.tan(math.radians(m["fov"]) / 2))
def proj(xy):
    xy = np.asarray(xy, float).reshape(-1, 2)
    return np.c_[m["W"] / 2 + f * (xy[:, 0] - m["cx"]) / m["z"], m["H"] / 2 + f * (xy[:, 1] - m["cy"]) / m["z"]]
S = np.array(tr["steps"], float)                              # t, x, y, yaw, speed, cur_node
road = np.array(tr["road_route"] or [tr["start"], tr["goal"]], float)
pathnodes = sorted({i for p in tr["paths"] for i in p["path"]})
pts = np.vstack([proj(road), proj(S[:, 1:3]), proj(NXY[pathnodes]) if pathnodes else np.zeros((0, 2)),
                 proj([tr["start"], tr["goal"]])])
cx, cy = pts.mean(0) if False else ((pts.min(0) + pts.max(0)) / 2)
half = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) / 2 + 40
x0, y0 = int(max(0, cx - half)), int(max(0, cy - half)); x1, y1 = int(cx + half), int(cy + half)
crop = np.ascontiguousarray(big[y0:y1, x0:x1])
OV = 720; sc = OV / max(crop.shape[:2])
base = cv2.resize(crop, (int(crop.shape[1] * sc), int(crop.shape[0] * sc)), interpolation=cv2.INTER_AREA)
base = cv2.cvtColor((0.45 * 255 + 0.55 * base).astype(np.uint8), cv2.COLOR_RGB2BGR)   # brighten, keep colour
canvas0 = np.full((OV, OV, 3), 255, np.uint8); canvas0[:base.shape[0], :base.shape[1]] = base
def P(xy): q = (proj(xy) - [x0, y0]) * sc; return q.astype(np.int32)
# static layers: all map nodes faint, road route
for q in P(NXY): cv2.circle(canvas0, tuple(q), 2, (150, 150, 150), -1, cv2.LINE_AA)
cv2.polylines(canvas0, [P(road)], False, (255, 140, 30), 4, cv2.LINE_AA)            # shortest road route: blue
gs, gg = P([tr["start"]])[0], P([tr["goal"]])[0]
cap = cv2.VideoCapture(vid_f); frames = []
while True:
    ok, fr = cap.read()
    if not ok: break
    frames.append(fr)
W = 1440; H = OV + 60
wr = cv2.VideoWriter(out_f, cv2.VideoWriter_fourcc(*"mp4v"), 20, (W, H))
def label(img, txt, org, s=0.7, c=(20, 20, 20), th=2):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, s, c, th, cv2.LINE_AA)
for i, fr in enumerate(frames):
    j = min(max(i - 1, 0), len(S) - 1)                       # state after step i
    t, x, y, yaw, spd, cur = S[j]
    paths = [p for p in tr["paths"] if p["t"] <= max(1, t)] or tr["paths"][:1]
    path = paths[-1]["path"] if paths else []
    ov = canvas0.copy()
    if path:
        q = P(NXY[path]); cv2.polylines(ov, [q], False, (0, 140, 255), 2, cv2.LINE_AA)
        for k_, qq in enumerate(q): cv2.circle(ov, tuple(qq), 6, (0, 165, 255), -1, cv2.LINE_AA); cv2.circle(ov, tuple(qq), 6, (30, 30, 30), 1, cv2.LINE_AA)
    if cur >= 0:
        cv2.circle(ov, tuple(P(NXY[int(cur)])[0]), 15, (255, 0, 255), 3, cv2.LINE_AA)
    trk = P(S[: j + 1, 1:3]); cv2.polylines(ov, [trk], False, (40, 40, 230), 3, cv2.LINE_AA)
    cv2.circle(ov, tuple(gs), 9, (60, 180, 60), -1, cv2.LINE_AA)
    cv2.drawMarker(ov, tuple(gg), (0, 215, 255), cv2.MARKER_STAR, 30, 3, cv2.LINE_AA)
    c = P([[x, y]])[0]; d = np.array([math.cos(yaw), math.sin(yaw)]) * 16 * 1.0
    tip = (c + d).astype(int); l = (c - d * 0.6 + np.array([-d[1], d[0]]) * 0.6).astype(int); r = (c - d * 0.6 - np.array([-d[1], d[0]]) * 0.6).astype(int)
    cv2.fillPoly(ov, [np.array([tip, l, r])], (20, 20, 200), cv2.LINE_AA)
    left = np.full((OV, 720, 3), 255, np.uint8)
    cam, sub = fr[:, :fr.shape[1] // 2], fr[:, fr.shape[1] // 2:]
    left[0:350, 5:715] = cv2.resize(cam, (710, 350)); left[370:720, 5:715] = cv2.resize(sub, (710, 350))
    frame = np.full((H, W, 3), 255, np.uint8); frame[:OV, :720] = left; frame[:OV, 720:] = ov
    label(frame, "Camera", (15, 340), 0.8, (255, 255, 255)); label(frame, "Subgoal Image" if cur >= 0 else "Goal Image", (15, 710), 0.8, (255, 255, 255))
    status = f"t {t/10:5.1f} s   Speed {spd:4.1f} km/h   Waypoint {path.index(int(cur)) if cur >= 0 and int(cur) in path else len(path)}/{len(path)}"
    if i == len(frames) - 1: status += f"   {'Reached' if tr['reached'] else 'Failed: ' + str(tr['end']).replace('_', ' ')}"
    label(frame, status, (15, OV + 40), 0.8)
    # legend
    lx = 728; ly = OV + 40
    for col, txt, kind in (((255, 140, 30), "Shortest Path", "line"), ((0, 165, 255), "Waypoints", "dot"), ((255, 0, 255), "Current Waypoint", "ring"),
                           ((40, 40, 230), "Car Track", "line"), ((0, 215, 255), "Goal", "star")):
        if kind == "line": cv2.line(frame, (lx, ly - 7), (lx + 30, ly - 7), col, 4, cv2.LINE_AA)
        elif kind == "dot": cv2.circle(frame, (lx + 15, ly - 7), 7, col, -1, cv2.LINE_AA)
        elif kind == "ring": cv2.circle(frame, (lx + 15, ly - 7), 9, col, 3, cv2.LINE_AA)
        else: cv2.drawMarker(frame, (lx + 15, ly - 7), col, cv2.MARKER_STAR, 22, 3, cv2.LINE_AA)
        label(frame, txt, (lx + 36, ly), 0.5, th=1); lx += 36 + int(9.2 * len(txt)) + 14
    wr.write(frame)
for _ in range(20): wr.write(frame)                          # hold the last frame 1 s
wr.release(); print(out_f, len(frames), "frames")
