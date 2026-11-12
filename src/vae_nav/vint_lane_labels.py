"""Lane-centre waypoint labels for ViNT (training labels only; the policy sees images).

ViNT's waypoints are used as AIM points: the driver (torch_rl.vint_driver) steers
at one of them, as ViNT's own PD controller does. Trained on the recorded car's
future positions, ViNT learns to aim where the noisy expert car WENT, which keeps
about 3/4 of a lane offset 5.6 m ahead (diagnostic 2026-11-07); the expert's own
controller aims at the lane centre. The label here is that aim point: for frame t,
the lane centre of the car's route at arc length s_t + j * spacing (j = 1..n),
with the lane's heading; entry 0 is the car itself (ViNT's local-frame origin).
A frame farther than max_gap_m from its lane keeps the recorded-path label.

All geometry is right-handed (x, y left), as vint_data writes traj_data.pkl.
"""
import math

import numpy as np


def _cum(poly):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def point_at(poly, s, cum=None):
    """Position and heading (radians) at arc length s, clamped to the polyline."""
    poly = np.asarray(poly, np.float64)
    cum = _cum(poly) if cum is None else cum
    s = float(np.clip(s, 0.0, cum[-1]))
    i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(poly) - 2))
    seg = poly[i + 1] - poly[i]
    L = float(np.hypot(*seg))
    f = 0.0 if L == 0.0 else (s - cum[i]) / L
    return poly[i] + f * seg, math.atan2(seg[1], seg[0])


def project(poly, p, cum=None, s_lo=None, s_hi=None):
    """Nearest point of the polyline to p -> (arc length, distance); optionally only
    over the segments overlapping [s_lo, s_hi] (monotone progress along a route)."""
    poly = np.asarray(poly, np.float64)
    cum = _cum(poly) if cum is None else cum
    a, b = poly[:-1], poly[1:]
    ok = np.ones(len(a), bool)
    if s_lo is not None:
        ok &= cum[1:] >= s_lo
    if s_hi is not None:
        ok &= cum[:-1] <= s_hi
    if not ok.any():
        ok[:] = True
    idx = np.flatnonzero(ok)
    a, b = a[idx], b[idx]
    d = b - a
    L2 = np.maximum((d ** 2).sum(1), 1e-12)
    f = np.clip(((np.asarray(p, np.float64) - a) * d).sum(1) / L2, 0.0, 1.0)
    q = a + f[:, None] * d
    gap = np.linalg.norm(q - p, axis=1)
    k = int(np.argmin(gap))
    return float(cum[idx[k]] + f[k] * math.sqrt(L2[k])), float(gap[k])


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def lane_centre_labels(xy, yaw, poly, spacing=1.1346, n=5, stride=3, max_gap_m=2.5,
                       back_m=2.0, ahead_m=15.0, max_branch_gap_m=3.0):
    """-> (positions (N, n+1, 2), yaws (N, n+1), ok (N,)) per frame of a trajectory
    (xy (N, 2), yaw (N,), right-handed) against its route's lane centre `poly`.
    A label point farther than max_branch_gap_m from where the car itself was at
    the same distance along its path (a wrong branch of the lane chain) sends the
    frame back to the recorded-path label."""
    xy = np.asarray(xy, np.float64)
    yaw = np.asarray(yaw, np.float64)
    poly = np.asarray(poly, np.float64)
    cum = _cum(poly)
    car_cum = _cum(xy) if len(xy) > 1 else np.zeros(1)
    N = len(xy)
    pos = np.zeros((N, n + 1, 2))
    yw = np.zeros((N, n + 1))
    ok = np.zeros(N, bool)
    s_prev = None
    for t in range(N):
        if s_prev is None:
            s, gap = project(poly, xy[t], cum)
        else:
            s, gap = project(poly, xy[t], cum, s_prev - back_m, s_prev + ahead_m)
            if gap > max_gap_m:
                s, gap = project(poly, xy[t], cum)
        if gap <= max_gap_m:
            s_prev = s
            ok[t] = True
            pos[t, 0], yw[t, 0] = xy[t], yaw[t]
            for j in range(1, n + 1):
                p, h = point_at(poly, s + j * spacing, cum)
                pos[t, j] = p
                yw[t, j] = yaw[t] + _wrap(h - yaw[t])
                d = car_cum[t] + j * spacing
                if len(xy) > 1 and d <= car_cum[-1]:
                    q, _ = point_at(xy, d, car_cum)
                    if float(np.hypot(*(p - q))) > max_branch_gap_m:
                        ok[t] = False
        if not ok[t]:
            # the recorded path, padded at the end as ViNT's _compute_actions pads
            sl = list(range(t, min(t + stride * n + 1, N), stride))
            sl += [sl[-1]] * (n + 1 - len(sl))
            pos[t], yw[t] = xy[sl], yaw[sl]
    return pos, yw, ok


def aim_from_plan(plan_xy, car_xy, spacing=1.1346, n=5, max_gap_m=4.0):
    """DAgger labels (EXPERT_RECORD_AIM=1): the lane-centre aim points j * spacing
    (j = 1..n) ahead of the car's projection on the expert's CURRENT plan -- the
    route it would drive from here, replanned when the car strays -- with the
    plan's heading there. CARLA coordinates in and out; yaw in CARLA's convention
    (atan2 in CARLA x/y, radians). -> (aims (n, 2), yaws (n,), ok); not ok without
    a plan or farther than max_gap_m from it (the frame is then left unlabelled)."""
    plan = np.asarray(plan_xy, np.float64).reshape(-1, 2)
    none = (np.zeros((n, 2)), np.zeros(n), False)
    if len(plan) < 2 or float(_cum(plan)[-1]) <= 0.0:
        return none
    cum = _cum(plan)
    s, gap = project(plan, np.asarray(car_xy, np.float64)[:2], cum)
    if gap > max_gap_m:
        return none
    aims, yaws = np.zeros((n, 2)), np.zeros(n)
    for j in range(1, n + 1):
        aims[j - 1], yaws[j - 1] = point_at(plan, s + j * spacing, cum)
    return aims, yaws, True


def lane_polyline(cmap, xy_carla, step=0.5, extra_m=20.0, probe_m=10.0):
    """The lane centre of the route a recorded car drove, in CARLA coordinates:
    chained carla.Waypoint.next(step) from the lane under the first position; at a
    branch, the successor whose next probe_m metres stay closest to the car's own
    path. Needs the carla module (an offline carla.Map from the town's .xodr)."""
    import carla

    xy_carla = np.asarray(xy_carla, np.float64)
    path_len = float(np.linalg.norm(np.diff(xy_carla, axis=0), axis=1).sum())
    w = cmap.get_waypoint(carla.Location(x=float(xy_carla[0, 0]), y=float(xy_carla[0, 1]), z=0.0),
                          project_to_road=True, lane_type=carla.LaneType.Driving)
    pts = [(w.transform.location.x, w.transform.location.y)]
    k = 0                                      # progress index along the car's path
    total = 0.0

    def path_gap(loc, lo):
        seg = xy_carla[lo:min(lo + 80, len(xy_carla))]
        return float(np.min(np.hypot(seg[:, 0] - loc.x, seg[:, 1] - loc.y)))

    while total < path_len + extra_m:
        nxt = w.next(step)
        if not nxt:
            break
        if len(nxt) > 1:
            def score(c):
                g, cc, m = 0.0, c, 0
                while m * step < probe_m:
                    g += path_gap(cc.transform.location, k)
                    nn = cc.next(step)
                    if not nn:
                        break
                    cc, m = nn[0], m + 1
                return g / max(m, 1)
            w = min(nxt, key=score)
        else:
            w = nxt[0]
        loc = w.transform.location
        pts.append((loc.x, loc.y))
        total += step
        seg = xy_carla[k:min(k + 40, len(xy_carla))]
        if len(seg):
            k += int(np.argmin(np.hypot(seg[:, 0] - loc.x, seg[:, 1] - loc.y)))
    return np.asarray(pts, np.float64)
