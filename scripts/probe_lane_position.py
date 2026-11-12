#!/usr/bin/env python3
"""Does the seg-VAE latent carry WHERE IN THE LANE the car is?

THE QUESTION THIS SETTLES. v43 leaves the road in 57-74% of long episodes and
94% of its faults are drifting over the centre line on open-road curves. That is
a lane-keeping failure. Before changing the reward, the action box or the data,
it is worth knowing whether the observation the policy receives even contains
the quantity it is failing to control. If distance-from-lane-centre is not
decodable from the 512-d latent, the policy is being asked to hold a lane it
cannot see, and no amount of behaviour cloning or reward shaping fixes that.

GROUND TRUTH comes from OpenDRIVE, not from the simulator: each pose is
projected to its lane centre and we take the SIGNED lateral offset (left
negative, right positive) and the heading error against the lane's own
direction. No CARLA server is needed.

THE SPLIT IS THE WHOLE EXPERIMENT. Kept samples are ~2.15 m apart and strongly
correlated with their neighbours, so a random train/test split leaks: a test
sample's neighbours sit in the training set two metres away and an R^2 near 1
means nothing. Splits here are:
  by-segment   whole driving segments held out within one town
  cross-town   fit on one town, test on another (the honest transfer number)
Both are reported, because the gap between them is itself informative.

CONTROLS, so a number cannot be read as more than it is:
  mean         predict the training mean. R^2 <= 0 means the probe learned nothing.
  state        the 9-d proprioceptive block (speed, heading, goal bearing). It
               has no camera in it, so whatever it decodes is what pose alone
               gives you and the latent must beat it to have earned anything.
  shuffled     latents paired with other samples' labels. Any R^2 above ~0 here
               means the split is leaking and the real numbers are inflated.

usage: probe_lane_position.py TRACE.pkl [TRACE2.pkl ...]
"""
import os
import pickle
import sys

import numpy as np

XODR = ("/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/"
        "OpenDrive/{town}.xodr")


def wrap_deg(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def labels_from_map(town, pos, yaw):
    """-> (lateral_offset_m, heading_error_deg, valid). Signed offset: the
    component of (pose - lane centre) along the lane's RIGHT vector."""
    sys.path.append("/home/robotlab/Apps/CARLA_0.9.15/PythonAPI/carla")
    import carla
    wmap = carla.Map(town, open(XODR.format(town=town)).read())
    off = np.full(len(pos), np.nan)
    herr = np.full(len(pos), np.nan)
    for i, (p, y) in enumerate(zip(pos, yaw)):
        wp = wmap.get_waypoint(carla.Location(x=float(p[0]), y=float(p[1]),
                                              z=0.0), project_to_road=True)
        if wp is None:
            continue
        c = wp.transform.location
        lane_yaw = np.radians(wp.transform.rotation.yaw)
        # CARLA is left-handed: right = (-sin yaw, cos yaw)
        rx, ry = -np.sin(lane_yaw), np.cos(lane_yaw)
        off[i] = (p[0] - c.x) * rx + (p[1] - c.y) * ry
        herr[i] = wrap_deg(np.degrees(y) - wp.transform.rotation.yaw)
    return off, herr, np.isfinite(off) & np.isfinite(herr)


def ridge(X, y, lam=10.0):
    X1 = np.c_[X, np.ones(len(X))]
    A = X1.T @ X1 + lam * np.eye(X1.shape[1])
    A[-1, -1] -= lam                                  # do not penalise the bias
    return np.linalg.solve(A, X1.T @ y)


def apply_w(X, w):
    return np.c_[X, np.ones(len(X))] @ w


def r2(y, yhat):
    ss = float(np.sum((y - yhat) ** 2))
    tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss / tot if tot > 0 else float("nan")


def load(path):
    with open(path, "rb") as fh:
        tr = pickle.load(fh)
    town = tr["meta"]["town"]
    pos = np.asarray(tr["positions"], np.float64)[:, :2]
    yaw = np.asarray(tr["yaws"], np.float64)
    seg = np.asarray(tr["segments"], int)
    lat = np.asarray(tr["obs_driven"]["latent"], np.float64)
    st = np.asarray(tr["obs_driven"]["state"], np.float64)
    off, herr, ok = labels_from_map(town, pos, yaw)
    print(f"[{os.path.basename(path)}] {town}: {ok.sum()}/{len(pos)} poses on a "
          f"lane | offset sd {np.nanstd(off[ok]):.2f} m, "
          f"heading error sd {np.nanstd(herr[ok]):.1f} deg")
    return dict(town=town, X=lat[ok], S=st[ok], y_off=off[ok],
                y_head=herr[ok], seg=seg[ok], name=os.path.basename(path))


def fit_report(tag, Xtr, ytr, Xte, yte, unit):
    w = ridge(Xtr, ytr)
    pred = apply_w(Xte, w)
    rmse = float(np.sqrt(np.mean((yte - pred) ** 2)))
    base = float(np.sqrt(np.mean((yte - ytr.mean()) ** 2)))
    print(f"    {tag:22s} R2 {r2(yte, pred):6.3f}   RMSE {rmse:6.2f} {unit}"
          f"   (predict-mean RMSE {base:6.2f})")
    return r2(yte, pred)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    ds = [load(p) for p in sys.argv[1:]]
    rng = np.random.default_rng(0)

    for d in ds:
        segs = np.unique(d["seg"])
        if len(segs) < 4:
            continue
        rng.shuffle(segs)
        te_seg = set(segs[: max(1, len(segs) // 3)].tolist())
        te = np.array([s in te_seg for s in d["seg"]])
        tr = ~te
        print(f"\n{d['town']} -- BY-SEGMENT split "
              f"({tr.sum()} train / {te.sum()} test samples, "
              f"{len(segs) - len(te_seg)}/{len(te_seg)} segments)")
        for name, y, unit in (("distance from lane centre", "y_off", "m"),
                              ("heading error", "y_head", "deg")):
            print(f"  {name}:")
            fit_report("latent (512-d)", d["X"][tr], d[y][tr], d["X"][te], d[y][te], unit)
            fit_report("state only (9-d)", d["S"][tr], d[y][tr], d["S"][te], d[y][te], unit)
            sh = rng.permutation(tr.sum())
            fit_report("latent, labels shuffled", d["X"][tr], d[y][tr][sh],
                       d["X"][te], d[y][te], unit)

    for a in ds:
        for b in ds:
            if a["town"] == b["town"]:
                continue
            print(f"\n{a['town']} -> {b['town']} -- CROSS-TOWN "
                  f"({len(a['X'])} train / {len(b['X'])} test)")
            for name, y, unit in (("distance from lane centre", "y_off", "m"),
                                  ("heading error", "y_head", "deg")):
                print(f"  {name}:")
                fit_report("latent (512-d)", a["X"], a[y], b["X"], b[y], unit)
                fit_report("state only (9-d)", a["S"], a[y], b["S"], b[y], unit)


if __name__ == "__main__":
    main()
