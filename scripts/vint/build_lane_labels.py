#!/usr/bin/env python3
"""Lane-centre waypoint labels for every expert episode (src/vae_nav/vint_lane_labels.py).

    build_lane_labels.py --paths paths.npz --xodr <OpenDrive dir> --out lane_labels.npz

paths.npz: per-step pos / speed of each il_data episode with town and seed (the
path fields only, no images). Positions and headings go through exactly the
converter's transform (vint_data.episode_to_traj: frame positions, right-handed,
headings from motion), so label row k belongs to frame k; the converter trims
fault tails afterwards. Needs the carla module (offline carla.Map, no server).
Prints, as a check, the lane offset of the label points (should be ~0).
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from vae_nav.vint_data import frame_positions, headings_from_motion, to_right_handed  # noqa: E402
from vae_nav.vint_lane_labels import lane_centre_labels, lane_polyline  # noqa: E402


def lane_offset(cmap, x, y):
    import carla
    w = cmap.get_waypoint(carla.Location(x=float(x), y=float(y), z=0.0), project_to_road=True,
                          lane_type=carla.LaneType.Driving)
    c = w.transform.location
    yaw = np.radians(w.transform.rotation.yaw)
    return float(np.dot([x - c.x, y - c.y], [-np.sin(yaw), np.cos(yaw)])), bool(w.is_junction)


def main(argv=None):
    import carla
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True)
    ap.add_argument("--xodr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-every", type=int, default=25)
    a = ap.parse_args(argv)
    z = np.load(a.paths)
    lens = z["lens"]
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    maps, names, P, Y, OK, L = {}, [], [], [], [], []
    check = []
    for i, (s, n) in enumerate(zip(starts, lens)):
        town, seed = str(z["town"][i]), int(z["seed"][i])
        if town not in maps:
            maps[town] = carla.Map(town, open(os.path.join(a.xodr, f"{town}.xodr")).read())
        fp = frame_positions(z["pos"][s:s + n])                     # CARLA, one per frame
        xy = to_right_handed(fp)
        yaw = headings_from_motion(xy)
        poly = to_right_handed(lane_polyline(maps[town], fp))
        pos, yw, ok = lane_centre_labels(xy, yaw, poly)
        names.append(f"{town}_{seed}")
        P.append(pos.astype(np.float32))
        Y.append(yw.astype(np.float32))
        OK.append(ok)
        L.append(len(xy))
        for t in range(0, len(xy), a.check_every):
            if ok[t]:
                for j in (1, 3, 5):
                    o, jx = lane_offset(maps[town], pos[t, j, 0], -pos[t, j, 1])   # back to CARLA
                    if not jx:
                        check.append(abs(o))
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(lens)} episodes", flush=True)
    ok_all = np.concatenate(OK)
    np.savez_compressed(a.out, names=np.array(names), lens=np.array(L),
                        position=np.concatenate(P), yaw=np.concatenate(Y), ok=ok_all)
    c = np.array(check)
    print(f"{len(names)} episodes, {len(ok_all)} frames, lane-centre labels on {ok_all.mean():.3f} "
          f"(the rest keep the recorded path) -> {a.out}")
    print(f"check: |lane offset| of label points (non-junction), median {np.median(c):.3f} m, "
          f"p95 {np.quantile(c, .95):.3f} m, share > 0.3 m {(c > 0.3).mean():.4f} (n={len(c)})")


if __name__ == "__main__":
    main()
