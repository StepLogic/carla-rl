#!/usr/bin/env python3
"""Let ViNT's loader read lane-centre labels, in place and idempotently.

1. ViNT_Dataset._compute_actions slices the recorded path for the waypoint
   labels. A trajectory whose traj_data.pkl carries cf_position (N, 6, 2) and
   cf_yaw (N, 6) (scripts/vint/add_lane_labels_to_tar.py, or the converter for a
   DAgger round) is labelled with those instead -- the lane centre ahead, where
   the expert aims; a trajectory without them is labelled exactly as before.
2. ViNT_Dataset._build_index leaves out a frame whose cf_ok is False: a DAgger
   frame with no expert plan there has no label (user 2026-11-07, "use DAGGER").

Each edit applies on its own, so a loader patched before edit 2 existed is
upgraded in place.

usage: patch_vint_cf_labels.py <visualnav train>/vint_train/data/vint_dataset.py
"""
import sys

EDITS = [
    ('        yaw = traj_data["yaw"][start_index:end_index:self.waypoint_spacing]\n'
     '        positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]\n',
     '        if "cf_position" in traj_data:  # vint-carla: lane-centre aim-point labels\n'
     '            yaw = traj_data["cf_yaw"][curr_time]\n'
     '            positions = traj_data["cf_position"][curr_time]\n'
     '        else:\n'
     '            yaw = traj_data["yaw"][start_index:end_index:self.waypoint_spacing]\n'
     '            positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]\n'),
    ('            for curr_time in range(begin_time, end_time):\n',
     '            for curr_time in range(begin_time, end_time):\n'
     '                if "cf_ok" in traj_data and not traj_data["cf_ok"][curr_time]:\n'
     '                    continue  # vint-carla: an unlabelled DAgger frame\n'),
]


def apply(src):
    for old, new in EDITS:
        if new in src:
            continue
        if src.count(old) != 1:
            raise SystemExit(f"expected {old.strip()[:60]!r}... exactly once, found "
                             f"{src.count(old)}; refusing to patch unknown code")
        src = src.replace(old, new)
    return src


def main():
    path = sys.argv[1]
    src = open(path).read()
    new = apply(src)
    if new == src:
        print(f"{path}: already reads lane-centre labels and skips unlabelled frames")
        return
    with open(path, "w") as fh:
        fh.write(new)
    print(f"{path}: reads cf_position / cf_yaw when present, skips cf_ok False frames")


if __name__ == "__main__":
    main()
