#!/usr/bin/env python3
"""Let ViNT's loader read lane-centre labels, in place and idempotently.

ViNT_Dataset._compute_actions slices the recorded path for the waypoint labels.
With this patch a trajectory whose traj_data.pkl carries cf_position (N, 6, 2)
and cf_yaw (N, 6) (scripts/vint/add_lane_labels_to_tar.py) is labelled with
those instead -- the lane centre ahead, where the expert aims; a trajectory
without them is labelled exactly as before.

usage: patch_vint_cf_labels.py <visualnav train>/vint_train/data/vint_dataset.py
"""
import sys

OLD = ('        yaw = traj_data["yaw"][start_index:end_index:self.waypoint_spacing]\n'
       '        positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]\n')
NEW = ('        if "cf_position" in traj_data:  # vint-carla: lane-centre aim-point labels\n'
       '            yaw = traj_data["cf_yaw"][curr_time]\n'
       '            positions = traj_data["cf_position"][curr_time]\n'
       '        else:\n'
       '            yaw = traj_data["yaw"][start_index:end_index:self.waypoint_spacing]\n'
       '            positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]\n')


def apply(src):
    if NEW in src:
        return src
    if src.count(OLD) != 1:
        raise SystemExit(f"expected ViNT's label slice exactly once, found {src.count(OLD)}; "
                         "refusing to patch unknown code")
    return src.replace(OLD, NEW)


def main():
    path = sys.argv[1]
    src = open(path).read()
    new = apply(src)
    if new == src:
        print(f"{path}: already reads lane-centre labels")
        return
    with open(path, "w") as fh:
        fh.write(new)
    print(f"{path}: reads cf_position / cf_yaw when present")


if __name__ == "__main__":
    main()
