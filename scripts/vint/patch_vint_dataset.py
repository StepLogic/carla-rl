#!/usr/bin/env python3
"""Fix ViNT's goal bound for waypoint_spacing > 1, in place and idempotently.

ViNT_Dataset._build_index stores the farthest goal in FRAMES
(max_dist_cat * waypoint_spacing), and _sample_goal multiplies the sampled
offset by waypoint_spacing AGAIN. Every released ViNT dataset uses spacing 1,
where the two agree; at our stride 3 goals landed up to 3x too far -- past the
episode end (the CPU probe's "AssertionError: 82 an 71") -- and distance labels
reached 60 against max_dist_cat 20. Stored as a count of STRIDES, every goal is
inside the episode and every distance label is <= max_dist_cat; spacing 1 is
unchanged.

usage: patch_vint_dataset.py <visualnav train>/vint_train/data/vint_dataset.py
"""
import sys

OLD = "max_goal_distance = min(self.max_dist_cat * self.waypoint_spacing, traj_len - curr_time - 1)"
NEW = ("max_goal_distance = min(self.max_dist_cat, (traj_len - curr_time - 1) "
       "// self.waypoint_spacing)")


def main():
    path = sys.argv[1]
    src = open(path).read()
    if NEW in src and OLD not in src:
        print(f"{path}: already patched")
        return
    if src.count(OLD) != 1:
        sys.exit(f"{path}: expected the original goal bound exactly once, found "
                 f"{src.count(OLD)}; refusing to patch unknown code")
    with open(path, "w") as fh:
        fh.write(src.replace(OLD, NEW))
    print(f"{path}: goal bound now counted in strides")


if __name__ == "__main__":
    main()
