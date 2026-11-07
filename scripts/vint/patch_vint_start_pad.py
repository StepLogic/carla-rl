#!/usr/bin/env python3
"""Let ViNT train on episode starts, opt-in per run, in place and idempotently.

ViNT_Dataset starts every trajectory's samples at frame context_size*spacing,
so the context a car has at a start -- its first frame repeated, which is what
the driver feeds (torch_rl.vint_driver / vint_topomap pad with the first frame)
-- never appears in training; v1 turned toward the goal from a standstill.
With VINT_START_PAD=1 in the environment, samples start at frame 0 and context
times below 0 read frame 0. Without it the sample index is unchanged, and the
clamp never fires (every context time is >= 0).

usage: patch_vint_start_pad.py <visualnav train>/vint_train/data/vint_dataset.py
"""
import sys

EDITS = [
    ("            begin_time = self.context_size * self.waypoint_spacing\n",
     "            begin_time = (0 if os.environ.get(\"VINT_START_PAD\") == \"1\"  # vint-carla\n"
     "                          else self.context_size * self.waypoint_spacing)\n"),
    ("            context = [(f_curr, t) for t in context_times]\n",
     "            context = [(f_curr, max(t, 0)) for t in context_times]  # vint-carla: start pad\n"),
]


def apply(src):
    if all(new in src for _, new in EDITS):
        return src
    for old, new in EDITS:
        if new in src:
            continue
        if src.count(old) != 1:
            raise SystemExit(f"expected {old.strip()!r} exactly once, found {src.count(old)}; "
                             "refusing to patch unknown code")
        src = src.replace(old, new)
    if "\nimport os\n" not in src and not src.startswith("import os\n"):
        raise SystemExit("the loader does not import os; refusing to patch")
    return src


def main():
    path = sys.argv[1]
    src = open(path).read()
    new = apply(src)
    if new == src:
        print(f"{path}: start padding already available (VINT_START_PAD=1)")
        return
    with open(path, "w") as fh:
        fh.write(new)
    print(f"{path}: VINT_START_PAD=1 trains from frame 0 with first-frame padding")


if __name__ == "__main__":
    main()
