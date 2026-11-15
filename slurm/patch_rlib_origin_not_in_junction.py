#!/usr/bin/env python3
"""ORIGIN_NOT_IN_JUNCTION=1 for rlib_integration's carla_core.pick_origin (user 2026-11-14: "ensure the
agent does not spawn in middle of junction").

The uniform branch already draws from waypoints outside junctions. The JUNCTION-BIASED branch does not: it
takes a waypoint INSIDE a junction and walks back `d` metres (5-25 m, the goal band), and in a wide junction
5 m back is still junction interior, so the car was spawned in the middle of one. With the switch on, a
set-back that lands in a junction is walked further back in 2 m steps (up to 40 m more) until it is on
ordinary road, so the origin is still BEFORE that junction and the route still drives through it.

Default off: every earlier run and every fixed evaluation episode set is unchanged (the random draws are
identical when the switch is off).

rlib_integration is not tracked in this worktree, so this is applied to each copy (code snapshot, cluster).
Idempotent; exits non-zero if the anchor is not found exactly once.

usage: patch_rlib_origin_not_in_junction.py <path to carla_core.py>
"""
import sys

EDITS = [
    ("ORIGIN_NOT_IN_JUNCTION (user 2026-11-14",
     "            prev = jwp.previous(float(d))\n"
     "            if prev:\n"
     "                return rnd.choice(prev), None, \"junction\"\n",
     "            prev = jwp.previous(float(d))\n"
     "            # ORIGIN_NOT_IN_JUNCTION (user 2026-11-14; slurm/patch_rlib_origin_not_in_junction.py):\n"
     "            # a set-back that is still junction interior is walked further back onto ordinary road.\n"
     "            if os.environ.get(\"ORIGIN_NOT_IN_JUNCTION\", \"0\") == \"1\":\n"
     "                _extra = 0.0\n"
     "                prev = [w for w in (prev or []) if not w.is_junction]\n"
     "                while not prev and _extra < 40.0:\n"
     "                    _extra += 2.0\n"
     "                    prev = [w for w in (jwp.previous(float(d) + _extra) or []) if not w.is_junction]\n"
     "            if prev:\n"
     "                return rnd.choice(prev), None, \"junction\"\n"),
]


def main(path):
    s = open(path).read()
    rc = 0
    for mark, old, new in EDITS:
        if mark in s:
            print(f"[patch] {mark.split(' (')[0]}: already applied")
            continue
        if s.count(old) != 1:
            print(f"[patch] {mark.split(' (')[0]}: anchor found {s.count(old)}x, expected 1"
                  " -- NOT patched", file=sys.stderr)
            rc = 1
            continue
        s = s.replace(old, new)
        print(f"[patch] {mark.split(' (')[0]}: applied")
    open(path, "w").write(s)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
