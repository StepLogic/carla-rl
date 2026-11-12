#!/usr/bin/env python3
"""Goal-sampler switches for rlib_integration's carla_core (user 2026-10-26).

GOAL_BRANCH=random -- past a junction origin.next(d) returns one waypoint per
  branch and the sampler took the LAST, usually straight across. Take a random
  one instead.
GOAL_PAST_JUNCTION=1 -- with JUNCTION_GOAL_FRAC the origin sat 5-25 m before a
  junction and the goal distance was drawn independently from the same band, so
  the goal mostly landed BEFORE the junction (one branch, straight ahead) or
  inside it (rejected): even with a random branch only 7/40 eval goals were off
  axis. Now the origin sits <= 8 m before the junction and the goal distance is
  drawn from the band values >= (metres to the junction entrance + 15), i.e.
  past the intersection on whichever branch.
Both default off: every earlier run is unchanged.

rlib_integration is a submodule populated only in the main checkout, which this
worktree must not edit, so slurm/sync_to_cluster.sh applies this to the CLUSTER
copy after syncing it. Idempotent per edit; exits non-zero if an anchor is not
found exactly once, so a changed upstream cannot be silently mis-patched.

usage: patch_rlib_goal_branch.py <path to carla_core.py>
"""
import sys

EDITS = [
    ("GOAL_BRANCH=random (user 2026-10-26",
     "                _cand = candidates[-1]          # furthest candidate\n",
     "                # GOAL_BRANCH=random (user 2026-10-26; applied by\n"
     "                # slurm/patch_rlib_goal_branch.py): a random branch past a\n"
     "                # junction instead of the last, usually straight across.\n"
     "                _cand = (random.choice(candidates)\n"
     "                         if os.environ.get(\"GOAL_BRANCH\", \"last\") == \"random\"\n"
     "                         else candidates[-1])          # furthest candidate\n"),
    ("GOAL_PAST_JUNCTION origin (user 2026-10-26",
     "        back = rnd.choice(lenghts) if lenghts else 20.0\n",
     "        back = rnd.choice(lenghts) if lenghts else 20.0\n"
     "        # GOAL_PAST_JUNCTION origin (user 2026-10-26; slurm/patch_rlib_goal_branch.py):\n"
     "        # sit close to the junction so a banded goal can land past it.\n"
     "        if os.environ.get(\"GOAL_PAST_JUNCTION\", \"0\") == \"1\":\n"
     "            back = rnd.choice([l for l in lenghts if l <= 8.0] or [5.0])\n"),
    ("GOAL_PAST_JUNCTION band (user 2026-10-26",
     "                _band = [d for d in lenghts if d <= _max_forward_m] or [_max_forward_m * 0.5]\n",
     "                _band = [d for d in lenghts if d <= _max_forward_m] or [_max_forward_m * 0.5]\n"
     "            # GOAL_PAST_JUNCTION band (user 2026-10-26; slurm/patch_rlib_goal_branch.py):\n"
     "            # a junction-biased origin draws only distances that end >= 15 m\n"
     "            # past the junction entrance, so the goal is beyond the turn.\n"
     "            if (_origin_kind == \"junction\"\n"
     "                    and os.environ.get(\"GOAL_PAST_JUNCTION\", \"0\") == \"1\"):\n"
     "                _dj, _w = 0.0, origin\n"
     "                while _dj < 30.0:\n"
     "                    _nx = _w.next(2.0)\n"
     "                    if not _nx:\n"
     "                        break\n"
     "                    _w, _dj = _nx[0], _dj + 2.0\n"
     "                    if _w.is_junction:\n"
     "                        break\n"
     "                _band = [d for d in lenghts if d >= _dj + 15.0] or _band\n"),
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
