#!/usr/bin/env python3
"""Add GOAL_BRANCH=random to rlib_integration's goal sampler (user 2026-10-26).

Past a junction, carla_core's origin.next(d) returns one waypoint per branch and
the sampler took the LAST, usually straight across: with JUNCTION_GOAL_FRAC=0.5
only 8/40 eval goals were off-axis (vs 7/40 at 0). GOAL_BRANCH=random takes a
random branch instead, so most junction goals become real turns. Default
("last") is the old behaviour.

rlib_integration is a submodule populated only in the main checkout, which this
worktree must not edit, so slurm/sync_to_cluster.sh applies this to the
CLUSTER copy after syncing it. Idempotent; exits non-zero if the line it
replaces is not found exactly once, so a changed upstream cannot be silently
mis-patched.

usage: patch_rlib_goal_branch.py <path to carla_core.py>
"""
import sys

OLD = "                _cand = candidates[-1]          # furthest candidate\n"
NEW = ("                # GOAL_BRANCH=random (user 2026-10-26; applied by\n"
       "                # slurm/patch_rlib_goal_branch.py): a random branch past a\n"
       "                # junction instead of the last, usually straight across.\n"
       "                _cand = (random.choice(candidates)\n"
       "                         if os.environ.get(\"GOAL_BRANCH\", \"last\") == \"random\"\n"
       "                         else candidates[-1])          # furthest candidate\n")
MARK = "GOAL_BRANCH=random (user 2026-10-26"


def main(path):
    s = open(path).read()
    if MARK in s:
        print(f"[patch] goal branch: already applied ({path})")
        return 0
    if s.count(OLD) != 1:
        print(f"[patch] goal branch: anchor found {s.count(OLD)}x, expected 1 -- NOT patched",
              file=sys.stderr)
        return 1
    open(path, "w").write(s.replace(OLD, NEW))
    print(f"[patch] goal branch: applied ({path})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
