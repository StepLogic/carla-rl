#!/usr/bin/env bash
# The local GPU slot handed over by the map benchmark session at 18:10 on 2026-11-12:
# ViNT's Town07 main-table rows first, then the junction bands (only if the ViNT runner
# finishes). One GPU job at a time; each runner refuses to start beside another CARLA.
set -u
R=/home/robotlab/scratch/carla-rl/.claude/worktrees/vint-carla/results
bash "$R/vint_town07/run_vint_t07.sh"
rc=$?
echo "=== $(date +%T) ViNT Town07 runner rc=$rc"
[ "$rc" = 0 ] || { echo "=== ViNT runner failed: bands NOT started"; exit "$rc"; }
for i in $(seq 1 60); do                       # let the last cell's own server exit
  [ "$(ps -eo args | grep -c '[C]arlaUE4-Linux')" = 0 ] && break
  sleep 2
done
bash "$R/junction_bands/run_bands.sh"
echo "=== $(date +%T) junction-band runner rc=$?"
