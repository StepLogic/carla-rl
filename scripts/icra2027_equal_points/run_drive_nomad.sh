#!/bin/bash
# USER 2026-09-13 "use traces as map for NoMaD and compare on same episodes - measure SR and SPL":
# uniform[2 m] trace map (547 nodes), NoMaD distance-head localisation + path on the two-way trace graph,
# baseline 300x300 camera + SetPointAgent PID, same 40 episodes and termination as the v6n maps.
# Env: nomad_carla (clone of oatomobile + gymnasium, opencv, tensorboard). Resumes from the JSONL.
# Usage: run_drive_nomad.sh <log> [extra drive_nomad.py args]
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl
LOG="$1"; shift
cd "$W" || exit 1
export CARLA_ROOT=/home/robotlab/Apps/CARLA_0.9.15 CARLA_QUALITY=Epic
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia
# SDL_VIDEODRIVER=dummy is set INSIDE drive_nomad.py after CARLA starts: exported here it reaches
# CarlaUE4, which segfaulted on start (2026-09-13 23:40).
export PYGAME_HIDE_SUPPORT_PROMPT=1
PY=/home/robotlab/anaconda3/envs/nomad_carla/bin/python
say(){ echo "=== [nomad $(date +%H:%M:%S)] $* ==="; }
kill_all_carla() {
  for _ in 1 2 3 4 5 6; do
    pkill -9 -f 'CarlaUE4-Linux-Shipping' 2>/dev/null
    pkill -9 -f 'CarlaUE4'                2>/dev/null
    pkill -9 -f 'CARLA_0.9.15'            2>/dev/null
    sleep 2
    if ! ps -eo args | grep -i '[C]arlaUE4' >/dev/null; then
      return 0
    fi
  done
  say "WARNING: a CARLA process survived six kill rounds:"
  ps -eo pid,args | grep -i '[C]arlaUE4' | cut -c1-120
  return 1
}
say "pre-flight: killing every CARLA on this machine"
kill_all_carla || exit 1
sleep 20
for ATTEMPT in 1 2 3 4 5 6 7 8; do
  say "attempt $ATTEMPT (log $LOG)"
  $PY -u drive_nomad.py "$@" >> "$LOG" 2>&1
  RC=$?
  if tail -3 "$LOG" | grep -aq "^\[nomad\] done:"; then
    say "DONE (rc $RC)"; kill_all_carla; exit 0
  fi
  say "attempt $ATTEMPT ended rc $RC without done; killing CARLA and resuming"
  kill_all_carla; sleep 30
done
say "FAILED after 8 attempts"; exit 1
