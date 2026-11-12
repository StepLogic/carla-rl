#!/bin/bash
# USER 2026-09-13 "lets drive these paths and compare success rate":
# two-way sequence maps (all 7), 40 episodes, option C (re-localise at each node within
# the next 3 plan hops), reversed edges driven as planned, success = 4 m of the true goal,
# end on collision or cap 2 x (plan m / 0.24) + 300, explore fallback after the plan.
# Usage: run_drive_plan_c.sh <out.jsonl> <log> [extra drive_plan_c.py args]
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl
OUT="$1"; LOG="$2"; shift 2
cd "$W" || exit 1
source /home/robotlab/anaconda3/etc/profile.d/conda.sh
conda activate carla-rl || { echo FATAL; exit 1; }
set +u; eval "$(grep -E '^export ' run_gated.sh)"
while IFS='=' read -r k v; do case "$k" in ''|'#'*) continue;; esac; export "$k=$v"; done < results/run_env/carla_gated_her_v6n.env
export CARLA_START_SERVER=1 CARLA_QUALITY=Epic CARLA_PORT=2000 CARLA_PORTS=2000
export EXPERT_PROB=0 RESUME=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export ENV_SKIP_IMAGE_FEATURES=1 PYTHONPATH=src
# no TimeLimit wrapper: the driver owns the step cap
export NO_TIMELIMIT=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia
say(){ echo "=== [drive $(date +%H:%M:%S)] $* ==="; }
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
  say "attempt $ATTEMPT -> $OUT (log $LOG)"
  python -u drive_plan_c.py --out "$OUT" "$@" >> "$LOG" 2>&1
  RC=$?
  if tail -3 "$LOG" | grep -aq "^\[drive\] done:\|^\[drive\] 0 episodes to drive"; then
    say "DONE (rc $RC)"; kill_all_carla; exit 0
  fi
  say "attempt $ATTEMPT ended rc $RC without done; killing CARLA and resuming"
  kill_all_carla; sleep 30
done
say "FAILED after 8 attempts"; exit 1
