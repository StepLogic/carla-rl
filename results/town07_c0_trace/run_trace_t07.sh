#!/usr/bin/env bash
# c0 (il_bc_c0 step_15000) collects map data in Town07: one unbroken drive with junction
# hops, the expert recovering only after faults (user 2026-11-12: "also can we try collect
# map data with trained policy and show me"). Host CARLA 0.9.15, windowed so it can be
# watched. Code: snapshot of merge/all-2026-11-12 (8d148d6).
#   bash run_trace_t07.sh                      # 3 km pilot
#   TRACE_DIST_M_T07=8000 RUN=... bash run_trace_t07.sh   # a longer drive
set -u
D=/home/robotlab/carla-sac-local
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/vint-carla/results/town07_c0_trace
export REPO=$W/code_8d148d6
export CLUSTER_PY=$D/venv/bin/python
export CARLA_LOCAL_ROOT=/home/robotlab/Apps/CARLA_0.9.15 CARLA_SIF=/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4.sh
export CARLA_OFFSCREEN=0 DISPLAY=${DISPLAY:-:1} CARLA_QUALITY=Epic
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export DEBIT_SRC=/home/robotlab/scratch/carla-rl/dependencies DEBIT_WEIGHTS=$D/weights/debit_tiny.pth
export CKPT_DIR=$W/checkpoints PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRACE_DIST_M_T07=${TRACE_DIST_M_T07:-3000}
RUN=${RUN:-c0_unbroken_Town07_pilot}
mkdir -p "$CKPT_DIR"
p=3100
# Never attach to someone else's server: abort if our port is already taken.
if ss -ltn | grep -q ":$p "; then echo "=== port $p busy -- abort"; exit 3; fi
echo "=== $(date +%T) TRACE c0 Town07 $RUN budget ${TRACE_DIST_M_T07} m port $p"
REWARD_OVERLAY=slurm/il_trace_unbroken_Town07.env ENV_TOWNS=Town07 RUN_NAME=$RUN \
  CARLA_PORT=$p EVAL_FROM=$D/checkpoints/il_bc_c0/step_15000 \
  bash "$REPO/slurm/train_sac.sbatch"
echo "=== $(date +%T) TRACE c0 Town07 $RUN rc=$?"
