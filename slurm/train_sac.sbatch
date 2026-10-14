#!/bin/bash
# Train the torch recurrent-SAC image-goal arm on one cluster GPU.
#
# usage:  sbatch slurm/train_sac.sbatch
#         MAX_STEPS=3000 LEARNING_STARTS=200 sbatch slurm/train_sac.sbatch  # pilot
#
# The env block below is v52's recipe at the 5-25 m band the user chose for
# the image goal, since DEBiT needs the view and the goal to overlap.
#
#SBATCH --job-name=sac_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log
set -uo pipefail

# EVERY ONE OF THESE LINES IS A MEASURED LESSON, NOT A PREFERENCE.
#
# 1. apptainer run --nv -e AND NOTHING ELSE. No binds, no Vulkan environment
#    variables. Binding the host's /usr/share/vulkan/icd.d displaces the image's
#    own nvidia_icd.json, which --nv already mounts, and CARLA then dies after
#    4-9 s; every such failure in October came from carrying that bind. The
#    Ubuntu-22.04 rebuild (carla0915_u22.sif) has no ICD at all -- do not use it.
# 2. 0.9.16 MOVED THE SIMULATOR TO /workspace/CarlaUE4.sh. In 0.9.15 it was
#    /home/carla/CarlaUE4.sh.
# 3. POLL THE PORT WHILE THE SERVER RUNS, and allow at least 150 s. Startup takes
#    11-76 s depending on cache warmth. A foreground `timeout` around CARLA and a
#    check afterwards scores a HEALTHY server as dead (exit 124 is the timeout,
#    not the simulator).
# 4. PYTHONNOUSERSITE=1, always. The cluster home's ~/.local site-packages leak
#    into the env and already hold numpy/torch/cv2, so pip reported "already
#    satisfied" and installed nothing; the env only worked while that leak was
#    present. With this set, what breaks breaks here rather than mid-run.
# 5. 0.9.16 ON THE CLUSTER vs 0.9.15 LOCALLY is a real difference in the results,
#    not a detail: 0.9.15's client ships cp27/cp37 wheels only, so the client and
#    torch cannot share an interpreter. Every cluster number carries it.

REPO=${REPO:-$HOME/carla-rl}
# THE REBUILT IMAGE IS THE DEFAULT. The official carlasim/carla:0.9.16
# image renders on the CPU here (plain ubuntu:20.04, Mesa ICDs, no NVIDIA
# one) and segfaults at Epic; carla0916nv.sif is that simulator on an
# nvidia/opengl glvnd base with the ICD baked in, and holds 1684 MiB on
# the GPU. Do not point this back at the official image.
SIF=${CARLA_SIF:-$HOME/carla-cluster/carla0916nv.sif}
PY=${CLUSTER_PY:-$HOME/carla-cluster/envs/carla310/bin/python}
export CARLA_PORT=${CARLA_PORT:-$((2000 + RANDOM % 900))}
# ALL TOWNS EXCEPT Town02 AND Town03 (user 2026-10-14); those two stay held
# out for the transfer read. The trainer rotates the map every TOWN_EVERY
# episodes on this one server.
export ENV_TOWNS=${ENV_TOWNS:-Town01,Town04,Town05,Town06,Town07,Town10HD}
export CARLA_QUALITY=${CARLA_QUALITY:-Epic}
export RUN_NAME=${RUN_NAME:-carla_sac_debit_v1}
export MAX_STEPS=${MAX_STEPS:-1000000}

echo "=== $(date +%T) node $(hostname) job ${SLURM_JOB_ID:-none} ==="
echo "=== town $ENV_TOWNS port $CARLA_PORT quality $CARLA_QUALITY ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
module load apptainer 2>/dev/null || echo "[warn] no apptainer module; using PATH"
for f in "$SIF" "$PY" "$REPO/slurm/smoke_sac_cluster.py"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

CARLA_LOG=${SLURM_JOB_NAME:-sac_smoke}_${SLURM_JOB_ID:-0}_carla.log
apptainer run --nv -e "$SIF" /workspace/CarlaUE4.sh \
  -RenderOffScreen -nosound -quality-level="$CARLA_QUALITY" \
  -carla-rpc-port="$CARLA_PORT" > "$CARLA_LOG" 2>&1 &
CARLA_PID=$!
echo "[carla] pid $CARLA_PID, log $CARLA_LOG"

# Poll WHILE it runs. 150 s, and the server's own liveness is checked each pass
# so a crashed start is reported as a crash rather than as a timeout.
up=0
for i in $(seq 1 150); do
  if ! kill -0 "$CARLA_PID" 2>/dev/null; then
    echo "FATAL: CARLA exited after ${i}s. Last lines:"; tail -25 "$CARLA_LOG"; exit 1
  fi
  if command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -q ":$CARLA_PORT\b"; then
    up=$i; break
  fi
  sleep 1
done
[ "$up" -gt 0 ] && echo "[carla] listening after ${up}s" || {
  echo "[carla] port never opened in 150s; continuing anyway -- the client's own "
  echo "        timeout is the real test, and 'ss' may be absent in this image."; }

# IS THE SIMULATOR ACTUALLY ON THE GPU? The container's log carries "lavapipe is
# not a conformant vulkan implementation", which is the SOFTWARE rasteriser
# announcing itself -- if CARLA fell back to it, Epic quality would be rendered
# on the CPU and the run would be unusably slow while looking healthy. The
# decisive check is whether the CarlaUE4 process holds GPU memory (a working
# start holds ~1.6 GB), so take it here rather than inferring from the log.
echo "=== GPU processes after startup ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null \
  | grep -iE "carla|pid" || echo "[warn] CARLA holds NO GPU memory -- suspect software rendering"

export PYTHONNOUSERSITE=1
# rlib_integration IS NESTED: the importable package is
# dependencies/rlib_integration/rlib_integration, and the directory above it has
# its own __init__.py. With only $REPO/dependencies on the path, `import
# rlib_integration` binds to that OUTER wrapper, which holds no carla_env, and
# the failure reads as a missing module rather than a path one. Locally the
# package is pip-installed in develop mode, so this never showed up there.
# $REPO/dependencies IS DELIBERATELY ABSENT. With it on the path, `import
# rlib_integration` binds to dependencies/rlib_integration/__init__.py -- an
# EMPTY wrapper -- and shadows the real package one level down, so the failure
# reads as a missing module rather than a shadowed one. Adding the inner path
# after it does not help: the first match wins. Locally the package is
# pip-installed in develop mode, which is why this only appears here. Nothing
# else needs $REPO/dependencies: debit_encoder puts croco on sys.path itself.
export PYTHONPATH="$REPO:$REPO/src:$REPO/dependencies/rlib_integration${PYTHONPATH:+:$PYTHONPATH}"
export DEBIT_SRC=${DEBIT_SRC:-$REPO/dependencies}
# The peer measured that importing a simulator client can corrupt multi-threaded
# torch CPU matmuls; they pin these in every worker and so do we.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export GATED=1 PERCEPTION=debit
export OBS_WITH_MODE=1 OBS_WITH_GOAL=1 OBS_WITH_STATE=1
export GATED_GOAL_REWARD=explore GOAL_EXPLORE_ARRIVAL_BONUS=50 EXPLORE_ARRIVAL_BONUS=10
export DISCOUNT=0.99 EXPLORE_DISCOUNT=0.99
export GOAL_DIST_MIN=5 GOAL_DIST_MAX=25 GOAL_DIST_START=25 GOAL_DIST_TARGET=25
export LANE_PROGRESS_MODE=heading LANE_PROGRESS_SCALE=1
export MAX_STEER=1.0 MAX_THROTTLE=1.0 INCREMENTAL_ACTIONS=0 TARGET_SPEED=20.0
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120 ENV_RGB_CHANNEL_FIX=1
export SHORT_HORIZON=1 LANE_FOLLOW=1 LANE_STRICT=1 LANE_EXPLORE=0
export GATED_EXPLORE_MAX_STEPS=1000 MAX_EPISODE_STEPS=2000
export EXPERT_PROB=0 BC_WEIGHT=0 EPISODE_HER=0

cd "$REPO" || exit 1
echo "=== $(date +%T) python ==="
"$PY" -u train_carla_sac.py
rc=$?
echo "=== $(date +%T) train rc=$rc ==="
kill "$CARLA_PID" 2>/dev/null
sleep 5
kill -9 "$CARLA_PID" 2>/dev/null
echo "=== carla log tail ==="; tail -15 "$CARLA_LOG"
exit $rc
