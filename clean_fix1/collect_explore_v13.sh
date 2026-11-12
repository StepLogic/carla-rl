#!/usr/bin/env bash
# OFFLINE EVAL of a v10-vel75 checkpoint (user 2026-09-23: "Evaluate
# checkpoint", after the 2M run finished rc=0). Mirrors eval_v9_town.sh
# exactly: one town, one server, one invocation; the port lock is refused
# if any trainer holds it; EVERY experiment-defining flag is replayed from
# the run's snapshot (results/run_env/carla_gated_her_v13_entunit_vc1.env)
# so the observation stack (PERCEPTION=vc1, VC1_READOUT=pyramid) and the
# reward regime (counting regime, HER_RECOST, GOAL_SPEED_BONUS=0.75,
# fines 10.0 both modes) match training; the checkpoint loader refuses any
# remaining mismatch before driving.
#
# THE BAND IS THE SNAPSHOT'S 5-200 m: the snapshot carries
# GOAL_DIST_MIN=5 / GOAL_DIST_MAX=200 (mixed-band recipe), so NOT passing
# --goal-min/--goal-max samples the same band training sampled. Do not
# pass a band unless you intend an off-distribution probe.
#
# Server self-starts via ensure_carla (Epic, windowed, CARLA_OFFSCREEN=0
# per the snapshot). GPU-gated on free VRAM.
#
# Usage: eval_v10_vel75.sh [checkpoint.pkl] [TOWN] [SUFFIX]
#   checkpoint defaults to checkpoints/carla_gated_her_v10_vel75_vc1/
#   checkpoint_2000000.pkl; TOWN defaults to Town05.
set -euo pipefail
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl/clean_fix1

# SNAP overrides the replayed snapshot (for evaluating another run's
# checkpoint on its own recipe, e.g. the 0.5-control run's).
SNAP="${SNAP:-results/run_env/carla_gated_her_v13_entunit_vc1.env}"

CK="${1:-checkpoints/carla_gated_her_v13_entunit_vc1/checkpoint_450000.pkl}"
TOWN="${2:-Town05}"
SUFFIX="${3:-}"
[ -f "$CK" ] || { echo "FATAL: no $CK"; exit 1; }
[ -f "$SNAP" ] || { echo "FATAL: no $SNAP"; exit 1; }
TAG=$(basename "$CK" .pkl)   # e.g. checkpoint_2000000

# Port lock, held for the whole eval (same lock the trainers use).
LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
flock -n 200 || { echo "REFUSING TO START: another CARLA job holds $LOCK." >&2; exit 1; }
echo "[Lock] acquired $LOCK (pid $$)"

# GPU gate: server (Epic ~3.6-5.9 GB) + eval torch VC-1 + jax.
NEED_MB="${NEED_MB:-7000}"
MAX_WAIT_H="${MAX_WAIT_H:-2}"
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "${free:-0}" -ge "$NEED_MB" ] && { echo "[GPU] ${free} MiB free -- starting"; break; }
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[GPU] gave up; ${free} MiB free." >&2; exit 1; }
  echo "[GPU] ${free} MiB free, need ${NEED_MB}; waiting"; sleep 60
done

source /home/robotlab/anaconda3/etc/profile.d/conda.sh
conda activate carla-rl || { echo "FATAL: no carla-rl env"; exit 1; }
# THE INTERPRETER IS THE IG-VENV, NOT BARE CONDA (same measured failure as
# the v9 eval: conda carla-rl alone lacks hydra/vc_models for PERCEPTION=vc1).
V=$PWD/dependencies/ig-venv
export PATH="$V/bin:$PATH"
export VIRTUAL_ENV="$V"
hash -r
python -c "import hydra, vc_models, torch" \
  || { echo "FATAL: this python lacks hydra/vc_models/torch for PERCEPTION=vc1"; exit 1; }
set -a
# shellcheck disable=SC1091
source "$SNAP"
set +a
# THE GATE LAYER'S OWN FLAGS, ABSENT FROM THE SNAPSHOT (run_gated.sh exports
# them only after train_v10.sh snapshots): the loader refuses the bare
# snapshot's obs width, and the action box must match training's mapping.
export GATED=1 OBS_WITH_MODE=1 OBS_WITH_GOAL=1
# IMAGE SIZE PINNED TO TRAINING (found 2026-09-26). run_gated.sh exports
# ENV_IMAGE_SIZE=120 AFTER train_v10.sh snapshots, so the .env this script
# replays never carries it and the env fell back to its default of 84. Every
# earlier eval from this script ran the policy on 84x84 frames it never saw.
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
# CHANNEL ORDER PINNED TO TRAINING. This checkpoint was trained on the
# BGR frames the double flip produced (verified 2026-09-26), so the fix
# must stay OFF here or the policy is fed out-of-distribution input.
export ENV_RGB_CHANNEL_FIX=0
export INCREMENTAL_ACTIONS=0 MAX_STEER=0.5 MAX_THROTTLE=0.4 TARGET_SPEED=20.0
export ENV_TOWNS="$TOWN" EVAL_TOWNS="$TOWN"   # provenance; --towns is what eval_arm reads
export CARLA_PORT=2000 CARLA_PORTS=2000
export DISPLAY=:1

mkdir -p results/v13_traces logs

# Explore-mode trace WITH PIXELS. The band flags are irrelevant here: explore
# mode zeroes the goal block, so there is no goal distance to restrict.
# TRACE_OUT overrides the path, for smoke tests that must not overwrite
# the real artefact.
OUT="${TRACE_OUT:-results/v13_traces/$(echo "$TOWN" | tr '[:upper:]' '[:lower:]')_explore_trace.pkl}"
echo "=== [v13-trace $(date +%H:%M:%S)] $TOWN: server self-starts via ensure_carla (Epic, windowed) ==="
python collect_explore_trace_v13.py --ckpt "$CK" --town "$TOWN" --port 2000 \
  --samples "${SAMPLES:-5000}" --candidate-spacing "${SPACING:-2.0}" \
  --collect-seed "${COLLECT_SEED:-0}" --action-mode "${ACTION_MODE:-sample}" \
  --out "$OUT" 2>&1 | tee -a "logs/v13_trace_${TOWN}.log"
echo "=== [v13-trace $(date +%H:%M:%S)] $TOWN done -> $OUT ==="
