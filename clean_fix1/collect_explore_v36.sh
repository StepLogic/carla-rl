#!/usr/bin/env bash
# EXPLORE-MODE TRACE with the v36 policy (user 2026-09-30: "stop v36 - run
# explore mode in town02 and town03 -> collect trace and plot driven path").
#
# Same collector as the v13 traces (collect_explore_trace_v13.py, imported
# helpers from eval_placement) so the artefacts are directly comparable; only
# the REPLAYED RECIPE differs, and every difference from the v13 wrapper is
# because v36 is a seg-VAE run, not a VC-1 one:
#   PERCEPTION=segvae + SEGVAE_CKPT   from the snapshot (v13 was vc1/pyramid)
#   ENV_RGB_CHANNEL_FIX=1             from the snapshot -- v36 trained on the
#                                     FIXED channel order, so unlike the v13
#                                     wrapper this must NOT be forced to 0.
#   AGENT_LATENT_DIMS                 from the snapshot (latent+goal:512:256)
# The script ASSERTS those three rather than trusting the file, because a
# silent mismatch feeds the policy out-of-distribution input and still drives.
#
# FLAGS run_gated.sh EXPORTS AFTER SNAPSHOTTING, so they are absent from the
# .env and must be pinned here or the env silently falls back to a default the
# policy never saw (this is the 84-vs-120 px trap, found 2026-09-26):
#   ENV_IMAGE_SIZE/VAE_IMAGE_SIZE 120, TARGET_SPEED 20.0, INCREMENTAL_ACTIONS 0,
#   MAX_STEER 0.5, MAX_THROTTLE 0.4, GATED/OBS_WITH_MODE/OBS_WITH_GOAL 1.
# Values copied from run_gated.sh lines 82-84, 269-270, 361, 367.
#
# Explore mode and the disabled 100 m arrival are set by the COLLECTOR itself
# (EVAL_FORCE_MODE=0, EXPLORE_GOAL_DIST_M=0), not here.
#
# Usage: collect_explore_v36.sh TOWN [CKPT] [SAMPLES]
set -euo pipefail
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl/clean_fix1

TOWN="${1:-Town02}"
CK="${2:-checkpoints/carla_gated_v36_segvae_nobc_noexpert/checkpoint_825000.pkl}"
SAMPLES="${3:-${SAMPLES:-5000}}"
SNAP="${SNAP:-results/run_env/carla_gated_v36_segvae_nobc_noexpert.env}"
[ -f "$CK" ]   || { echo "FATAL: no $CK"; exit 1; }
[ -f "$SNAP" ] || { echo "FATAL: no $SNAP"; exit 1; }

# Port lock, held for the whole collection (the same lock the trainers take).
LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
flock -n 200 || { echo "REFUSING TO START: another CARLA job holds $LOCK." >&2; exit 1; }
echo "[Lock] acquired $LOCK (pid $$)"

# GPU gate: one Epic windowed server (~5-9 GB) + jax + the seg-VAE.
NEED_MB="${NEED_MB:-9000}"
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
# THE INTERPRETER IS THE IG-VENV, not bare conda (the conda env lacks the
# stack; same measured failure as every earlier eval script).
V=$PWD/dependencies/ig-venv
export PATH="$V/bin:$PATH"
export VIRTUAL_ENV="$V"
hash -r
python - <<'EOF' || { echo "FATAL: this python cannot load the seg-VAE stack"; exit 1; }
import sys, os
sys.path.insert(0, "src")
import jax, flax, numpy  # noqa: F401
from vae_nav import seg_vae  # noqa: F401
print(f"[Check] jax {jax.__version__}, seg_vae import OK")
EOF

set -a
# shellcheck disable=SC1091
source "$SNAP"
set +a

# THE THREE RECIPE ASSERTIONS (see header).
[ "${PERCEPTION:-}" = "segvae" ] || { echo "FATAL: snapshot PERCEPTION=${PERCEPTION:-unset}, expected segvae"; exit 1; }
[ -f "${SEGVAE_CKPT:-/nonexistent}" ] || { echo "FATAL: SEGVAE_CKPT missing: ${SEGVAE_CKPT:-unset}"; exit 1; }
[ "${ENV_RGB_CHANNEL_FIX:-}" = "1" ] || { echo "FATAL: snapshot ENV_RGB_CHANNEL_FIX=${ENV_RGB_CHANNEL_FIX:-unset}, expected 1 for v36"; exit 1; }

export GATED=1 OBS_WITH_MODE=1 OBS_WITH_GOAL=1
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
# USER 2026-09-30: "no marking data at all and offroad false - non terminal".
# A crossing the sensor reported with NO marking type no longer ends an explore
# episode while the car is ON the road; off_road is judged separately and still
# ends it, and painted markings are untouched. Default elsewhere stays 1.
export LANE_UNKNOWN_MARKS_FATAL=0
# FIXED WEATHER (user 2026-09-30: "keep fixe weather", superseding "keep on
# weather"). The snapshot carries ENV_WEATHER_RANDOM=1 because v36 TRAINED under
# randomised skies; a trace is an evaluation artefact, and varying the sky across
# segments would confound every appearance-based use of it (localisation, the
# RGB bank, any figure of the driven path). Overridden AFTER sourcing the
# snapshot, so this wins.
export ENV_WEATHER_RANDOM=0
unset ENV_WEATHER_PRESETS ENV_WEATHER_SEED
export INCREMENTAL_ACTIONS=0 MAX_STEER=0.5 MAX_THROTTLE=0.4 TARGET_SPEED=20.0
export ENV_TOWNS="$TOWN" EVAL_TOWNS="$TOWN"
export CARLA_PORT=2000 CARLA_PORTS=2000
# Epic windowed, matching the v13 traces so the two are comparable, and the
# user's standing local rule that a training/collection run shows its render.
export CARLA_QUALITY=Epic CARLA_OFFSCREEN=0
unset CARLA_QUALITY_PORTS CARLA_OFFSCREEN_PORTS
export DISPLAY="${DISPLAY:-:1}"

mkdir -p results/v36_traces logs
OUT="${TRACE_OUT:-results/v36_traces/$(echo "$TOWN" | tr '[:upper:]' '[:lower:]')_explore_trace.pkl}"
echo "=== [v36-trace $(date +%H:%M:%S)] $TOWN ckpt=$(basename "$CK") samples=$SAMPLES -> $OUT ==="
python collect_explore_trace_v13.py --ckpt "$CK" --town "$TOWN" --port 2000 \
  --samples "$SAMPLES" --candidate-spacing "${SPACING:-2.0}" \
  --collect-seed "${COLLECT_SEED:-0}" --action-mode "${ACTION_MODE:-sample}" \
  --out "$OUT" 2>&1 | tee -a "logs/v36_trace_${TOWN}.log"
echo "=== [v36-trace $(date +%H:%M:%S)] $TOWN done -> $OUT ==="
