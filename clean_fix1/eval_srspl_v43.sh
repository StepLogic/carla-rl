#!/usr/bin/env bash
# EXPLORE-MODE TRACE with the v36 policy (user 2026-09-30: "stop v36 - run
# explore mode in town02 and town03 -> collect trace and plot driven path").
#
# Same collector as the v13 traces (eval_sr_spl_v43.py, imported
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
CK="${2:-checkpoints/carla_gated_v43_segvae_nocaps/checkpoint_550000.pkl}"
SAMPLES="${3:-${SAMPLES:-5000}}"
SNAP="${SNAP:-results/run_env/carla_gated_v43_segvae_nocaps.env}"
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
[ "${ENV_RGB_CHANNEL_FIX:-}" = "1" ] || { echo "FATAL: snapshot ENV_RGB_CHANNEL_FIX=${ENV_RGB_CHANNEL_FIX:-unset}, expected 1 for v43"; exit 1; }

# RUN_GATED.SH'S POST-SNAPSHOT EXPORTS. The snapshot is written BEFORE run_gated.sh
# runs, so everything it exports is absent from SNAP. EXPLORE_DISCOUNT then falls back
# to DISCOUNT -- which is 1.0 for v43 -- and gated_config refuses it. v36 never hit
# this because its DISCOUNT was 0.99. Pinned to what training actually ran with.
export EXPLORE_DISCOUNT=0.99 STEP_COST=1.0
export GOAL_NOT_IN_JUNCTION=1 LANE_FOLLOW=1 LANE_STRICT=1 SHORT_HORIZON=1
export DELTA_MODE=positive CHAIN_MAX_GOALS=1 LANE_EXPLORE=0 LANE_INVASION_NOOP=0
export GATED_EXPLORE_MAX_STEPS=1000 MAX_EPISODE_STEPS=2000
export EXPLORE_EXEMPT_MARKS=NONE EXPLORE_LANE_MODE=terminal
# The oracle's PID saturates above what the action box can reach; v43's
# box is uncapped, so this is the only thing governing its speed.
export EXPERT_KIND=basic EXPERT_TARGET_SPEED="${EXPERT_TARGET_SPEED:-20}"
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
# ACTION BOX FROM THE SNAPSHOT (2026-10-02). v36 hardcoded 0.5/0.4 here, AFTER
# sourcing SNAP. v43 trained at 1.0/1.0, and the caps define the action-space Box
# plus the action-history observation normalisation -- hardcoding them would drive a
# DIFFERENT policy than the checkpoint, silently clipped.
export INCREMENTAL_ACTIONS=0 TARGET_SPEED=20.0
export MAX_STEER="${MAX_STEER:?snapshot must define MAX_STEER}"
export MAX_THROTTLE="${MAX_THROTTLE:?snapshot must define MAX_THROTTLE}"
echo "[Trace] action box from snapshot: steer +/-$MAX_STEER throttle 0-$MAX_THROTTLE"
export ENV_TOWNS="$TOWN" EVAL_TOWNS="$TOWN"
export CARLA_PORT=2000 CARLA_PORTS=2000
# Epic windowed, matching the v13 traces so the two are comparable, and the
# user's standing local rule that a training/collection run shows its render.
export CARLA_QUALITY=Epic CARLA_OFFSCREEN=0
unset CARLA_QUALITY_PORTS CARLA_OFFSCREEN_PORTS
export DISPLAY="${DISPLAY:-:1}"

mkdir -p results/v43_traces logs

export EVAL_FORCE_MODE=1
export EVAL_FAULT_ENDS="${EVAL_FAULT_ENDS:-fault}"   # USER RULE: faults are terminal
MAPS="${MAPS:-results/v43_traces/maps/$(echo "$TOWN" | tr '[:upper:]' '[:lower:]')_oracle_trace_maps.json}"
SC="${SC:-results/v43_traces/maps/$(echo "$TOWN" | tr '[:upper:]' '[:lower:]')_oracle_trace_maps_shortcuts.json}"
TRACE="${TRACE:-results/v43_traces/$(echo "$TOWN" | tr '[:upper:]' '[:lower:]')_oracle_trace.pkl}"
mkdir -p results/v43_srspl
OUT="${OUT:-results/v43_srspl/$(echo "$TOWN" | tr '[:upper:]' '[:lower:]')_srspl.json}"
EPS="${EPS:-results/v43_srspl/$(echo "$TOWN" | tr '[:upper:]' '[:lower:]')_episodes_ge50m.json}"
echo "=== [v43-srspl $(date +%H:%M:%S)] $TOWN ckpt=$(basename "$CK") -> $OUT ==="
python eval_sr_spl_v43.py --town "$TOWN" --maps "$MAPS" --shortcuts "$SC" \
  --trace "$TRACE" --ckpt "$CK" --out "$OUT" --episodes "$EPS" \
  --n-episodes "${N_EPISODES:-30}" --min-geodesic "${MIN_GEODESIC:-50}" \
  --port 2000 2>&1 | tee -a "logs/v43_srspl_${TOWN}.log"
rc=${PIPESTATUS[0]}
echo "=== [v43-srspl $(date +%H:%M:%S)] $TOWN done rc=$rc -> $OUT ==="
exit $rc
