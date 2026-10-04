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
SNAP="${SNAP:-results/run_env/carla_gated_v42_segvae_sorb_her.env}"

CK="${1:-checkpoints/carla_gated_v42_segvae_sorb_her/checkpoint_1000000.pkl}"
TOWN="${2:-Town05}"
SUFFIX="${3:-}"
[ -f "$CK" ] || { echo "FATAL: no $CK"; exit 1; }
[ -f "$SNAP" ] || { echo "FATAL: no $SNAP"; exit 1; }
TAG=$(basename "$CK" .pkl)   # e.g. checkpoint_2000000

# Port lock, held for the whole eval (same lock the trainers use).
# EVAL_PORT (2026-09-26): a sweep running BESIDE a training run must not use
# 2000, which the trainer holds -- the sweep driver kills its own port before
# every stage. The lock is per port, so each server keeps its own.
PORT="${EVAL_PORT:-2000}"
OUTDIR="${EVAL_OUTDIR:-results/v42_eval}"
MODE="${EVAL_MODE:-goal}"
LOCK=/tmp/carla_rl_port${PORT}.lock
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
# RUN_GATED.SH'S POST-SNAPSHOT EXPORTS (2026-10-01). The training snapshot is written
# BEFORE run_gated.sh runs, so everything it exports is missing from SNAP. v42's eval
# died on EXPLORE_DISCOUNT (unset -> falls back to DISCOUNT=1.0 -> config guard).
# The env-behaviour flags below were never set by earlier eval wrappers either; they
# are pinned here to exactly what training ran with.
export EXPLORE_DISCOUNT=0.99 STEP_COST=1.0
export GOAL_NOT_IN_JUNCTION=1 LANE_FOLLOW=1 LANE_STRICT=1 SHORT_HORIZON=1 DELTA_MODE=positive
export CHAIN_MAX_GOALS=1 LANE_EXPLORE=0 LANE_INVASION_NOOP=0
export GATED_EXPLORE_MAX_STEPS=1000 MAX_EPISODE_STEPS=2000
export EXPLORE_EXEMPT_MARKS=NONE EXPLORE_LANE_MODE=terminal VAE_CKPT=checkpoints/lane_vae_120/best.pkl
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
# v17 TRAINED ON RGB (ENV_RGB_CHANNEL_FIX=1 in its snapshot): the fix must be ON
# here, the opposite of eval_v13.sh, whose checkpoint trained on BGR.
export ENV_RGB_CHANNEL_FIX=1
# TRAINING'S REWARD, EXACTLY (2026-09-27, for the per-step Q trace). run_gated.sh
# exports these after the snapshot; without them the env publishes no
# route_remaining_m and GatedReward drops the K*progress shaping, so a traced
# reward would not be the reward the critic was fitted to. Reward/info only --
# the observation does not change, and success rates are unaffected.
export ROUTE_POTENTIAL=1 POTENTIAL_K=0   # v42 trained without route shaping (K=0)
# run_gated.sh sets the explore arm's progress term AFTER the snapshot, so the
# replayed .env lacks it and the env fell back to projected x1 (training:
# heading x10). Reward only -- the traced explore reward must be training's.
export LANE_PROGRESS_MODE=heading LANE_PROGRESS_SCALE=10
# v25 TRAINED ON TWO SERVERS (Town01 Low offscreen @2000, Town05 Epic @2010);
# this eval runs ONE server, so the per-port settings must not leak in, and
# each town gets the quality it trained at. Towns v25 never saw run at Epic.
unset CARLA_QUALITY_PORTS CARLA_OFFSCREEN_PORTS
export ENV_STAGGER_S=0 GPU_RESERVE_BEFORE_CARLA=0
if [ "$TOWN" = "Town01" ]; then export CARLA_QUALITY=Low; else export CARLA_QUALITY=Epic; fi
export INCREMENTAL_ACTIONS=0 MAX_STEER=0.5 MAX_THROTTLE=0.4 TARGET_SPEED=20.0
export ENV_TOWNS="$TOWN" EVAL_TOWNS="$TOWN"   # provenance; --towns is what eval_arm reads
export CARLA_PORT="$PORT" CARLA_PORTS="$PORT"
# EVAL_CARLA_OFFSCREEN (2026-09-26): must be applied AFTER the snapshot is
# sourced, because the snapshot carries CARLA_OFFSCREEN=0 and would silently
# win over a plain env var. -RenderOffScreen drops the viewport WINDOW; the
# camera sensor still renders at the same -quality-level, so the frames the
# policy sees are unchanged. Used when an eval shares the GPU with a training
# run: a windowed Epic eval server took 5.4 GB and squeezed the trainer's.
[ -n "${EVAL_CARLA_OFFSCREEN:-}" ] && export CARLA_OFFSCREEN="$EVAL_CARLA_OFFSCREEN"
# JAX MEMORY, AS IN TRAINING. Without these JAX preallocates 75% of the card
# (~9 GB). Harmless when the eval had the GPU alone; beside a live training
# run it would starve that run's CARLA. Allocation policy does not change
# what the policy computes.
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.2}"
export DISPLAY=:1

mkdir -p "$OUTDIR" logs
# OPTIONAL GOAL BAND (band-restricted eval, user-approved 2026-09-23):
# GOAL_MIN/GOAL_MAX select the band, SUFFIX separates the out/log names;
# unset = the snapshot's band (5-200 m mixed, what training sampled).
BAND_ARGS=()
if [ -n "${GOAL_MIN:-}" ] && [ -n "${GOAL_MAX:-}" ]; then
  BAND_ARGS=(--goal-min "$GOAL_MIN" --goal-max "$GOAL_MAX")
fi
echo "=== [v17-eval $(date +%H:%M:%S)] $TOWN $TAG: server self-starts via ensure_carla (Epic, windowed) ==="
python eval_arm.py --arm "gated_v42_${TAG}_${MODE}" --ckpt "$CK" --mode "$MODE" \
  --episodes "${EPISODES:-60}" --towns "$TOWN" --seed 0 --max-steps 1000 --goal-conditioned 1 \
  "${BAND_ARGS[@]+"${BAND_ARGS[@]}"}" \
  --out "$OUTDIR/$(echo "$TOWN" | tr '[:upper:]' '[:lower:]')_${TAG}${SUFFIX}_${MODE}.json" \
  --port "$PORT" 2>&1 | tee "logs/v42_eval_${TOWN}_${TAG}${SUFFIX}_${MODE}.log"
echo "=== [v17-eval $(date +%H:%M:%S)] $TOWN $TAG done ==="