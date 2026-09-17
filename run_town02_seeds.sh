#!/usr/bin/env bash
# Town02 evaluation of the time-to-goal policy, FOUR SEEDS.
#
# FOUR IS THE FLOOR, not a nicety. It is the transferable rule from the peer
# protocol at ~/scratch/robot-learning-tools/experiments.txt: "a criterion that
# wins on three traces and loses on the fourth has not won", and that has
# already happened on their side (+4.3%, p = 0.11). Our own single-seed Town02
# run today gave 0.720 / 0.400 / 0.120 across the bands; whether that is the
# policy or the draw is exactly what four seeds answers.
#
# WHAT IS DELIBERATELY NOT PORTED from that file, on both its authors' explicit
# warnings: the checkpoint of record, the 164-d env block (GOAL_KIND=imgpoint,
# POINT_GOAL_MODE=polar_sincos, STATE_LATENT_DIM=256) and sgm[250/300]. Those
# were measured on a Habitat critic and do not transfer -- the same algorithm
# gave three different cutoffs against three different critics.
#
# ONE SERVER, ONE PORT, REUSED. Started here rather than by CarlaCore so the
# port is known and the map can be verified by querying it, and at EPIC quality
# because the frozen lane VAE was trained on Epic frames -- CarlaCore defaults
# to Low (carla_core.py:30) and nothing asserts it.
set -uo pipefail
cd "$(dirname "$0")"

CKPT="${CKPT:-checkpoints/carla_ttg_laneterm/checkpoint_150000.pkl}"
TOWN="${TOWN:-Town02}"
EPISODES="${EPISODES:-25}"
SEEDS="${SEEDS:-1234 1235 1236 1237}"
STAMP=$(date +%Y%m%d_%H%M)
CARLA_SH=/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4.sh
export DISPLAY=:1

say() { echo "=== [seeds $(date +%H:%M:%S)] $* ==="; }

pkill -9 -f '[C]arlaUE4-Linux-Shipping' 2>/dev/null; sleep 5
say "starting CARLA on port 2000 at Epic"
# setsid, or the wrapper exits and takes the binary with it.
setsid "$CARLA_SH" -carla-port=2000 -quality-level=Epic -windowed \
    -ResX=640 -ResY=360 > logs/carla_seeds_${STAMP}.log 2>&1 < /dev/null &
sleep 45

for SEED in $SEEDS; do
  OUT="results/town02_seed${SEED}_${STAMP}.json"
  say "seed $SEED -> $OUT"
  env CARLA_START_SERVER=0 CARLA_PORT=2000 CARLA_PORTS=2000 DRAW_GOAL=1 \
      ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120 \
      VAE_CKPT=checkpoints/lane_vae_120/best.pkl \
      INCREMENTAL_ACTIONS=0 MAX_STEER=0.5 MAX_THROTTLE=0.4 MIN_THROTTLE=0.05 \
      OBS_WITH_VECTOR=0 OBS_WITH_GOAL=1 \
      TIME_TO_GOAL=1 STEP_COST=1.0 CHAIN_GOALS=1 \
      TTG_MAX_STEPS=750 MAX_EPISODE_STEPS=750 MAX_DISTANCE_M=1200 \
      LANE_WEDGE=1 LANE_STRICT=0 WEDGE_ON_OFF_ROAD=1 STALL_STEPS=0 \
      COLLISION_TERMINAL=1 ROUTE_POTENTIAL=1 POTENTIAL_K=1.0 \
      SHORT_HORIZON=1 JUNCTION_GOAL_FRAC=0.7 TARGET_SPEED=20.0 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
      python -u eval_policy.py --ckpt "$CKPT" --towns "$TOWN" \
        --episodes "$EPISODES" --port 2000 --seed "$SEED" \
        --bands "20-40,40-70,70-113" --max-steps 750 \
        --out "$OUT" 2>&1 | grep -E "^  ${TOWN}|MAP|Traceback|Error" || true
  say "seed $SEED done (exit ${PIPESTATUS[0]})"
done

pkill -9 -f '[C]arlaUE4-Linux-Shipping' 2>/dev/null
say "all seeds done -- results/town02_seed*_${STAMP}.json"
