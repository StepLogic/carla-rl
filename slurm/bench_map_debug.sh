#!/usr/bin/env bash
# ONE map-benchmark cell for debugging the map navigator (user 2026-11-06, "lets
# debug uniform map first"): c0 step 15000, the fixed episode file, one town,
# one band, one placement of MAPROOT, with whatever EVAL_MAP_* knobs the caller
# exports (EVAL_MAP_LOOKAHEAD_M, EVAL_MAP_RESET_HIDDEN, ...). bench_c0_single.sh's
# protocol otherwise. Skipped if its eval.json exists.
#   TOWN=Town02 BAND=hard ARM=uniform LABEL=dbg_la12 PORT=8500 \
#     EVAL_MAP_LOOKAHEAD_M=12 sbatch ... --export=ALL slurm/bench_map_debug.sh
set -u
cd ~/carla-rl
: "${TOWN:?}" "${BAND:?}" "${ARM:?}" "${LABEL:?}" "${PORT:?}"
# CKPT: the policy (default c0 step 15000). Run Q (user 2026-11-08) passes its own
# step dir; its config.json carries critic LayerNorm, read at load.
C=${CKPT:-$HOME/carla-rl/checkpoints/il_bc_c0/step_15000}
TAG=$(basename "$C")
MAPROOT=${MAPROOT:-$HOME/carla-rl/checkpoints/maps_expert}
# BENCH_BASE_OVERLAY: default il_eval_chw.env (10 km/h arrival, from v52_reward.env);
# slurm/il_eval_chw_anyspeed.env = arrival at any speed (user 2026-11-09).
export BENCH_BASE_OVERLAY=${BENCH_BASE_OVERLAY:-slurm/il_eval_chw.env} GOAL_CAM_MATCH=1
# EPISODES: the fixed episode file (default: the development set). The frozen
# evaluation uses the fresh, never-seen set (slurm/bench_probe_fresh.sh).
export EVAL_EPISODES_FILE=${EPISODES:-$HOME/carla-rl/logs/bench_long_episodes.json}
export EVAL_MAP_FALLBACK=${EVAL_MAP_FALLBACK:-mapless}
RUN=bench_${LABEL}_${ARM}_${TOWN}_${BAND}
# ARM=mapless: no map, the true goal image (bench_c0.sh's mapless arm).
if [ "$ARM" = mapless ]; then out=checkpoints/$RUN/eval_$TAG; MAPDIR=""
else out=checkpoints/$RUN/eval_${TAG}_map-${ARM}; MAPDIR=$MAPROOT/$ARM; fi
if [ -f "$out/eval.json" ]; then echo "=== have $RUN"; exit 0; fi
echo "=== MAPDEBUG $RUN port $PORT lookahead=${EVAL_MAP_LOOKAHEAD_M:-0} reset_hidden=${EVAL_MAP_RESET_HIDDEN:-0} fallback=$EVAL_MAP_FALLBACK off_road_junction_m=${OFF_ROAD_JUNCTION_M:-0} maps=${MAPDIR:-none} ==="
EVAL_VIDEO_EVERY=10 REWARD_OVERLAY=slurm/bench_$BAND.env ENV_TOWNS=$TOWN WRONG_GOAL_MODE=0 \
  EVAL_MAP_DIR=$MAPDIR RUN_NAME=$RUN CARLA_PORT=$PORT EVAL_FROM=$C bash slurm/train_sac.sbatch
echo "=== MAPDEBUG $RUN rc=$? ==="
