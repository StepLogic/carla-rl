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
C=$HOME/carla-rl/checkpoints/il_bc_c0/step_15000
MAPROOT=${MAPROOT:-$HOME/carla-rl/checkpoints/maps_expert}
export BENCH_BASE_OVERLAY=slurm/il_eval_chw.env GOAL_CAM_MATCH=1
export EVAL_EPISODES_FILE=$HOME/carla-rl/logs/bench_long_episodes.json
export EVAL_MAP_FALLBACK=${EVAL_MAP_FALLBACK:-mapless}
RUN=bench_${LABEL}_${ARM}_${TOWN}_${BAND}
out=checkpoints/$RUN/eval_step_15000_map-${ARM}
if [ -f "$out/eval.json" ]; then echo "=== have $RUN"; exit 0; fi
echo "=== MAPDEBUG $RUN port $PORT lookahead=${EVAL_MAP_LOOKAHEAD_M:-0} reset_hidden=${EVAL_MAP_RESET_HIDDEN:-0} fallback=$EVAL_MAP_FALLBACK maps=$MAPROOT ==="
EVAL_VIDEO_EVERY=10 REWARD_OVERLAY=slurm/bench_$BAND.env ENV_TOWNS=$TOWN WRONG_GOAL_MODE=0 \
  EVAL_MAP_DIR=$MAPROOT/$ARM RUN_NAME=$RUN CARLA_PORT=$PORT EVAL_FROM=$C bash slurm/train_sac.sbatch
echo "=== MAPDEBUG $RUN rc=$? ==="
