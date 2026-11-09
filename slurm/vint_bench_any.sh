#!/usr/bin/env bash
# ANY-SPEED copy of ~/carla-rl-vint/slurm/vint_bench_map.sh (user 2026-11-09: "the
# only condition is the agent reaches the goal"; "rerun ViNT"). Byte-identical to
# the ViNT tree's script (md5 d6502831...) except BENCH_BASE_OVERLAY, which points
# at this tree's slurm/il_eval_chw_anyspeed.env through a path relative to the
# ViNT repo (bench_<band>.env sources "$REPO/$BENCH_BASE_OVERLAY"); that overlay
# sources the ViNT tree's own il_eval_chw.env and sets GOAL_ARRIVAL_MAX_SPEED=0.
# Runs the CELLS loop of vint_bench_cells.sh itself, so nothing in the ViNT tree
# is edited.
SELF=$HOME/carla-rl/slurm/vint_bench_any.sh
if [ "${1:-}" != one ]; then

CELLS=${CELLS:-Town02:easy Town02:medium Town02:hard Town03:easy Town03:medium Town03:hard}
p0=$PORT; k=0
for cell in $CELLS; do
  TOWN=${cell%%:*} BAND=${cell##*:} PORT=$((p0 + 10 * k)) bash "$SELF" one
  k=$((k + 1))
done
echo "=== VINT_BENCH_ANY $LABEL $ARM done ==="
exit 0
fi
# bench_map_debug.sh for ViNT (EVAL_POLICY=vint): one map-benchmark cell (TOWN x BAND
# x ARM) with the frozen settings of docs/map_benchmark_fresh_2026-11-07.md, driven by
# a ViNT checkpoint dir (CKPT, scripts/vint/export_vint_ckpt.py), from the vint-carla
# shadow tree. bench_map_debug.sh hard-codes c0 and stays untouched.
#
#   CKPT=$HOME/carla-rl/checkpoints/vint_carla_expert/epoch_1 TOWN=Town02 BAND=hard \
#   ARM=uniform LABEL=vint_fresh_image EVAL_MAP_LOCALISE=image PORT=3100 \
#   EPISODES=$HOME/carla-rl/logs/bench_fresh_episodes.json \
#   sbatch -p gpu --gres=gpu:1 -c 4 --mem=48G -t 10:00:00 --export=ALL slurm/vint_bench_map.sh
#
# Arms (as the c0 runs): ARM=mapless; ARM=uniform EVAL_MAP_LOCALISE=image (main);
# ARM=uniform EVAL_MAP_LOCALISE=pose EVAL_MAP_JOINT_LOCALISE=1 (labelled oracle).
# EPISODES has no default on purpose: debug on the development set
# (logs/bench_long_episodes.json), then run the frozen fresh set once.
# Difference from the c0 arm: ViNT keeps its frame context at a subgoal switch
# (subgoal_reset) and approaches only the final goal (vint_driver.final_goal).
set -u
REPO=${REPO:-$HOME/carla-rl-vint}
cd "$REPO" || exit 1
: "${TOWN:?}" "${BAND:?}" "${ARM:?}" "${LABEL:?}" "${PORT:?}" "${CKPT:?}"
: "${EPISODES:?set EPISODES: the development set to debug, the frozen fresh set once}"
MAPROOT=${MAPROOT:-$HOME/carla-rl/checkpoints/maps_expert_jx_img}
export REPO EVAL_POLICY=vint
export PYTHONPATH="$HOME/vint/pkgs:$HOME/vint/visualnav/train${PYTHONPATH:+:$PYTHONPATH}"
export BENCH_BASE_OVERLAY=../carla-rl/slurm/il_eval_chw_anyspeed.env GOAL_CAM_MATCH=1
export EVAL_EPISODES_FILE=$EPISODES
# The frozen settings, common to all arms (overridable only on purpose).
export OFF_ROAD_JUNCTION_M=${OFF_ROAD_JUNCTION_M:-3.5} EVAL_MAP_FALLBACK=${EVAL_MAP_FALLBACK:-mapless}
export EVAL_MAP_RESET_HIDDEN=${EVAL_MAP_RESET_HIDDEN:-1} EVAL_MAP_REPLAN=${EVAL_MAP_REPLAN:-3}
TAG=$(basename "$CKPT")
RUN=bench_${LABEL}_${ARM}_${TOWN}_${BAND}
# ARM=vinttopo (user 2026-11-07, "ViNT-native topomap"): ViNT's own map of the
# expert-tour trace, TOPOMAP=<dir of <Town>_topomap.npz>; no benchmark map.
if [ "$ARM" = mapless ]; then out=$HOME/carla-rl/checkpoints/$RUN/eval_$TAG; MAPDIR=""
elif [ "$ARM" = vinttopo ]; then
  : "${TOPOMAP:?ARM=vinttopo needs TOPOMAP}"
  export EVAL_VINT_TOPOMAP=$TOPOMAP
  out=$HOME/carla-rl/checkpoints/$RUN/eval_${TAG}_vinttopo-$(basename "$TOPOMAP"); MAPDIR=""
else out=$HOME/carla-rl/checkpoints/$RUN/eval_${TAG}_map-${ARM}; MAPDIR=$MAPROOT/$ARM; fi
if [ -f "$out/eval.json" ]; then echo "=== have $RUN"; exit 0; fi
echo "=== VINT_BENCH $RUN ckpt=$CKPT port $PORT topomap=${EVAL_VINT_TOPOMAP:-none} localise=${EVAL_MAP_LOCALISE:-pose} joint=${EVAL_MAP_JOINT_LOCALISE:-0} reset_hidden=$EVAL_MAP_RESET_HIDDEN replan=$EVAL_MAP_REPLAN fallback=$EVAL_MAP_FALLBACK off_road_junction_m=$OFF_ROAD_JUNCTION_M maps=${MAPDIR:-none} episodes=$EVAL_EPISODES_FILE ==="
EVAL_VIDEO_EVERY=10 REWARD_OVERLAY=${BAND_OVERLAY:-slurm/bench_$BAND.env} ENV_TOWNS=$TOWN WRONG_GOAL_MODE=0 \
  EVAL_MAP_DIR=$MAPDIR RUN_NAME=$RUN CARLA_PORT=$PORT EVAL_FROM=$CKPT bash slurm/train_sac.sbatch
echo "=== VINT_BENCH $RUN rc=$? ==="
