#!/usr/bin/env bash
# ViNT Town07 rerun with the ALIGNED NAVIGATOR (user 2026-11-15, relayed by the map benchmark
# session: "turn this on Off: ignoring no-data crossings. Those still end the episode, so the
# cells stay comparable with the current table - we rerun vint and sac local next").
# Comparable with that session's c0 rerun (label na25):
#   - code: snapshot of vint-rerun-2026-11-15 @27b7fb6 (ViNT + worktree-sac-image-goal @7a4ee1d);
#     image_nav.py, carla_benchmark.py, train_env_config_v4_image_goal.py and image_map.py are
#     byte-identical to the c0 rerun's tree (code_t07_na);
#   - EVAL_MAP_NEXT_ADVANCE=1 (Habitat's next-node rule, commit 4e706cd); inert for the
#     ViNT-topomap arm, which follows ViNT's own navigate.py rule, and for mapless;
#   - ANY speed (il_eval_chw_anyspeed.env), as the main table;
#   - LANE_UNKNOWN_MARKS_FATAL UNSET (default 1: a no-data crossing ends the episode);
#   - z=2.5 maps (t07/maps_t07_z25) for all three bands;
#   - leg budget 150 TICKS in every case: AR=2 -> LEG_TIMEOUT 75 / LEG_STEPS 50,
#     AR=1 -> the defaults 150 / 100.
# VINT_AR is REQUIRED (the user decides ViNT's action repeat); the label carries it.
# Final ViNT dagger3_e1, run once; the Town07 ViNT topomap built 2026-11-12 (drive 0).
# Each cell is skipped if its eval.json exists. Refuses to start beside another CARLA.
set -u
: "${VINT_AR:?set VINT_AR=1 or 2 (the user's decision on ViNT's action repeat)}"
case $VINT_AR in
  2) LEG_TIMEOUT=75; LEG_STEPS=50 ;;
  1) LEG_TIMEOUT=150; LEG_STEPS=100 ;;
  *) echo "VINT_AR must be 1 or 2"; exit 2 ;;
esac
D=/home/robotlab/carla-sac-local
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/vint-carla
V=$W/results/vint_town07_nx
export REPO=$V/code_27b7fb6 CLUSTER_PY=$D/venv/bin/python CKPT_DIR=$V/checkpoints
export CARLA_LOCAL_ROOT=/home/robotlab/Apps/CARLA_0.9.15 CARLA_SIF=/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4.sh
export CARLA_OFFSCREEN=0 DISPLAY=${DISPLAY:-:1} CARLA_QUALITY=Epic
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export DEBIT_SRC=/home/robotlab/scratch/carla-rl/dependencies DEBIT_WEIGHTS=$D/weights/debit_tiny.pth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EVAL_POLICY=vint
export PYTHONPATH=/home/robotlab/vint/pkgs:/home/robotlab/vint/visualnav/train
CKPT=/home/robotlab/vint/final/dagger3_e1
TOPO=$W/results/vint_town07/vint_topomap/dagger3_e1
MAPROOT=$D/t07/maps_t07_z25
FRESH=$D/logs/bench_t07_fresh_episodes.json
TURNS=$D/logs/bench_t07_turns_episodes.json
L=$V/logs
mkdir -p "$CKPT_DIR" "$L"
say() { echo "=== $(date +%T) $*"; }
if ps -eo args | grep -q '[C]arlaUE4-Linux'; then say "another CARLA is running -- abort"; exit 3; fi
[ -f "$TOPO/Town07_topomap.npz" ] || { say "no Town07 ViNT topomap at $TOPO -- abort"; exit 1; }
cd "$REPO" || exit 1
export BENCH_BASE_OVERLAY=slurm/il_eval_chw_anyspeed.env GOAL_CAM_MATCH=1
export OFF_ROAD_JUNCTION_M=3.5 EVAL_MAP_FALLBACK=mapless EVAL_MAP_RESET_HIDDEN=1 EVAL_MAP_REPLAN=3
export EVAL_MAP_NEXT_ADVANCE=1 ACTION_REPEAT=$VINT_AR
export EVAL_MAP_LEG_TIMEOUT=$LEG_TIMEOUT EVAL_MAP_LEG_STEPS=$LEG_STEPS
unset LANE_UNKNOWN_MARKS_FATAL
say "rule: any speed ($BENCH_BASE_OVERLAY) | LANE_UNKNOWN_MARKS_FATAL unset (1) | next-advance ON | AR=$VINT_AR | leg timeout $LEG_TIMEOUT / leg steps $LEG_STEPS (150 ticks) | code $(basename "$REPO")"
LABEL=vintd3_nx_ar$VINT_AR
port=3200
TAG=$(basename "$CKPT")
for spec in easy:mapless easy:uniform easy:vinttopo medium:mapless medium:uniform medium:vinttopo \
            turns:mapless turns:uniform turns:vinttopo; do
  BAND=${spec%%:*} ARM=${spec#*:}
  EP=$FRESH; [ "$BAND" = turns ] && EP=$TURNS
  RUN=bench_${LABEL}_${ARM}_Town07_${BAND}
  MAPDIR=""; unset EVAL_VINT_TOPOMAP EVAL_MAP_LOCALISE
  case $ARM in
    mapless)  out=$CKPT_DIR/$RUN/eval_$TAG ;;
    uniform)  out=$CKPT_DIR/$RUN/eval_${TAG}_map-uniform; MAPDIR=$MAPROOT/uniform
              export EVAL_MAP_LOCALISE=image ;;
    vinttopo) out=$CKPT_DIR/$RUN/eval_${TAG}_vinttopo-$(basename "$TOPO")
              export EVAL_VINT_TOPOMAP=$TOPO ;;
  esac
  if [ -f "$out/eval.json" ]; then say "have $RUN"; continue; fi
  port=$((port + 15))
  if ss -ltn | grep -q ":$port "; then say "port $port busy -- abort"; exit 3; fi
  say "cell $RUN port $port"
  EVAL_EPISODES_FILE=$EP EVAL_VIDEO_EVERY=10 REWARD_OVERLAY=slurm/bench_$BAND.env ENV_TOWNS=Town07 \
    WRONG_GOAL_MODE=0 EVAL_MAP_DIR=$MAPDIR RUN_NAME=$RUN CARLA_PORT=$port EVAL_FROM=$CKPT \
    bash slurm/train_sac.sbatch > "$L/$RUN.log" 2>&1
  say "cell $RUN rc=$? $(grep -o 'SR=[0-9.]*' "$L/$RUN.log" | tail -1)"
done
say "VINT TOWN07 NX (AR=$VINT_AR) DONE"
