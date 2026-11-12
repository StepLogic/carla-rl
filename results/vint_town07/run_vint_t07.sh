#!/usr/bin/env bash
# ViNT's Town07 main-table rows (user 2026-11-12: Town07 replaces Town03 in the main tables).
# The final ViNT (dagger3_e1, chosen on training-town validation), run ONCE, reported
# whatever it shows. Harness = the cluster ViNT rows': a read-only local copy of the
# cluster's ~/carla-rl-vint (REPO) and the settings of ~/carla-rl/slurm/vint_bench_any.sh
# (md5 d6502831 for its vint_bench_map.sh twin), with the map benchmark's Town07 inputs:
#   easy/medium  /home/robotlab/carla-sac-local/logs/bench_t07_fresh_episodes.json
#                (medium = the NEW >= 1-turn medium: not poolable with ViNT's old Town02/03
#                 medium rows until those are rerun on the medturn sets)
#   hard         /home/robotlab/carla-sac-local/logs/bench_t07_turns_episodes.json
#   maps         /home/robotlab/carla-sac-local/t07/maps_t07_z25 (z=2.5, image gap nodes)
# Steps: 1. ViNT topomap of the expert tour (build_topomap.py, the Town02/03 recipe);
#        2. nine cells: {easy, medium, hard} x {mapless, uniform + image, ViNT topomap}.
# Each step is skipped if its output exists. Refuses to start beside another CARLA.
set -u
D=/home/robotlab/carla-sac-local
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/vint-carla
V=$W/results/vint_town07
export REPO=/home/robotlab/vint/carla-rl-vint CLUSTER_PY=$D/venv/bin/python CKPT_DIR=$V/checkpoints
export CARLA_LOCAL_ROOT=/home/robotlab/Apps/CARLA_0.9.15 CARLA_SIF=/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4.sh
export CARLA_OFFSCREEN=0 DISPLAY=${DISPLAY:-:1} CARLA_QUALITY=Epic
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export DEBIT_SRC=/home/robotlab/scratch/carla-rl/dependencies DEBIT_WEIGHTS=$D/weights/debit_tiny.pth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EVAL_POLICY=vint
export PYTHONPATH=/home/robotlab/vint/pkgs:/home/robotlab/vint/visualnav/train
CKPT=/home/robotlab/vint/final/dagger3_e1
TRACE=$D/checkpoints/trace_expert_jx/trace_expert_Town07
TOPO=$V/vint_topomap/dagger3_e1
MAPROOT=$D/t07/maps_t07_z25
FRESH=$D/logs/bench_t07_fresh_episodes.json
TURNS=$D/logs/bench_t07_turns_episodes.json
L=$V/logs
mkdir -p "$CKPT_DIR" "$TOPO" "$L"
say() { echo "=== $(date +%T) $*"; }
if ps -eo args | grep -q '[C]arlaUE4-Linux'; then say "another CARLA is running -- abort"; exit 3; fi
cd "$REPO" || exit 1

# 1. the ViNT topomap of the Town07 expert tour
if [ ! -f "$TOPO/Town07_topomap.npz" ]; then
  say "topomap Town07 from $TRACE"
  $CLUSTER_PY scripts/vint/build_topomap.py --trace "$TRACE" --ckpt "$CKPT" \
    --out "$TOPO/Town07_topomap.npz" > "$L/topomap_Town07.log" 2>&1
  say "topomap rc=$?"
  [ -f "$TOPO/Town07_topomap.npz" ] || { say "no topomap -- abort"; exit 1; }
fi

# 2. the cells, with the frozen settings of vint_bench_any.sh
export BENCH_BASE_OVERLAY=slurm/il_eval_chw_anyspeed.env GOAL_CAM_MATCH=1
export OFF_ROAD_JUNCTION_M=3.5 EVAL_MAP_FALLBACK=mapless EVAL_MAP_RESET_HIDDEN=1 EVAL_MAP_REPLAN=3
port=3200
TAG=$(basename "$CKPT")
for spec in easy:mapless easy:uniform easy:vinttopo medium:mapless medium:uniform medium:vinttopo \
            turns:mapless turns:uniform turns:vinttopo; do
  BAND=${spec%%:*} ARM=${spec#*:}
  EP=$FRESH; [ "$BAND" = turns ] && EP=$TURNS
  LABEL=vintd3_t07
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
say "VINT TOWN07 DONE"
