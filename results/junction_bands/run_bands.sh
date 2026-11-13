#!/usr/bin/env bash
# Junction-turn bands, as pre-specified in docs/junction_bands_prespec_2026-11-12.md:
# c0 (il_bc_c0 step_15000) mapless, true and wrong goal on the same seeds, Town07 and
# Town02, local CARLA 0.9.15, with the map benchmark's own harness (the cluster-code
# snapshot code_t07 + slurm/bench_map_local.sh, the settings of its Town07 cells).
#   1. Town07 probe blocks 42000-44999 (the 3+ band needs more than one block)
#   2. Town07 3+ episode files (first 100 + a 20-seed pilot set after them)
#   3. pilot: Town07, 20 true-goal episodes per band (seeds after the evaluated 100)
#   4. full: 3 bands x 2 towns x {true, wrong} goal, 100 episodes each
# Every step is skipped if its output exists, so a re-run resumes. Outputs only under
# results/junction_bands/. Refuses to start beside another CARLA (the GPU is shared).
set -u
D=/home/robotlab/carla-sac-local
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/vint-carla
J=$W/results/junction_bands
export REPO=$D/code_t07 CLUSTER_PY=$D/venv/bin/python CKPT_DIR=$J/checkpoints
export CARLA_LOCAL_ROOT=/home/robotlab/Apps/CARLA_0.9.15 CARLA_SIF=/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4.sh
export CARLA_OFFSCREEN=0 DISPLAY=${DISPLAY:-:1} CARLA_QUALITY=Epic
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export DEBIT_SRC=/home/robotlab/scratch/carla-rl/dependencies DEBIT_WEIGHTS=$D/weights/debit_tiny.pth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C0=$D/checkpoints/il_bc_c0/step_15000
mkdir -p "$CKPT_DIR" "$J/logs"
say() { echo "=== $(date +%T) $*"; }
if ps -eo args | grep -q '[C]arlaUE4-Linux'; then say "another CARLA is running -- abort"; exit 3; fi
port=3100
next_port() {      # a fresh port per step; abort if it is taken (never attach to another server)
  port=$((port + 15))
  if ss -ltn | grep -q ":$port "; then say "port $port busy -- abort"; exit 3; fi
}

# 1. Town07 probe blocks
for seed in 42000 43000 44000; do
  run=jb_probe_turns_Town07_b$seed
  if [ -f "$CKPT_DIR/$run/eval_step_15000/probe.json" ]; then say "have $run"; continue; fi
  next_port
  say "probe Town07 seeds $seed.. port $port"
  ( cd "$REPO" && GOAL_CAM_MATCH=1 EVAL_PROBE=1 EVAL_PROBE_TURNS=1 EVAL_EPISODES=1000 EVAL_SEED=$seed \
      BENCH_BASE_OVERLAY=slurm/il_eval_chw_anyspeed.env REWARD_OVERLAY=slurm/bench_turns.env \
      ENV_TOWNS=Town07 RUN_NAME=$run CARLA_PORT=$port EVAL_FROM=$C0 \
      bash "$REPO/slurm/train_sac.sbatch" ) > "$J/logs/$run.log" 2>&1
  say "probe $run rc=$?"
  [ -f "$CKPT_DIR/$run/eval_step_15000/probe.json" ] || { say "no probe.json -- abort"; exit 1; }
done

# 2. Town07 3+ files
if [ ! -f "$J/episodes/Town07_turns3plus.json" ]; then
  bash "$J/build_t07_3plus.sh" || { say "Town07 3+ selection failed -- abort"; exit 1; }
fi

# 3 + 4. eval cells: c0 mapless, the benchmark's settings
export BENCH_BASE_OVERLAY=slurm/il_eval_chw_anyspeed.env OFF_ROAD_JUNCTION_M=3.5 \
  EVAL_MAP_FALLBACK=mapless MAPROOT=$J/no_map EVAL_MAP_RESET_HIDDEN=1 EVAL_MAP_REPLAN=3 \
  EVAL_MAP_LOCALISE=image CKPT=$C0
cell() {           # town episodes-file label wrong-goal
  local town=$1 file=$2 label=$3 wg=$4
  local sfx=""; [ "$wg" = 1 ] && sfx=_wrong        # the trainer names a wrong-goal run eval_<tag>_wrong
  if [ -f "$CKPT_DIR/bench_${label}_mapless_${town}_turns/eval_step_15000$sfx/eval.json" ]; then
    say "have $label $town"; return 0
  fi
  [ -f "$J/episodes/$file.json" ] || { say "no $file.json -- skip"; return 0; }
  next_port
  say "cell $town $label (wrong goal $wg) port $port"
  EPISODES=$J/episodes/$file.json WRONG_GOAL_MODE=$wg TOWN=$town BAND=turns ARM=mapless \
    LABEL=$label PORT=$port bash "$REPO/slurm/bench_map_local.sh" > "$J/logs/eval_${label}_${town}.log" 2>&1
  say "cell $town $label rc=$? $(grep -o 'SR=[0-9.]*' "$J/logs/eval_${label}_${town}.log" | tail -1)"
}
for b in turns1 turns2 turns3plus; do cell Town07 Town07_${b}_pilot jbpilot_$b 0; done
for t in Town07 Town02; do
  for b in turns1 turns2 turns3plus; do
    cell $t ${t}_$b jb_$b 0
    cell $t ${t}_$b jb_${b}wrong 1
  done
done
say "BANDS DONE"
