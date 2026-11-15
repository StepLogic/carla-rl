#!/usr/bin/env bash
# ViNT on Town02 + Town03 with the ALIGNED navigator (next-node advance): the cluster half of
# results/vint_town07_nx (label vintd3_nx_ar1, Town07, local, 2026-11-15), so the three towns pool.
# Same code snapshot (code_27b7fb6 = branch vint-rerun-2026-11-15 @27b7fb6, deployed to the cluster
# as ~/carla-rl-vint-nx), same settings as run_vint_t07_nx.sh; only the towns, episode sets, maps
# and topomaps differ, and those match the c0 rerun of the same day (label na25, ~/carla-rl-na).
#
# Requested via the mapping session after the user's "run 1" (c0 Town02/Town03 next-advance
# cells, jobs 466595-466618); this is the ViNT block's share of that pooling. ONE job runs all
# 18 cells serially (user 2026-11-07: "make single jobs get enough data"). A cell whose eval.json
# exists is skipped, so a resubmission resumes.
#
#   cd ~/carla-rl-vint-nx && VINT_AR=1 sbatch -J vint_nx_t0203 -p gpu --gres=gpu:1 -c 4 --mem=48G \
#     -t 14:00:00 --exclude=node-g104,node-g106 --dependency=afterany:<last GPU job, any session> \
#     -o logs/vint_nx_t0203_%j.log --export=ALL slurm/vint_nx_cells.sh
set -u
: "${VINT_AR:?set VINT_AR=1 (user decision 2026-11-15, ViNT at its native rate) or 2}"
case $VINT_AR in
  2) LEG_TIMEOUT=75; LEG_STEPS=50 ;;
  1) LEG_TIMEOUT=150; LEG_STEPS=100 ;;
  *) echo "VINT_AR must be 1 or 2"; exit 2 ;;
esac
REPO=$HOME/carla-rl-vint-nx
cd "$REPO" || exit 1
export REPO CKPT_DIR=$REPO/checkpoints EVAL_POLICY=vint
export PYTHONPATH="$HOME/vint/pkgs:$HOME/vint/visualnav/train${PYTHONPATH:+:$PYTHONPATH}"
CKPT=$HOME/carla-rl/checkpoints/vint_final/dagger3_e1
TOPO=$HOME/carla-rl/checkpoints/vint_topomap/dagger3_e1
MAPROOT=$HOME/carla-rl/checkpoints/maps_expert_jx_img_z25
FRESH=$HOME/carla-rl/logs/bench_fresh_episodes.json
TURNS=$HOME/carla-rl/logs/bench_turns_episodes.json
L=$REPO/logs
mkdir -p "$CKPT_DIR" "$L"
say() { echo "=== $(date +%T) $*"; }
for f in "$CKPT" "$FRESH" "$TURNS" "$HOME/carla-rl/logs/bench_Town02_medturn_episodes.json" \
         "$HOME/carla-rl/logs/bench_Town03_medturn_episodes.json" "$TOPO/Town02_topomap.npz" \
         "$TOPO/Town03_topomap.npz" "$MAPROOT/uniform/Town02.json" "$MAPROOT/uniform/Town03.json"; do
  [ -e "$f" ] || { say "preflight: missing $f -- abort"; exit 1; }
done
grep -q EVAL_MAP_NEXT_ADVANCE train_carla_sac.py || { say "preflight: no EVAL_MAP_NEXT_ADVANCE in $REPO -- abort"; exit 1; }
export BENCH_BASE_OVERLAY=slurm/il_eval_chw_anyspeed.env GOAL_CAM_MATCH=1
export OFF_ROAD_JUNCTION_M=3.5 EVAL_MAP_FALLBACK=mapless EVAL_MAP_RESET_HIDDEN=1 EVAL_MAP_REPLAN=3
export EVAL_MAP_NEXT_ADVANCE=1 ACTION_REPEAT=$VINT_AR
export EVAL_MAP_LEG_TIMEOUT=$LEG_TIMEOUT EVAL_MAP_LEG_STEPS=$LEG_STEPS
unset LANE_UNKNOWN_MARKS_FATAL
say "node $(hostname) job ${SLURM_JOB_ID:-none} | rule: any speed ($BENCH_BASE_OVERLAY) | LANE_UNKNOWN_MARKS_FATAL unset (1) | next-advance ON | AR=$VINT_AR | leg timeout $LEG_TIMEOUT / leg steps $LEG_STEPS (150 ticks) | maps $MAPROOT | code $REPO"
free_port() {   # first port >= $1 in this session's range (3100-3399) with no listener
  local p=$1
  while ss -ltn 2>/dev/null | grep -q ":$p "; do p=$((p + 5)); done
  [ "$p" -le 3399 ] || { say "no free port in 3100-3399 -- abort"; exit 3; }
  echo "$p"
}
LABEL=vintd3_nx_ar$VINT_AR
TAG=$(basename "$CKPT")
port=${PORT0:-3100}
for TOWN in ${TOWNS:-Town02 Town03}; do
  for spec in easy:mapless easy:uniform easy:vinttopo medium:mapless medium:uniform medium:vinttopo \
              turns:mapless turns:uniform turns:vinttopo; do
    BAND=${spec%%:*} ARM=${spec#*:}
    case $BAND in
      easy) EP=$FRESH ;;
      medium) EP=$HOME/carla-rl/logs/bench_${TOWN}_medturn_episodes.json ;;
      turns) EP=$TURNS ;;
    esac
    RUN=bench_${LABEL}_${ARM}_${TOWN}_${BAND}
    MAPDIR=""; unset EVAL_VINT_TOPOMAP EVAL_MAP_LOCALISE
    case $ARM in
      mapless)  out=$CKPT_DIR/$RUN/eval_$TAG ;;
      uniform)  out=$CKPT_DIR/$RUN/eval_${TAG}_map-uniform; MAPDIR=$MAPROOT/uniform
                export EVAL_MAP_LOCALISE=image ;;
      vinttopo) out=$CKPT_DIR/$RUN/eval_${TAG}_vinttopo-$(basename "$TOPO")
                export EVAL_VINT_TOPOMAP=$TOPO ;;
    esac
    port=$((port + 10))
    if [ -f "$out/eval.json" ]; then say "have $RUN"; continue; fi
    p=$(free_port "$port") || exit 3
    say "cell $RUN port $p episodes $(basename "$EP")"
    EVAL_EPISODES_FILE=$EP EVAL_VIDEO_EVERY=10 REWARD_OVERLAY=slurm/bench_$BAND.env ENV_TOWNS=$TOWN \
      WRONG_GOAL_MODE=0 EVAL_MAP_DIR=$MAPDIR RUN_NAME=$RUN CARLA_PORT=$p EVAL_FROM=$CKPT \
      bash slurm/train_sac.sbatch > "$L/$RUN.log" 2>&1
    rc=$?
    say "cell $RUN rc=$rc $(grep -o 'SR=[0-9.]*' "$L/$RUN.log" | tail -1) | $(grep -m1 -o 'next-node advance [A-Za-z]*' "$L/$RUN.log") $(grep -m1 -o 'leg timeout [0-9]*' "$L/$RUN.log")"
  done
done
say "VINT NX TOWN02/TOWN03 (AR=$VINT_AR) DONE"
