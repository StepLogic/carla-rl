#!/usr/bin/env bash
# THE CARLA REPLICATION OF THE GIBSON BENCHMARK on c0, MAP ARMS ON THE
# SINGLE-TRACE EVALUATION MAPS (user 2026-11-06: "CAN YOU CREATE MAP FROM A SINGLE
# TRACE?" -> "I am talking about evaluations").
#
# slurm/bench_c0.sh's protocol, UNCHANGED, with one difference: the maps.
#   * the same fixed episode file (100 per town per bin; 10-25 / 25-50 / 50-100 m
#     route), the same c0 checkpoint and eval overlay, success within 1.0 m and
#     slow, deterministic action, faults end the episode, 1,000 steps max,
#     localised by true pose (start/goal node within 8 m, switch within 4 m);
#   * maps: checkpoints/maps_c0step/<placement>/ (scripts/build_single_trace_maps.py),
#     built from c0's OWN deterministic explore trace with every env step kept
#     (slurm/il_trace_step*.env; user 2026-11-06, "NO drive with new policy")
#     instead of v30's trace thinned to 2 m;
#   * no mapless arm here: it is the same episodes, so bench_c0.sh's mapless
#     results (checkpoints/bench_c0_{true,wrong}_<Town>_<band>) are the baseline.
# One array task per (town, placement); every eval is skipped if its eval.json
# exists, so a killed task resumes. Ports 6000 + 100 i (bench_c0.sh uses 4000+).
#   sbatch --array=0-7 --gres=gpu:l40:1 ... slurm/bench_c0_single.sh
set -u
cd ~/carla-rl
C=$HOME/carla-rl/checkpoints/il_bc_c0/step_15000
EP=$HOME/carla-rl/logs/bench_long_episodes.json
# MAPROOT / MAP_LABEL: which single-trace maps, and the label in the run names.
# Default c0's segmented per-step trace; the unbroken drive (user 2026-11-06,
# "i want one unbroke drive"): MAPROOT=$HOME/carla-rl/checkpoints/maps_c0unbroken MAP_LABEL=c0unbroken
MAPROOT=${MAPROOT:-$HOME/carla-rl/checkpoints/maps_c0step}
MAP_LABEL=${MAP_LABEL:-c0step}
export BENCH_BASE_OVERLAY=slurm/il_eval_chw.env
export GOAL_CAM_MATCH=1 EVAL_EPISODES_FILE=$EP
TASKS=(Town02:entropy Town03:entropy Town02:uniform Town03:uniform
       Town02:random Town03:random Town02:sgm_matched Town03:sgm_matched)
i=${SLURM_ARRAY_TASK_ID:?run as a Slurm array task}
spec=${TASKS[$i]}
town=${spec%%:*}
arm=${spec##*:}
PORT=$((6000 + 100 * i))
echo "=== BENCH c0 single-trace task $i: $town $arm port $PORT maps $MAPROOT/$arm ==="
if [ ! -f "$MAPROOT/$arm/$town.json" ]; then
  echo "=== no map $MAPROOT/$arm/$town.json: nothing to drive"; exit 1
fi
for band in easy medium hard; do
  out=checkpoints/bench_${MAP_LABEL}_${arm}_${town}_${band}/eval_step_15000_map-${arm}
  if [ -f "$out/eval.json" ]; then echo "=== have $arm $town $band"; continue; fi
  echo "=== BENCHMAP c0 single $arm $town $band ==="
  EVAL_VIDEO_EVERY=20 REWARD_OVERLAY=slurm/bench_$band.env ENV_TOWNS=$town \
    WRONG_GOAL_MODE=0 EVAL_MAP_DIR=$MAPROOT/$arm RUN_NAME=bench_${MAP_LABEL}_${arm}_${town}_${band} \
    CARLA_PORT=$PORT EVAL_FROM=$C bash slurm/train_sac.sbatch
  echo "=== BENCHMAP c0 single $arm $town $band rc=$? ==="
done
echo "=== BENCH c0 single TASK $i DONE ==="
