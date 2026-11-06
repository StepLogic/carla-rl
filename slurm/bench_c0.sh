#!/usr/bin/env bash
# THE CARLA REPLICATION OF THE GIBSON BENCHMARK on the behaviour-cloning checkpoint
# c0 = il_bc_c0 step 15000 (Task 7's selection; user 2026-11-06, "evaluating BC
# train checkpoints with mapping as described by Gibson eval set episodes count").
#
# The protocol of v30's runs (logs/bench_run.sh + logs/bench_map_run.sh on the
# cluster, 2026-11-02; docs/v30_maps_2026-11-02.md), UNCHANGED:
#   * v30's fixed episode file: 100 episodes per town per bin (Gibson's per-bin
#     count), bins 10-25 / 25-50 / 50-100 m by ROUTE distance, chosen by the probe
#   * success = within 1.0 m and slow (2.0 m also scored), deterministic action,
#     faults end the episode, 1,000 steps max
#   * mapless: the true goal image and the wrong-goal control, paired per episode
#   * map arms: the true goal image, node to node over checkpoints/maps_v30det/
#     <placement>/ (built from the deterministic v30 trace), localised by true
#     pose, the placements v30 was benchmarked with
# Towns: c0's held-out towns that have maps -- Town02 and Town03 (c0 trained on
# Town01/04/05/10HD; no Town06/Town07 maps exist).
#
# BENCH_BASE_OVERLAY = c0's own eval overlay (MONO_CHW + SPEED_OBS, which c0 needs):
# slurm/bench_<band>.env layers on it and its GOAL_DIST_MIN/MAX/STEP win. The
# goal-sampling settings are v30's (JUNCTION_GOAL_FRAC 0.85, GOAL_PAST_JUNCTION,
# GOAL_NOT_IN_JUNCTION, GOAL_BRANCH, via v22/v16), so a seed is the SAME episode
# as in v30's benchmark. c0's own MAX_DISTANCE_M (1e9, v30 200) and
# GOAL_KEEP_ON_PASS (1, v30 0) apply: outcome rules only, never the draw.
#
# One array task per (town, arm). "mapless" drives the true and the wrong goal on
# all three bins; a placement name drives that map, true goal. Every eval is
# skipped if its eval.json exists, so a killed task resumes.
#   sbatch --array=0-11%6 --gres=gpu:rtx2080:1 ... slurm/bench_c0.sh
set -u
cd ~/carla-rl
C=$HOME/carla-rl/checkpoints/il_bc_c0/step_15000
EP=$HOME/carla-rl/logs/bench_long_episodes.json
MAPROOT=$HOME/carla-rl/checkpoints/maps_v30det
export BENCH_BASE_OVERLAY=slurm/il_eval_chw.env
export GOAL_CAM_MATCH=1 EVAL_EPISODES_FILE=$EP
TASKS=(Town02:mapless Town03:mapless
       Town02:entropy Town03:entropy Town02:uniform Town03:uniform
       Town02:random Town03:random Town02:sgm Town03:sgm
       Town02:sgm_matched Town03:sgm_matched)
i=${SLURM_ARRAY_TASK_ID:?run as a Slurm array task}
spec=${TASKS[$i]}
town=${spec%%:*}
arm=${spec##*:}
PORT=$((4000 + 100 * i))
echo "=== BENCH c0 task $i: $town $arm port $PORT ==="
for band in easy medium hard; do
  if [ "$arm" = mapless ]; then
    for g in true wrong; do
      WG=0
      out=checkpoints/bench_c0_${g}_${town}_${band}/eval_step_15000
      if [ "$g" = wrong ]; then WG=1; out=${out}_wrong; fi
      if [ -f "$out/eval.json" ]; then echo "=== have $g $town $band"; continue; fi
      echo "=== BENCH c0 $g $town $band ==="
      EVAL_VIDEO_EVERY=10 REWARD_OVERLAY=slurm/bench_$band.env ENV_TOWNS=$town \
        WRONG_GOAL_MODE=$WG RUN_NAME=bench_c0_${g}_${town}_${band} CARLA_PORT=$PORT \
        EVAL_FROM=$C bash slurm/train_sac.sbatch
      echo "=== BENCH c0 $g $town $band rc=$? ==="
    done
  else
    out=checkpoints/bench_c0map_${arm}_${town}_${band}/eval_step_15000_map-${arm}
    if [ -f "$out/eval.json" ]; then echo "=== have $arm $town $band"; continue; fi
    echo "=== BENCHMAP c0 $arm $town $band ==="
    EVAL_VIDEO_EVERY=20 REWARD_OVERLAY=slurm/bench_$band.env ENV_TOWNS=$town \
      WRONG_GOAL_MODE=0 EVAL_MAP_DIR=$MAPROOT/$arm RUN_NAME=bench_c0map_${arm}_${town}_${band} \
      CARLA_PORT=$PORT EVAL_FROM=$C bash slurm/train_sac.sbatch
    echo "=== BENCHMAP c0 $arm $town $band rc=$? ==="
  fi
done
echo "=== BENCH c0 TASK $i DONE ==="
