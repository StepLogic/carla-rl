#!/usr/bin/env bash
# Record (or, with EXPERT_RECORD_DRY=1, only benchmark) CARLA's autopilot in the
# imitation-first training towns, one train_sac.sbatch call per town.
#
#   EXPERT_RECORD_DRY=1 EXPERT_RECORD_EPISODES=80 RUN_PREFIX=il_m0 bash slurm/il_record.sh   # M0 gate
#   EXPERT_RECORD_EPISODES=200 bash slurm/il_record.sh                                       # M1 data
#
# Seeds and ports are fixed per town, so a subset (IL_TOWNS="Town05") never
# shifts another town's seed range. Town02/Town03 are held out: run_record
# refuses them. Pass-through: EXPERT_RECORD_DRY, EXPERT_RECORD_EPISODES,
# EXPERT_DRIVER, EVAL_FROM, EXPERT_RECORD_ROOT, EXPERT_SEED_BASE.
cd ~/carla-rl
export EXPERT_RECORD=1
BASE=${EXPERT_SEED_BASE:-100000}
for town in ${IL_TOWNS:-Town01 Town04 Town05 Town10HD}; do
  case $town in
    Town01) i=0;; Town04) i=1;; Town05) i=2;; Town10HD) i=3;;
    *) echo "=== il_record: $town is not a training town, skipped"; continue;;
  esac
  echo "=== IL_RECORD $town seeds from $((BASE + 10000 * i)) port 34${i}0 ==="
  REWARD_OVERLAY=slurm/il_record.env ENV_TOWNS=$town EXPERT_SEED0=$((BASE + 10000 * i)) \
    RUN_NAME=${RUN_PREFIX:-il_record}_$town CARLA_PORT=34${i}0 bash slurm/train_sac.sbatch
  echo "=== IL_RECORD $town rc=$? ==="
done
echo "=== IL_RECORD DONE ==="
