#!/usr/bin/env bash
# DAgger round (imitation-first plan, Task 9): the selected clone DRIVES
# (EXPERT_DRIVER=policy) while the expert, replanning whenever the car is more
# than 3 m off its route, LABELS every step -- so the data now covers the states
# the clone actually reaches: standing starts it never leaves, goals it passes
# too fast, drifts towards the kerb. Episodes join M1's in the same data root.
#
#   DAGGER_ROUND=1 EVAL_FROM=$HOME/carla-rl/checkpoints/il_bc_r0/step_7500 bash slurm/il_dagger.sh
#
# One town per call is fine (IL_TOWNS="Town05"): seeds and ports are fixed per
# town and round, so a subset never shifts another's range. Town02/Town03 are
# held out: run_record refuses them.
cd ~/carla-rl
: "${DAGGER_ROUND:?}" "${EVAL_FROM:?}"
export EXPERT_RECORD=1 EXPERT_DRIVER=policy EXPERT_RECORD_DRY=0 EVAL_FROM
export EXPERT_RECORD_ROOT=$HOME/carla-rl/checkpoints/il_data
export EXPERT_RECORD_EPISODES=${EXPERT_RECORD_EPISODES:-120}
for town in ${IL_TOWNS:-Town01 Town04 Town05 Town10HD}; do
  case $town in
    Town01) i=0;; Town04) i=1;; Town05) i=2;; Town10HD) i=3;;
    *) echo "=== il_dagger: $town is not a training town, skipped"; continue;;
  esac
  seed0=$((200000 + 10000 * i + 100000 * DAGGER_ROUND))
  echo "=== IL_DAGGER round $DAGGER_ROUND $town seeds from $seed0 port 35${i}0 policy=$EVAL_FROM ==="
  REWARD_OVERLAY=slurm/il_record.env ENV_TOWNS=$town EXPERT_SEED0=$seed0 \
    RUN_NAME=il_dagger_r${DAGGER_ROUND}_$town CARLA_PORT=35${i}0 bash slurm/train_sac.sbatch
  echo "=== IL_DAGGER round $DAGGER_ROUND $town rc=$? ==="
done
echo "=== IL_DAGGER DONE ==="
