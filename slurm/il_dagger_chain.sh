#!/usr/bin/env bash
# B1 (plan 2026-11-09): one DAgger-in-chains round for ONE training town per job.
#   TOWN=Town05 ROUND=1 EPISODES=300 sbatch ... --wrap='bash slurm/il_dagger_chain.sh'
# c0 (or POLICY) drives; the expert labels. Seeds 900000 + 10000*town + 100000*(round-1),
# even only; shards go to checkpoints/il_data_dagger_chain<ROUND>/<Town>/ (separate
# from M1, so a clone chooses its data by root). Town02/Town03: run_record refuses.
set -u
cd ~/carla-rl
: "${TOWN:?}" "${ROUND:?}"
POLICY=${POLICY:-$HOME/carla-rl/checkpoints/il_bc_c0/step_15000}
case $TOWN in Town01) i=0;; Town04) i=1;; Town05) i=2;; Town10HD) i=3;;
  *) echo "=== $TOWN is not a training town"; exit 2;; esac
seed0=$((900000 + 10000 * i + 100000 * (ROUND - 1)))
port=$((5960 + 10 * i))
export EXPERT_RECORD=1 EXPERT_DRIVER=policy EXPERT_RECORD_DRY=0 EVAL_FROM=$POLICY
export EXPERT_RECORD_ROOT=${ROOT:-$HOME/carla-rl/checkpoints/il_data_dagger_chain$ROUND}
export EXPERT_RECORD_EPISODES=${EPISODES:-300}
echo "=== DAGGER-CHAIN round $ROUND $TOWN seeds from $seed0 port $port policy $POLICY -> $EXPERT_RECORD_ROOT"
REWARD_OVERLAY=slurm/il_dagger_chain.env ENV_TOWNS=$TOWN EXPERT_SEED0=$seed0 \
  RUN_NAME=il_dagger_chain_r${ROUND}_$TOWN CARLA_PORT=$port bash slurm/train_sac.sbatch
echo "=== DAGGER-CHAIN round $ROUND $TOWN rc=$? ==="
