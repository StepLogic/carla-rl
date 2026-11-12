#!/usr/bin/env bash
# SINGLE-TRACE EVALUATION MAPS, step 1: the expert's closed coverage tour of each
# held-out town, the clone c0 riding along for its entropy (slurm/expert_trace.env,
# train_carla_sac.run_expert_trace). One town per job:
#   TRACE_TOWNS=Town02 sbatch --gres=gpu:l40:1 ... --wrap='bash slurm/expert_trace.sh'
# A finished trace (index.json done) is skipped; an unfinished one resumes at its
# last written chunk as a new drive. Ports are fixed per town.
set -u
cd ~/carla-rl
C=${TRACE_CLONE:-$HOME/carla-rl/checkpoints/il_bc_c0/step_15000}
for town in ${TRACE_TOWNS:-Town02 Town03}; do
  case $town in Town02) port=7100;; Town03) port=7200;; *) port=7300;; esac
  echo "=== EXPERT TRACE $town port $port clone $C ==="
  REWARD_OVERLAY=${TRACE_ENV:-slurm/expert_trace.env} ENV_TOWNS=$town EVAL_FROM=$C \
    RUN_NAME=expert_trace_$town CARLA_PORT=$port bash slurm/train_sac.sbatch
  echo "=== EXPERT TRACE $town rc=$? ==="
done
echo "=== EXPERT TRACE DONE ==="
