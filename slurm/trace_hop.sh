#!/usr/bin/env bash
# THE EVALUATION-MAP TRACE WITH JUNCTION HOPS (user 2026-11-06: "Goal hop per
# junction", "but save observations and entropy"). One town per job, so a town's
# trace is its own resumable job:
#
#   TRACE_TOWNS=Town02 sbatch -p gpu --gres=gpu:l40:1 -c 4 --mem=48G -t 12:00:00 \
#     -J trace1hop_c0_Town02 -o logs/trace1hop_c0_Town02_%j.log slurm/trace_hop.sh
#
# The no-hop twin is slurm/il_trace_step_<Town>.env (jobs 464221/464222,
# checkpoints/trace1step_il_bc_c0_s15000_<Town>): the SAME budget, the same
# continuation rule and the same per-step recording, hops off -- it is the
# comparison, not a discard. Ports 7200/7300 so neither twin can collide with the
# other (7100/7101) or with the SGM reads (7400/7500).
set -u
cd ~/carla-rl
C=$HOME/carla-rl/checkpoints/il_bc_c0/step_15000
for town in ${TRACE_TOWNS:-Town02 Town03}; do
  case $town in Town02) port=7200;; Town03) port=7300;; *) port=7350;; esac
  echo "=== TRACEHOP $town port $port ==="
  REWARD_OVERLAY=slurm/il_trace_hop_$town.env ENV_TOWNS=$town \
    RUN_NAME=trace1hop_il_bc_c0_s15000_$town CARLA_PORT=$port EVAL_FROM=$C \
    bash slurm/train_sac.sbatch
  echo "=== TRACEHOP $town rc=$? ==="
done
echo "=== TRACEHOP DONE ==="
