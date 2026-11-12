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
# comparison, not a discard. PORTS: Town02 7700, Town03 7300. 7200 is TAKEN --
# the no-hop Town03 twin (464222) ran its CARLA there, and the first Town02 hop job
# (464230) connected to that server, reset its hero and killed both. A port's
# owner is in the job log's "=== town ... port N" line; check before reusing one.
set -u
cd ~/carla-rl
C=$HOME/carla-rl/checkpoints/il_bc_c0/step_15000
for town in ${TRACE_TOWNS:-Town02 Town03}; do
  case $town in Town02) port=7700;; Town03) port=7300;; *) port=7800;; esac
  echo "=== TRACEHOP $town port $port ==="
  REWARD_OVERLAY=slurm/il_trace_hop_$town.env ENV_TOWNS=$town \
    RUN_NAME=trace1hop_il_bc_c0_s15000_$town CARLA_PORT=$port EVAL_FROM=$C \
    bash slurm/train_sac.sbatch
  echo "=== TRACEHOP $town rc=$? ==="
done
echo "=== TRACEHOP DONE ==="
