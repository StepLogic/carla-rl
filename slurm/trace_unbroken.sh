#!/usr/bin/env bash
# THE UNBROKEN EVALUATION-MAP DRIVE (slurm/il_trace_unbroken.env). One town per job:
#   UNBROKEN=Town02 sbatch -p gpu --gres=gpu:l40:1 -c 4 --mem=64G -t 24:00:00 \
#     -J unbroken_c0_Town02 -o logs/unbroken_c0_Town02_%j.log slurm/trace_unbroken.sh
# UNBROKEN=pilot runs the 600 m Town02 preflight.
# PORTS 8100 / 8200 / 8300: 7200 belonged to a no-hop trace and one job sharing
# it killed another (823a646); check the logs' "port N" lines before reusing one.
set -u
cd ~/carla-rl
C=$HOME/carla-rl/checkpoints/il_bc_c0/step_15000
for n in ${UNBROKEN:-Town02 Town03}; do
  case $n in Town02) port=8100; town=Town02;; Town03) port=8200; town=Town03;;
             pilot) port=8300; town=Town02;; *) echo "unknown $n"; exit 1;; esac
  echo "=== UNBROKEN $n ($town) port $port ==="
  REWARD_OVERLAY=slurm/il_trace_unbroken_$n.env ENV_TOWNS=$town \
    RUN_NAME=unbroken_il_bc_c0_s15000_$n CARLA_PORT=$port EVAL_FROM=$C \
    bash slurm/train_sac.sbatch
  echo "=== UNBROKEN $n rc=$? ==="
done
