#!/usr/bin/env bash
# SINGLE-TRACE EVALUATION MAPS, step 2 (the SGM arm only): v30's goal critic read
# between the expert trace's frames (slurm/sgm_pairs_single.env). One town per job:
#   SGM_TOWNS=Town03 sbatch --gres=gpu:l40:1 ... --wrap='bash slurm/sgm_pairs_single.sh'
# Then: scripts/build_single_trace_maps.py --sgm --out checkpoints/maps_single
set -u
cd ~/carla-rl
C=$HOME/carla-rl/checkpoints/sac_debitmono_t05t01_v30/step_300000
for town in ${SGM_TOWNS:-Town02 Town03}; do
  case $town in Town02) port=7400;; Town03) port=7500;; *) port=7600;; esac
  out=$HOME/carla-rl/checkpoints/maps_single/sgm_pairs_$town.npz
  if [ -f "$out" ]; then echo "=== have $town"; continue; fi
  echo "=== SGMPAIRS single $town port $port ==="
  REWARD_OVERLAY=slurm/sgm_pairs_single.env ENV_TOWNS=$town \
    EVAL_SGM_TRACE=$HOME/carla-rl/checkpoints/trace_expert/trace_expert_$town EVAL_SGM_OUT=$out \
    RUN_NAME=sgm_pairs_single_$town CARLA_PORT=$port EVAL_FROM=$C bash slurm/train_sac.sbatch
  echo "=== SGMPAIRS single $town rc=$? ==="
done
echo "=== SGMPAIRS single DONE ==="
