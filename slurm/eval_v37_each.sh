#!/usr/bin/env bash
# v37 checkpoint evals, run INSIDE the training job (srun --overlap), usage:
#   JOB=<training job id> bash slurm/eval_v37_each.sh
#   * every 100k: Town02 standing set (5-25 m, 85% past a junction), true vs
#     wrong goal image, paired gap (~/paired_wrong.py);
#   * every 200k: Town02 at 25-100 m, true goal;
#   * every 500k: Town05 (training town), standing set.
# Fixed seeds, deterministic, faults end the episode, matched goal camera,
# one goal per episode (chaining is training-only).
cd ~/carla-rl
export GOAL_CAM_MATCH=1
C=$HOME/carla-rl/checkpoints/sac_debitmono_t05t01_v37
D=$HOME/carla-rl/checkpoints/sac_debitmono_t05t01_v37_evalT02
for step in $(seq 100000 100000 10000000); do
  CK=$C/step_$step/params.pt
  until [ -f "$CK" ]; do
    squeue -j "$JOB" -h -t RUNNING | grep -q . || exit 0
    sleep 120
  done
  sleep 60
  echo "=== CKPT v37_$step Town02 TRUE ==="
  REWARD_OVERLAY=slurm/v37_eval.env ENV_TOWNS=Town02 RUN_NAME=sac_debitmono_t05t01_v37_evalT02 \
    CARLA_PORT=3100 EVAL_FROM=$C/step_$step EVAL_EPISODES=40 bash slurm/train_sac.sbatch
  echo "=== CKPT v37_$step Town02 WRONG ==="
  REWARD_OVERLAY=slurm/v37_eval.env ENV_TOWNS=Town02 RUN_NAME=sac_debitmono_t05t01_v37_evalT02 \
    CARLA_PORT=3170 WRONG_GOAL_MODE=1 EVAL_FROM=$C/step_$step EVAL_EPISODES=40 bash slurm/train_sac.sbatch
  python3 -I ~/paired_wrong.py $D/eval_step_$step $D/eval_step_${step}_wrong 2>&1 | sed 's/^/[Gap] /'
  if [ $((step % 200000)) -eq 0 ]; then
    echo "=== CKPT v37_$step Town02 LONG 25-100m ==="
    REWARD_OVERLAY=slurm/v37_eval_long.env ENV_TOWNS=Town02 \
      RUN_NAME=sac_debitmono_t05t01_v37_evalT02_long CARLA_PORT=3190 \
      EVAL_FROM=$C/step_$step EVAL_EPISODES=40 bash slurm/train_sac.sbatch
  fi
  if [ $((step % 500000)) -eq 0 ]; then
    echo "=== CKPT v37_$step Town05 TRUE ==="
    REWARD_OVERLAY=slurm/v37_eval.env ENV_TOWNS=Town05 RUN_NAME=sac_debitmono_t05t01_v37 \
      CARLA_PORT=3150 EVAL_FROM=$C/step_$step EVAL_EPISODES=40 bash slurm/train_sac.sbatch
  fi
done
