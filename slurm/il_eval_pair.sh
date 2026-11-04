#!/usr/bin/env bash
# A paired goal-image eval (imitation-first plan, Tasks 7-9): the same EVAL_SEED
# episodes with the TRUE goal image, then with a WRONG one (WRONG_GOAL_MODE=1),
# then ~/paired_wrong.py on the pair. Each half is skipped if its eval.json
# already exists, so a killed job resumes.
#
#   OVERLAY=slurm/il_eval.env CKPT=<.../step_N> TOWN=Town02 RUN=<run name> \
#     N=100 PORT=3600 bash slurm/il_eval_pair.sh
#
# Faults always end an eval episode in this trainer (_fault, train_carla_sac.py).
# Optional: EVAL_SEED (the trainer's default when unset).
set -u
cd ~/carla-rl
: "${OVERLAY:?}" "${CKPT:?}" "${TOWN:?}" "${RUN:?}" "${N:?}" "${PORT:?}"
D=$HOME/carla-rl/checkpoints/$RUN
TAG=$(basename "$CKPT")
for wg in 0 1; do
  out=$D/eval_$TAG; [ "$wg" = 1 ] && out=${out}_wrong
  if [ -f "$out/eval.json" ]; then echo "=== have $out"; continue; fi
  echo "=== EVAL_PAIR $RUN $TOWN wrong=$wg n=$N ==="
  REWARD_OVERLAY=$OVERLAY ENV_TOWNS=$TOWN RUN_NAME=$RUN CARLA_PORT=$((PORT + 50 * wg)) \
    WRONG_GOAL_MODE=$wg EVAL_FROM=$CKPT EVAL_EPISODES=$N bash slurm/train_sac.sbatch
  echo "=== EVAL_PAIR $RUN $TOWN wrong=$wg rc=$? ==="
done
python3 -I ~/paired_wrong.py "$D/eval_$TAG" "$D/eval_${TAG}_wrong" 2>&1 | sed "s/^/[Gap] /"
echo "=== EVAL_PAIR DONE ==="
