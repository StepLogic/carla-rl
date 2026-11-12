#!/usr/bin/env bash
# il_eval_pair.sh for ViNT (EVAL_POLICY=vint, torch_rl/vint_driver.py): the same
# EVAL_SEED episodes with the TRUE goal image, then with a WRONG one, then
# ~/paired_wrong.py on the pair -- c0's protocol, run from the vint-carla shadow
# tree with ViNT's code and packages on the path.
#
#   OVERLAY=slurm/il_eval_chw.env CKPT=<checkpoints/vint_.../epoch_N> TOWN=Town02 \
#     RUN=<run name> N=100 PORT=3100 bash slurm/vint_eval_pair.sh
#
# Faults always end an eval episode (_fault, train_carla_sac.py). Each half is
# skipped if its eval.json exists, so a killed job resumes. Ports: this session
# owns 3100-3399 on the cluster (each half uses PORT and PORT+50).
set -u
REPO=${REPO:-$HOME/carla-rl-vint}
cd "$REPO" || exit 1
: "${OVERLAY:?}" "${CKPT:?}" "${TOWN:?}" "${RUN:?}" "${N:?}" "${PORT:?}"
export REPO EVAL_POLICY=vint
export PYTHONPATH="$HOME/vint/pkgs:$HOME/vint/visualnav/train${PYTHONPATH:+:$PYTHONPATH}"
D=$HOME/carla-rl/checkpoints/$RUN
TAG=$(basename "$CKPT")
for wg in 0 1; do
  out=$D/eval_$TAG; [ "$wg" = 1 ] && out=${out}_wrong
  if [ -f "$out/eval.json" ]; then echo "=== have $out"; continue; fi
  echo "=== VINT_EVAL_PAIR $RUN $TOWN wrong=$wg n=$N ckpt=$CKPT ==="
  REWARD_OVERLAY=$OVERLAY ENV_TOWNS=$TOWN RUN_NAME=$RUN CARLA_PORT=$((PORT + 50 * wg)) \
    WRONG_GOAL_MODE=$wg EVAL_FROM=$CKPT EVAL_EPISODES=$N bash slurm/train_sac.sbatch
  echo "=== VINT_EVAL_PAIR $RUN $TOWN wrong=$wg rc=$? ==="
done
python3 -I ~/paired_wrong.py "$D/eval_$TAG" "$D/eval_${TAG}_wrong" 2>&1 | sed "s/^/[Gap] /"
echo "=== VINT_EVAL_PAIR DONE ==="
