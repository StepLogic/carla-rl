#!/usr/bin/env bash
# The TRUE-goal half of slurm/vint_eval_pair.sh only (user 2026-11-07, "can you make
# it fast"): for choosing an epoch quickly. Every ViNT epoch so far had a wrong-goal
# arrival rate of 0, so the wrong half is run for the chosen epoch, not during the
# search. Same output folder as the pair's first half (eval_<tag>), so a later
# full pair skips it.
#
#   OVERLAY=slurm/il_eval_val_chw.env CKPT=<dir> TOWN=Town05 RUN=<run> N=40 PORT=3100 \
#     bash slurm/vint_eval_true.sh
set -u
REPO=${REPO:-$HOME/carla-rl-vint}
cd "$REPO" || exit 1
: "${OVERLAY:?}" "${CKPT:?}" "${TOWN:?}" "${RUN:?}" "${N:?}" "${PORT:?}"
export REPO EVAL_POLICY=vint
export PYTHONPATH="$HOME/vint/pkgs:$HOME/vint/visualnav/train${PYTHONPATH:+:$PYTHONPATH}"
out=$HOME/carla-rl/checkpoints/$RUN/eval_$(basename "$CKPT")
if [ -f "$out/eval.json" ]; then echo "=== have $out"; exit 0; fi
echo "=== VINT_EVAL_TRUE $RUN $TOWN n=$N ckpt=$CKPT ==="
REWARD_OVERLAY=$OVERLAY ENV_TOWNS=$TOWN RUN_NAME=$RUN CARLA_PORT=$PORT \
  WRONG_GOAL_MODE=0 EVAL_FROM=$CKPT EVAL_EPISODES=$N bash slurm/train_sac.sbatch
echo "=== VINT_EVAL_TRUE $RUN $TOWN rc=$? ==="
