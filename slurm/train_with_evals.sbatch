#!/usr/bin/env bash
# Training with its evals INSIDE the job (user 2026-11-07: "evals should be in
# training not separate jobs"). Starts slurm/eval_sidecar.sh in the background,
# runs the unchanged trainer (slurm/train_sac.sbatch) in the foreground, then lets
# the sidecar finish the evals of checkpoints already saved before the job ends.
#
#   REWARD_OVERLAY=... RUN_NAME=... EVAL_STEPS="100000 250000 ..." \
#   EVAL_VARIANTS="slurm/il_eval_val_chw_jr.env:jr35 slurm/il_eval_val_chw_jr_nb.env:jr35nb" \
#   sbatch ... slurm/train_with_evals.sbatch
#
# The sidecar logs to logs/<RUN_NAME>_evals_<jobid>.log; results go to
# checkpoints/<RUN_NAME>_val_<tag>_<Town>/eval_step_<N>/eval.json as before.
set -u
cd ~/carla-rl
: "${RUN_NAME:?}" "${EVAL_STEPS:?}"
JID=${SLURM_JOB_ID:-local}
export EVAL_DONE_FLAG=$HOME/carla-rl/logs/.${RUN_NAME}_${JID}.trained
rm -f "$EVAL_DONE_FLAG"
bash slurm/eval_sidecar.sh > "logs/${RUN_NAME}_evals_${JID}.log" 2>&1 &
SIDECAR=$!
echo "=== $(date +%T) eval sidecar pid $SIDECAR -> logs/${RUN_NAME}_evals_${JID}.log"
bash slurm/train_sac.sbatch
rc=$?
touch "$EVAL_DONE_FLAG"
echo "=== $(date +%T) training rc=$rc; waiting for the eval sidecar to finish"
wait "$SIDECAR"
echo "=== $(date +%T) eval sidecar rc=$?"
exit $rc
