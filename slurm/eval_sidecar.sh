#!/usr/bin/env bash
# IN-JOB EVALS (user 2026-11-07: "evals should be in training not separate jobs").
# Runs beside the trainer INSIDE the same Slurm job (slurm/train_with_evals.sbatch
# starts it) and evaluates each checkpoint in EVAL_STEPS as soon as it is saved,
# through the very code path the standalone evals use (slurm/il_eval_pair.sh: true
# goal, then a wrong goal, N episodes from EVAL_SEED, faults end the episode), so the
# numbers are comparable with c0's. One eval at a time, on its own ports (CARLA
# servers are killed by port only, never by name -- torch_rl/carla_server.py).
#
#   RUN_NAME        the training run (checkpoints/<RUN_NAME>/step_<N>)
#   EVAL_STEPS      "100000 250000 ..."
#   EVAL_TOWNS      default "Town05 Town01"
#   EVAL_VARIANTS   "<overlay>:<tag> ..." -> checkpoints/<RUN_NAME>_val_<tag>_<Town>
#   EVAL_N          episodes per half (default 40)
#   EVAL_PORT       first port (default 5600; each eval uses PORT and PORT+50)
#   EVAL_DONE_FLAG  the wrapper touches it when training exits: a checkpoint that
#                   is not saved by then never will be, so stop waiting for it.
set -u
cd ~/carla-rl
: "${RUN_NAME:?}" "${EVAL_STEPS:?}"
TOWNS=${EVAL_TOWNS:-"Town05 Town01"}
VARIANTS=${EVAL_VARIANTS:-"slurm/il_eval_val_chw_jr.env:jr35"}
N=${EVAL_N:-40}
PORT0=${EVAL_PORT:-5600}
echo "=== SIDECAR $(date +%T) run $RUN_NAME steps [$EVAL_STEPS] towns [$TOWNS] variants [$VARIANTS] n=$N"
for s in $EVAL_STEPS; do
  c=$HOME/carla-rl/checkpoints/$RUN_NAME/step_$s
  # config.json is written AFTER params.pt (agent.save): it marks a whole checkpoint.
  until [ -f "$c/config.json" ]; do
    if [ -n "${EVAL_DONE_FLAG:-}" ] && [ -f "$EVAL_DONE_FLAG" ]; then
      echo "=== SIDECAR $(date +%T): training ended before $c; stopping"
      exit 0
    fi
    sleep 60
  done
  k=0
  for v in $VARIANTS; do
    ov=${v%%:*}
    tag=${v##*:}
    for t in $TOWNS; do
      echo "=== SIDECAR $(date +%T): step $s $t $tag"
      OVERLAY=$ov CKPT=$c TOWN=$t RUN=${RUN_NAME}_val_${tag}_$t N=$N \
        PORT=$((PORT0 + 100 * k)) bash slurm/il_eval_pair.sh
      k=$((k + 1))
    done
  done
done
echo "=== SIDECAR DONE $(date +%T) ==="
