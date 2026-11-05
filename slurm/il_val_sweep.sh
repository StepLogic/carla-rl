#!/usr/bin/env bash
# Validation sweep for a cloning run (imitation-first plan, Task 7 Step 8):
# every checkpoint of $BC_RUN, as soon as it is saved, on ONE training town,
# true and wrong goal, n=40, under slurm/il_eval_val.env (seeds from 50000).
# Runs beside the cloning job, so checkpoint choice is ready when it ends.
#
#   BC_RUN=il_bc_r0 TOWN=Town05 PORT=3900 bash slurm/il_val_sweep.sh
#
# Then: python3 -I scripts/select_checkpoint.py --out logs/<run>_selected.txt \
#         checkpoints/<run>_val_Town05/eval_step_* checkpoints/<run>_val_Town01/eval_step_*
set -u
cd ~/carla-rl
: "${BC_RUN:?}" "${TOWN:?}" "${PORT:?}"
STEPS=${STEPS:-"2500 5000 7500 10000 12500 15000 17500 20000"}
for s in $STEPS; do
  c=$HOME/carla-rl/checkpoints/$BC_RUN/step_$s
  waited=0
  # config.json is written AFTER params.pt (agent.save), so it marks a whole checkpoint.
  until [ -f "$c/config.json" ]; do
    sleep 60
    waited=$((waited + 60))
    if [ "$waited" -ge "${MAX_WAIT_S:-14400}" ]; then
      echo "=== VAL_SWEEP gave up waiting for $c"
      exit 1
    fi
  done
  OVERLAY=${VAL_OVERLAY:-slurm/il_eval_val.env} CKPT=$c TOWN=$TOWN RUN=${BC_RUN}_val_$TOWN N=40 PORT=$PORT \
    bash slurm/il_eval_pair.sh
done
echo "=== VAL_SWEEP DONE ==="
