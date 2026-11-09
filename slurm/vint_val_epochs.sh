#!/usr/bin/env bash
# ONE job validates several epochs of a ViNT training run (user 2026-11-07: "do not
# create so many jobs - make single jobs get enough data"): for each epoch, export
# <RUN_DIR>/<E>.pth (symlink) to checkpoints/<TAG>/epoch_<E>, then the true-goal
# validation (slurm/vint_eval_true.sh, EVAL_SEED 50000, n=N) in each town, one
# after another, then a summary line per epoch.
#
#   RUN_DIR=~/vint/visualnav/train/logs/vint_carla/<run> CONFIG=scripts/vint/<cfg>.yaml \
#   TAG=vint_carla_dagger2 EPOCHS="0 1 2" PORT=3100 \
#   sbatch -p gpu --gres=gpu:rtx2080:1 -c 4 --mem=48G -t 6:00:00 --export=ALL slurm/vint_val_epochs.sh
# VAL_OVERLAY / VAL_SUFFIX (user 2026-11-09, train/eval parity): another validation
# rule, e.g. VAL_OVERLAY=slurm/il_eval_val_chw_any1m.env VAL_SUFFIX=_any1m (1 m, any
# speed), written to <TAG>_val<VAL_SUFFIX>_<town> so it never collides with earlier runs.
set -u
REPO=${REPO:-$HOME/carla-rl-vint}
cd "$REPO" || exit 1
: "${RUN_DIR:?}" "${CONFIG:?}" "${TAG:?}" "${EPOCHS:?}" "${PORT:?}"
TOWNS=${TOWNS:-Town05 Town01}
N=${N:-40}
VAL_OVERLAY=${VAL_OVERLAY:-slurm/il_eval_val_chw.env}
VAL_SUFFIX=${VAL_SUFFIX:-}
PY=${PY:-$HOME/carla-cluster/envs/carla310/bin/python}
for E in $EPOCHS; do
  CK=$HOME/carla-rl/checkpoints/$TAG/epoch_$E
  PYTHONPATH=$HOME/vint/pkgs $PY scripts/vint/export_vint_ckpt.py --ckpt "$RUN_DIR/$E.pth" \
    --config "$CONFIG" --summary "$HOME/carla-rl/checkpoints/vint_data/carla_expert_summary.json" \
    --out "$CK" --link || { echo "=== export of epoch $E failed"; continue; }
  for T in $TOWNS; do
    OVERLAY=$VAL_OVERLAY CKPT=$CK TOWN=$T RUN=${TAG}_val${VAL_SUFFIX}_$T N=$N PORT=$PORT \
      bash slurm/vint_eval_true.sh
  done
done
echo "=== VINT_VAL_EPOCHS SUMMARY $TAG (true goal, n=$N per town, $VAL_OVERLAY) ==="
for E in $EPOCHS; do
  line="epoch $E:"
  for T in $TOWNS; do
    f=$HOME/carla-rl/checkpoints/${TAG}_val${VAL_SUFFIX}_$T/eval_epoch_$E/eval.json
    sr=$($PY -c "import json; print(json.load(open('$f'))['success_rate'])" 2>/dev/null || echo NA)
    line="$line $T $sr"
  done
  echo "$line"
done
