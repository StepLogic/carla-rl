#!/usr/bin/env bash
# PREPARED, NOT RUN (2026-11-09): the user asked for all ViNT jobs to be cancelled
# before this was submitted. Submit only on the user's request.
#
# ONE job re-validates every ViNT finalist at evaluation parity (user 2026-11-09:
# success = within 1 m of the goal at ANY speed). Town05 + Town01, EVAL_SEED 50000+,
# n=40, true goal, overlay slurm/il_eval_val_chw_any1m.env. Results go to
# ~/carla-rl/checkpoints/<TAG>_val_any1m_<town>/eval_epoch_<E>/eval.json.
# The finalists were selected under the old rule (2 m, <= 10 km/h).
#
# First sync slurm/il_eval_val_chw_any1m.env, slurm/vint_val_epochs.sh and this
# file to ~/carla-rl-vint (tmp + mv), then:
#   sbatch -p gpu --gres=gpu:rtx2080:1 -c 4 --mem=48G -t 12:00:00 --exclude=node-g106 \
#     -J vint_val_any1m -o ~/carla-rl/logs/vint_val_any1m_%j.out --export=ALL \
#     ~/carla-rl-vint/slurm/vint_val_any1m.sh
# vint_val_epochs.sh re-exports checkpoints/<TAG>/epoch_<E> (same .pth symlink, same
# vint.json) before each eval; the standalone copies in checkpoints/vint_final/ are
# not touched.
set -u
cd ~/carla-rl-vint || exit 1
L=$HOME/vint/visualnav/train/logs/vint_carla
for spec in "vint_carla_dagger3:vint_carla_dagger3_2026_11_08_07_54_56:0 1 2" \
            "vint_carla_dagger2:vint_carla_dagger2_2026_11_07_23_56_53:2" \
            "vint_carla_dagger1:vint_carla_dagger1_2026_11_07_21_01_46:1"; do
  IFS=: read -r TAG RUN EPOCHS <<< "$spec"
  RUN_DIR=$L/$RUN CONFIG=scripts/vint/$TAG.yaml TAG=$TAG EPOCHS="$EPOCHS" PORT=3100 \
    VAL_OVERLAY=slurm/il_eval_val_chw_any1m.env VAL_SUFFIX=_any1m bash slurm/vint_val_epochs.sh
done
echo "=== VINT_VAL_ANY1M done ==="
