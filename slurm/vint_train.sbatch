#!/usr/bin/env bash
# ViNT fine-tuned on c0's expert data (scripts/vint/vint_carla_expert.yaml; user
# 2026-11-07). Runs from the vint-carla shadow tree; the ViNT code, its extra
# packages and the released weights live in ~/vint, outside the synced tree.
#
#   cd ~/carla-rl-vint && sbatch --partition=gpu --nodelist=<node> --gres=gpu:1 \
#     --cpus-per-task=16 --mem=64gb --time=12:00:00 --job-name=vint_train \
#     --output=logs/%x_%j.log slurm/vint_train.sbatch
#
# MODE=probe runs scripts/vint/probe_vint_setup.py (PROBE_TRAJ trajectories) instead
# of training. The data comes as ONE tar (checkpoints/vint_data/carla_expert.tar) and
# is unpacked on the node's local disk, where ViNT builds its LMDB image cache.
set -uo pipefail
REPO=${REPO:-$HOME/carla-rl-vint}
VINT=${VINT:-$HOME/vint}
PY=${PY:-$HOME/carla-cluster/envs/carla310/bin/python}
DATA_TAR=${DATA_TAR:-$HOME/carla-rl/checkpoints/vint_data/carla_expert.tar}
WEIGHTS=${WEIGHTS:-$VINT/weights/vint_release.pth}
CONFIG=${CONFIG:-$REPO/scripts/vint/vint_carla_expert.yaml}
TRAIN=$VINT/visualnav/train
# scripts/vint/stubs first: its wandb stands in for the cluster's, which cannot import.
export PYTHONPATH="$REPO/scripts/vint/stubs:$VINT/pkgs:$TRAIN:$VINT/visualnav/diffusion_policy${PYTHONPATH:+:$PYTHONPATH}"
echo "=== $(date +%T) node $(hostname) job ${SLURM_JOB_ID:-local} mode ${MODE:-train} config $CONFIG ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU visible"

LOCAL=${TMPDIR:-/tmp}/vint_${SLURM_JOB_ID:-$$}
mkdir -p "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT
t0=$(date +%s)
tar -xf "$DATA_TAR" -C "$LOCAL" || { echo "=== tar failed"; exit 1; }
echo "=== data unpacked in $(( $(date +%s) - t0 )) s: $(du -sh "$LOCAL" | cut -f1) | $(df -h "$LOCAL" | tail -1)"

# ViNT's goal bound is wrong for waypoint_spacing > 1 (scripts/vint/patch_vint_dataset.py).
$PY "$REPO/scripts/vint/patch_vint_dataset.py" "$TRAIN/vint_train/data/vint_dataset.py" || exit 1

# ViNT reads metric_waypoint_spacing for the dataset from data_config.yaml.
SPACING=$($PY -c "import json; print(json.load(open('$LOCAL/carla_expert/summary.json'))['metric_waypoint_spacing_m'])")
DC=$TRAIN/vint_train/data/data_config.yaml
grep -q '^carla_expert:' "$DC" || printf '\ncarla_expert:\n  metric_waypoint_spacing: %s\n' "$SPACING" >> "$DC"
echo "=== data_config: $(grep -A1 '^carla_expert:' "$DC" | tr '\n' ' ') (measured $SPACING)"

sed -e "s#__DATA__#$LOCAL#g" -e "s#__WEIGHTS__#$WEIGHTS#g" "$CONFIG" > "$LOCAL/config.yaml"
cd "$TRAIN" || exit 1
if [ "${MODE:-train}" = probe ]; then
  $PY "$REPO/scripts/vint/probe_vint_setup.py" "$LOCAL/config.yaml" "${PROBE_TRAJ:-4}"
else
  $PY train.py --config "$LOCAL/config.yaml"
fi
rc=$?
echo "=== $(date +%T) vint ${MODE:-train} rc=$rc ==="
exit $rc
