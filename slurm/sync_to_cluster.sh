#!/usr/bin/env bash
# Put the torch recurrent-SAC arm on the ACG cluster.
#
# User 2026-10-12: "focus on moving the recurrent SAC policy to the cluster".
# ~/carla-rl does NOT exist on the cluster (the old slurm/tasks.sh refers to a
# copy that is gone), so this script creates it.
#
# WHAT GOES AND WHAT STAYS. Code, the DEBiT-tiny weights (105 MB) and the croco
# checkout (1.4 MB) go. Checkpoints, replays, logs, results, the .sif images and
# the local venvs do NOT: the images are already on the cluster
# (~/carla-cluster/carla0916.sif) and the venvs are built there
# (~/carla-cluster/envs/carla310), so copying them would waste an hour and
# overwrite a working environment with one built against a different driver.
#
# The JAX lineage comes along as files but is never imported by this arm; the
# cluster env has no jaxlib and torch_rl/carla_env_build.py exists precisely so
# nothing in this arm touches train_carla_tqc.
#
# usage: slurm/sync_to_cluster.sh [--dry-run]
set -euo pipefail
HOST=${CLUSTER_HOST:-login1.acg.maine.edu}
SRC=$(cd "$(dirname "$0")/.." && pwd)
DEST=${CLUSTER_DEST:-carla-rl}
DRY=${1:-}

TINY_SRC=${DEBIT_TINY:-/home/robotlab/scratch/debit/out/ckpt/hab_bl/imgnav/debit_tiny.pth}
[ -f "$TINY_SRC" ] || { echo "FATAL: no DEBiT-tiny weights at $TINY_SRC"; exit 1; }

echo "[sync] $SRC -> $HOST:~/$DEST"
rsync -az --delete ${DRY:+--dry-run} --info=stats1 \
  --exclude '.git/' --exclude '.claude/' \
  --exclude 'checkpoints/' --exclude 'logs/' --exclude 'results/' \
  --exclude 'evidence/' --exclude 'icra-2027/' --exclude 'paper/' \
  --exclude 'data/' --exclude 'archive/' --exclude 'wandb/' \
  --exclude '*.sif' --exclude '*venv*/' --exclude '__pycache__/' \
  --exclude '*.pkl' --exclude '*.pth' --exclude '*.mp4' --exclude '*.png' \
  --exclude '/debit' --exclude '/dependencies/croco' \
  --exclude '/dependencies/debit_weights' \
  "$SRC"/ "$HOST":"$DEST"/

# The assets the arm cannot run without, sent explicitly because the rules above
# exclude every .pth and every symlink target outside the tree. The destination
# directories are made FIRST: rsync does not create a missing parent for a
# single-file transfer, and under --dry-run it never creates anything, so
# without this the dry run fails on a directory the real run would have made.
echo "[sync] DEBiT-tiny + croco"
# Those three paths are SYMLINKS in the local worktree (into the main checkout,
# where the untracked assets live). rsync -a would copy the links themselves and
# leave dangling absolute paths on the cluster, so they are excluded above and
# the real directories are made here -- replacing a link if an earlier sync left
# one, which is why this is rm -f and not a bare mkdir.
# Remove them ONLY if an earlier sync left a symlink; once they are real
# directories a plain `rm -f` fails on them and, under set -e, kills the script
# before the assets are sent.
ssh "$HOST" "cd ~/$DEST && \
  for p in debit dependencies/croco dependencies/debit_weights; do \
    [ -L \"\$p\" ] && rm -f \"\$p\"; done; \
  mkdir -p dependencies/debit_weights dependencies/croco"
# shellcheck disable=SC2086
rsync -az ${DRY:+--dry-run} --info=stats1 \
  "$TINY_SRC" "$HOST":"$DEST"/dependencies/debit_weights/debit_tiny.pth
# shellcheck disable=SC2086
rsync -az ${DRY:+--dry-run} --info=stats1 --copy-links \
  /home/robotlab/scratch/carla-rl/dependencies/croco/ \
  "$HOST":"$DEST"/dependencies/croco/

# rlib_integration is a git SUBMODULE and is not checked out in this worktree:
# the directory exists and is EMPTY, so it synced as an empty directory and the
# cluster env died on "No module named rlib_integration.carla_env" three stages
# into a smoke run. It is populated in the main checkout, so it comes from there
# like croco and the DEBiT weights.
echo "[sync] rlib_integration (submodule, empty in this worktree)"
# shellcheck disable=SC2086
rsync -az ${DRY:+--dry-run} --info=stats1 --copy-links --exclude '__pycache__/' \
  /home/robotlab/scratch/carla-rl/dependencies/rlib_integration/ \
  "$HOST":"$DEST"/dependencies/rlib_integration/

if [ -z "$DRY" ]; then
  ssh "$HOST" "mkdir -p ~/$DEST/logs ~/$DEST/checkpoints && \
    ls -la ~/$DEST/dependencies/debit_weights/ && \
    test -f ~/$DEST/dependencies/croco/models/croco.py && echo '[sync] croco ok' && \
    test -f ~/$DEST/torch_rl/gated_recurrent_sac.py && echo '[sync] learner ok'"
fi
echo "[sync] done"
