#!/usr/bin/env bash
# ONE job collects a ViNT DAgger round in every training town, one after another
# (user 2026-11-07: "do not create so many jobs - make single jobs get enough
# data"): slurm/vint_dagger.sh per town, its own port each (PORT + 10 * index).
#
#   DAGGER_ROUND=3 EVAL_FROM=<ckpt dir> EXPERT_RECORD_EPISODES=150 PORT=3150 \
#   sbatch -p gpu --gres=gpu:1 -c 4 --mem=48G -t 12:00:00 --export=ALL slurm/vint_dagger_towns.sh
set -u
REPO=${REPO:-$HOME/carla-rl-vint}
cd "$REPO" || exit 1
: "${DAGGER_ROUND:?}" "${EVAL_FROM:?}" "${PORT:?}"
p0=$PORT
k=0
for T in ${TOWNS:-Town01 Town04 Town05 Town10HD}; do
  TOWN=$T PORT=$((p0 + 10 * k)) bash slurm/vint_dagger.sh
  k=$((k + 1))
done
echo "=== VINT_DAGGER_TOWNS round $DAGGER_ROUND done ==="
