#!/usr/bin/env bash
# ONE job runs every cell of one map-benchmark arm (user 2026-11-07: "do not create
# so many jobs - make single jobs get enough data"): slurm/vint_bench_map.sh for
# each TOWN:BAND in CELLS, one after another, each cell on its own port
# (PORT + 10 * index, so a slow server shutdown cannot block the next cell).
# A cell whose eval.json exists is skipped by vint_bench_map.sh, so a rerun resumes.
#
#   CKPT=<ckpt dir> ARM=mapless|vinttopo [TOPOMAP=<dir>] LABEL=<label> PORT=3100 \
#   EPISODES=$HOME/carla-rl/logs/bench_long_episodes.json \
#   sbatch -p gpu --gres=gpu:1 -c 4 --mem=48G -t 24:00:00 --export=ALL slurm/vint_bench_cells.sh
set -u
REPO=${REPO:-$HOME/carla-rl-vint}
cd "$REPO" || exit 1
: "${CKPT:?}" "${ARM:?}" "${LABEL:?}" "${PORT:?}" "${EPISODES:?}"
CELLS=${CELLS:-Town02:easy Town02:medium Town02:hard Town03:easy Town03:medium Town03:hard}
p0=$PORT
k=0
for cell in $CELLS; do
  TOWN=${cell%%:*} BAND=${cell##*:} PORT=$((p0 + 10 * k)) bash slurm/vint_bench_map.sh
  k=$((k + 1))
done
echo "=== VINT_BENCH_CELLS $LABEL $ARM done ==="
