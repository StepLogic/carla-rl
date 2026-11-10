#!/usr/bin/env bash
# B3 (plan 2026-11-09): write c1's shard list (M1 expert shards with id <= 199999 in
# the four training towns + every DAgger-in-chains shard of round ROUND), then clone
# with c0's recipe, validating checkpoints inside the job at the benchmark rule (1 m,
# any speed, junction off-road rule). One GPU (the L40): one job.
#   sbatch -p gpu --gres=gpu:l40:1 ... --wrap='bash slurm/il_bc_c1.sh'
set -u
cd ~/carla-rl
LIST=$HOME/carla-rl/logs/c1_shards.txt
: > "$LIST"
for T in Town01 Town04 Town05 Town10HD; do
  for f in checkpoints/il_data/$T/ep_*.npz; do
    id=$(basename "$f" .npz)
    id=${id#ep_}
    if [ "$id" -le 199999 ]; then echo "$HOME/carla-rl/$f" >> "$LIST"; fi
  done
  ls "$HOME"/carla-rl/checkpoints/il_data_dagger_chain"${ROUND:-1}"/$T/ep_*.npz 2>/dev/null >> "$LIST"
done
echo "=== c1 shard list: $(wc -l < "$LIST") shards ($(grep -c il_data_dagger_chain "$LIST") DAgger-in-chains) -> $LIST"
export BC_SHARD_LIST=$LIST
REWARD_OVERLAY=slurm/il_bc_c1.env RUN_NAME=${RUN_NAME:-il_bc_c1} \
  EVAL_STEPS=${EVAL_STEPS:-"5000 7500 10000 12500 15000"} \
  EVAL_VARIANTS="slurm/il_eval_val_chw_jr_any1m.env:any1m" EVAL_PORT=${EVAL_PORT:-5100} \
  bash slurm/train_with_evals.sbatch
