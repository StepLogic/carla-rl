#!/bin/bash
#SBATCH --job-name=c0_val_ar2_rules
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=6:00:00
#SBATCH --output=/home/egyaase/carla-rl-frl/logs/%x_%j.log
# THE CONTROL FOR THE RL ROW: c0 (il_bc_c0 step_15000) on the SAME validation the learner-C runs
# report, at the SAME control rate (ACTION_REPEAT=2), on the SAME simulator (cluster CARLA 0.9.16),
# under BOTH arrival rules -- any speed and slow (<= 10 km/h) -- run LIVE, never filtered.
#   checkpoints/c0_val_<tag>_ar2_<Town>/eval_step_15000{,_wrong}/eval.json
# Why: the cluster's c0_valany1m_Town0{1,5} is action repeat 1 and any speed only, and the local
# c0_local_k2_val_* pair is CARLA 0.9.15. Neither is a like-for-like control for a learner-C row.
#   REPO=$HOME/carla-rl-frl/f73455d sbatch --export=ALL --dependency=afterany:<tail> \
#     $REPO/slurm/c0_val_ar2_rules_job.sbatch
set -u
REPO=${REPO:-$HOME/carla-rl-frl/f73455d}
cd "$REPO" || exit 1
export CKPT_DIR=$HOME/carla-rl/checkpoints
export DEBIT_SRC=$HOME/carla-rl/dependencies
export DEBIT_WEIGHTS=$HOME/carla-rl/dependencies/debit_weights/debit_tiny.pth
export CARLA_QUALITY=Epic PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C0=${C0:-$CKPT_DIR/il_bc_c0/step_15000}
N=${EVAL_N:-40}
PORT=${EVAL_PORT:-5200}
PY=$HOME/carla-cluster/envs/carla310/bin/python
VARIANTS=${EVAL_VARIANTS:-"slurm/il_eval_val_chw_jr_any1m_ar2.env:any1m slurm/il_eval_val_chw_jr_strict1m_ar2.env:strict1m"}
echo "=== $(date +%T) node $(hostname) job ${SLURM_JOB_ID:-local} code $REPO c0 $C0"
[ -d "$C0" ] || { echo "FATAL: no c0 at $C0"; exit 1; }
k=0
for v in $VARIANTS; do
  ov=${v%%:*}; tag=${v##*:}
  [ -f "$REPO/$ov" ] || { echo "FATAL: no overlay $ov in $REPO"; exit 1; }
  for t in Town01 Town05; do
    R=c0_val_${tag}_ar2_$t
    for wg in 0 1; do
      out=$CKPT_DIR/$R/eval_step_15000
      [ "$wg" = 1 ] && out=${out}_wrong
      if [ -f "$out/eval.json" ]; then echo "=== have $out"; continue; fi
      p=$((PORT + 50 * (k % 4)))
      echo "=== $(date +%T) EVAL c0 $tag $t wrong=$wg port $p"
      REWARD_OVERLAY=$ov ENV_TOWNS=$t RUN_NAME=$R CARLA_PORT=$p WRONG_GOAL_MODE=$wg \
        EVAL_FROM=$C0 EVAL_EPISODES=$N bash "$REPO/slurm/train_sac.sbatch"
      echo "=== $(date +%T) EVAL c0 $tag $t wrong=$wg rc=$?"
      pkill -9 -f -- "-carla-rpc-por[t]=$p( |$)" 2>/dev/null || true
      k=$((k + 1))
    done
    "$PY" -I "$REPO/scripts/paired_wrong.py" "$CKPT_DIR/$R/eval_step_15000" \
      "$CKPT_DIR/$R/eval_step_15000_wrong" 2>&1 | sed "s/^/[Gap $tag $t] /"
  done
done
echo "=== $(date +%T) C0 AR2 RULE PAIR DONE"
