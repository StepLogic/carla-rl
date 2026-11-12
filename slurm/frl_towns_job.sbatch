#!/bin/bash
#SBATCH --job-name=frl_t05t01
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=24:00:00
#SBATCH --output=/home/egyaase/carla-rl-frl/logs/%x_%j.log
# LEARNER C, TOWN05 + TOWN01, ON THE CLUSTER (user 2026-11-11: "confirmation of learner implementation and
# hyperparameters so we can start town05 and town01 train"). The code is THIS tree ($REPO, one commit, deployed
# as its own directory so the shared ~/carla-rl is untouched); the run is slurm/train_sac.sbatch with
# OVERLAY (default slurm/cluster_frl_c0emb_t05t01_ar2_xp.env). EVALS INSIDE THE JOB (user rule 2026-11-07), one
# at a time while training runs: every checkpoint in EVAL_STEPS, Town05 then Town01, true then wrong goal,
# EVAL_N episodes each, at action repeat 2 (slurm/il_eval_val_chw_jr_any1m_ar2.env) -- the local runs' evals.
#   REPO=$HOME/carla-rl-frl/<sha> RUN=<name> [OVERLAY=...] [EVAL_STEPS=...] [EVAL_N=40] \
#     sbatch --export=ALL $REPO/slurm/frl_towns_job.sbatch
set -u
: "${REPO:?}" "${RUN:?}"
cd "$REPO" || exit 1
export CKPT_DIR=$HOME/carla-rl/checkpoints
export DEBIT_SRC=$HOME/carla-rl/dependencies
export DEBIT_WEIGHTS=$HOME/carla-rl/dependencies/debit_weights/debit_tiny.pth
export CARLA_QUALITY=Epic PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OVERLAY=${OVERLAY:-slurm/cluster_frl_c0emb_t05t01_ar2_xp.env}
STEPS=${EVAL_STEPS:-"50000 100000 150000 200000 250000"}
N=${EVAL_N:-40}
EV=${EVAL_OVERLAY:-slurm/il_eval_val_chw_jr_any1m_ar2.env}
TRAIN_PORT=${TRAIN_PORT:-4400}
EVAL_PORT=${EVAL_PORT:-5600}
PY=$HOME/carla-cluster/envs/carla310/bin/python
LOGS=$HOME/carla-rl-frl/logs
JID=${SLURM_JOB_ID:-local}
DONE=$LOGS/.${RUN}_${JID}.trained
rm -f "$DONE"
echo "=== $(date +%T) node $(hostname) job $JID run $RUN code $REPO overlay $OVERLAY"
echo "=== evals: steps [$STEPS] n=$N overlay $EV ports $EVAL_PORT/$((EVAL_PORT + 50)) -> $LOGS/${RUN}_evals_$JID.log"
evals() {
  for s in $STEPS; do
    c=$CKPT_DIR/$RUN/step_$s
    until [ -f "$c/config.json" ]; do
      if [ -f "$DONE" ] && [ ! -f "$c/config.json" ]; then
        echo "=== EVALS $(date +%T): training ended before $c"; return
      fi
      sleep 60
    done
    for t in Town05 Town01; do
      R=${RUN}_val_any1m_$t
      for wg in 0 1; do
        out=$CKPT_DIR/$R/eval_step_$s
        [ "$wg" = 1 ] && out=${out}_wrong
        if [ -f "$out/eval.json" ]; then echo "=== have $out"; continue; fi
        p=$((EVAL_PORT + 50 * wg))
        echo "=== EVALS $(date +%T): step $s $t wrong=$wg port $p"
        REWARD_OVERLAY=$EV ENV_TOWNS=$t RUN_NAME=$R CARLA_PORT=$p WRONG_GOAL_MODE=$wg \
          EVAL_FROM=$c EVAL_EPISODES=$N bash "$REPO/slurm/train_sac.sbatch"
        echo "=== EVALS $(date +%T): step $s $t wrong=$wg rc=$?"
        pkill -9 -f -- "-carla-rpc-por[t]=$p( |\$)" 2>/dev/null || true
      done
      "$PY" -I "$REPO/scripts/paired_wrong.py" "$CKPT_DIR/$R/eval_step_$s" \
        "$CKPT_DIR/$R/eval_step_${s}_wrong" 2>&1 | sed "s/^/[Gap] /"
    done
  done
  echo "=== EVALS DONE $(date +%T)"
}
evals > "$LOGS/${RUN}_evals_$JID.log" 2>&1 &
EVPID=$!
REWARD_OVERLAY=$OVERLAY RUN_NAME=$RUN CARLA_PORT=$TRAIN_PORT bash "$REPO/slurm/train_sac.sbatch"
rc=$?
touch "$DONE"
echo "=== $(date +%T) training rc=$rc; waiting for the evals (pid $EVPID)"
wait "$EVPID"
echo "=== $(date +%T) evals finished"
exit $rc
