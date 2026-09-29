#!/usr/bin/env bash
# v29 checkpoint sweep (user 2026-09-29 "proceed"). One worker per port/town:
#   eval_v29_sweep.sh PORT TOWN MODE EPISODES "CKPT_STEPS..." [START_DELAY_S]
# Uses clean_fix1/eval_v25.sh with v29's snapshot (seg-VAE perception, frame
# sync, no priv, POTENTIAL_K=0, Town01 Low / others Epic). Resumable by artefact.
PORT=$1; TOWN=$2; MODE=$3; EPS=$4; STEPS=$5; DELAY=${6:-0}
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl/clean_fix1 || exit 1
OUT=results/v29_eval
CKD=checkpoints/carla_gated_v29_segvae_bc_vc1free
export SNAP=results/run_env/carla_gated_v29_segvae_bc_vc1free.env
# user 2026-09-29: the first fined crossing, off-road or collision ends an eval episode
export EVAL_FAULT_ENDS="${EVAL_FAULT_ENDS:-fault}"
PAT="carla-port=${PORT}"
say(){ echo "=== [v29sweep:$PORT $(date '+%m-%d %H:%M:%S')] $* ==="; }
kill_my(){ for _ in 1 2 3 4 5; do pkill -9 -f -- "-${PAT}" 2>/dev/null; sleep 2; pgrep -f -- "-${PAT}" >/dev/null || return 0; done; return 1; }
mkdir -p "$OUT"; sleep "$DELAY"
for ck in $STEPS; do
  f="$OUT/$(echo "$TOWN" | tr A-Z a-z)_checkpoint_${ck}_${MODE}.json"
  if [ -s "$f" ]; then say "SKIP $f"; continue; fi
  kill_my || { say "FAILED: server on $PORT survived"; exit 1; }; sleep 8
  say "START $ck $TOWN $MODE"
  EVAL_PORT=$PORT EVAL_Q_TRACE=1 EVAL_MODE=$MODE EPISODES=$EPS EVAL_OUTDIR=$OUT \
    bash eval_v25.sh "$CKD/checkpoint_${ck}.pkl" "$TOWN"
  [ -s "$f" ] && say "DONE $ck $TOWN $MODE" || say "FAILED: no $f"
done
kill_my
say "WORKER DONE"
