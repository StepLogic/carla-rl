#!/usr/bin/env bash
# v36 checkpoint sweep (user 2026-09-30 "take over both"). One worker per port/town:
#   eval_v36_sweep.sh PORT TOWN MODE EPISODES "CKPT_STEPS..." [START_DELAY_S]
#
# Identical in structure to eval_v29_sweep.sh -- same eval_v25.sh entry, same
# fault-ends rule, same resumable-by-artefact behaviour -- with v29's checkpoint
# dir and snapshot swapped for v36's. Everything experiment-defining is replayed
# from v36's OWN snapshot, so perception (segvae), the counting regime, the
# 5-200 m band and fines 10.0 all match how it trained.
#
# THE BAND IS NOT PASSED. v36 trained on GOAL_DIST_MIN=5 / GOAL_DIST_MAX=200 and
# the snapshot carries both, so omitting --goal-min/--goal-max samples exactly
# the training band. The success-by-goal-distance split is recovered AFTERWARDS
# from the per-step qtrace (EVAL_Q_TRACE=1) by eval_distance_breakdown.py --
# each episode's initial route distance is its t=0 remaining_m. That is why this
# runs ONE wide-band sweep rather than one sweep per band: same information, a
# quarter of the GPU time, and no off-distribution probe.
#
# DOES NOT LAUNCH ANYTHING ON IMPORT. Run it deliberately, and not while the
# trainer holds the port: eval_v25.sh takes the same flock and will refuse.
PORT=$1; TOWN=$2; MODE=$3; EPS=$4; STEPS=$5; DELAY=${6:-0}
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl/clean_fix1 || exit 1
OUT=results/v36_eval
CKD=checkpoints/carla_gated_v36_segvae_nobc_noexpert
export SNAP=results/run_env/carla_gated_v36_segvae_nobc_noexpert.env
# user 2026-09-29: the first fined crossing, off-road or collision ends an eval episode
export EVAL_FAULT_ENDS="${EVAL_FAULT_ENDS:-fault}"
PAT="carla-port=${PORT}"
say(){ echo "=== [v36sweep:$PORT $(date '+%m-%d %H:%M:%S')] $* ==="; }
kill_my(){ for _ in 1 2 3 4 5; do pkill -9 -f -- "-${PAT}" 2>/dev/null; sleep 2; pgrep -f -- "-${PAT}" >/dev/null || return 0; done; return 1; }

[ -d "$CKD" ] || { say "FATAL: no checkpoint dir $CKD"; exit 1; }
[ -f "$SNAP" ] || { say "FATAL: no snapshot $SNAP"; exit 1; }
mkdir -p "$OUT"; sleep "$DELAY"
for ck in $STEPS; do
  f="$OUT/$(echo "$TOWN" | tr A-Z a-z)_checkpoint_${ck}_${MODE}.json"
  if [ -s "$f" ]; then say "SKIP $f"; continue; fi
  [ -f "$CKD/checkpoint_${ck}.pkl" ] || { say "SKIP (no checkpoint_${ck}.pkl)"; continue; }
  kill_my || { say "FAILED: server on $PORT survived"; exit 1; }; sleep 8
  say "START $ck $TOWN $MODE"
  EVAL_PORT=$PORT EVAL_Q_TRACE=1 EVAL_MODE=$MODE EPISODES=$EPS EVAL_OUTDIR=$OUT \
    bash eval_v25.sh "$CKD/checkpoint_${ck}.pkl" "$TOWN"
  [ -s "$f" ] && say "DONE $ck $TOWN $MODE" || say "FAILED: no $f"
done
kill_my
say "WORKER DONE"
