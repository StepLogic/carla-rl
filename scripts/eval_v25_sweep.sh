#!/usr/bin/env bash
# v25 checkpoint sweep (user 2026-09-28 "proceed", then "evaluate explore mode";
# MODE=goal|explore, EPISODES): 150k/300k/450k x Town05
# (Epic) and Town01 (Low), 40 goal episodes each, seed-0 fixed set, per-step Q
# trace (image-only q AND privileged q_p). One server, one eval at a time.
# Resumable by artefact.
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl/clean_fix1 || exit 1
OUT=results/v25_eval
CKD=checkpoints/carla_gated_v25_t01stagger_t05epic_vc1
say(){ echo "=== [v25sweep $(date '+%m-%d %H:%M:%S')] $* ==="; }
kill_my_carla(){ for _ in 1 2 3 4 5; do pkill -9 -f 'carla-port=2000' 2>/dev/null; sleep 2;
  pgrep -f 'carla-port=2000' >/dev/null || return 0; done; return 1; }
MODE="${MODE:-goal}"; EPISODES="${EPISODES:-40}"; TOWNS="${TOWNS:-Town05 Town01}"
mkdir -p "$OUT"
for ck in 150000 300000 450000; do
  for town in $TOWNS; do
    f="$OUT/$(echo $town | tr A-Z a-z)_checkpoint_${ck}_${MODE}.json"
    if [ -s "$f" ]; then say "SKIP $f"; continue; fi
    kill_my_carla || { say "FAILED: CARLA survived the kill"; exit 1; }; sleep 8
    say "START $ck $town $MODE"
    EVAL_Q_TRACE=1 EVAL_MODE=$MODE EPISODES=$EPISODES EVAL_OUTDIR=$OUT bash eval_v25.sh "$CKD/checkpoint_${ck}.pkl" "$town"
    [ -s "$f" ] && say "DONE $ck $town" || say "FAILED: no $f"
  done
done
kill_my_carla
say "SWEEP DONE"
