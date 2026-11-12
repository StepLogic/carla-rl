#!/usr/bin/env bash
# Per-step Q vs faults on v17's checkpoint_150000 (user 2026-09-27: "does q
# function predict fatal states correctly?" -> "do b"). Goal mode Town05 60
# episodes (the same seed-0 set as results/v17_eval_150k), explore mode Town05
# 30, each with EVAL_Q_TRACE=1, then scripts/q_fault_report.py. v18 is stopped.
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl/clean_fix1 || exit 1
OUT=results/v17_qtrace_150k
CK=checkpoints/carla_gated_her_v17_sorb_fixed_vc1/checkpoint_150000.pkl
say(){ echo "=== [qmeasure $(date '+%m-%d %H:%M:%S')] $* ==="; }
kill_my_carla(){ for _ in 1 2 3 4 5; do pkill -9 -f 'carla-port=2000' 2>/dev/null; sleep 2;
  pgrep -f 'carla-port=2000' >/dev/null || return 0; done; return 1; }
mkdir -p "$OUT"
for spec in "goal 60" "explore 30"; do
  set -- $spec
  f="$OUT/town05_checkpoint_150000_qtrace_$1.json.qtrace.jsonl"
  if [ -s "$f" ]; then say "SKIP $f"; continue; fi
  kill_my_carla || { say "FAILED: CARLA survived the kill"; exit 1; }; sleep 10
  say "START $1 ($2 episodes)"
  EVAL_Q_TRACE=1 EVAL_MODE=$1 EPISODES=$2 EVAL_OUTDIR=$OUT bash eval_v17.sh "$CK" Town05 _qtrace
  [ -s "$OUT/town05_checkpoint_150000_qtrace_$1.json.qtrace.jsonl" ] || { say "FAILED: no trace for $1"; exit 1; }
  say "DONE $1"
done
kill_my_carla
cd .. && python3 scripts/q_fault_report.py 0.99 clean_fix1/$OUT/*.qtrace.jsonl
say "Q MEASUREMENT DONE"
