#!/usr/bin/env bash
# user 2026-09-24: "v10-vel75 - evaluate in town02 and town03 to test generalization".
# One town per invocation with a FRESH server between them: eval_arm.py's ensure_carla
# attaches to whatever answers on the port, so re-using a live server would drive the
# previous town's map. The kill is port-scoped so a peer's server on another port lives.
# Band is the snapshot's own 5-200 m, matching the existing Town05 baseline.
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl || exit 1
CK=checkpoints/carla_gated_her_v10_vel75_vc1/checkpoint_2000000.pkl
say(){ echo "=== [gen $(date '+%m-%d %H:%M:%S')] $* ==="; }
kill_my_carla(){ for _ in 1 2 3 4 5; do pkill -9 -f 'carla-port=2000' 2>/dev/null; sleep 2;
  pgrep -f 'carla-port=2000' >/dev/null || return 0; done; return 1; }

for TOWN in Town02 Town03; do
  OUT=results/v10_vel75_eval/$(echo $TOWN | tr 'A-Z' 'a-z')_checkpoint_2000000.json
  if [ -s "$OUT" ]; then say "$TOWN already done -> $OUT"; continue; fi
  say "$TOWN starting (fresh server)"
  kill_my_carla || { say "FAILED: port-2000 CARLA survived"; exit 1; }
  sleep 10
  bash eval_v10_vel75.sh "$CK" "$TOWN" || { say "FAILED: $TOWN eval returned $?"; exit 1; }
  [ -s "$OUT" ] || { say "FAILED: $TOWN produced no $OUT"; exit 1; }
  say "$TOWN done -> $OUT"
done
kill_my_carla
say "GENERALIZATION EVAL DONE"
