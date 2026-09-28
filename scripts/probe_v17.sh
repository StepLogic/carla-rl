#!/usr/bin/env bash
# Probe capture on v17@150k, goal mode, Town05, same 60 seed-0 episodes as the
# Q measurement, then scripts/probe_fault.py. v18nh is paused (GPU alone).
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl/clean_fix1 || exit 1
OUT=results/v17_probe_150k
CK=checkpoints/carla_gated_her_v17_sorb_fixed_vc1/checkpoint_150000.pkl
say(){ echo "=== [probe $(date '+%m-%d %H:%M:%S')] $* ==="; }
for _ in 1 2 3 4 5; do pkill -9 -f 'carla-port=2000' 2>/dev/null; sleep 2; pgrep -f 'carla-port=2000' >/dev/null || break; done
sleep 10; mkdir -p "$OUT"
say "START goal 60"
EVAL_Q_TRACE=1 EVAL_Q_TRACE_OBS=1 EVAL_MODE=goal EPISODES="${EPISODES:-60}" EVAL_OUTDIR=$OUT \
  bash eval_v17.sh "$CK" Town05 _probe
T="$OUT/town05_checkpoint_150000_probe_goal.json.qtrace.jsonl"
[ -s "${T%.qtrace.jsonl}.probe.npz" ] || { say "FAILED: no probe npz"; exit 1; }
pkill -9 -f 'carla-port=2000' 2>/dev/null
cd .. && python3 scripts/probe_fault.py "clean_fix1/$T"
say "PROBE DONE"
