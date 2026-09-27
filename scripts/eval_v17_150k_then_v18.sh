#!/usr/bin/env bash
# Evaluate v17's checkpoint_150000 -- its last clean point: at 152658 the trainer
# was OOM-killed by a test run and resumed with a fresh replay buffer -- then
# start v18. User 2026-09-27: "stop v17 - evaluate the 150k checkpoint", then
# "lets see if 1) it transfers to other towns 2) the goal mode is strong with no
# lane crossings and offroad 3) explore mode can explore".
#
#   1+2  goal mode, 60 fixed episodes (seed 0), 5-200 m band from the snapshot:
#        Town05 (trained), Town01 and Town02 (never seen), Town03 (the held-out town)
#   3    explore mode, 30 episodes: Town05 and Town03
# eval_arm now records per-episode rule accounting (vae_nav.eval_faults): off-road,
# collision and every crossing classified junction/edge/lane_change/illegal, and a
# CLEAN arrival = arrived with no off-road, no collision, no illegal crossing.
#
# ALONE ON THE GPU: v17 is stopped and v18's queue is held until the sweep ends --
# an Epic eval beside an Epic training run does not fit the 12 GB card. One town
# per fresh server (the env attaches to whatever answers on the port). The kill is
# port-scoped. Resumable by artefact.
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl/clean_fix1 || exit 1
OUTDIR=results/v17_eval_150k
CK=checkpoints/carla_gated_her_v17_sorb_fixed_vc1/checkpoint_150000.pkl
say(){ echo "=== [v17eval $(date '+%m-%d %H:%M:%S')] $* ==="; }
kill_my_carla(){ for _ in 1 2 3 4 5; do pkill -9 -f 'carla-port=2000' 2>/dev/null; sleep 2;
  pgrep -f 'carla-port=2000' >/dev/null || return 0; done; return 1; }
stage(){  # <town> <mode> <episodes>
  local town="$1" mode="$2" eps="$3"
  local out="$OUTDIR/$(echo "$town" | tr 'A-Z' 'a-z')_checkpoint_150000_${mode}.json"
  if [ -s "$out" ]; then say "SKIP (exists): $out"; return 0; fi
  say "START $mode $town ($eps episodes)"
  kill_my_carla || { say "FAILED: port-2000 CARLA survived the kill"; return 1; }
  sleep 10
  EVAL_MODE="$mode" EPISODES="$eps" EVAL_OUTDIR="$OUTDIR" bash eval_v17.sh "$CK" "$town"
  [ -s "$out" ] || { say "FAILED: no artefact $out"; return 1; }
  say "DONE $mode $town -> $out"
}
mkdir -p "$OUTDIR"
rc=0
for t in Town05 Town01 Town02 Town03; do stage "$t" goal 60 || rc=1; done
for t in Town05 Town03; do stage "$t" explore 30 || rc=1; done
kill_my_carla
say "EVAL SWEEP $([ $rc = 0 ] && echo DONE || echo 'FINISHED WITH FAILURES')"

# ---- then v18, via its own queue (v17's wrapper is gone, so it launches at once) ----
cd .. || exit 1
say "starting v18 via scripts/queue_v18_after_v17.sh"
exec bash scripts/queue_v18_after_v17.sh
