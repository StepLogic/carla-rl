#!/usr/bin/env bash
# OFFLINE EVAL SWEEP for v13 (carla_gated_her_v13_entunit_vc1), which finished
# rc=0 at 450k on 2026-09-25 12:20. Analysis only -- no reward or baseline
# change, so it runs locally under the "analysis/offline builds local" rule.
#
# FOUR STAGES, one eval per stage, 60 fixed episodes each, seed 0:
#   1. Town05 full band (the snapshot's 5-200 m)      -> the headline number,
#      directly comparable to v10-vel75's Town05 row.
#   2. Town05 5-20 m     } the band split that tests the constant-hazard fit:
#   3. Town05 50-200 m   } SR = exp(-lambda*d) predicted long-band 0.28 before
#      the entropy fix; if the crossing reduction is real the long band should
#      move up and the short band should barely change.
#   4. Town05 full band at checkpoint 175000 -- the mid-run goal-arm peak, to
#      separate "the final policy is best" from "the run plateaued early".
#
# PINNED TO clean_fix1: v13 TRAINED there, where gated_reward.py is the HEAD
# counting regime (fault caps TRUNCATE, mask=1). The worktree root carries
# another session's UNCOMMITTED bill regime (caps TERMINATE with a
# remaining-clock bill), and a first launch from the root printed exactly
# that banner -- the eval would have scored the policy under a reward it
# never trained on. Every script here does sys.path.insert(0, W/"src"), so
# the CWD is what pins the source tree; PYTHONPATH does not.
#
# Each stage: port-scoped kill (a peer's server on another port lives), fresh
# server, then eval_v13.sh, which self-starts CARLA and replays every
# experiment-defining flag from v13's own snapshot.
#
# RESUMABLE BY ARTEFACT, not by marker: a stage is done only when its JSON
# exists and is non-empty, because the supervisor kills background jobs
# spuriously and wrappers can exit 0 after a failed attempt.
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl/clean_fix1 || exit 1

CKDIR=checkpoints/carla_gated_her_v13_entunit_vc1
say(){ echo "=== [v13sweep $(date '+%m-%d %H:%M:%S')] $* ==="; }
kill_my_carla(){ for _ in 1 2 3 4 5; do pkill -9 -f 'carla-port=2000' 2>/dev/null; sleep 2;
  pgrep -f 'carla-port=2000' >/dev/null || return 0; done; return 1; }

# stage: <ckpt-step> <suffix> <goal-min|-> <goal-max|->
stage(){
  local step="$1" suffix="$2" gmin="$3" gmax="$4"
  local ck="$CKDIR/checkpoint_${step}.pkl"
  local out="results/v13_eval/town05_checkpoint_${step}${suffix}.json"
  [ -f "$ck" ] || { say "FAILED: no $ck"; return 1; }
  if [ -s "$out" ]; then say "SKIP (artefact exists): $out"; return 0; fi
  say "START step=$step band=${gmin}-${gmax} -> $out"
  kill_my_carla || { say "FAILED: port-2000 CARLA survived the kill"; return 1; }
  sleep 10
  if [ "$gmin" = "-" ]; then
    SUFFIX="$suffix" bash eval_v13.sh "$ck" Town05 "$suffix"
  else
    GOAL_MIN="$gmin" GOAL_MAX="$gmax" bash eval_v13.sh "$ck" Town05 "$suffix"
  fi
  local rc=$?
  [ -s "$out" ] || { say "FAILED: rc=$rc and no artefact at $out"; return 1; }
  say "DONE step=$step band=${gmin}-${gmax} -> $out"
}

mkdir -p results/v13_eval
rc_all=0
stage 450000 ""          -   -    || rc_all=1
stage 450000 "_b5_20"    5   20   || rc_all=1
stage 450000 "_b50_200"  50  200  || rc_all=1
stage 175000 ""          -   -    || rc_all=1
kill_my_carla
if [ "$rc_all" = 0 ]; then say "V13 SWEEP DONE"; else say "V13 SWEEP FAILED (see stages above)"; fi
exit "$rc_all"
