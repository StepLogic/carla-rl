#!/usr/bin/env bash
# v13 RE-EVAL AT THE TRAINING IMAGE SIZE, beside the live v14 run.
# User 2026-09-26: "run v13 with changes locally".
#
# WHY IT IS RE-RUN. v13 TRAINED at ENV_IMAGE_SIZE=120 ([Env] image_size=120 in
# its log), but run_gated.sh sets that AFTER train_v10.sh snapshots, so every
# earlier offline eval replaying the snapshot ran at the env default of 84. The
# first sweep (results/v13_eval/) is kept untouched as evidence of that; this
# one writes to results/v13_eval_120/. Channel order stays BGR
# (ENV_RGB_CHANNEL_FIX=0): v13 was TRAINED on the swapped frames.
#
# SAME FOUR STAGES as scripts/eval_v13_sweep.sh, so every row pairs with an
# 84-px row: 450k full band, 450k 5-20 m, 450k 50-200 m, 175k full band.
#
# BESIDE A TRAINING RUN, SO FOUR GUARDS:
#   1. PORT 2030, never 2000. v14 holds 2000-2002, and this driver kills its
#      own port before every stage; on 2000 that kill would take v14's server.
#   2. JAX memory limits (set in eval_v13.sh): without them JAX preallocates
#      ~75% of the card.
#   3. The eval server runs OFFSCREEN (EVAL_CARLA_OFFSCREEN=1): a windowed
#      Epic eval server took 5.4 GB on the first try and v14's server shrank
#      3480 -> 2704 MiB alongside it.
#   4. A VRAM WATCHDOG that kills THIS EVAL -- its python and its port-2030
#      server, never anything on 2000 -- if free memory falls under
#      MIN_FREE_MB (2000: the eval yields LONG before
#      v14 is squeezed). Losing an eval stage costs minutes and resumes from its
#      artefact; losing v14's server costs a training attempt.
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl/clean_fix1 || exit 1

PORT=2030
OUTDIR=results/v13_eval_120
MIN_FREE_MB="${MIN_FREE_MB:-2000}"
CKDIR=checkpoints/carla_gated_her_v13_entunit_vc1
say(){ echo "=== [v13sweep120 $(date '+%m-%d %H:%M:%S')] $* ==="; }
kill_my_carla(){ for _ in 1 2 3 4 5; do pkill -9 -f "carla-port=${PORT}" 2>/dev/null; sleep 2;
  pgrep -f "carla-port=${PORT}" >/dev/null || return 0; done; return 1; }

watchdog(){
  while sleep 10; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${free:-99999}" -lt "$MIN_FREE_MB" ]; then
      say "WATCHDOG: ${free} MiB free < ${MIN_FREE_MB}; killing THIS eval (port ${PORT}) to protect v14"
      pkill -9 -f "eval_arm.py.*--port ${PORT}" 2>/dev/null
      pkill -9 -f "carla-port=${PORT}" 2>/dev/null
    fi
  done
}
watchdog & WD=$!
trap 'kill $WD 2>/dev/null' EXIT

stage(){
  local step="$1" suffix="$2" gmin="$3" gmax="$4"
  local ck="$CKDIR/checkpoint_${step}.pkl"
  local out="$OUTDIR/town05_checkpoint_${step}${suffix}.json"
  [ -f "$ck" ] || { say "FAILED: no $ck"; return 1; }
  if [ -s "$out" ]; then say "SKIP (artefact exists): $out"; return 0; fi
  say "START step=$step band=${gmin}..${gmax} -> $out"
  kill_my_carla || { say "FAILED: port-${PORT} CARLA survived the kill"; return 1; }
  sleep 10
  if [ "$gmin" = "-" ]; then
    EVAL_CARLA_OFFSCREEN=1 EVAL_PORT=$PORT EVAL_OUTDIR=$OUTDIR bash eval_v13.sh "$ck" Town05 "$suffix"
  else
    EVAL_CARLA_OFFSCREEN=1 EVAL_PORT=$PORT EVAL_OUTDIR=$OUTDIR GOAL_MIN="$gmin" GOAL_MAX="$gmax" \
      bash eval_v13.sh "$ck" Town05 "$suffix"
  fi
  local rc=$?
  [ -s "$out" ] || { say "FAILED: rc=$rc and no artefact at $out"; return 1; }
  say "DONE step=$step -> $out"
}

mkdir -p "$OUTDIR"
rc_all=0
stage 450000 ""          -   -    || rc_all=1
stage 450000 "_b5_20"    5   20   || rc_all=1
stage 450000 "_b50_200"  50  200  || rc_all=1
stage 175000 ""          -   -    || rc_all=1
kill_my_carla
if [ "$rc_all" = 0 ]; then say "V13 SWEEP (120 px) DONE"; else say "V13 SWEEP (120 px) FAILED"; fi
exit "$rc_all"
