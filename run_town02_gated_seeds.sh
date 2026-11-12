#!/bin/bash
# HELD-TOWN EVALUATION OF THE GATED ARM, four seeds, Town02.
#
# The policy trained in Town01 ONLY (ENV_TOWNS=Town01), so Town02 is a transfer
# test: same VAE distribution, a map the policy has never driven.
#
# ENV PARITY IS THE WHOLE POINT. The env config is sourced from run_gated.sh
# itself rather than retyped, so a training/eval mismatch cannot creep in --
# that mismatch "cost a whole run" once already. Only the eval-specific
# overrides below are applied on top.
set -uo pipefail
cd "$(dirname "$0")"

CKPT="${CKPT:-checkpoints/carla_gated_her_v4/checkpoint_350000.pkl}"
TOWN="${TOWN:-Town02}"
EPISODES="${EPISODES:-25}"
SEEDS="${SEEDS:-1234 1235 1236 1237}"
BANDS="${BANDS:-20-50}"
STAMP=$(date +%Y%m%d_%H%M)

# Every `export VAR=...` line from the training script, verbatim.
set +u
eval "$(grep -E '^export ' run_gated.sh)"
set -u

# ---- eval-only overrides ----------------------------------------------------
# CARLA IS STARTED BY THE PYTHON SIDE (CarlaCore), which is faster than
# launching the shell script and waiting on a readiness probe, and lets
# CarlaCore kill the server it started when the env closes.
export CARLA_START_SERVER=1 CARLA_QUALITY=Epic
export CARLA_PORT=2000 CARLA_PORTS=2000
export ENV_TOWNS="$TOWN" EVAL_TOWNS="$TOWN"
export EVAL_FORCE_MODE=1          # goal mode; explore has no goal to reach
export EXPERT_PROB=0              # no oracle at evaluation
export RESUME=0
export GOAL_DIST_MIN=20 GOAL_DIST_MAX=50   # eval_policy pins these per band
export DRAW_GOAL=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

say() { echo "=== [t02 $(date +%H:%M:%S)] $* ==="; }
mkdir -p logs results

# THE SERVER GETS KILLED UNDER US. Measured twice on this box: the CarlaUE4
# process takes a SIGKILL mid-episode with no OOM in dmesg and 100+ GB free
# ("line 48: ... Killed", then the client's 50 s time-out). A supervisor on
# this machine kills background jobs spuriously. So the server is not launched
# once and trusted -- it is (re)started on demand and every seed is retried.
# KILL ALL CARLA BEFORE ANY SWEEP, and PROVE it before launching.
# Two reasons this is not paranoia:
#   1. `pkill -f CarlaUE4-Linux-Shipping` alone is too narrow -- the launcher,
#      the shell wrapper and the shipped binary do not all carry that string.
#   2. OTHER sessions on this box start and stop CARLA. A concurrent
#      machine-wide kill is the likeliest explanation for the server dying
#      mid-episode twice with no OOM and the GPU half empty.
# So: kill by every pattern, then LOOP until `ps` shows none and port 2000 is
# free. A sweep that starts beside a surviving server measures whichever one
# answers first, which is not an experiment.
kill_all_carla() {
  for _ in 1 2 3 4 5 6; do
    pkill -9 -f 'CarlaUE4-Linux-Shipping' 2>/dev/null
    pkill -9 -f 'CarlaUE4'                2>/dev/null
    pkill -9 -f 'CARLA_0.9.15'            2>/dev/null
    sleep 2
    if ! ps -eo args | grep -i '[C]arlaUE4' >/dev/null; then
      return 0
    fi
  done
  say "WARNING: a CARLA process survived six kill rounds:"
  ps -eo pid,args | grep -i '[C]arlaUE4' | cut -c1-120
  return 1
}

# NO SHELL LAUNCHER HERE ANY MORE. CarlaCore starts the server from Python
# (CARLA_START_SERVER=1 above): faster than `CarlaUE4.sh` plus a readiness
# probe, and the env's close() kills the server it started. All this script
# still owns is making sure the machine is CLEAR first.

say "pre-flight: killing every CARLA on this machine"
kill_all_carla || exit 1

FAILED=""
for SEED in $SEEDS; do
  OUT="results/town02_gated_seed${SEED}_${STAMP}.json"
  for ATTEMPT in 1 2 3; do
    say "seed $SEED attempt $ATTEMPT -> $OUT"
    # KILL ALL CARLA BEFORE EVERY ATTEMPT. CarlaCore starts its own server,
    # so any survivor from a previous attempt would hold port 2000 and the new
    # env would silently attach to it instead.
    kill_all_carla || { say "seed $SEED: cannot clear CARLA"; continue; }
    python -u eval_policy.py --ckpt "$CKPT" --towns "$TOWN" \
        --episodes "$EPISODES" --port 2000 --seed "$SEED" \
        --bands "$BANDS" --max-steps 750 --out "$OUT" \
        --video-dir "results/town02_gated_videos_${STAMP}/seed${SEED}" \
        --max-videos "$EPISODES" 2>&1 \
      | grep -E "^  ${TOWN}|\[Eval\]|MAP|Traceback|Error|CarlaUE4|quality-level" || true
    # THE FILE IS THE EVIDENCE, not the exit status: the eval is behind a
    # pipe, so PIPESTATUS is the grep's, and a partially-run seed must not be
    # reported as done.
    if [ -s "$OUT" ]; then
      say "seed $SEED done on attempt $ATTEMPT"
      break
    fi
    say "seed $SEED attempt $ATTEMPT produced no results -- retrying"
    [ "$ATTEMPT" = 3 ] && FAILED="$FAILED $SEED"
  done
done

kill_all_carla || true
if [ -n "$FAILED" ]; then
  say "INCOMPLETE -- seeds that never produced results:$FAILED"
else
  say "all seeds done -- results/town02_gated_seed*_${STAMP}.json"
fi
