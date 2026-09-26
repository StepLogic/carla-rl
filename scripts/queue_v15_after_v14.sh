#!/usr/bin/env bash
# v15 = v14's recipe + GOAL-MODE LANE TERMINAL, launched when v14 finishes.
# User 2026-09-26: "make lane invasion terminal and bill remaining steps in goal
# mode", then "queue that for v15".
#
# RECIPE: v14's launch verbatim (v13's command + AGENT_LATENT_DIMS latent+goal
# 512:256 + ENV_RGB_CHANNEL_FIX=1) plus GATED_GOAL_LANE_TERMINAL=1 -- a fatal
# goal-mode crossing (junction-forgiving, off-road excluded) ends the episode
# with the exact bill -step_cost*(1+remaining), mask 0; commit 948a89d.
#
# LAUNCHER IS train_v15.sh, NOT train_v10.sh: train_v10.sh exports
# GATED_GOAL_LANE_TERMINAL=0 unconditionally, which would have silently discarded
# the override. It could not be edited in place because it IS v14's running
# wrapper (bash reads by byte offset; a mid-run edit killed a retry loop before).
#
# SEQUENCED, not concurrent: two Epic runs do not fit this 12 GB card (measured
# today -- an eval server beside v14 took free VRAM to 1.5 GB).
#
# PROVES THE CONFIG FROM THE LIVE RUN. After launch it waits for the first
# episode and requires, from v15's OWN log and snapshot: the lane-terminal
# banner, GATED_GOAL_LANE_TERMINAL=1 in the .env, the latent stack and the RGB
# fix. Any miss kills v15 -- a run with the wrong reward is worse than no run.
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl || exit 1
V14_WRAPPER_PID="${V14_WRAPPER_PID:-3963809}"
RUN=carla_gated_her_v15_laneterm_vc1
LOG=logs/${RUN}.log
SNAP=results/run_env/${RUN}.env
say(){ echo "=== [v15queue $(date '+%m-%d %H:%M:%S')] $* ==="; }

say "waiting for v14 wrapper PID $V14_WRAPPER_PID"
while kill -0 "$V14_WRAPPER_PID" 2>/dev/null; do sleep 60; done
say "v14 wrapper exited; its last lines:"
tail -3 logs/train_v14_lat512_wrapper.log
ls checkpoints/carla_gated_her_v14_lat512_rgb_vc1/checkpoint_450000.pkl >/dev/null 2>&1 \
  && say "v14 reached 450000" \
  || say "WARNING: v14 did NOT reach 450000; launching v15 anyway (independent run)"

# PREFLIGHT from the live machine, not from assumptions.
sleep 30
n_tr=$(pgrep -fc 'train_carla_tq[c]' || true)
if [ "${n_tr:-0}" -gt 0 ]; then say "ABORT: a trainer is still running ($n_tr)"; exit 1; fi
if [ -d "checkpoints/$RUN" ]; then say "ABORT: checkpoints/$RUN exists; refusing to clobber"; exit 1; fi
say "preflight ok: no trainer; GPU $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) free"

cd clean_fix1 || exit 1
env RUN_NAME=$RUN TARGET_ENTROPY_UNIT=1 GOAL_SPEED_BONUS=0.75 INIT_FROM= \
    ACTOR_FREEZE_STEPS=0 MAX_STEPS=450000 TQC_VALUE_CEILING=10.0 \
    AGENT_LATENT_DIMS="latent+goal:512:256" ENV_RGB_CHANNEL_FIX=1 \
    GATED_GOAL_LANE_TERMINAL=1 \
    nohup bash ./train_v15.sh > logs/train_v15_laneterm_wrapper.log 2>&1 &
W=$!
cd .. || exit 1
say "v15 launched, wrapper PID $W"

# Wait for the first episode line (or a crash), up to 15 min.
for _ in $(seq 1 90); do
  grep -aqE "^\[Ep +[0-9]|Traceback" "$LOG" 2>/dev/null && break
  kill -0 "$W" 2>/dev/null || break
  sleep 10
done

fail=""
grep -aq "^\[Gated\] goal lane invasion TERMINAL" "$LOG" || fail="$fail no-lane-terminal-banner"
grep -q "^GATED_GOAL_LANE_TERMINAL=1$" "$SNAP" 2>/dev/null || fail="$fail snapshot-not-1"
grep -aq "^\[AgentLatent\] per-key projection widths {'latent': (512, 256), 'goal': (512, 256)}" "$LOG" || fail="$fail no-latent-stack"
grep -aq "^\[RGB\] ENV_RGB_CHANNEL_FIX=1" "$LOG" || fail="$fail no-rgb-fix"
grep -aq "\[Env\] image_size=120" "$LOG" || fail="$fail image-size-not-120"
grep -aq "^\[Ep +[0-9]" "$LOG" || fail="$fail no-episodes"

if [ -n "$fail" ]; then
  say "CONFIG CHECK FAILED:$fail -- killing v15's whole tree"
  pkill -9 -P "$W" 2>/dev/null; kill -9 "$W" 2>/dev/null
  for p in $(pgrep -f 'train_carla_tq[c]'); do kill -9 "$p"; done
  exit 1
fi
say "v15 CONFIRMED from its live log + snapshot: lane terminal ON, latent 512:256, RGB fix, 120 px"
grep -aE "^\[Gated\] goal lane invasion TERMINAL|^\[AgentLatent\]|^\[RGB\]|\[Env\] image_size|^\[TargetEntropy\]" "$LOG" | cut -c1-160
grep -E "GATED_GOAL_LANE_TERMINAL|AGENT_LATENT_DIMS|ENV_RGB_CHANNEL_FIX|TARGET_ENTROPY_UNIT" "$SNAP"
grep -a "^\[Ep" "$LOG" | tail -1 | cut -c1-160
