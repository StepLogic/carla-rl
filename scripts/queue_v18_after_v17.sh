#!/usr/bin/env bash
# v18 = v17's recipe (v13's minimum-time reward on the corrected pipeline)
#       + GOAL_LANE_GRADED=1   (legal crossings free/cheap, illegal full fine)
#       + ENV_WEATHER_RANDOM=1 (14 daytime/sunset presets, per episode)
#       + ENV_TOWNS=Town05,Town01, one server rotating every 50 episodes.
# User 2026-09-27: suggestions for transfer -> "proceed", "add town01 to training
# towns", "proceed on auto". Commits 51d2f96 (code) and 6610b10 (tests).
#
# Launches from clean_fix1/train_v18.sh: train_v17.sh pinned ENV_TOWNS=Town05 and
# TOWN_ROTATE_EPISODES=0 with UNCONDITIONAL exports, which would have made v18 a
# single-town run while its launch line asked for two.
#
# VERIFIES FROM THE LIVE RUN, in two phases. Every regex below is ERE (grep -E) and
# was tested against real or format-exact log lines before this was armed -- v15's
# queue killed a healthy run with an untested BRE pattern.
#   phase 1 (start-up): config banners + snapshot; any miss KILLS v18.
#   phase 2 (first rotation to Town01): the hindsight planner must be rebuilt on
#     Town01's map. A miss means hindsight progress would be measured on Town05's
#     roads for every Town01 episode, so it also KILLS v18.
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl || exit 1
V17_WRAPPER_PID="${V17_WRAPPER_PID:-3646450}"
RUN=carla_gated_her_v18_graded_weather_2town_vc1
LOG=logs/${RUN}.log
SNAP=results/run_env/${RUN}.env
say(){ echo "=== [v18queue $(date '+%m-%d %H:%M:%S')] $* ==="; }
kill_v18(){
  say "KILLING v18: $*"
  for p in $(pgrep -f 'bash \./train_v1[8]\.sh'); do kill -9 "$p"; done
  for p in $(pgrep -f 'clean_fix1/run_gate[d]\.sh'); do kill -9 "$p"; done
  for p in $(pgrep -f 'python -u train_carla_tq[c]'); do kill -9 "$p"; done
  for p in $(pgrep -f 'carla-port=200[0]'); do kill -9 "$p"; done
  exit 1
}

say "waiting for v17 wrapper PID $V17_WRAPPER_PID"
while kill -0 "$V17_WRAPPER_PID" 2>/dev/null; do sleep 60; done
say "v17 wrapper exited; its last lines:"
tail -3 logs/train_v17_sorb_fixed_wrapper.log
sleep 30
n_tr=$(pgrep -fc 'python -u train_carla_tq[c]' || true)
[ "${n_tr:-0}" -gt 0 ] && { say "ABORT: a trainer is still running ($n_tr)"; exit 1; }
[ -d "checkpoints/$RUN" ] && { say "ABORT: checkpoints/$RUN exists; refusing to clobber"; exit 1; }
for p in $(pgrep -f 'carla-port=200[0]'); do kill -9 "$p"; done     # v17 leftovers only
say "preflight ok: GPU $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) free"

cd clean_fix1 || exit 1
env RUN_NAME=$RUN TARGET_ENTROPY_UNIT=1 GOAL_SPEED_BONUS=0.75 INIT_FROM= \
    ACTOR_FREEZE_STEPS=0 MAX_STEPS=450000 TQC_VALUE_CEILING=10.0 \
    AGENT_LATENT_DIMS="latent+goal:512:256" ENV_RGB_CHANNEL_FIX=1 \
    GATED_GOAL_REWARD=sorb ROUTE_REMAINING_MODE=project GATED_GOAL_LANE_TERMINAL=0 \
    GOAL_LANE_GRADED=1 GOAL_LANE_PENALTY_BROKEN=2.0 ENV_WEATHER_RANDOM=1 \
    ENV_TOWNS=Town05,Town01 TOWN_ROTATE_EPISODES=50 \
    setsid nohup bash ./train_v18.sh > logs/train_v18_wrapper.log 2>&1 < /dev/null &
cd .. || exit 1
say "v18 launched"

# ---- phase 1: start-up ---------------------------------------------------------
for _ in $(seq 1 90); do
  grep -aqE '^\[Ep +[0-9]|Traceback' "$LOG" 2>/dev/null && break
  sleep 10
done
fail=""
chk(){ grep -aqE "$2" "$1" 2>/dev/null || fail="$fail $3"; }
chk "$LOG" '^\[Gated\] goal lane fines GRADED'                      graded-banner
chk "$LOG" 'COUNTING regime'                                        counting-regime
chk "$LOG" '^\[Weather\] randomised per episode over 14 presets'    weather-banner
chk "$LOG" "^\[Rotate\] ENV_TOWNS is set; rotation restricted to \['Town05', 'Town01'\]" rotation-towns
chk "$LOG" "^\[Rotate\] every 50 episodes per env"                  rotation-every-50
chk "$LOG" "^\[AgentLatent\] per-key projection widths \{'latent': \(512, 256\), 'goal': \(512, 256\)\}" latent-stack
chk "$LOG" '^\[RGB\] ENV_RGB_CHANNEL_FIX=1'                         rgb-fix
chk "$LOG" '\[Env\] image_size=120'                                 image-120
chk "$LOG" 'route planner for hindsight progress on Carla/Maps/Town05' her-planner-town05
chk "$LOG" '^\[Ep +[0-9]'                                           episodes
chk "$SNAP" '^GOAL_LANE_GRADED=1$'                                  snap-graded
chk "$SNAP" '^ENV_WEATHER_RANDOM=1$'                                snap-weather
chk "$SNAP" '^ENV_TOWNS=Town05,Town01$'                             snap-towns
chk "$SNAP" '^TOWN_ROTATE_EPISODES=50$'                             snap-rotate
chk "$SNAP" '^GATED_GOAL_REWARD=sorb$'                              snap-reward
grep -aqE 'Traceback' "$LOG" && fail="$fail traceback"
n_weather=$(grep -aoE '^\[Weather\] episode [0-9]+: [A-Za-z]+' "$LOG" | awk '{print $4}' | sort -u | wc -l)
[ "$n_weather" -ge 3 ] || fail="$fail weather-not-varying($n_weather)"
[ -n "$fail" ] && kill_v18 "phase-1 checks failed:$fail"
say "PHASE 1 OK: graded fines, weather varying ($n_weather distinct presets in the first draws), Town05+Town01 rotation every 50, latent stack, RGB fix, 120 px, HER planner on Town05"
grep -aE '^\[Weather\] episode [0-9]+:' "$LOG" | head -5
grep -aE '^\[Ep +[0-9]' "$LOG" | tail -1 | cut -c1-170

# ---- phase 2: the first rotation to Town01 -----------------------------------------
say "waiting for the first rotation to Town01 (~50 episodes)"
for _ in $(seq 1 360); do                   # up to 6 h
  grep -aqE '^\[Rotate\] env0: -> Town01 after' "$LOG" && break
  kill -0 "$(pgrep -f 'bash \./train_v1[8]\.sh' | head -1)" 2>/dev/null || { say "v18 wrapper gone before any rotation"; exit 1; }
  sleep 60
done
grep -aqE '^\[Rotate\] env0: -> Town01 after' "$LOG" || { say "no rotation within 6 h"; exit 1; }
rot_line=$(grep -anE '^\[Rotate\] env0: -> Town01 after' "$LOG" | head -1 | cut -d: -f1)
say "rotated to Town01 at log line $rot_line; waiting for Town01 episodes to flush"
for _ in $(seq 1 60); do
  [ "$(awk -v L="$rot_line" 'NR>L' "$LOG" | grep -acE '^\[Ep +[0-9]')" -ge 1 ] && break
  sleep 30
done
if awk -v L="$rot_line" 'NR>L' "$LOG" | grep -aqE 'route planner for hindsight progress on Carla/Maps/Town01'; then
  say "PHASE 2 OK: HER rebuilt its planner on Town01 after the rotation"
  awk -v L="$rot_line" 'NR>L' "$LOG" | grep -aE '^\[Rotate\]|route planner for hindsight|^\[Preflight\]|town=Town01' | head -5 | cut -c1-170
else
  awk -v L="$rot_line" 'NR>L' "$LOG" | grep -aE '^\[Ep +[0-9]' | head -1 | grep -q . \
    && kill_v18 "phase-2: Town01 episodes flushed but HER never planned on Town01" \
    || say "phase-2 INCONCLUSIVE: no Town01 episode line yet; not killing"
fi
