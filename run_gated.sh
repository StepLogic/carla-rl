#!/usr/bin/env bash
# THE GATED ARM: one policy serving both the goal-reaching and the goal-free
# task, switched by a bit drawn once per episode.
#
# This is the THIRD LEG OF A THREE-WAY ABLATION, against:
#   run_lanefollow.sh         goal-only   144-d obs   checkpoints/carla_lanefollow
#   run_lanefollow_nogoal.sh  goal-free   132-d obs   checkpoints/carla_lanefollow_nogoal
# Everything except the gate is deliberately copied from run_lanefollow.sh, so
# that a difference in the result is attributable to the gate and not to a
# setting that drifted between scripts.
#
# THIS IS A FROM-SCRATCH RUN, NOT A FINE-TUNE. The gated observation is 145-d
# (goal 12, latent 128, vector 4, mode 1) and matches neither existing arm, so
# policy_io.load_agent_state refuses both -- correctly. That is the main reason
# the three-way comparison costs a full training budget.
#
# DO NOT EXPECT THE GATED ARM TO WIN. Perception is a frozen autoencoder outside
# the policy, so the only shared trainable parameters are small heads, and the
# collision rates of both arms were already at the floor. The value here is the
# comparison, not a performance claim.
set -uo pipefail
cd "$(dirname "$0")"

# ---- one CARLA job at a time -------------------------------------------------
# Held for the script's lifetime. Non-blocking on purpose: a run that WAITED on
# the lock would sit holding its VRAM reservation while another job ran, which
# is the deadlock this convention exists to avoid. Refuse and exit instead.
LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING TO START: another CARLA job holds $LOCK." >&2
  ps -eo pid,etime,args | grep -E "[t]rain_carla_tqc|[r]un_" >&2
  exit 1
fi
echo "[Lock] acquired $LOCK (pid $$)"

# ---- wait for the shared card ------------------------------------------------
# One 12.3 GB card, three sessions. Our floor is ~5.1 GB (the CARLA server takes
# ~2.8 GB BEFORE jax starts, jax adds ~2.3 GB) and another user's Habitat jobs
# take ~4.8 GB per stage. Poll for free memory; NEVER gate on a process name --
# /proc/PID/comm is capped at 15 chars, so the 23-char CARLA binary name is
# truncated and an exact-name match can never fire.
# 9500, NOT 6000, AND THE OLD FIGURE WAS MEASURED WRONG.
#
# The 6000 threshold every script in this repo uses came from an estimate of
# "CARLA ~2.8 GB before jax, plus ~2.3 GB jax". MEASURED on this run:
#
#     CarlaUE4-Linux-Shipping   5448 MiB     (not 2800)
#     our process total          9221 MiB    (CARLA + jax together)
#
# So the real requirement is about 9.2 GB, not 5.1 GB, and a 6000 MiB gate
# passes on a card that cannot actually hold the run. That is exactly what
# happened: the gate saw 6066 MiB free, started, and jax then died with
# RESOURCE_EXHAUSTED trying to allocate 16 MB for a cudnn convolution.
#
# CONSEQUENCE WORTH STATING PLAINLY: at 9.2 GB for us and ~4.8 GB for the
# Habitat jobs on the same 12.3 GB card, the two CANNOT run together. This is a
# scheduling constraint, not a tuning problem. Waiting is correct; shrinking
# CARLA's quality level to fit is NOT a free fix, because the frozen
# autoencoder was trained on frames rendered at this quality and changing it
# shifts the input distribution out from under it.
NEED_MB="${NEED_MB:-9500}"
MAX_WAIT_H="${MAX_WAIT_H:-6}"
POLL_S="${POLL_S:-60}"
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${free:-0}" -ge "$NEED_MB" ]; then
    echo "[GPU] ${free} MiB free -- starting"; break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "[GPU] GAVE UP after ${MAX_WAIT_H}h; only ${free} MiB free." >&2; exit 1
  fi
  echo "[GPU] ${free} MiB free, need ${NEED_MB}; waiting ${POLL_S}s"
  sleep "$POLL_S"
done

export DISPLAY=:1
export CARLA_PORT="${CARLA_PORT:-2000}" CARLA_PORTS="${CARLA_PORTS:-2000}"

# ---- the gate ----------------------------------------------------------------
export GATED=1
export OBS_WITH_MODE=1            # the 1-d mode key, its own projection
export OBS_WITH_GOAL=1            # PRESENT and zeroed in explore mode; one
                                  # actor cannot take two input shapes
export EXPLORE_PROB=0.5           # strictly between 0 and 1, or one critic
                                  # never trains
export GOAL_BATCH_FRAC=0.5        # batch composition FIXED on sample, so the
                                  # env-step skew cannot starve either critic
export GATED_EXPLORE_MAX_STEPS=1000  # the goal-free arm's cap (user 2026-09-18: 500 -> 1000)
# ---- the goal arm's reward: SORB minimum time --------------------------------
# -1 per step until arrival, 0 on arrival, as in the Habitat peer. The goal
# critic then IS the cost-to-go: Q = -(steps to goal). That reading is what a
# topological map builder needs and no shaped reward provides.
# ---- GOAL MODE MATCHES THE TIME-TO-GOAL ARM ---------------------------------
# Ported 2026-09-10. gated_reward.goal_reward_sorb is a pure-function MIRROR of
# time_to_goal_reward.TimeToGoalReward, and the mirror had drifted: these two
# were added there and not here, so goal mode would have trained on the older
# reward while everything claimed the arms matched.
#
# Set in the PARENT, not in run_gated_her.sh, so both arms share one goal reward
# and the hindsight ablation still differs by exactly one flag.
#
# COLLISION_TERMINAL: a crash ends the episode and is billed every remaining
# step, so it costs exactly what running out of time costs. The bill is what
# stops the terminal becoming an escape.
# GUARDED so a caller can override (v7 needs these settable). Unset by every existing caller, so the default below
# is exactly the previous behaviour.
export COLLISION_TERMINAL="${COLLISION_TERMINAL:-1}"
# ROUTE_POTENTIAL / POTENTIAL_K: potential-based shaping on ALONG-ROAD progress,
# K per metre. Policy-invariant (Ng et al. 1999), telescopes to a bounded
# constant so it cannot be farmed. Road distance rather than euclidean because a
# straight-line potential points through buildings -- on the CORL Town02 set one
# pair is 39.8 m apart and 621.6 m by road.
#
# K=1.0 against STEP_COST=1.0 keeps K/c well under the 3.53 ceiling, so -V stays
# POSITIVE and usable as an SGM edge weight without subtracting the potential;
# it is a linear rescale of true steps, so path ranking is exact. The graph's
# CUTOFF must be recalibrated for that scale instead.
export ROUTE_POTENTIAL=1
export POTENTIAL_K=1.0

export GATED_GOAL_REWARD=sorb
export STEP_COST=1.0
export GATED_GOAL_MAX_STEPS=750  # goal return therefore lies in [-750, 0],
                                 # ported to match TTG_MAX_STEPS
export WEDGE_TRUNCATE="${WEDGE_TRUNCATE:-1}"  # a wedge ends the episode BILLED EXACTLY for
                                  # the time it would have burned, so ending
                                  # early is worth precisely nothing

# CROSSING A LANE MARKING IS A NO-OP: no penalty, no termination, in either
# mode. Goal mode under sorb already ignored lane state, so this only changes
# the explore half, where a crossing previously cost 5.0 and ended the episode.
#
# OFF-ROAD IS STILL FATAL and that is deliberate. The explore progress term pays
# the same for driving in circles as for driving down the road, so with the lane
# term gone, off-road is the only remaining reason to stay on it.
#
# COMPARABILITY: carla_lanefollow_nogoal was trained WITH the lane penalty. Its
# explore half and this one now differ by more than the gate, so either that
# baseline is retrained with LANE_INVASION_NOOP=1 or the writeup states that the
# explore arms are not directly comparable.
# LANE_INVASION_NOOP IS AN EXPLORE-MODE FLAG, AND ONLY THAT. Reverted to 1
# after briefly setting it to 0 while porting the goal-mode setup: it does not
# touch goal mode at all.
#
# GOAL MODE GETS ITS LANE RULE FROM LANE_WEDGE BELOW, in the env config -- an
# illegal crossing wedges the vehicle and goal_reward_sorb bills the remaining
# clock, priced purely in TIME so V keeps meaning steps-to-goal. There is no
# lane term in the goal reward at all.
#
# LANE_DEPARTURE_PENALTY (5.0) is applied at exactly one site,
# gated_reward.py:257, inside explore_reward(). Setting this to 0 therefore
# changed the EXPLORE half -- from "a crossing costs nothing and ends nothing"
# to "-5.0 and terminal" -- which was no part of porting the goal reward.
# CROSSING A LANE IS FATAL IN BOTH MODES, priced differently in each because
# the two returns mean different things.
#
#   explore : LANE_INVASION_NOOP=0 -> terminal, and explore_reward charges
#             LANE_DEPARTURE_PENALTY (5.0). A fine is harmless here: the explore
#             return is a discounted COVERAGE RATE, not a distance, so nothing
#             downstream reads it as one.
#   goal    : GATED_GOAL_LANE_TERMINAL=1 -> terminal and billed the remaining
#             clock, in TIME, with no fine. A penalty would make V mean "steps
#             to goal PLUS a fine" and corrupt every graph built from it.
#
# Both use the same junction-forgiving filter (lane_departure_is_fatal), so the
# two halves cannot disagree about what counts as a crossing.
# RESTORED TO 1 (user decision 2026-09-11, "exploration worked earlier in this
# repo"). It did: every explore half that covered ground -- gated_her's
# ~300-step episodes, the lanefollow_nogoal traces that ring Town02 -- trained
# with LANE_INVASION_NOOP=1. Under 0 (v4) the explore half learned to cross a
# painted line at step ~37 because termination capped its loss; under the
# NONE-exempt/junction-forgiving variant (first v5 attempt, warm-started from
# v4) it stayed in that trap, still dying on painted road lines at 40 steps.
# With NOOP=1 a crossing costs nothing and ends nothing in EXPLORE MODE ONLY;
# off-road still ends it. Goal mode is untouched: GATED_GOAL_LANE_TERMINAL=1.
# EXPLORE-MODE LANE RULE (user decisions 2026-09-11, in order): no-op rejected
# ("not a good idea"); then "lane crossing should end episode". So:
#   EXPLORE_LANE_MODE=terminal  a painted crossing on open road ENDS the
#                               explore episode and is charged
#                               LANE_DEPARTURE_PENALTY; off-road ends it too;
#                               NONE (unpainted) exempt; junction turns forgiven.
# KNOWN RISK, on record: this is the structure under which v4's explore half
# learned to cross a line at step ~37, because ending the episode CAPS the
# loss at ~-10 while driving badly costs more. Whether v5 escapes that with
# the NONE/junction exemptions and the v4 warm start is what its explore
# episode length will show.
export LANE_INVASION_NOOP=0
export EXPLORE_LANE_MODE="${EXPLORE_LANE_MODE:-terminal}"
# PROGRESS WITHOUT THE ROAD (user decision 2026-09-11, "remove projection to
# road"): the explore progress term is displacement along the CAR'S OWN
# HEADING, not onto the lane direction -- no map anywhere in the explore
# reward. Consequence, on record: driving in the opposite lane no longer
# earns negative progress; lane discipline now rests on the terminal crossing
# and off-road rules alone.
export LANE_PROGRESS_MODE="${LANE_PROGRESS_MODE:-heading}"
# SCALE THE PROGRESS TERM BY 10 (user decision 2026-09-11): ~2.2 per step at
# cruising speed, ~+1100 over the 500-step cap, so a terminal crossing at
# step 37 forfeits ~1000 -- the exit is no longer cheap. The explore critic is
# unclipped (gated_tqc_learner: the goal window never touches it), so returns
# of that size are representable.
export LANE_PROGRESS_SCALE="${LANE_PROGRESS_SCALE:-10}"
# LANE_WEDGE IS DELIBERATELY NOT SET HERE, and that was a real mistake earlier.
#
# It lives in the env's get_done_status, which runs for EVERY episode and knows
# nothing about the mode -- so enabling it froze the vehicle on an illegal
# crossing during EXPLORE episodes too, where a crossing is meant to cost
# nothing and end nothing. That silently destroys explore data while goal mode
# looks correct.
#
# The goal-mode lane terminal is done in the gated wrapper instead, where the
# mode is known. Behaviourally identical for goal mode, since the episode ends
# on the same step either way and it is the BILL that does the work.
export GATED_GOAL_LANE_TERMINAL="${GATED_GOAL_LANE_TERMINAL:-1}"
# OFF-ROAD NO LONGER ENDS AN EPISODE. Removed at the user's instruction after
# it was measured killing the control: the scripted PID oracle failed 43% of its
# episodes on off_road and scored 0.000 success, against 0.980 for the same
# oracle in the time-to-goal arm. The probe is get_waypoint(project_to_road
# =False) is None, which the config itself records as least reliable inside
# junctions -- and JUNCTION_GOAL_FRAC=0.7 routes most episodes through one.
#
# THE COST IS REAL AND IS NOT GUARDED ANYWHERE ELSE. off_road is the only signal
# that catches leaving the road where nothing is painted, and CARLA junction
# interiors have no markings at all -- not one crossing has ever been tagged
# @junc in any run here. So a grass shortcut now costs only time, and under a
# minimum-time objective it is REWARDED whenever it is shorter. The rate still
# appears in [Why] as off_road, off_road@junc and off_road@road, so if the
# policy starts cutting corners it will be visible rather than silent.
export WEDGE_ON_OFF_ROAD=0

# TWO CRITICS, and under SORB this IS the Habitat rationale. The discounts
# genuinely differ and the returns differ in SIGN, so one critic could not serve
# both even at equal scale.
#
# DISCOUNT=1.0 IS REQUIRED, NOT A PREFERENCE. At 0.99 the horizon is
# 1/(1-gamma) = 100 steps, so Q saturates near -100 and cannot separate a
# 400-step goal from a 1000-step one -- measured 0.18 apart in this repo. The
# entire point of a minimum-time reward is that Q reads as time.
#
# It is SAFE here only because every episode terminates or truncates, so the
# undiscounted return is bounded by construction, and the goal critic's target
# is clipped to that window. The clip is applied PER CRITIC in the learner --
# never through TQC_TARGET_MIN/MAX, which would clip the explore critic's
# POSITIVE targets to zero.
# CALLER-OVERRIDABLE. These were unconditional, so a caller exporting
# DISCOUNT=0.99 before invoking this script was silently overwritten back to
# 1.0 -- the run then trained with the undiscounted goal critic while its
# launcher said otherwise. Caught 2026-09-11 on the v6m launch, from the
# [Gated] banner reading discount=1.0. gated_config.check_gated_run still
# refuses a discounted sorb reward unless GATED_ALLOW_DISCOUNTED_SORB=1, so
# honouring the caller here cannot make the expensive mistake silently.
export DISCOUNT="${DISCOUNT:-1.0}"
export EXPLORE_DISCOUNT="${EXPLORE_DISCOUNT:-0.99}"
# The explore critic starts uninformed. Damping its actor gradient for the
# first updates stops the actor following a critic that knows nothing.
export EXPLORE_ACTOR_WARMUP="${EXPLORE_ACTOR_WARMUP:-5000}"

# ---- the task, copied from run_lanefollow.sh ---------------------------------
export LANE_FOLLOW=1              # LaneFollowReward sits BELOW GatedReward for
                                  # info["is_success"] and the goal-arm
                                  # termination rule; its reward is discarded
export DELTA_MODE=positive
export INCREMENTAL_ACTIONS=0
export MAX_STEER=0.5 MAX_THROTTLE=0.4   # ported: 0.6 settled ~22 km/h
export STEER_STEP=0.01 THROTTLE_STEP=0.01
# v8 gated runs ship the 9-d state block instead of the vector (OBS_WITH_VECTOR=0); default unchanged.
export OBS_WITH_VECTOR="${OBS_WITH_VECTOR:-1}"   # REQUIRED: BOTH gated rewards are speed-gated
# OVERRIDABLE so run_gated_her.sh can flip exactly this one flag and nothing
# else. The two arms must differ by hindsight alone or the comparison measures
# the difference between two scripts rather than the effect of relabelling.
export EPISODE_HER="${EPISODE_HER:-0}"
export CHAIN_MAX_GOALS=1
export EXPERT_PROB=0.4           # ported: scripted PID on 40% of episodes
# GOAL-MODE EPISODES ONLY (decided 2026-09-11). In explore mode the oracle
# drives to a goal explore does not have and then parks for the rest of the
# cap -- rows the explore critic should not see. Until 2026-09-11 the oracle
# coin shared its RNG stream with the mode coin and the oracle ONLY drove
# explore episodes; see tests/test_expert_coin_is_not_the_mode_coin.py.
export EXPERT_MODES="${EXPERT_MODES:-goal}"
# ---- THE ORACLE MUST BE ABLE TO DRIVE THE TASK -------------------------------
# It is the solvability floor: if a GlobalRoutePlanner route followed by a tuned
# PID cannot complete an episode, no policy will, and a low success rate says
# nothing about learning. Measured before these three: 36% collisions, a lane
# crossing on 36% of episodes, and 0.000 success.
#
# 1. A REACHABLE SPEED TARGET. ExpertOracle defaults to 25 km/h; this vehicle
#    settles at 22.4 km/h at throttle 0.6 and lower at our 0.4 cap, so the
#    longitudinal PID held full throttle permanently and never lifted for a
#    corner. Falls back to TARGET_SPEED so the two cannot disagree.
#
#    MEASURED DIRECTLY on vehicle.micro.microlino, throttle held from rest:
#        0.4 -> 11.8 km/h     0.6 -> 24.0 km/h     1.0 -> 83.7 km/h
#    (the 0.6 figure corroborates the 22.4 km/h already recorded in the config).
#    So at our MAX_THROTTLE=0.4 the ceiling is 11.8 and the 25 default was more
#    than DOUBLE it. 9.0 leaves ~24% headroom, which is what lets the PID
#    regulate instead of sitting on the stop -- a saturated controller cannot
#    slow for a corner no matter what the curvature term asks for.
export EXPERT_TARGET_SPEED=9.0
# 4. CARLA'S OWN BasicAgent, not our wrapper. BENCHMARKED on this exact task
#    clipped to our action space: 25/25 reached the route end, 0 collisions,
#    0 off-road, 0 lane crossings (bench_basic_agent.py). Our ExpertOracle was
#    colliding on 36% with the same planner and PID -- the difference is
#    BasicAgent's LocalPlanner. It hands over to the PID for the last few
#    metres, because the planner's route end snaps up to 2 m short of the goal
#    and only 2/25 reached the env's 2 m radius without that.
export EXPERT_KIND=basic
# 2. PURE-PURSUIT AIM POINT instead of a fixed 3 nodes (6 m at 2 m sampling) at
#    every speed. A fixed aim point on a tight turn steers at a node past the
#    apex, which cuts the corner and crosses the inside line.
export EXPERT_LOOKAHEAD_GAIN=1.2
export EXPERT_LOOKAHEAD_MIN_M=3.0 EXPERT_LOOKAHEAD_MAX_M=12.0
# 3. SLOW DOWN FOR CURVATURE, from the heading change over the next 6 route
#    nodes -- so the car sees the corner before it is in it, rather than
#    arriving carrying whatever the preceding straight allowed.
export EXPERT_CURVE_SLOWDOWN=2.0 EXPERT_CURVE_NODES=6 EXPERT_CURVE_MIN_FRAC=0.35
#
# All three read only the vehicle's own speed and the route the oracle already
# planned. No new information source, and nothing here reaches the policy's
# observation -- the expert's rows enter the buffer as ordinary off-policy
# transitions.
# 1 -- ANY LANE INVASION TERMINATES, at the user's instruction. No marking
# filter, no junction exemption.
#
# The filter it bypasses was largely dead anyway: only NONE and Broken have ever
# been observed in this map, so ILLEGAL_MARKS (Solid/SolidSolid/SolidBroken/
# Curb/Grass) never matched and a kerb-hop reported as NONE, which the filter
# forgave.
#
# THE RECORDED RISK: making NONE fatal at 100 m goals took the scripted PID
# oracle to 0.000 over its last 100 episodes. At 20-50 m the exposure is
# smaller, but lane_any/orc is the number that says whether the task is still
# solvable, and it should be read before trusting anything else from this run.
export LANE_STRICT=1
# EXPLORE MODE IS LESS STRICT (user decision 2026-09-11): off-road is fatal;
# a crossing is fatal unless every marking crossed is unpainted (NONE). Goal
# mode keeps any-crossing-fatal via GATED_GOAL_LANE_TERMINAL. v4 trained with
# every explore crossing fatal and its explore half learned to end episodes at
# step ~37 (median 1.7 m at evaluation); this is the v5 change.
export EXPLORE_EXEMPT_MARKS="${EXPLORE_EXEMPT_MARKS:-NONE}"
export EXPLORE_OFF_ROAD_FATAL="${EXPLORE_OFF_ROAD_FATAL:-1}"
export LANE_EXPLORE=0             # the explorer is a SEPARATE arm; gating
                                  # against it would vary four things at once

# ---- environment -------------------------------------------------------------
# TOWN01, train and eval. Changed from Town02 2026-09-10 at the user's
# instruction. Note this is only meaningful now that make_env actually
# honours it -- before today's fix the carla block's hardcoded "Town01"
# won regardless, so every gated checkpoint on disk was trained in Town01
# while its logs said Town02. This run makes the label true rather than
# changing where the car drives.
# GUARDED: this line silently discarded a caller's ENV_TOWNS, so a launcher asking for two towns trained on one
# while its .env snapshot -- taken in the PARENT, before this file runs -- recorded both. Provenance that asserts a
# run which never happened is worse than no provenance.
export ENV_TOWNS="${ENV_TOWNS:-Town01}" EVAL_TOWNS="${EVAL_TOWNS:-Town01}"
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
export PERCEPTION="${PERCEPTION:-vae}"             # 128-d, shared with BOTH lane arms. MobileNet
                                  # is 384-d and would change the obs width.
export VAE_CKPT=checkpoints/lane_vae_120/best.pkl
export MAX_EPISODE_STEPS=2000     # the GOAL arm's cap; explore truncates at 500
export MAX_DISTANCE_M=200
export TARGET_SPEED=20.0
export SHORT_HORIZON=1
export GOAL_DIST_MIN=20.0 GOAL_DIST_START=50.0 GOAL_DIST_TARGET=50.0
export JUNCTION_GOAL_FRAC="${JUNCTION_GOAL_FRAC:-0.7}"    # 70% of episodes ROUTED THROUGH a junction
# ...BUT THE GOAL ITSELF IS NEVER INSIDE ONE. These two are not in tension:
# JUNCTION_GOAL_FRAC places the ORIGIN one goal-distance BEFORE a junction so
# the route passes through it, while this rejects a goal that LANDS in one.
#
# Asking the car to stop inside an intersection is asking for something a route
# follower will not do, and it is scored by a 2 m radius test in the one place
# where lane polygons meet with gaps -- so off_road fires and the lane sensor
# fires on the turn itself, both while the car is driving correctly.
export GOAL_NOT_IN_JUNCTION=1

# ---- run ---------------------------------------------------------------------
export POLICY_FOLDER="${POLICY_FOLDER:-carla_gated}"
export MAX_STEPS="${MAX_STEPS:-350000}"
export RESUME="${RESUME:-1}"
# 10.0, ported: kappa is the Huber knee and must track the reward scale.
# At STEP_COST=1.0 the goal return spans [-1000, 0]; at kappa=1 essentially
# the whole TD error distribution sits in Huber's LINEAR region.
export TQC_KAPPA=10.0
                                  # would divide the critic gradient by 60
export TEMP_LR=3e-4
export EVAL_VIDEO=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
# A free-memory check is necessary but NOT sufficient: jax can still die with
# RESOURCE_EXHAUSTED while instantiating a CUDA command buffer, which no byte
# count predicts. This removes that allocation.
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_command_buffer="
# v8 (2026-09-21): torch VC-1 + jax must share the ~1 GB left beside two
# windowed CARLA servers; the cudnn autotune probe dies there
# (RESOURCE_EXHAUSTED on a 16-18 MB workspace). The picker flag falls back to
# a default algorithm instead of probing -- selection only, no numerics.
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_strict_conv_algorithm_picker=false"

CARLA_SH=/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4.sh
# pgrep -x CANNOT match: /proc/PID/comm is capped at 15 chars, so the 23-char
# 'CarlaUE4-Linux-Shipping' truncates and an exact-name match never fires. That
# is why servers survived cleanup and sat holding ~2.8 GB after their job left.
# The bracket also stops a concurrent kill_servers matching THIS script's argv.
kill_servers() {
  for _ in 1 2 3 4 5; do
    mapfile -t pids < <(pgrep -f '[C]arlaUE4-Linux-Shipping')
    [ ${#pids[@]} -eq 0 ] && return 0
    for p in "${pids[@]}"; do kill -9 "$p" 2>/dev/null; done
    sleep 3
  done
  pgrep -f '[C]arlaUE4-Linux-Shipping' >/dev/null && \
    echo "WARNING: a CARLA server survived 5 kill attempts" >&2
}
cleanup() { kill_servers; echo "=== gated run done ==="; }
trap cleanup EXIT

mkdir -p logs
for ATTEMPT in 1 2 3 4 5; do
  echo "=== gated attempt $ATTEMPT ==="
  kill_servers
  # The subshell drops fd 200 FIRST: an orphaned grep inheriting the lock once
  # held it for 8.5 hours.
  # RUN_GATED_START_SERVER=0 leaves server startup to train_carla_tqc.py (ensure_carla), which starts ONE server PER
  # PORT in CARLA_PORTS and honours CARLA_QUALITY and CARLA_OFFSCREEN. This block only ever started port 2000, at a
  # HARDCODED Epic/windowed, and python then ATTACHED to it -- so a caller's CARLA_QUALITY=Low silently never reached
  # the port-2000 env, while the run's .env recorded Low. In a two-server run the Epic windowed server (6846 MiB)
  # plus a Low one starved the card and sensors timed out. Default 1 keeps every existing caller unchanged, and the
  # quality now follows CARLA_QUALITY (default Epic, as before).
  if [ "${RUN_GATED_START_SERVER:-1}" = "1" ]; then
    ( exec 200>&-; "$CARLA_SH" -carla-port=2000 -quality-level="${CARLA_QUALITY:-Epic}" \
          -windowed -ResX=640 -ResY=360 2>&1 \
        | grep -v --line-buffered "Invalid session: no stream available" \
        > logs/carla_gated.log ) &
    sleep 40
  fi
  if python -u train_carla_tqc.py; then
    echo "=== gated finished cleanly ==="; break
  fi
  echo "=== attempt $ATTEMPT died; restarting server ==="
  sleep 10
done
