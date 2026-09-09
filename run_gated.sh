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
export CARLA_PORT=2000 CARLA_PORTS=2000

# ---- the gate ----------------------------------------------------------------
export GATED=1
export OBS_WITH_MODE=1            # the 1-d mode key, its own projection
export OBS_WITH_GOAL=1            # PRESENT and zeroed in explore mode; one
                                  # actor cannot take two input shapes
export EXPLORE_PROB=0.5           # strictly between 0 and 1, or one critic
                                  # never trains
export GOAL_BATCH_FRAC=0.5        # batch composition FIXED on sample, so the
                                  # env-step skew cannot starve either critic
export GATED_EXPLORE_MAX_STEPS=500   # the goal-free arm's cap
# ---- the goal arm's reward: SORB minimum time --------------------------------
# -1 per step until arrival, 0 on arrival, as in the Habitat peer. The goal
# critic then IS the cost-to-go: Q = -(steps to goal). That reading is what a
# topological map builder needs and no shaped reward provides.
export GATED_GOAL_REWARD=sorb
export STEP_COST=1.0
export GATED_GOAL_MAX_STEPS=1000  # goal return therefore lies in [-1000, 0]
export WEDGE_TRUNCATE=1           # a wedge ends the episode BILLED EXACTLY for
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
export LANE_INVASION_NOOP=1

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
export DISCOUNT=1.0
export EXPLORE_DISCOUNT=0.99
# The explore critic starts uninformed. Damping its actor gradient for the
# first updates stops the actor following a critic that knows nothing.
export EXPLORE_ACTOR_WARMUP="${EXPLORE_ACTOR_WARMUP:-5000}"

# ---- the task, copied from run_lanefollow.sh ---------------------------------
export LANE_FOLLOW=1              # LaneFollowReward sits BELOW GatedReward for
                                  # info["is_success"] and the goal-arm
                                  # termination rule; its reward is discarded
export DELTA_MODE=positive
export INCREMENTAL_ACTIONS=0
export MAX_STEER=0.5 MAX_THROTTLE=0.6
export STEER_STEP=0.01 THROTTLE_STEP=0.01
export OBS_WITH_VECTOR=1          # REQUIRED: BOTH gated rewards are speed-gated
# OVERRIDABLE so run_gated_her.sh can flip exactly this one flag and nothing
# else. The two arms must differ by hindsight alone or the comparison measures
# the difference between two scripts rather than the effect of relabelling.
export EPISODE_HER="${EPISODE_HER:-0}"
export CHAIN_MAX_GOALS=1
export EXPERT_PROB=0
export LANE_STRICT=1
export LANE_EXPLORE=0             # the explorer is a SEPARATE arm; gating
                                  # against it would vary four things at once

# ---- environment -------------------------------------------------------------
export ENV_TOWNS=Town02 EVAL_TOWNS=Town02
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
export PERCEPTION=vae             # 128-d, shared with BOTH lane arms. MobileNet
                                  # is 384-d and would change the obs width.
export VAE_CKPT=checkpoints/lane_vae_120/best.pkl
export MAX_EPISODE_STEPS=2000     # the GOAL arm's cap; explore truncates at 500
export MAX_DISTANCE_M=200
export TARGET_SPEED=20.0
export SHORT_HORIZON=1
export GOAL_DIST_MIN=5.0 GOAL_DIST_START=20.0 GOAL_DIST_TARGET=20.0

# ---- run ---------------------------------------------------------------------
export POLICY_FOLDER="${POLICY_FOLDER:-carla_gated}"
export MAX_STEPS="${MAX_STEPS:-350000}"
export RESUME="${RESUME:-1}"
export TQC_KAPPA=1.0              # returns of order 10; the derived K_GOAL/10
                                  # would divide the critic gradient by 60
export TEMP_LR=3e-4
export EVAL_VIDEO=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
# A free-memory check is necessary but NOT sufficient: jax can still die with
# RESOURCE_EXHAUSTED while instantiating a CUDA command buffer, which no byte
# count predicts. This removes that allocation.
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

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
  ( exec 200>&-; "$CARLA_SH" -carla-port=2000 -quality-level=Epic \
        -windowed -ResX=640 -ResY=360 2>&1 \
      | grep -v --line-buffered "Invalid session: no stream available" \
      > logs/carla_gated.log ) &
  sleep 40
  if python -u train_carla_tqc.py; then
    echo "=== gated finished cleanly ==="; break
  fi
  echo "=== attempt $ATTEMPT died; restarting server ==="
  sleep 10
done
