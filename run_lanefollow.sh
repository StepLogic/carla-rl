#!/usr/bin/env bash
# Lane following with a goal: per-step distance reward, gated on speed,
# with ABSOLUTE actions -- steer and throttle written straight from the policy.
#
# Supervisor mirrors run_rl_120.sh, including the two failures that cost time:
#   * flock, held for the script's lifetime, so two trainers can never share one
#     CARLA server (that happened three times and silently corrupted metrics)
#   * the server pipeline wrapped in a subshell that drops fd 200 FIRST -- an
#     orphaned `grep` inheriting the lock once held it for 8.5 hours
#   * server log truncated per attempt and filtered, after an unfiltered append
#     reached 87 GB and filled the root filesystem
set -uo pipefail
cd "$(dirname "$0")"

LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING TO START: another CARLA job holds $LOCK." >&2
  ps -eo pid,etime,args | grep -E "[t]rain_carla_tqc|[r]un_rl|[r]un_lanefollow|[e]xplore_trace|[e]val_policy" >&2
  exit 1
fi
echo "[Lock] acquired $LOCK (pid $$)"

export DISPLAY=:1
export CARLA_PORT=2000 CARLA_PORTS=2000

# ---- the lane-following task -------------------------------------------------
export LANE_FOLLOW=1              # use LaneFollowReward, not SparseGoalReward
export DELTA_MODE=positive        # clip(delta_distance, 0, None)
# ABSOLUTE actions. The action IS the control, not a delta on it, so the
# action space becomes Box([-0.5, 0.0], [0.5, 0.6]) instead of [-1, 1]^2.
# The tradeoff is documented in train_env_config_v4_image_goal.py:353 and was
# MEASURED on the trained goal-conditioned agent: writing the control straight
# from the action each 0.05 s tick allowed p95 |acceleration| of 196 m/s^2,
# 79.5% of steps above the 4 m/s^2 "harsh" threshold, steering reversal on
# 32.6% of steps, and speeds to 123 km/h. Under incremental actions those are
# unreachable by construction; here smoothness has to be bought by the reward
# (k_steer_change) instead of being structural.
export INCREMENTAL_ACTIONS=0
export MAX_STEER=0.5 MAX_THROTTLE=0.6
# INERT under INCREMENTAL_ACTIONS=0 -- compute_action only reads these on the
# incremental branch. Left set so flipping the flag back needs one edit.
export STEER_STEP=0.01 THROTTLE_STEP=0.01
export OBS_WITH_VECTOR=1          # REQUIRED: the reward is speed-gated, so the
                                  # policy must be able to observe its speed
export EPISODE_HER=0              # no hindsight: the reward is dense now
export CHAIN_MAX_GOALS=1
export EXPERT_PROB=0
export LANE_STRICT=1

# ---- environment -------------------------------------------------------------
export ENV_TOWNS=Town02 EVAL_TOWNS=Town02
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
export VAE_CKPT=checkpoints/lane_vae_120/best.pkl
export MAX_EPISODE_STEPS=2000
export MAX_DISTANCE_M=200         # the upstream repo's 200 m episode target
# 20 km/h, not the config's 5.0. MEASURED: at the 0.6 throttle cap this vehicle
# settles at a 22.4 km/h median, so a 5 km/h target gated the distance reward
# off on 86.1% of steps. At 20 the gate discriminates -- the agent must hold
# just under it to keep being paid, which is what a speed-gated reward is for.
export TARGET_SPEED=20.0
export SHORT_HORIZON=1
export GOAL_DIST_MIN=5.0 GOAL_DIST_START=20.0 GOAL_DIST_TARGET=20.0

# ---- run ---------------------------------------------------------------------
export POLICY_FOLDER=carla_lanefollow
export MAX_STEPS="${MAX_STEPS:-350000}"   # the upstream repo's reported budget
export RESUME="${RESUME:-1}"
# KAPPA MUST TRACK THE REWARD SCALE. tqc_kappa() derives it as K_GOAL/10 = 60,
# which is right for the sparse +600 arrival reward and badly wrong here:
# quantiles.py:72 computes `huber / kappa`, so with the lane-follow reward
# (~0.1 per step, MEASURED TD errors of 0.038) every error sits in the quadratic
# branch and the critic gradient is divided by 60. That is why critic_loss
# printed as 0.00 at 22k steps. 1.0 is the standard TQC value and is correct for
# returns of order 10.
export TQC_KAPPA=1.0
export TEMP_LR=3e-4
export EVAL_VIDEO=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

CARLA_SH=/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4.sh
# pgrep -x CANNOT match here: /proc/PID/comm is capped at 15 chars, so
# 'CarlaUE4-Linux-Shipping' (23) is truncated to 'CarlaUE4-Linux-' and an
# exact-name match never fires. That is why servers survived this trap and
# sat holding ~2.8 GB of GPU after their job had exited. -f matches argv.
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
cleanup() { kill_servers; echo "=== lane-follow run done ==="; }
trap cleanup EXIT

for ATTEMPT in 1 2 3 4 5; do
  echo "=== lane-follow attempt $ATTEMPT ==="
  kill_servers
  ( exec 200>&-; "$CARLA_SH" -carla-port=2000 -quality-level=Epic \
        -windowed -ResX=640 -ResY=360 2>&1 \
      | grep -v --line-buffered "Invalid session: no stream available" \
      > logs/carla_lanefollow.log ) &
  sleep 40
  if python -u train_carla_tqc.py; then
    echo "=== lane-follow finished cleanly ==="; break
  fi
  echo "=== attempt $ATTEMPT died; restarting server ==="
  sleep 10
done
