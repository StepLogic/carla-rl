#!/usr/bin/env bash
# GOAL-FREE lane following. The goal is removed from BOTH halves.
#
# WHAT THE POLICY SEES:   {latent (128,), vector (4,)}  = 132 dims
#   latent  frozen lane/obstacle VAE encoding of the 120x120 camera frame
#   vector  [steer/max_steer, throttle/max_throttle, speed/target_speed, 1.0]
# No goal offset, no goal bearing, no action history, no map, no memory.
#
# WHAT IT IS PAID FOR: staying in lane and making speed-gated progress, where
# progress is UNPROJECTED |displacement| (LANE_PROGRESS_MODE=raw, set below).
# LaneExploreReward with novelty disabled -- base 4e-3 step cost, raw
# steer_change, speed-gated progress, -3 collision, -5 lane departure. The -3
# overspeed term in that module is DEAD: it is gated on info["done_speed"], and
# the env hardcodes that False ("Speed-based termination (currently disabled)").
# No delta_distance, no arrival bonus, no checkpoint bonus, no wrong-junction
# penalty.
#
# WHY IT IS GENUINELY GOAL-FREE. The upstream reward projects displacement onto
# core.destination.get_forward_vector() while in a junction -- the GOAL's
# heading -- which would leave it secretly goal-conditioned on exactly the
# episodes that cross junctions. That vector is not read at all here.
#
# CANNOT RESUME FROM carla_lanefollow. The observation drops from 144 to 132
# dims, so the actor's pytree differs and policy_io.load_agent_state refuses.
# Hence its own POLICY_FOLDER; the goal-conditioned checkpoints stay untouched.
set -uo pipefail
cd "$(dirname "$0")"

LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING TO START: another CARLA job holds $LOCK." >&2
  exit 1
fi
echo "[Lock] acquired $LOCK (pid $$)"

export DISPLAY=:1
export CARLA_PORT=2000 CARLA_PORTS=2000

# ---- the goal-free task --------------------------------------------------------
export LANE_FOLLOW_NOGOAL=1
export OBS_WITH_GOAL=0            # drop the 12-d goal block from the observation
# Every goal-dependent reward OFF. Belt-and-braces against a stale export from a
# shell: VAEGoalWrapper raises if any of these is on while OBS_WITH_GOAL=0, so a
# mismatch is a loud failure rather than a critic quietly fitting noise.
export LANE_FOLLOW=0 TIME_TO_GOAL=0 LANE_EXPLORE=0
export K_DIST=0.0 K_SPEED=0.0 K_LIVING=0.0
export EPISODE_HER=0              # no goal, so no hindsight relabelling
export CHAIN_MAX_GOALS=1
export EXPERT_PROB=0

# ---- control -------------------------------------------------------------------
export OBS_WITH_VECTOR=1          # REQUIRED: the lane reward is speed-gated, so
                                  # the policy must be able to observe its speed
export INCREMENTAL_ACTIONS=0      # absolute actions, as in the goal-conditioned run
export MAX_STEER=0.5 MAX_THROTTLE=0.6
export TARGET_SPEED=20.0

# ---- lane rules ----------------------------------------------------------------
# LaneExploreReward re-derives `done` itself with the junction exemption
# (illegal markings fatal anywhere, Broken/NONE fatal only OUTSIDE a junction,
# off_road always fatal), so the car can turn through a junction without the
# episode ending. Without that, every turn reads as a catastrophe.
export LANE_STRICT=0
export LANE_WEDGE=0

# ---- progress term: NO PROJECTION ----------------------------------------------
# raw = |displacement|. The reward no longer reads wp_forward, so it contains no
# map information at all.
#
# WHAT THIS COSTS, stated plainly: |displacement| is maximised by moving fast in
# ANY direction, so a car driving tight circles earns exactly what a car driving
# down the road earns. The lane-departure penalty is now the only thing that
# distinguishes them, and inside a junction that penalty is forgiven by design.
# If circling appears in the eval videos, LANE_PROGRESS_MODE=heading keeps the
# reward map-free while paying only FORWARD motion (projection onto the car's own
# yaw, which is proprioceptive rather than privileged).
export LANE_PROGRESS_MODE=raw

# ---- environment ---------------------------------------------------------------
export ENV_TOWNS=Town02 EVAL_TOWNS=Town02
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
export VAE_CKPT=checkpoints/lane_vae_120/best.pkl
export MAX_EPISODE_STEPS=500
export MAX_DISTANCE_M=2000        # no goal to arrive at, so let it keep driving
export SHORT_HORIZON=1

# ---- run -----------------------------------------------------------------------
export POLICY_FOLDER=carla_lanefollow_nogoal
export MAX_STEPS="${MAX_STEPS:-350000}"
export RESUME="${RESUME:-1}"
# gamma < 1 and kappa 1: the return here is a discounted lane-progress rate of
# order 10, not a +600 arrival, so the derived K_GOAL/10 kappa would divide the
# critic gradient by 60 -- the bug that printed critic_loss as 0.00.
export DISCOUNT=0.99
export TQC_KAPPA=1.0
export TEMP_LR=3e-4
export EVAL_VIDEO=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

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
}
cleanup() { kill_servers; echo "=== goal-free lane-follow run done ==="; }
trap cleanup EXIT

for ATTEMPT in 1 2 3 4 5; do
  echo "=== nogoal attempt $ATTEMPT ==="
  kill_servers
  ( exec 200>&-; "$CARLA_SH" -carla-port=2000 -quality-level=Epic \
        -windowed -ResX=640 -ResY=360 2>&1 \
      | grep -v --line-buffered "Invalid session: no stream available" \
      > logs/carla_lanefollow_nogoal.log ) &
  sleep 40
  if python -u train_carla_tqc.py; then
    echo "=== nogoal finished cleanly ==="; break
  fi
  echo "=== attempt $ATTEMPT died; restarting server ==="
  sleep 10
done
