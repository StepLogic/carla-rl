#!/usr/bin/env bash
# GOAL-FREE LANE-FOLLOWING EXPLORER.
#
# The exploration arm. Drives without a goal, rewarded for staying in lane and
# for seeing views it has not seen. Its job is to produce a trace to build the
# graph over, and -- the research question -- a value function whose Q-gap is
# FLAT at junctions and PEAKED on straight road, because junction ambiguity is
# only topology when no goal decides the branch.
#
# Reward: the goal-free half of the upstream nisheeth-golakiya/carla-rl reward
# (base 4e-3, raw steer_change, speed-gated lane projection, -3 collision,
# -5 lane departure, -3 overspeed), plus a bounded image-novelty bonus.
set -uo pipefail
cd "$(dirname "$0")"

LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING TO START: another CARLA job holds $LOCK." >&2
  ps -eo pid,etime,args | grep -E "[t]rain_carla|[r]un_|[w]atch_policy" >&2
  exit 1
fi
echo "[Lock] acquired $LOCK (pid $$)"

export DISPLAY=:1
export CARLA_PORT=2000 CARLA_PORTS=2000

# ---- the arm -------------------------------------------------------------------
export LANE_EXPLORE=1
# Every goal-arm reward must be OFF. make_env returns early on LANE_EXPLORE, so
# these are belt-and-braces against a stale export leaking in from a shell.
export TIME_TO_GOAL=0 LANE_FOLLOW=0
export K_DIST=0.0 K_SPEED=0.0 K_LIVING=0.0

# ---- perception: MobileNetV3, not the VAE --------------------------------------
# Frozen mobilenet_v3_small(ImageNet).features[:4], pooled 4x4 -> 384-d.
# The raw flatten is 5400-d, which is 43 GB of replay at 1M capacity.
export PERCEPTION=mobilenet
export MOBILENET_POOL=4
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120

# ---- novelty -------------------------------------------------------------------
# NOVELTY_PERSIST=1 IS LOAD-BEARING, NOT A TUNING KNOB. Turn diversity at
# junctions emerges through the critic: the branch leading to unseen road holds
# more future novelty, so Q(s,turn) > Q(s,straight) once straight is covered.
# With a per-episode reset every episode starts blank, every branch looks
# equally unseen, and that pressure does not exist at all.
export NOVELTY_PERSIST=1
export NOVELTY_ADMIT_SIM=0.9
export NOVELTY_BUFFER_MAX=2048
# Ceiling must stay under the ~0.283/step lane-progress term or a car driving
# tight circles through fresh headings out-earns one driving forward. The peer
# paid for this rule: "any seen bonus must stay below one entered cell per step".
export K_NOVELTY=0.2

# ---- episode -------------------------------------------------------------------
export LANE_EXPLORE_MAX_STEPS=500
export MAX_EPISODE_STEPS=500
# gamma < 1: the explore return is a discounted coverage-rate, NOT a cost-to-go,
# so the goal arm's gamma=1 does not apply and would not be bounded here.
export DISCOUNT=0.99
export TQC_KAPPA=1.0

# ---- lane rules ----------------------------------------------------------------
# Junction-interior crossings forgiven, or every turn would end the episode and
# junctions would read as catastrophic rather than ambiguous.
export LANE_STRICT=0
export LANE_WEDGE=0
export MAX_DISTANCE_M=2000

# ---- environment ---------------------------------------------------------------
export ENV_TOWNS=Town02 EVAL_TOWNS=Town02
export INCREMENTAL_ACTIONS=0
export MAX_STEER=0.5 MAX_THROTTLE=0.6
export TARGET_SPEED=20.0
export SHORT_HORIZON=1
# No goal, so no goal sampling, no curriculum, no expert, no hindsight.
export EXPERT_PROB=0
export EPISODE_HER=0
export CHAIN_MAX_GOALS=1

# ---- dead reckoning ------------------------------------------------------------
# UNFITTED starting points for vehicle.micro.microlino. Run
# dead_reckoning.calibrate() on the first recorded trace before trusting any
# drift figure -- an unfitted gain manufactures drift that looks like physics.
export DR_WHEELBASE_M=1.5
export DR_MAX_STEER_RAD=1.22

# ---- run -----------------------------------------------------------------------
export POLICY_FOLDER="${POLICY_FOLDER:-carla_lane_explorer}"
export MAX_STEPS="${MAX_STEPS:-500000}"
export RESUME="${RESUME:-1}"
export TEMP_LR=3e-4
export EVAL_VIDEO=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

CARLA_SH=/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4.sh
kill_servers() {
  for _ in 1 2 3 4 5; do
    mapfile -t p < <(pgrep -x CarlaUE4-Linux-Shipping); [ ${#p[@]} -eq 0 ] && return 0
    for q in "${p[@]}"; do kill -9 "$q" 2>/dev/null; done; sleep 3
  done
}
cleanup() { kill_servers; echo "=== lane explorer run done ==="; }
trap cleanup EXIT

for ATTEMPT in 1 2 3 4 5; do
  echo "=== lane explorer attempt $ATTEMPT ==="
  kill_servers
  # fd 200 dropped FIRST: an orphaned grep inheriting this lock once held it
  # for 8.5 hours after the job it belonged to had exited.
  ( exec 200>&-; "$CARLA_SH" -carla-port=2000 -quality-level=Epic \
      -windowed -ResX=640 -ResY=360 2>&1 \
    | grep -v --line-buffered "Invalid session: no stream available" \
    > logs/carla_lane_explorer.log ) &
  sleep 40
  if python -u train_carla_tqc.py; then
    echo "=== lane explorer finished cleanly ==="; break
  fi
  echo "=== attempt $ATTEMPT died; restarting server ==="
  sleep 10
done
