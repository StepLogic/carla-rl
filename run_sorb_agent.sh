#!/usr/bin/env bash
# SORB-style agent: SHORT-HORIZON policy whose VALUE FUNCTION is the product.
#
#   r = -0.1              every step
#   r =  0   + terminate  on arrival
#            + truncate   at 300 steps
#
# V = -0.1 * steps-to-goal, bounded in [-30, 0] by construction. The graph
# supplies long-range navigation; the policy only has to be right within H=30 m.
#
# Design: docs/superpowers/specs/2026-09-01-sorb-agent-carla-design.md
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

# ---- the MDP -----------------------------------------------------------------
export TIME_TO_GOAL=1
export STEP_COST=0.1
# 300 steps is 2.8x the ~106 a 30 m goal needs at the MEASURED 0.283 m/step.
# It is also what bounds the undiscounted return: 6.7x tighter than the 2000
# under which q reached -280 against a -200 floor.
export TTG_MAX_STEPS=300
export MAX_EPISODE_STEPS=300
export DISCOUNT=1.0
export TQC_KAPPA=1.0
# The achievable return range, enforced. Every quantile of a distribution
# supported on [-30, 0] lies inside it, so a target outside is unrealisable and
# clipping adds information rather than bias.
export TQC_TARGET_MIN=-30
export TQC_TARGET_MAX=0

# Sparse-reward terms are dead under TIME_TO_GOAL; pinned so nothing carries.
export K_DIST=0.0 K_SPEED=0.0 K_LIVING=0.0

# ---- NO WEDGE ------------------------------------------------------------------
# An absorbing failure makes V mean "distance to goal OR death", and the graph
# reads V as travel time. Collisions and lane crossings cost only the time they
# lose, which is self-calibrating and needs no coefficient.
export LANE_WEDGE=0

# ---- goals: 2-30 m, FIXED --------------------------------------------------------
# V must be accurate everywhere the graph queries it, and graph edges between
# adjacent trace nodes are mostly SHORT. Training only on 20-30 m would leave
# most edges extrapolated.
#
# START == TARGET makes the curriculum inert: nothing can advance, the demote
# branch needs curr_max > START, and the stall branch needs curr_max < TARGET.
export GOAL_DIST_MIN=2.0 GOAL_DIST_START=30.0 GOAL_DIST_TARGET=30.0
export GOAL_DIST_STEP=1.0
export JUNCTION_GOAL_FRAC=0.6
export SHORT_HORIZON=1
export MAX_DISTANCE_M=200

# ---- actions: ABSOLUTE, for parity with the Habitat peer -------------------------
export INCREMENTAL_ACTIONS=0
export MAX_STEER=0.5 MAX_THROTTLE=0.6
export OBS_WITH_VECTOR=0

# ---- environment -----------------------------------------------------------------
export ENV_TOWNS=Town02 EVAL_TOWNS=Town02
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
export VAE_CKPT=checkpoints/lane_vae_120/best.pkl
export LANE_STRICT=0
export TARGET_SPEED=20.0
export EPISODE_HER=0
export CHAIN_MAX_GOALS=1
export EXPERT_PROB=0.2

# ---- run -------------------------------------------------------------------------
export POLICY_FOLDER="${POLICY_FOLDER:-carla_sorb_h30}"
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
cleanup() { kill_servers; echo "=== sorb agent run done ==="; }
trap cleanup EXIT

for ATTEMPT in 1 2 3 4 5; do
  echo "=== sorb agent attempt $ATTEMPT ==="
  kill_servers
  # fd 200 dropped FIRST: an orphaned grep inheriting this lock once held it
  # for 8.5 hours after the job it belonged to had exited.
  ( exec 200>&-; "$CARLA_SH" -carla-port=2000 -quality-level=Epic \
      -windowed -ResX=640 -ResY=360 2>&1 \
    | grep -v --line-buffered "Invalid session: no stream available" \
    > logs/carla_sorb_h30.log ) &
  sleep 40
  if python -u train_carla_tqc.py; then
    echo "=== sorb agent finished cleanly ==="; break
  fi
  echo "=== attempt $ATTEMPT died; restarting server ==="
  sleep 10
done
