#!/usr/bin/env bash
# FOUR-WAY ABLATION: goal-only, goal-free, gated, gated+hindsight.
#
# ONE CARLA SERVER, FOUR PROCESSES, ONE EPISODE SET. The arms have different
# observation widths (144 / 132 / 145 / 145) selected by environment variables
# that several modules read AT IMPORT, so each arm gets a fresh process. They
# share one server because starting four costs ~5.4 GB and four minutes each for
# nothing.
#
# THE EPISODE SET IS THE SAME FOR EVERY ARM. EVAL_SEED is passed to
# build_eval_episodes, which draws from an RNG never touched by training. Change
# it and you change the comparison for all four together, which is the only safe
# way to change it.
#
# WHAT IS DELIBERATELY IDENTICAL ACROSS ARMS: town, image size, the frozen
# autoencoder, the action space, the step cap. Everything that differs is listed
# per arm below and nowhere else, so a difference in the table is attributable.
#
# SUCCESS IS NOT REPORTED FOR THE GOAL-FREE ARM. It measures arrival at a goal
# that policy cannot see and is not paid for; the existing run reports 0.000 over
# 60 episodes for exactly that reason. A zero in a comparison table reads as
# "worse" rather than "not applicable", so eval_arm.py prints n/a instead.
set -uo pipefail
cd "$(dirname "$0")"

LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING TO START: another CARLA job holds $LOCK." >&2
  exit 1
fi
echo "[Lock] acquired $LOCK (pid $$)"

# Same measured requirement as training: ~9.2 GB. See run_gated.sh.
NEED_MB="${NEED_MB:-9500}"
MAX_WAIT_H="${MAX_WAIT_H:-6}"
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "${free:-0}" -ge "$NEED_MB" ] && { echo "[GPU] ${free} MiB free -- starting"; break; }
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[GPU] GAVE UP; ${free} MiB free." >&2; exit 1; }
  echo "[GPU] ${free} MiB free, need ${NEED_MB}; waiting"
  sleep 60
done

export DISPLAY=:1
export CARLA_PORT=2000 CARLA_PORTS=2000

# ---- shared, and identical for every arm -------------------------------------
export ENV_TOWNS=Town02 EVAL_TOWNS=Town02
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
export PERCEPTION=vae
export VAE_CKPT=checkpoints/lane_vae_120/best.pkl
export INCREMENTAL_ACTIONS=0 MAX_STEER=0.5 MAX_THROTTLE=0.6
export TARGET_SPEED=20.0 SHORT_HORIZON=1
export GOAL_DIST_MIN=5.0 GOAL_DIST_START=20.0 GOAL_DIST_TARGET=20.0
export MAX_EPISODE_STEPS=2000 MAX_DISTANCE_M=200
export EXPERT_PROB=0 EPISODE_HER=0 CHAIN_MAX_GOALS=1
export EVAL_VIDEO=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

EPISODES="${EPISODES:-60}"
EVAL_SEED="${EVAL_SEED:-0}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-1000}"
OUT_DIR="${OUT_DIR:-results/four_way_eval}"
mkdir -p "$OUT_DIR" logs

CARLA_SH=/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4.sh
kill_servers() {
  for _ in 1 2 3 4 5; do
    mapfile -t pids < <(pgrep -f '[C]arlaUE4-Linux-Shipping')
    [ ${#pids[@]} -eq 0 ] && return 0
    for p in "${pids[@]}"; do kill -9 "$p" 2>/dev/null; done
    sleep 3
  done
}
cleanup() { kill_servers; echo "=== four-way eval done ==="; }
trap cleanup EXIT

kill_servers
( exec 200>&-; "$CARLA_SH" -carla-port=2000 -quality-level=Epic \
      -windowed -ResX=640 -ResY=360 2>&1 \
    | grep -v --line-buffered "Invalid session: no stream available" \
    > logs/carla_four_way_eval.log ) &
sleep 40

# ---- the four arms -----------------------------------------------------------
# Each block lists ONLY what differs. Anything not set here is inherited from
# the shared block above, which is what makes the comparison attributable.

# DISTANCE BANDS. 5-20 m is the training range; everything beyond it is
# EXTRAPOLATION and is expected to be weak.
#
# THEY EXIST BECAUSE 5-20 m SEPARATES NOTHING. In-distribution success saturated
# at 1.000 for both gated arms within their first evaluation, so a table built on
# that band shows four numbers all approximately one.
#
# AND BECAUSE THEY TEST HINDSIGHT'S ACTUAL CLAIM. Relabelling supplies goal data
# at distances the pinned curriculum cannot produce. Evaluating only inside the
# curriculum could not show that benefit even if it were real.
BANDS="${BANDS:-5-20,20-40,40-70}"

run_arm() {
  local name="$1" ckpt="$2" goal_conditioned="$3"
  if [ ! -f "$ckpt" ]; then
    echo "[Arm $name] SKIPPED: no checkpoint at $ckpt" >&2
    return 0
  fi
  local band lo hi
  for band in ${BANDS//,/ }; do
    lo="${band%%-*}"; hi="${band##*-}"
    echo "=== arm: $name  band: ${lo}-${hi}m  ($ckpt) ==="
    python -u eval_arm.py --arm "$name" --ckpt "$ckpt" \
        --episodes "$EPISODES" --seed "$EVAL_SEED" \
        --max-steps "$EVAL_MAX_STEPS" \
        --goal-conditioned "$goal_conditioned" \
        --goal-min "$lo" --goal-max "$hi" --mode goal \
        --minimum-time "${5:-0}" \
        --out "$OUT_DIR/${name}__${lo}-${hi}m.json" \
      || echo "[Arm $name band ${lo}-${hi}m] FAILED" >&2
  done

  # THE EXPLORE HALF, for gated arms only. A gated policy is supposed to serve
  # BOTH modes, and measuring only the goal half can report an arm as better
  # while its explore behaviour has degraded -- which is exactly what the second
  # training run showed, its explore return falling from 62.5 to 40.1 while its
  # goal half improved. Reporting one without the other would be misleading.
  #
  # ONCE, NOT PER BAND: explore mode has no goal, so distance bands are
  # meaningless there.
  if [ "${4:-0}" = "1" ]; then
    echo "=== arm: $name  EXPLORE half  ($ckpt) ==="
    python -u eval_arm.py --arm "$name" --ckpt "$ckpt" \
        --episodes "$EPISODES" --seed "$EVAL_SEED" \
        --max-steps "$EVAL_MAX_STEPS" --mode explore \
        --minimum-time "${5:-0}" \
        --out "$OUT_DIR/${name}__explore.json" \
      || echo "[Arm $name explore] FAILED" >&2
  fi
}

# 1. GOAL-ONLY: the shaped lane-follow reward, 144-d.
( export GATED=0 OBS_WITH_MODE=0 OBS_WITH_GOAL=1 OBS_WITH_VECTOR=1 \
         LANE_FOLLOW=1 LANE_FOLLOW_NOGOAL=0 LANE_EXPLORE=0 DELTA_MODE=positive
  run_arm goal_only "${CKPT_GOAL:-checkpoints/carla_lanefollow/checkpoint_50000.pkl}" 1 )

# 2. GOAL-FREE: no goal block at all, 132-d. Success is not applicable.
( export GATED=0 OBS_WITH_MODE=0 OBS_WITH_GOAL=0 OBS_WITH_VECTOR=1 \
         LANE_FOLLOW=0 LANE_FOLLOW_NOGOAL=1 LANE_EXPLORE=0 K_DIST=0.0
  run_arm goal_free "${CKPT_NOGOAL:-checkpoints/carla_lanefollow_nogoal/checkpoint_350000.pkl}" 0 )

# 3. GATED: 145-d, evaluated in GOAL mode (make_env forces it under eval_mode).
( export GATED=1 OBS_WITH_MODE=1 OBS_WITH_GOAL=1 OBS_WITH_VECTOR=1 \
         LANE_FOLLOW=1 LANE_EXPLORE=0 GATED_GOAL_REWARD=sorb STEP_COST=1.0 \
         GATED_GOAL_MAX_STEPS=1000 GATED_EXPLORE_MAX_STEPS=500 \
         LANE_INVASION_NOOP=1 DISCOUNT=1.0 EXPLORE_DISCOUNT=0.99 \
         EXPLORE_PROB=0.5 GOAL_BATCH_FRAC=0.5 TQC_KAPPA=1.0
  run_arm gated "${CKPT_GATED:-checkpoints/carla_gated/checkpoint_350000.pkl}" 1 1 1 )

# 4. GATED + HINDSIGHT: identical environment to arm 3. ONLY the checkpoint
#    differs, which is what isolates relabelling from the gate itself.
( export GATED=1 OBS_WITH_MODE=1 OBS_WITH_GOAL=1 OBS_WITH_VECTOR=1 \
         LANE_FOLLOW=1 LANE_EXPLORE=0 GATED_GOAL_REWARD=sorb STEP_COST=1.0 \
         GATED_GOAL_MAX_STEPS=1000 GATED_EXPLORE_MAX_STEPS=500 \
         LANE_INVASION_NOOP=1 DISCOUNT=1.0 EXPLORE_DISCOUNT=0.99 \
         EXPLORE_PROB=0.5 GOAL_BATCH_FRAC=0.5 TQC_KAPPA=1.0
  run_arm gated_her "${CKPT_GATED_HER:-checkpoints/carla_gated_her/checkpoint_350000.pkl}" 1 1 1 )

echo
echo "=== FOUR-WAY COMPARISON ==="
python summarize_four_way.py "$OUT_DIR"
