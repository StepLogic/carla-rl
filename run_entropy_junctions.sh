#!/usr/bin/env bash
# Roll out the trained lane-following policy and record policy entropy per step,
# labelled by junction occupancy.
#
# THE ENV MUST MATCH THE ONE THE POLICY WAS TRAINED IN, exactly. The observation
# is assembled from these variables (OBS_WITH_VECTOR adds the speed block,
# INCREMENTAL_ACTIONS decides the action space), so a single mismatch feeds the
# policy a differently-shaped or differently-scaled input and the entropy trace
# describes a policy that was never trained. Everything below is copied from
# run_lanefollow.sh; only the goal band differs, and that is the point.
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

# ---- identical to run_lanefollow.sh -------------------------------------------
export LANE_FOLLOW=1
export DELTA_MODE=positive
export INCREMENTAL_ACTIONS=0
export MAX_STEER=0.5 MAX_THROTTLE=0.6
export STEER_STEP=0.01 THROTTLE_STEP=0.01
export OBS_WITH_VECTOR=1
export EPISODE_HER=0
export CHAIN_MAX_GOALS=1
export EXPERT_PROB=0
export LANE_STRICT=1
export ENV_TOWNS=Town02 EVAL_TOWNS=Town02
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
export VAE_CKPT=checkpoints/lane_vae_120/best.pkl
export MAX_EPISODE_STEPS=2000
export MAX_DISTANCE_M=200
export TARGET_SPEED=20.0
export SHORT_HORIZON=1
export TQC_KAPPA=1.0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# 0.25, not 0.4, AND command buffers off. MEASURED FAILURE 2026-09-08 20:38:
# this rollout is a TWO-process GPU job (CARLA ~2.6 GB + JAX), sharing the card
# with the peer's Habitat training (~3.5 GB). At 0.4 the JAX cap alone is
# ~4.9 GB, and the first jitted call died with RESOURCE_EXHAUSTED while
# instantiating a CUDA command buffer -- not while allocating arrays, which is
# why a free-memory check passed and the job still OOM'd. The XLA flag is the
# remedy the error message itself names; it trades a little kernel-launch
# overhead for not needing a contiguous graph allocation at all.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

# ---- what differs: the goal band ----------------------------------------------
# 40-113 m instead of the curriculum's current 5-20 m rung. Long goals are what
# force the route across junctions at all -- at 5-20 m the measured junction
# occupancy was ~1% of steps, which is far too few to compare against.
CKPT="${CKPT:-checkpoints/carla_lanefollow/checkpoint_50000.pkl}"
EPISODES="${EPISODES:-10}"
OUT="${OUT:-results/entropy_junctions.json}"

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
cleanup() { kill_servers; echo "=== entropy run done ==="; }
trap cleanup EXIT

kill_servers
# fd 200 dropped FIRST: an orphaned grep inheriting this lock once held it for
# 8.5 hours after the job it belonged to had exited.
( exec 200>&-; "$CARLA_SH" -carla-port=2000 -quality-level=Epic \
      -windowed -ResX=640 -ResY=360 2>&1 \
    | grep -v --line-buffered "Invalid session: no stream available" \
    > logs/carla_entropy.log ) &
sleep 40

python -u entropy_at_junctions.py \
    --ckpt "$CKPT" \
    --town Town02 \
    --episodes "$EPISODES" \
    --goal-min 40 --goal-max 113 \
    --out "$OUT"
