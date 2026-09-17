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

# ---- identical to run_lanefollow_nogoal.sh -------------------------------------
# A MISMATCH HERE IS SILENT AND FATAL. The policy's observation is assembled from
# these variables; OBS_WITH_GOAL=0 makes it 132-d. Build the env with the goal on
# and the agent is a different shape, load_agent_state refuses, and the run dies
# -- which is the good case. Get LANE_STRICT or the reward wrong instead and it
# runs happily while measuring a policy that was never trained.
export LANE_FOLLOW_NOGOAL=1
export OBS_WITH_GOAL=0
export LANE_FOLLOW=0 TIME_TO_GOAL=0 LANE_EXPLORE=0
export K_DIST=0.0 K_SPEED=0.0 K_LIVING=0.0
export LANE_PROGRESS_MODE=raw
export INCREMENTAL_ACTIONS=0
export MAX_STEER=0.5 MAX_THROTTLE=0.6
export OBS_WITH_VECTOR=1
export EPISODE_HER=0
export CHAIN_MAX_GOALS=1
export EXPERT_PROB=0
export LANE_STRICT=0
export LANE_WEDGE=0
export ENV_TOWNS=Town02 EVAL_TOWNS=Town02
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
export VAE_CKPT=checkpoints/lane_vae_120/best.pkl
# 2000, not the 500 used in TRAINING. Trace collection wants long drives: at
# 500 steps an episode covers ~220 m and crosses 4-5 junctions, and the merged
# map needed 59 episodes to reach 23 distinct junctions. A 4x cap should give
# more junctions per episode and more distinct places per rollout.
#
# TWO CONSEQUENCES TO READ THE RESULTS WITH.
#   * The policy was TRAINED with a 500-step cap, so everything past step 500 is
#     extrapolation. It is the same policy, but it has never been optimised for
#     that regime and the later part of a trace is not equivalent to the earlier.
#   * Episodes that previously TRUNCATED at the cap will now keep driving until
#     they fail, so the mix of termination reasons shifts toward lane_any and
#     off_road. Since ~11 of 17 non-junction nodes sat within 10 m of a
#     termination, a longer cap should raise the share of failure-driven spikes,
#     not lower it. Precision against junctions may therefore FALL even as the
#     number of junctions found rises.
export MAX_EPISODE_STEPS=2000
export MAX_DISTANCE_M=2000
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

# ---- NO GOAL BAND ---------------------------------------------------------------
# There is no goal, so there is nothing to tune to force the route across
# junctions. The policy simply drives until it leaves the lane or hits the 500-
# step cap, and whatever junctions it meets are the ones its own behaviour took
# it to. That is a WEAKER instrument than the goal-conditioned version -- we no
# longer choose the junction exposure -- but it is the only honest one here, and
# it is exactly the condition the peer measured as favourable: in EXPLORE mode
# their signal tracked place and openness (+0.24), while in GOAL mode it went
# flat and anti-correlated (-0.15).
CKPT="${CKPT:-checkpoints/carla_lanefollow_nogoal/checkpoint_50000.pkl}"
EPISODES="${EPISODES:-10}"
OUT="${OUT:-results/entropy_nogoal.json}"

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
cleanup() { kill_servers; echo "=== goal-free entropy run done ==="; }
trap cleanup EXIT

kill_servers
# fd 200 dropped FIRST: an orphaned grep inheriting this lock once held it for
# 8.5 hours after the job it belonged to had exited.
( exec 200>&-; "$CARLA_SH" -carla-port=2000 -quality-level=Epic \
      -windowed -ResX=640 -ResY=360 2>&1 \
    | grep -v --line-buffered "Invalid session: no stream available" \
    > logs/carla_entropy_nogoal.log ) &
sleep 40

python -u entropy_at_junctions.py \
    --ckpt "$CKPT" \
    --town Town02 \
    --episodes "$EPISODES" \
    --goal-min 5 --goal-max 20 \
    --out "$OUT"
