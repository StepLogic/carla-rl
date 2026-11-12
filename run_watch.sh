#!/usr/bin/env bash
# Watch the trained policy drive, on screen. Env MUST match how it trained --
# incremental actions and the vector observation key -- or load_agent_state
# rejects the checkpoint.
set -uo pipefail
cd "$(dirname "$0")"
LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
flock -n 200 || { echo "another CARLA job holds the lock" >&2; exit 1; }
export DISPLAY=:1 CARLA_PORT=2000
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
export VAE_CKPT=checkpoints/lane_vae_120/best.pkl
export INCREMENTAL_ACTIONS=1 OBS_WITH_VECTOR=1
export MAX_STEER=0.5 MAX_THROTTLE=0.6 STEER_STEP=0.01 THROTTLE_STEP=0.01
export LANE_FOLLOW=0 K_DIST=1.0 DELTA_MODE=positive
export K_DIST_GATE_MODE=positive K_DIST_TARGET_SPEED=20.0
export K_SPEED=0.02 SPEED_LIMIT_KMH=30.0 K_LIVING=0.005
export TARGET_SPEED=20.0 MAX_EPISODE_STEPS=500
export SHORT_HORIZON=1 LANE_STRICT=1 EPISODE_HER=0 CHAIN_MAX_GOALS=1 EXPERT_PROB=0
export GOAL_DIST_MIN=20.0 GOAL_DIST_MAX=40.0     # long enough to watch it drive
# Chase camera: behind and above, pitched down so the car is IN frame.
export SPECTATOR_BACK=7.0 SPECTATOR_UP=3.0 SPECTATOR_PITCH=-12.0
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
for p in $(pgrep -f '[C]arlaUE4-Linux-Shipping'); do kill -9 "$p" 2>/dev/null; done
sleep 6
( exec 200>&-; /home/robotlab/Apps/CARLA_0.9.15/CarlaUE4.sh -carla-port=2000 \
    -quality-level=Epic -windowed -ResX=1280 -ResY=720 2>&1 \
  | grep -v --line-buffered "Invalid session" > logs/carla_watch.log ) &
sleep 40
python -u watch_policy.py \
  --ckpt checkpoints/carla_sparse_perstep/checkpoint_150000.pkl \
  --town Town02 --episodes 8 \
  --record results/watch_sparse_perstep.mp4
