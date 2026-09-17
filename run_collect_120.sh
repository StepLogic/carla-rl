#!/usr/bin/env bash
# Town02 and Town05 first (they are the RL town and the cleanest map),
# then Town01 and Town04, all at Epic quality and 120x120.
#
# The resolution moves as ONE CHAIN: the env's post_process_image size and the
# VAE's input size are both 120 here. Raising only the VAE side would have
# upsampled 84x84 of information into 120x120 of pixels.
#
# Written to a SEPARATE directory: the v2 shards are 64x64 and cannot be
# concatenated with these.
set -uo pipefail
export DISPLAY=:1
export MIN_MOVE_M=1.5 STUCK_TICKS=600 COLLECT_STRIDE=1
export N_TRAFFIC=0 N_WALKERS=0 HERO_BP=vehicle.micro.microlino
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
export CARLA_PORT=2000
CARLA_SH=/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4.sh
mkdir -p logs data/lane_vae_120

for TOWN in Town02 Town05 Town01 Town04; do
  SHARD="data/lane_vae_120/shard_${TOWN}_Epic.npz"
  [ -f "$SHARD" ] && { echo "== $TOWN cached"; continue; }
  for ATTEMPT in 1 2; do
    echo "== collecting $TOWN Epic 120x120 (attempt $ATTEMPT) =="
    for p in $(pgrep -f '[C]arlaUE4-Linux-Shipping'); do kill -9 "$p" 2>/dev/null; done
    sleep 8
    "$CARLA_SH" -carla-port=2000 -quality-level=Epic -windowed -ResX=640 -ResY=360 \
        >> "logs/carla_server_2000.log" 2>&1 &
    sleep 35
    CARLA_QUALITY=Epic timeout 3600 python -u src/vae_nav/collect_lane_data.py \
        --town "$TOWN" --frames 3000 --out data/lane_vae_120 --tag Epic --size 120 \
        >> "logs/collect120_${TOWN}.log" 2>&1 \
      && { echo "   ok $TOWN"; break; } \
      || echo "   attempt $ATTEMPT failed for $TOWN (see logs/collect120_${TOWN}.log)"
  done
done

for p in $(pgrep -f '[C]arlaUE4-Linux-Shipping'); do kill -9 "$p" 2>/dev/null; done
sleep 5
echo "== assembling =="
ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120 python -u src/vae_nav/collect_lane_data.py \
    --assemble-only --out data/lane_vae_120 2>&1 | tail -12
echo "== done =="
