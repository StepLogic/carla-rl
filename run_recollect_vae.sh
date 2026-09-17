#!/usr/bin/env bash
# Re-collect the VAE corpus with the displacement gate.
#
# The previous corpus was ~30,000 frames holding ~1,000-1,500 distinct views:
# the only sampling gate counted simulator ticks, so a hero stopped at a light
# or wedged behind a bus wrote the same frame thousands of times. Town02 was
# the worst -- 91.9% of its frames contained no lane pixels at all because the
# camera spent the drive against the side of a bus.
#
# MIN_MOVE_M makes n frames mean n samples at least that far apart.
set -uo pipefail
export DISPLAY=:1
export MIN_MOVE_M=1.5 STUCK_TICKS=600 COLLECT_STRIDE=1 N_TRAFFIC=0 N_WALKERS=0 HERO_BP=vehicle.micro.microlino
export CARLA_PORT=2000
CARLA_SH=/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4.sh
mkdir -p logs data/lane_vae_v2

for TOWN in Town02 Town01 Town04 Town05; do
  for Q in Epic Low; do
    SHARD="data/lane_vae_v2/shard_${TOWN}_${Q}.npz"
    [ -f "$SHARD" ] && { echo "== $TOWN/$Q cached"; continue; }
    echo "== collecting $TOWN quality=$Q =="
    pkill -f CarlaUE4 2>/dev/null; sleep 8
    "$CARLA_SH" -carla-port=2000 -quality-level="$Q" -windowed -ResX=640 -ResY=360 \
        >> "logs/carla_server_2000.log" 2>&1 &
    sleep 35
    CARLA_QUALITY="$Q" timeout 3600 python -u src/vae_nav/collect_lane_data.py \
        --town "$TOWN" --frames 3000 --out data/lane_vae_v2 --tag "$Q" \
        >> "logs/collect_${TOWN}_${Q}.log" 2>&1 \
      && echo "   ok $TOWN/$Q" || echo "   FAILED $TOWN/$Q (see logs/collect_${TOWN}_${Q}.log)"
  done
done

pkill -f CarlaUE4 2>/dev/null; sleep 5
echo "== assembling corpus =="
python -u src/vae_nav/collect_lane_data.py --assemble-only --out data/lane_vae_v2 \
    2>&1 | tail -20
