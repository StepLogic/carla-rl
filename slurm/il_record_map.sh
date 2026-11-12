#!/usr/bin/env bash
# B2 (plan 2026-11-09): map-subgoal recording for ONE training town per job -- the
# towns with a training-town map (Town01, Town05). The expert drives; the goal input
# is the next node image of MAPROOT/<Town>.json (default: the uniform z=2.5 training
# map the A3 baselines used). EPISODES = routes RECORDED (seeds tried until then, at
# most EXPERT_RECORD_MAP_TRIES x as many). Seeds 2000000 + 100000*town +
# 1000000*(round-1), even only; shards (one per leg, id seed*100+leg) go to
# checkpoints/il_data_map<ROUND>/<Town>/. Town02/Town03: run_record refuses them.
#   TOWN=Town05 ROUND=1 EPISODES=150 sbatch -p gpu --gres=gpu:l40:1 ... --wrap='bash slurm/il_record_map.sh'
set -u
cd ~/carla-rl
: "${TOWN:?}" "${ROUND:?}"
case $TOWN in Town01) i=0;; Town04) i=1;; Town05) i=2;; Town10HD) i=3;;
  *) echo "=== $TOWN is not a training town"; exit 2;; esac
seed0=$((2000000 + 100000 * i + 1000000 * (ROUND - 1)))
port=$((5900 + 10 * i))
MAPROOT=${MAPROOT:-$HOME/carla-rl/checkpoints/maps_train_jx_img_z25/uniform}
if [ ! -f "$MAPROOT/$TOWN.json" ]; then echo "=== no map $MAPROOT/$TOWN.json"; exit 2; fi
export EXPERT_RECORD=1 EXPERT_DRIVER=expert EXPERT_RECORD_DRY=0 EVAL_MAP_DIR=$MAPROOT
export EXPERT_RECORD_ROOT=${ROOT:-$HOME/carla-rl/checkpoints/il_data_map$ROUND}
export EXPERT_RECORD_EPISODES=${EPISODES:-150}
echo "=== MAP-SUBGOAL RECORD round $ROUND $TOWN seeds from $seed0 port $port map $MAPROOT -> $EXPERT_RECORD_ROOT"
REWARD_OVERLAY=slurm/il_record_map.env ENV_TOWNS=$TOWN EXPERT_SEED0=$seed0 \
  RUN_NAME=${RUN_NAME:-il_record_map_r${ROUND}_$TOWN} CARLA_PORT=$port bash slurm/train_sac.sbatch
echo "=== MAP-SUBGOAL RECORD round $ROUND $TOWN rc=$? ==="
