#!/bin/bash
# v37 = v36 + the 2026-09-30 reward decisions.
#
# RECONSTRUCTED LAUNCHER. train_v29.sh -- the script v36 actually ran -- was deleted from this
# worktree mid-session. Its two jobs were (a) export the run's flags and (b) write the run_env
# snapshot that every later eval replays; run_gated.sh does neither. This script rebuilds both
# from v36's OWN snapshot (results/run_env/carla_gated_v36_segvae_nobc_noexpert.env, 84 vars),
# which is the record of what actually ran and is a better basis than the lost script.
#
# EVERY v36 FLAG IS RESTATED EXPLICITLY below rather than sourced, so this file is the whole
# provenance of the run and a diff against v36's snapshot is the whole change set.
#
# NOT SET HERE, DELIBERATELY: ENV_IMAGE_SIZE / VAE_IMAGE_SIZE. run_gated.sh:356 hardcodes both
# to 120 AFTER the snapshot is written, so the snapshot records the pre-raise value. Setting
# them here would be ignored at best and would train at 84 px at worst.
#
# Usage:  ./train_v37.sh
# Smoke:  RUN_NAME=carla_gated_v37_smoke MAX_STEPS=2000 ./train_v37.sh
set -u
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl/clean_fix1
cd "$W" || exit 1

# THE INTERPRETER. run_gated.sh:436 runs a bare `python -u train_carla_tqc.py` with no override
# variable, so the venv must be on PATH before it is called -- the launcher this file replaces
# did that and it was lost with it. v36's wrapper log records the interpreter it used:
# clean_fix1/dependencies/ig-venv/bin/python (3.10.16).
#
# THE GUARD IS THE POINT. Without it a missing interpreter is not an error: run_gated.sh's retry
# loop swallows "python: command not found" five times and then prints "gated run done" and
# exits 0 -- a wrapper reporting success having trained nothing. Measured, on this file's first
# smoke test. Fail loudly here instead.
export PATH="$W/dependencies/ig-venv/bin:$PATH"
command -v python >/dev/null 2>&1 || { echo "FATAL: no python on PATH ($W/dependencies/ig-venv/bin missing?)"; exit 1; }
python -c "import jax, flax" 2>/dev/null || { echo "FATAL: $(command -v python) cannot import jax/flax"; exit 1; }
echo "=== [v37] python=$(command -v python) ($(python -V 2>&1)) ==="

export RUN_NAME="${RUN_NAME:-carla_gated_v37_segvae_nonefine}"
export POLICY_FOLDER="$RUN_NAME"
export MAX_STEPS="${MAX_STEPS:-1000000}"

# ---- inherited from v36, unchanged ------------------------------------------------------
export ACTOR_FREEZE_STEPS=0 ACTOR_Q_NORM=1 AGENT_AUGMENT_PIXELS=0 AGENT_ENCODER=none
export AGENT_LATENT_DIM=512 AGENT_LATENT_DIMS=latent+goal:512:256
export BC_WEIGHT=0 EXPERT_PROB=0                 # pure RL (EXPERT_PROB override verified live)
export BUFFER_PIXELS_FP16=0
export CARLA_CLIENT_TIMEOUT=180 CARLA_OFFSCREEN=0 CARLA_OFFSCREEN_PORTS=2000
export CARLA_PORT=2000 CARLA_PORTS=2000,2010 CARLA_QUALITY=Low
export CARLA_QUALITY_PORTS=2000:Low,2010:Epic CARLA_ROOT=/home/robotlab/Apps/CARLA_0.9.15
export CNN_FEATURES=64,128,512 CNN_FILTERS=3,3,1 CNN_STRIDES=2,2,1
export COLLISION_TERMINAL=0                      # unreachable under GOAL_TRUNCATE_COUNTS=1
export ENV_RGB_CHANNEL_FIX=1 ENV_SKIP_IMAGE_FEATURES=1
export ENV_STAGGER_S=90 ENV_STAGGER_TOWNS=Town01 ENV_TOWNS=Town01,Town05 ENV_WEATHER_RANDOM=1
export EPISODE_HER=0 HER_RECOST=1
export EVAL_INTERVAL=25000 EVAL_TOWNS=Town05 N_EVAL_EPISODES=0 N_EVAL_EXPLORE=0
export EXPLORE_GOAL_DIST_M=100 EXPLORE_OFF_ROAD_FATAL=1 EXPLORE_OFF_ROAD_PENALTY=10.0
export GATED_GOAL_LANE_TERMINAL=0 GATED_GOAL_MAX_STEPS=1000 GATED_GOAL_REWARD=sorb
export GOAL_ARRIVAL_BONUS=100.0 GOAL_COLLISION_PENALTY=10.0
export GOAL_CROSSING_TRUNCATE_STEPS=20
export GOAL_DIST_MAX=200 GOAL_DIST_MIN=5 GOAL_DIST_START=200 GOAL_DIST_TARGET=200
export GOAL_LANE_GRADED=1 GOAL_LANE_PENALTY=10.0 GOAL_LANE_PENALTY_BROKEN=10.0
export GOAL_OFF_ROAD_PENALTY=10.0 GOAL_TRUNCATE_COUNTS=1
export INIT_FROM= JUNCTION_GOAL_FRAC=0.5 STRAIGHT_GOAL_FRAC=0.3
export LANE_DEPARTURE_PENALTY=10.0
export MAX_DISTANCE_M=500
export OBS_LATENT_AS_PIXELS=0 OBS_PRIV=0 OBS_STATE_COLLISION=0 OBS_WITH_STATE=1 OBS_WITH_VECTOR=0
export PERCEPTION=segvae POTENTIAL_K=0 ROUTE_REMAINING_MODE=project
export REPLAY_SAVE=1 RUN_GATED_START_SERVER=0    # python starts the servers, not the shell
export SEGVAE_CKPT=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl/results/seg_vae_v1_dec30/seg_vae.pkl
export SENSOR_FRAME_SYNC=1 TARGET_ENTROPY_UNIT=1
export TOWN_LIST=configs/town_split_v7.json TOWN_ROTATE_EPISODES=0
export TQC_VALUE_CEILING=100.0 TQC_VALUE_FLOOR=-3500
export VC1_READOUT=pyramid WEDGE_TRUNCATE=0
export TARGET_SPEED=20.0                         # user 2026-09-30: "leave target speed alone"

# ---- v37 CHANGES ------------------------------------------------------------------------
# 1. Minimum-time contract restored: a clean step costs exactly -1 regardless of speed.
export GOAL_SPEED_BONUS=0.0                      # was 0.75
# 2. An unpainted (NONE) crossing is no longer free in EITHER mode. CARLA traces the four
#    bounding-box corners, so a wheel on a kerb reports as NONE and was costing nothing;
#    5.0 sits below the painted fines because it is a weaker claim about a weaker boundary.
export GOAL_LANE_PENALTY_NONE=5.0                # new
export EXPLORE_LANE_PENALTY_NONE=5.0             # new
# 3. Collisions end the episode in both modes; explore also pays 10 on contact (was 3).
export EXPLORE_COLLISION_FATAL=1                 # new
export LANE_COLLISION_PENALTY=10.0               # unchanged value, stated for provenance
export GOAL_COLLISION_TRUNCATE_STEPS=1           # was 20 -> first collision step is terminal
# 4. Off-road ends a goal episode on the first step (was 20).
export GOAL_OFFROAD_TRUNCATE_STEPS=1             # was 20
# 5. Explore arrival pays 10, not the stale 2.0 that was SMALLER than one step of progress
#    (~2.2), which made arriving mildly discouraged. Target stays 100 m: measured explore
#    speed covers 100-117 m inside the step cap, so 200 m would be unreachable.
export EXPLORE_ARRIVAL_BONUS=10.0                # was 2.0
# 6. Buffer no longer evicts. v36 saturated at step 488,820 and held 46% of its experience
#    by 646k. 97 GB RAM is available; the cost scales linearly from v36's 300k.
export REPLAY_CAPACITY=1000000                   # was 300000
# 7. Undiscounted goal critic, so Q IS the cost-to-go. gated_config REFUSES sorb with a
#    discount unless GATED_ALLOW_DISCOUNTED_SORB=1 -- v36 set that escape hatch and ran at
#    0.99, where Q saturates near -100 and cannot separate a 400-step goal from a 1000-step
#    one. The explore critic stays discounted: it has no terminal state and an undiscounted
#    explore return has no bounded fixed point.
export DISCOUNT=1.0                              # was 0.99
unset GATED_ALLOW_DISCOUNTED_SORB                # no longer needed; let the guard be live

# ---- snapshot (what run_gated.sh does NOT do) --------------------------------------------
mkdir -p results/run_env
SNAP="results/run_env/${RUN_NAME}.env"
env | grep -E '^(ACTOR_|AGENT_|BC_|BUFFER_|CARLA_|CNN_|COLLISION_|DISCOUNT|ENV_|EPISODE_|EVAL_|EXPERT_|EXPLORE_|GATED_|GOAL_|HER_|INIT_FROM|JUNCTION_|LANE_|MAX_|N_EVAL|OBS_|PERCEPTION|POTENTIAL_|REPLAY_|ROUTE_|RUN_GATED_|SEGVAE_|SENSOR_|STRAIGHT_|TARGET_|TOWN_|TQC_|VC1_|WEDGE_)' \
  | sort > "$SNAP"
for k in GATED_GOAL_REWARD DISCOUNT REPLAY_CAPACITY GOAL_ARRIVAL_BONUS GOAL_SPEED_BONUS \
         GOAL_LANE_PENALTY_NONE EXPLORE_LANE_PENALTY_NONE EXPLORE_COLLISION_FATAL \
         EXPLORE_ARRIVAL_BONUS GOAL_OFFROAD_TRUNCATE_STEPS GOAL_COLLISION_TRUNCATE_STEPS \
         EXPERT_PROB PERCEPTION; do
  grep -qE "^${k}=" "$SNAP" || { echo "FATAL: $k missing from $SNAP"; exit 1; }
done
echo "=== [v37 $(date +%H:%M:%S)] snapshot -> $SNAP ($(wc -l < "$SNAP") vars) ==="
echo "=== [v37] RUN_NAME=$RUN_NAME MAX_STEPS=$MAX_STEPS ==="

exec bash ./run_gated.sh
