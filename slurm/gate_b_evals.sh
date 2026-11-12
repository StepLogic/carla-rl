#!/usr/bin/env bash
# Gate B (plan 2026-11-09): one clone's training-town evals in ONE town, all with
# EVAL_TRAJ=1. TAG names the clone (c0, c1, ...) in every run name, so two clones
# whose chosen checkpoints share a step name never collide (slurm/a3_c0_baselines.sh
# hard-codes c0's names; TAG=c0 here finds A3's chain result instead of re-running it).
# The standard validation pair is not here: c1's runs inside its training job.
#   PART=chains  chain eval, ahead style (the spec metric), 100 chains from seed 53000
#   PART=turns   the multi-turn validation set, uniform map + image localisation
#                (Gate B's metric; c0's is A3's bench_valturns_uniform_<Town>_turns)
#   PART=oracle  the same set, uniform map + TRUE POSE with the mapping session's oracle
#                recipe: replans only on the 150-step leg timeout (no 15 m off-route
#                rule), no 8 m localisation radius, nearest aligned start / nearest goal
#                node. Supplementary: A3's pose arm dropped the map (memory 2026-11-10).
#   TAG=c1 CKPT=.../il_bc_c1/step_N TOWN=Town01 PART=turns PORT=5600 \
#     sbatch -p gpu --gres=gpu:l40:1 ... --wrap='bash slurm/gate_b_evals.sh'
set -u
cd ~/carla-rl
: "${TAG:?}" "${CKPT:?}" "${TOWN:?}" "${PART:?}" "${PORT:?}"
export EVAL_TRAJ=1
case $PART in
  chains)
    run=${TAG}_chain_ahead_$TOWN
    if [ -f "checkpoints/$run/eval_$(basename "$CKPT")/eval.json" ]; then echo "=== have $run"; exit 0; fi
    echo "=== GATE B chains $TAG $TOWN ($CKPT)"
    REWARD_OVERLAY=slurm/il_eval_chain_chw_jr_any.env ENV_TOWNS=$TOWN RUN_NAME=$run \
      CARLA_PORT=$PORT EVAL_FROM=$CKPT EVAL_EPISODES=100 EVAL_SEED=53000 bash slurm/train_sac.sbatch
    ;;
  turns|oracle)
    export EPISODES=${EPISODES:-$HOME/carla-rl/logs/val_turns_episodes_$TOWN.json}
    export MAPROOT=${MAPROOT:-$HOME/carla-rl/checkpoints/maps_train_jx_img_z25}
    export CKPT BENCH_BASE_OVERLAY=slurm/il_eval_chw_anyspeed.env OFF_ROAD_JUNCTION_M=3.5 \
      EVAL_MAP_FALLBACK=mapless EVAL_MAP_RESET_HIDDEN=1 EVAL_MAP_REPLAN=3
    if [ "$PART" = turns ]; then
      EVAL_MAP_LOCALISE=image TOWN=$TOWN BAND=turns ARM=uniform LABEL=valturns_$TAG \
        PORT=$PORT bash slurm/bench_map_debug.sh
    else
      EVAL_MAP_LOCALISE=pose EVAL_MAP_OFFROUTE_M=1000000000 EVAL_MAP_LEG_STEPS=150 \
        EVAL_MAP_START_M=1000000 EVAL_MAP_GOAL_M=1000000 EVAL_MAP_POSE_TOP1=1 \
        EVAL_MAP_JOINT_LOCALISE=0 TOWN=$TOWN BAND=turns ARM=uniform LABEL=valoracle_$TAG \
        PORT=$PORT bash slurm/bench_map_debug.sh
    fi
    ;;
  *) echo "PART must be chains, turns or oracle"; exit 2 ;;
esac
echo "=== GATE B $TAG $TOWN $PART DONE"
