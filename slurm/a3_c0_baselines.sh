#!/usr/bin/env bash
# A3 (plan 2026-11-09): c0's baselines in ONE training town, evals in sequence, all
# with EVAL_TRAJ=1 (per-step trajectories for the failure taxonomy, A4).
#   PART=val    standard validation pair (40 true + 40 wrong goals) at 1 m / any speed,
#               then the chain evals (ahead; past_junction), 100 chains each
#   PART=turns  the multi-turn validation set: no map, uniform map + image
#               localisation, uniform map + true-pose localisation (oracle)
#   TOWN=Town01 PART=val PORT=5600 sbatch ... --wrap='bash slurm/a3_c0_baselines.sh'
set -u
cd ~/carla-rl
: "${TOWN:?}" "${PART:?}" "${PORT:?}"
C=${CKPT:-$HOME/carla-rl/checkpoints/il_bc_c0/step_15000}
export EVAL_TRAJ=1
if [ "$PART" = val ]; then
  OVERLAY=slurm/il_eval_val_chw_jr_any1m.env CKPT=$C TOWN=$TOWN RUN=c0_valany1m_$TOWN N=40 \
    PORT=$PORT bash slurm/il_eval_pair.sh
  for st in ahead:il_eval_chain_chw_jr_any.env:53000 pj:il_eval_chain_chw_jr_any_pj.env:54000; do
    tag=${st%%:*}; rest=${st#*:}; ov=${rest%%:*}; seed=${rest##*:}
    run=c0_chain_${tag}_$TOWN
    if [ -f "checkpoints/$run/eval_$(basename "$C")/eval.json" ]; then echo "=== have $run"; continue; fi
    echo "=== A3 chain $tag $TOWN seed $seed"
    REWARD_OVERLAY=slurm/$ov ENV_TOWNS=$TOWN RUN_NAME=$run CARLA_PORT=$PORT EVAL_FROM=$C \
      EVAL_EPISODES=100 EVAL_SEED=$seed bash slurm/train_sac.sbatch
    echo "=== A3 chain $tag $TOWN rc=$?"
  done
elif [ "$PART" = turns ]; then
  : "${EPISODES:?}" "${MAPROOT:?}"
  export BENCH_BASE_OVERLAY=slurm/il_eval_chw_anyspeed.env OFF_ROAD_JUNCTION_M=3.5 \
    EVAL_MAP_FALLBACK=mapless EVAL_MAP_RESET_HIDDEN=1 EVAL_MAP_REPLAN=3 EPISODES MAPROOT
  TOWN=$TOWN BAND=turns ARM=mapless LABEL=valturns PORT=$PORT bash slurm/bench_map_debug.sh
  EVAL_MAP_LOCALISE=image TOWN=$TOWN BAND=turns ARM=uniform LABEL=valturns PORT=$PORT \
    bash slurm/bench_map_debug.sh
  EVAL_MAP_LOCALISE=pose EVAL_MAP_JOINT_LOCALISE=1 TOWN=$TOWN BAND=turns ARM=uniform \
    LABEL=valturnspose PORT=$PORT bash slurm/bench_map_debug.sh
else
  echo "PART must be val or turns"; exit 2
fi
echo "=== A3 $TOWN $PART DONE"
