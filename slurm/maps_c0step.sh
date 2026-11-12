#!/usr/bin/env bash
# SINGLE-TRACE EVALUATION MAPS, step 1 and step 3: build the placements from c0's
# OWN per-step explore trace (checkpoints/trace1step_il_bc_c0_s15000_<Town>).
#
#   step 1 (CPU, no CARLA):  entropy / uniform / random
#   step 3 (CPU, after slurm/sgm_pairs_single.sh wrote maps_c0step/sgm_pairs_<Town>.npz):
#           SGM_STEP=1 -> sgm_matched only
#
# One town per job so a town's build cannot overwrite another's summary (9d830d4):
#   MAP_TOWNS=Town02 sbatch -p haswell -c 4 --mem=64G -t 4:00:00 \
#       -J maps_c0step_Town02 -o logs/maps_c0step_Town02_%j.log slurm/maps_c0step.sh
# MAP_ARGS: extra builder flags, e.g. "--gap-m 8 --max-turn-deg 90" (the Habitat
# fixes, 2026-11-06); the same flags must go to the SGM_STEP=1 build.
set -u
cd ~/carla-rl
PY=${CLUSTER_PY:-$HOME/carla-cluster/envs/carla310/bin/python}
export PYTHONNOUSERSITE=1
TRACE_FMT=${TRACE_FMT:-"checkpoints/trace1step_il_bc_c0_s15000_{town}/trace_step_15000"}
OUT=${MAP_OUT:-$HOME/carla-rl/checkpoints/maps_c0step}
EXTRA=${SGM_STEP:+--sgm}
for town in ${MAP_TOWNS:-Town02 Town03}; do
  echo "=== MAPS c0step $town ${EXTRA:-entropy/uniform/random} ==="
  $PY scripts/build_single_trace_maps.py ${EXTRA:-} ${MAP_ARGS:-} \
    --trace-fmt "$TRACE_FMT" --towns "$town" --out "$OUT"
  echo "=== MAPS c0step $town rc=$? ==="
done
echo "=== MAPS c0step DONE ==="
