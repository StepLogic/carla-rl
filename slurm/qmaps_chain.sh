#!/usr/bin/env bash
# Run Q's own maps (user 2026-11-08): after Q's expert tour of each town, the same
# pipeline as c0's maps_expert_jx_img (defaults reproduce its stored meta), then
# Q's 9 map cells per town.
set -u
cd ~/carla-rl
PY="PYTHONNOUSERSITE=1 $HOME/carla-cluster/envs/carla310/bin/python"
G="-p gpu --gres=gpu:1 -c 4"
TR=checkpoints/trace_expert_jx_q500; O=checkpoints/maps_expert_jx_q500
declare -A TJ=([Town02]=465013 [Town03]=465014)
S="-p gpu --gres=gpu:1 -c 4 --mem=48G -t 10:00:00 --export=ALL"
export CKPT=$HOME/carla-rl/checkpoints/sac_scratch_2t_u5_orterm_idle_nb_a02/step_500000 \
  OFF_ROAD_JUNCTION_M=3.5 EVAL_MAP_FALLBACK=mapless MAPROOT=$HOME/carla-rl/$O \
  EPISODES=$HOME/carla-rl/logs/bench_fresh_episodes.json EVAL_MAP_RESET_HIDDEN=1 EVAL_MAP_REPLAN=3 \
  EVAL_MAP_LOCALISE=image
unset GOAL_ARRIVAL_MAX_SPEED
p=11400
for T in Town02 Town03; do
  pr=$(sbatch --parsable $G --mem=96G -t 3:00:00 --dependency=afterok:${TJ[$T]} -J qsgmpairs_$T -o logs/qsgmpairs_${T}_%j.log --wrap="cd ~/carla-rl && $PY scripts/sgm_distance_pairs.py --trace $TR/trace_expert_$T --town $T --model checkpoints/sgm_distance --out $O")
  bd=$(sbatch --parsable $G --mem=160G -t 8:00:00 --dependency=afterok:$pr -J qmapbuild_$T -o logs/qmapbuild_${T}_%j.log --wrap="cd ~/carla-rl && $PY scripts/build_image_sim_maps.py --trace $TR/trace_expert_$T --town $T --out $O --placements entropy uniform sgm_matched")
  ds=$(sbatch --parsable $G --mem=160G -t 3:00:00 --dependency=afterok:$bd -J qmapdesc_$T -o logs/qmapdesc_${T}_%j.log --wrap="cd ~/carla-rl && for pl in entropy uniform sgm_matched; do $PY scripts/build_map_descriptors.py --map $O/\$pl/$T.json --trace $TR/trace_expert_$T || exit 1; done")
  E=""
  for b in easy medium hard; do for pl in uniform entropy sgm_matched; do
    j=$(TOWN=$T BAND=$b ARM=$pl LABEL=qfresh PORT=$p sbatch --parsable $S --dependency=afterok:$ds -J q_${pl}_${T}_$b -o logs/qfresh_${pl}_${T}_${b}_%j.log slurm/bench_map_debug.sh); p=$((p+100)); E="$E $j"
  done; done
  echo "$T trace ${TJ[$T]} pairs $pr build $bd desc $ds evals$E"
done
