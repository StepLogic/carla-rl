#!/usr/bin/env bash
# DENSE reference map (user 2026-11-11): every candidate frame is a node, no gap
# nodes needed (coverage is complete by construction). One build + descriptors +
# the hard multi-turn eval per town. A30, one job at a time, chained.
set -u
cd ~/carla-rl
PY="PYTHONNOUSERSITE=1 $HOME/carla-cluster/envs/carla310/bin/python"
G="-p gpu --gres=gpu:a30:1 -c 4"
O=checkpoints/maps_expert_jx_img_dense
S="-p gpu --gres=gpu:a30:1 -c 4 --mem=48G -t 10:00:00 --export=ALL"
export BENCH_BASE_OVERLAY=slurm/il_eval_chw_anyspeed.env OFF_ROAD_JUNCTION_M=3.5 \
  EVAL_MAP_FALLBACK=mapless MAPROOT=$HOME/carla-rl/$O \
  EPISODES=$HOME/carla-rl/logs/bench_turns_episodes.json EVAL_MAP_RESET_HIDDEN=1 \
  EVAL_MAP_REPLAN=3 EVAL_MAP_LOCALISE=image
prev=""; p=9000
for T in Town02 Town03; do
  dep=""; [ -n "$prev" ] && dep="--dependency=afterany:$prev"
  b=$(sbatch --parsable $G --mem=160G -t 6:00:00 $dep -J dense_build_$T -o logs/dense_build_${T}_%j.log \
      --wrap="cd ~/carla-rl && $PY scripts/build_image_sim_maps.py --trace checkpoints/trace_expert_jx/trace_expert_$T \
              --town $T --out $O --placements uniform --uniform-spacing 0.5 --no-image-gaps")
  d=$(sbatch --parsable $G --mem=160G -t 4:00:00 --dependency=afterok:$b -J dense_desc_$T -o logs/dense_desc_${T}_%j.log \
      --wrap="cd ~/carla-rl && $PY scripts/build_map_descriptors.py --map $O/uniform/$T.json --trace checkpoints/trace_expert_jx/trace_expert_$T")
  e=$(TOWN=$T BAND=turns ARM=uniform LABEL=dense PORT=$p sbatch --parsable $S --dependency=afterok:$d \
      -J dense_eval_$T -o logs/dense_eval_${T}_%j.log slurm/bench_map_debug.sh)
  p=$((p+100)); prev=$e
  echo "$T build $b desc $d eval $e"
done
