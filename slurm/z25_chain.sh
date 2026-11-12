#!/usr/bin/env bash
# Entropy map at MAD z = 2.5 (Leys et al. 2013's default; user 2026-11-09 "redraw
# with z-2.5", "proceed with 1"), 15 m window unchanged. All three maps rebuilt
# together: entropy sets the key count, uniform and SGM forced to it; image gap
# nodes and edges as before. Then descriptors, then the hard multi-turn set with
# image localisation and any-speed arrival.
set -u
cd ~/carla-rl
PY="PYTHONNOUSERSITE=1 $HOME/carla-cluster/envs/carla310/bin/python"
G="-p gpu --gres=gpu:1 -c 4"
O=checkpoints/maps_expert_jx_img_z25
mkdir -p $O && cp checkpoints/maps_expert_jx_img/sgm_pairs_Town0?.npz $O/
S="-p gpu --gres=gpu:1 -c 4 --mem=48G -t 10:00:00 --export=ALL"
export BENCH_BASE_OVERLAY=slurm/il_eval_chw_anyspeed.env OFF_ROAD_JUNCTION_M=3.5 EVAL_MAP_FALLBACK=mapless \
  MAPROOT=$HOME/carla-rl/$O EPISODES=$HOME/carla-rl/logs/bench_turns_episodes.json EVAL_MAP_RESET_HIDDEN=1 \
  EVAL_MAP_REPLAN=3 EVAL_MAP_LOCALISE=image
p=9000
for T in Town02 Town03; do
  bd=$(sbatch --parsable $G --mem=160G -t 4:00:00 -J z25build_$T -o logs/z25build_${T}_%j.log --wrap="cd ~/carla-rl && $PY scripts/build_image_sim_maps.py --trace checkpoints/trace_expert_jx/trace_expert_$T --town $T --out $O --placements entropy uniform sgm_matched --mad-z 2.5")
  ds=$(sbatch --parsable $G --mem=160G -t 3:00:00 --dependency=afterok:$bd -J z25desc_$T -o logs/z25desc_${T}_%j.log --wrap="cd ~/carla-rl && for pl in entropy uniform sgm_matched; do $PY scripts/build_map_descriptors.py --map $O/\$pl/$T.json --trace checkpoints/trace_expert_jx/trace_expert_$T || exit 1; done")
  E=""
  for pl in uniform entropy sgm_matched; do
    E="$E $(TOWN=$T BAND=turns ARM=$pl LABEL=turnsz25 PORT=$p sbatch --parsable $S --dependency=afterok:$ds -J tz25_${pl}_$T -o logs/turnsz25_${pl}_${T}_%j.log slurm/bench_map_debug.sh)"; p=$((p+100))
  done
  echo "$T build $bd desc $ds evals$E"
done
