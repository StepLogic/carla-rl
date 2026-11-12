#!/usr/bin/env bash
# z = 2.5 ADOPTED (user 2026-11-09): every paper row that uses our maps, on z25 maps.
#  1. image table, easy + medium: uniform / entropy / SGM on maps_expert_jx_img_z25
#     (the hard multi-turn cells already ran: label turnsz25)
#  2. ViNT on our uniform z25 map: easy + medium (fresh set) and hard (turns set)
#  3. oracle table: key-only z25 maps (--no-image-gaps), true pose, image-arm rules,
#     easy + medium + hard
# All with any-speed arrival.
set -u
cd ~/carla-rl
PY="PYTHONNOUSERSITE=1 $HOME/carla-cluster/envs/carla310/bin/python"
G="-p gpu --gres=gpu:1 -c 4"
S="-p gpu --gres=gpu:1 -c 4 --mem=48G -t 10:00:00 --export=ALL"
FRESH=$HOME/carla-rl/logs/bench_fresh_episodes.json; TURNS=$HOME/carla-rl/logs/bench_turns_episodes.json
IMG=$HOME/carla-rl/checkpoints/maps_expert_jx_img_z25; KEYS=checkpoints/maps_expert_jx_keys_z25
COMMON="BENCH_BASE_OVERLAY=slurm/il_eval_chw_anyspeed.env OFF_ROAD_JUNCTION_M=3.5 EVAL_MAP_FALLBACK=mapless EVAL_MAP_RESET_HIDDEN=1 EVAL_MAP_REPLAN=3"
p=9000
# 1. image table easy/medium
I=""
for T in Town02 Town03; do for b in easy medium; do for pl in uniform entropy sgm_matched; do
  I="$I $(env $COMMON MAPROOT=$IMG EPISODES=$FRESH EVAL_MAP_LOCALISE=image TOWN=$T BAND=$b ARM=$pl LABEL=imagez25 PORT=$p \
        sbatch --parsable $S -J iz25_${pl}_${T}_$b -o logs/imagez25_${pl}_${T}_${b}_%j.log slurm/bench_map_debug.sh)"; p=$((p+100))
done; done; done
echo "image easy/medium:$I"
# 2. ViNT on our uniform z25 map
V1=$(env CKPT=$HOME/carla-rl/checkpoints/vint_final/dagger3_e1 MAPROOT=$IMG EPISODES=$FRESH LABEL=vintd3_z25 ARM=uniform EVAL_MAP_LOCALISE=image \
       CELLS="Town02:easy Town02:medium Town03:easy Town03:medium" PORT=12000 \
       sbatch --parsable $S -t 24:00:00 -J vz25_uniform_em -o logs/vintd3_z25_uniform_em_%j.log slurm/vint_bench_any.sh)
V2=$(env CKPT=$HOME/carla-rl/checkpoints/vint_final/dagger3_e1 MAPROOT=$IMG EPISODES=$TURNS LABEL=vintd3_z25 ARM=uniform EVAL_MAP_LOCALISE=image \
       CELLS="Town02:turns Town03:turns" BAND_OVERLAY=../carla-rl/slurm/bench_turns.env PORT=12100 \
       sbatch --parsable $S -t 24:00:00 -J vz25_uniform_turns -o logs/vintd3_z25_uniform_turns_%j.log slurm/vint_bench_any.sh)
echo "vint uniform z25: $V1 $V2"
# 3. oracle: key-only z25 maps, then easy/medium/hard
mkdir -p $KEYS && cp checkpoints/maps_expert_jx_img/sgm_pairs_Town0?.npz $KEYS/
O=""
for T in Town02 Town03; do
  bd=$(sbatch --parsable $G --mem=160G -t 4:00:00 -J kz25build_$T -o logs/kz25build_${T}_%j.log --wrap="cd ~/carla-rl && $PY scripts/build_image_sim_maps.py --trace checkpoints/trace_expert_jx/trace_expert_$T --town $T --out $KEYS --placements entropy uniform sgm_matched --mad-z 2.5 --no-image-gaps")
  echo "$T key-only build $bd"
  for b in easy medium turns; do
    EP=$FRESH; [ $b = turns ] && EP=$TURNS
    for pl in uniform entropy sgm_matched; do
      O="$O $(env $COMMON MAPROOT=$HOME/carla-rl/$KEYS EPISODES=$EP EVAL_MAP_LOCALISE=pose EVAL_MAP_JOINT_LOCALISE=0 EVAL_MAP_POSE_TOP1=1 \
             EVAL_MAP_START_M=1000000 EVAL_MAP_GOAL_M=1000000 EVAL_MAP_LEG_STEPS=150 EVAL_MAP_OFFROUTE_M=1000000000 \
             TOWN=$T BAND=$b ARM=$pl LABEL=oraclez25 PORT=$p \
             sbatch --parsable $S --dependency=afterok:$bd -J oz25_${pl}_${T}_$b -o logs/oraclez25_${pl}_${T}_${b}_%j.log slurm/bench_map_debug.sh)"; p=$((p+100))
    done
  done
done
echo "oracle:$O"
