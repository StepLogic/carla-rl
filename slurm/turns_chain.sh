#!/usr/bin/env bash
# Hard-goal set (>= 2 junction turns, 50-150 m) replacing the hard band (user
# 2026-11-09). After the probes: select 100/town, then every arm, any-speed arrival.
set -u
cd ~/carla-rl
PY="PYTHONNOUSERSITE=1 $HOME/carla-cluster/envs/carla310/bin/python"
EP=$HOME/carla-rl/logs/bench_turns_episodes.json
# Town03 yields ~9% qualifying routes (89 of 1000 from 41000), so a second Town03
# probe of 1000 seeds from 42000 runs first; the selection merges both files.
C0=$HOME/carla-rl/checkpoints/il_bc_c0/step_15000
P3B=$(sbatch --parsable -p gpu --gres=gpu:1 -c 4 --mem=48G -t 6:00:00 --export=ALL -J turnprobe_Town03b -o logs/turnprobe_Town03b_%j.log --wrap="cd ~/carla-rl && export GOAL_CAM_MATCH=1 EVAL_PROBE=1 EVAL_PROBE_TURNS=1 EVAL_EPISODES=1000 EVAL_SEED=42000 BENCH_BASE_OVERLAY=slurm/il_eval_chw_anyspeed.env; REWARD_OVERLAY=slurm/bench_turns.env ENV_TOWNS=Town03 RUN_NAME=bench_probe_turns_Town03b CARLA_PORT=10200 EVAL_FROM=$C0 bash slurm/train_sac.sbatch")
echo "probe Town03b $P3B"
sel=$(sbatch --parsable -p gpu --gres=gpu:1 -c 1 --mem=4G -t 0:20:00 --dependency=afterok:465250:465251:$P3B -J turns_select -o logs/turns_select_%j.log --wrap="cd ~/carla-rl && $PY scripts/select_turn_episodes.py $EP checkpoints/bench_probe_turns_Town02/eval_step_15000/probe.json checkpoints/bench_probe_turns_Town03/eval_step_15000/probe.json checkpoints/bench_probe_turns_Town03b/eval_step_15000/probe.json")
echo "select $sel"
S="-p gpu --gres=gpu:1 -c 4 --mem=48G -t 10:00:00 --export=ALL --dependency=afterok:$sel"
export BENCH_BASE_OVERLAY=slurm/il_eval_chw_anyspeed.env OFF_ROAD_JUNCTION_M=3.5 EVAL_MAP_FALLBACK=mapless \
  MAPROOT=$HOME/carla-rl/checkpoints/maps_expert_jx_img EPISODES=$EP EVAL_MAP_RESET_HIDDEN=1 EVAL_MAP_REPLAN=3
p=9000; B=""
for T in Town02 Town03; do
  j=$(TOWN=$T BAND=turns ARM=mapless LABEL=turns PORT=$p sbatch --parsable $S -J turns_mapless_$T -o logs/turns_mapless_${T}_%j.log slurm/bench_map_debug.sh); p=$((p+100)); B="$B $j"
  j=$(WRONG_GOAL_MODE=1 TOWN=$T BAND=turns ARM=mapless LABEL=turnswrong PORT=$p sbatch --parsable $S -J turnsw_mapless_$T -o logs/turnswrong_mapless_${T}_%j.log slurm/bench_map_debug.sh); p=$((p+100)); B="$B $j"
  for pl in uniform entropy sgm_matched; do
    j=$(EVAL_MAP_LOCALISE=image TOWN=$T BAND=turns ARM=$pl LABEL=turns PORT=$p sbatch --parsable $S -J turns_${pl}_$T -o logs/turns_${pl}_${T}_%j.log slurm/bench_map_debug.sh); p=$((p+100)); B="$B $j"
  done
done
echo "c0:$B"
V=""
export CKPT=$HOME/carla-rl/checkpoints/vint_final/dagger3_e1 LABEL=vintd3_turns CELLS="Town02:turns Town03:turns" BAND_OVERLAY=../carla-rl/slurm/bench_turns.env
V="$V $(ARM=mapless PORT=12000 sbatch --parsable $S -t 24:00:00 --hold -J vturns_mapless -o logs/vintd3_turns_mapless_%j.log slurm/vint_bench_any.sh)"
V="$V $(ARM=uniform EVAL_MAP_LOCALISE=image PORT=12100 sbatch --parsable $S -t 24:00:00 --hold -J vturns_uniform_img -o logs/vintd3_turns_uniform_image_%j.log slurm/vint_bench_any.sh)"
V="$V $(ARM=vinttopo TOPOMAP=$HOME/carla-rl/checkpoints/vint_topomap/dagger3_e1 PORT=12200 sbatch --parsable $S -t 24:00:00 --hold --exclude=node-g104 -J vturns_vinttopo -o logs/vintd3_turns_vinttopo_%j.log slurm/vint_bench_any.sh)"
echo "vint:$V"
# Oracle (true pose, key-only maps, image-arm rules) on the hard-turns set (user 2026-11-09).
export EVAL_MAP_LOCALISE=pose EVAL_MAP_JOINT_LOCALISE=0 EVAL_MAP_POSE_TOP1=1 EVAL_MAP_START_M=1000000 \
  EVAL_MAP_GOAL_M=1000000 EVAL_MAP_LEG_STEPS=150 EVAL_MAP_OFFROUTE_M=1000000000 \
  MAPROOT=$HOME/carla-rl/checkpoints/maps_expert_jx_keys EPISODES=$EP
unset CKPT LABEL CELLS BAND_OVERLAY
p=9600; O=""
for T in Town02 Town03; do for pl in uniform entropy sgm_matched; do
  O="$O $(TOWN=$T BAND=turns ARM=$pl LABEL=turnsoracle PORT=$p sbatch --parsable $S -J turnso_${pl}_$T -o logs/turnsoracle_${pl}_${T}_%j.log slurm/bench_map_debug.sh)"; p=$((p+100))
done; done
echo "oracle:$O"
