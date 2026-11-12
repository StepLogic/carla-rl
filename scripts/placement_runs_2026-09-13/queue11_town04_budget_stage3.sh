#!/bin/bash
# QUEUE 11, 2026-09-13 ~01:05. Town04 10k-trace placement, stage 3 only.
# queue10 printed "E SKIPPED: ... chose nothing" at 00:54 -- MISLEADING: I had
# killed the slow pure-Python calibration at 00:43 and restarted it with the
# compiled reachability (identical numbers, pinned by
# tests/test_directed_reachability_fast.py), so no calibration.json existed yet.
# This waits for that restarted calibration, then runs stage 3 if it chose.
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl
cd "$W" || exit 1
source /home/robotlab/anaconda3/etc/profile.d/conda.sh
conda activate carla-rl || { echo FATAL; exit 1; }
export ENV_SKIP_IMAGE_FEATURES=1 PYTHONPATH=src
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia
Q="$W/results/queue_markers"; T04=results/placement_v6n_town04_budget10k
say(){ echo "=== [Q11 $(date +%H:%M:%S)] $* ==="; }

say "waiting for the Town04 calibration"
while pgrep -f '[c]alibrate_critic.py.*town04_budget10k' >/dev/null; do sleep 30; done
grep -aE "CHOSEN|NO setting" $T04/calibration.txt | cut -c1-200
c=$(python -c "import json;c=json.load(open('$T04/calibration.json')).get('chosen');print(c if c else '')" 2>/dev/null)
if [ -z "$c" ]; then
  say "STOP: Town04 calibration chose nothing (or wrote no file) -- needs a decision, not a retry"
  exit 1
fi
say "calibration chose $c"
while pgrep -f '[e]val_placement.py' >/dev/null || pgrep -f '[s]weep_eval.py' >/dev/null || pgrep -f '[t]rain_carla_tqc.py' >/dev/null; do sleep 30; done
for _ in 1 2 3 4 5 6; do pkill -9 -f 'Linux-Shipping' 2>/dev/null; sleep 2; pgrep -f 'Linux-Shippin[g]' >/dev/null || break; done
sleep 45

say "START Town04 stage 3 -> logs/q11_town04_stage3.log"
RUN_ENV=results/run_env/carla_gated_her_v6n.env MAD_Z=2 CKPT=checkpoints/carla_gated_her_v6n/checkpoint_250000.pkl \
  ARM=v6n TOWN=Town04 OUT=$T04 SEEDS='7' COLLECT_TOTAL_STEPS=10000 EPISODES=60 \
  STAGES='3' EXPLORER=policy_explore LANE_RULE=strict ./run_placement_eft.sh > logs/q11_town04_stage3.log 2>&1
n=$(ls $T04/v6n__*__20-50m__s7.json 2>/dev/null | wc -l)
grep -aE "\[Result\]" logs/q11_town04_stage3.log | cut -c1-140
[ "$n" -ge 6 ] && { date +%H:%M > "$Q/budget10k_town04_placement"; say "DONE ($n rows)"; } || say "INCOMPLETE ($n/6 rows)"
