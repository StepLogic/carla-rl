#!/bin/bash
# QUEUE 9, 2026-09-12 21:50. Town04 v6n placement, STAGE 3 ONLY.
# queue4's Town04 stage failed at 21:32: calibration found no cutoff with
# directed reach >= 0.99 on all 4 traces (best 0.9536). Reach was still
# climbing at the grid top (71.6 steps = 1.5x the band median), so the grid
# was widened to 3x (map building; rule unchanged: min-reach 0.99, smallest
# cutoff first). Traces and critic matrices are reused -- stages 1-2 are not
# rerun. Waits for queue8 AND the stage-4 script it execs, so the two never
# share CARLA.
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl
cd "$W" || exit 1
source /home/robotlab/anaconda3/etc/profile.d/conda.sh
conda activate carla-rl || { echo FATAL; exit 1; }
export ENV_SKIP_IMAGE_FEATURES=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia
Q="$W/results/queue_markers"; OUT=results/placement_v6n_town04
say(){ echo "=== [Q9 $(date +%H:%M:%S)] $* ==="; }
busy(){ pgrep -f "tmp/[q]ueue8\.sh" >/dev/null || pgrep -f "tmp/[q]ueue6_stage4\.sh" >/dev/null \
     || pgrep -f "[c]alibrate_critic.py" >/dev/null || pgrep -f "[t]rain_carla_tqc.py" >/dev/null \
     || pgrep -f "[s]weep_eval.py" >/dev/null || pgrep -f "[e]val_placement.py" >/dev/null; }

[ -f "$Q/placement_v6n_town04" ] && { say "already done"; exit 0; }
say "armed; waiting for queue8, stage 4 and the Town04 recalibration"
for _ in $(seq 1 5760); do busy || { sleep 30; busy || break; }; sleep 30; done

chosen=$(python -c "import json;print(json.load(open('$OUT/calibration.json')).get('chosen'))" 2>/dev/null)
if [ -z "$chosen" ] || [ "$chosen" = "None" ]; then
  say "STOP: $OUT/calibration.json has no chosen setting even on the 3x grid -- needs a decision, not a retry"
  exit 1
fi
say "calibration chose: $chosen"
for _ in 1 2 3 4 5 6; do pkill -9 -f 'CarlaUE4-Linux-Shipping' 2>/dev/null; sleep 2; ps -eo args | grep -qi '[C]arlaUE4' || break; done
sleep 45

say "START placement_v6n_town04 stage 3 -> logs/q9_placement_v6n_town04.log"
RUN_ENV=results/run_env/carla_gated_her_v6n.env MAD_Z=2 CKPT=checkpoints/carla_gated_her_v6n/checkpoint_250000.pkl \
  ARM=v6n TOWN=Town04 OUT=$OUT SEEDS='7 8 9 10' COLLECT_EPISODES=30 COLLECT_STEPS=500 EPISODES=60 \
  STAGES='3' EXPLORER=policy_explore LANE_RULE=strict ./run_placement_eft.sh > logs/q9_placement_v6n_town04.log 2>&1
n=$(ls $OUT/*__20-50m__s*.json 2>/dev/null | wc -l)
grep -aE "\[Result\]" logs/q9_placement_v6n_town04.log | cut -c1-120
if [ "$n" -ge 24 ]; then date +%H:%M > "$Q/placement_v6n_town04"; say "DONE placement_v6n_town04 ($n rows)"; else say "INCOMPLETE: $n/24 rows"; fi
say "QUEUE 9 COMPLETE"
