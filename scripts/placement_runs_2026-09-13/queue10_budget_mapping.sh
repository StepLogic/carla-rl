#!/bin/bash
# QUEUE 10, 2026-09-12 ~23:10. USER: "Yes proceed" to
#   (1) build maps on the Town02 10k-step continued trace (seed 7)
#   (2) collect Town04 the same way (one seed, 10,000 steps, collision-only)
# Order keeps CARLA single-tenant and uses the CPU in parallel:
#   A. Town02 calibration (CPU, background)  +  B. Town04 stage 1 (CARLA)
#   C. Town04 calibration (CPU, background)  +  D. Town02 stage 3 (CARLA)
#   E. Town04 stage 3 (CARLA)
# CALIBRATION GRID: the default (band median x 0.5..1.5) plus x1.75, 2, 2.5, 3,
# from the start. Same rule (min reach 0.99 on every trace, smallest cutoff
# first); widened because Town02 chose at its old grid top and Town04 needed
# more. Stage 3 refuses a calibration with no chosen setting.
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl
cd "$W" || exit 1
source /home/robotlab/anaconda3/etc/profile.d/conda.sh
conda activate carla-rl || { echo FATAL; exit 1; }
export ENV_SKIP_IMAGE_FEATURES=1 PYTHONPATH=src
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia
Q="$W/results/queue_markers"; mkdir -p "$Q"
T02=results/placement_v6n_town02_budget10k
T04=results/placement_v6n_town04_budget10k
E6N=results/run_env/carla_gated_her_v6n.env
V6N=checkpoints/carla_gated_her_v6n/checkpoint_250000.pkl
say(){ echo "=== [Q10 $(date +%H:%M:%S)] $* ==="; }

calibrate(){  # OUT -- widened grid, writes calibration.json/.txt
  local out=$1
  local E
  E=$(python - "$out/trace_s7.pkl" <<'EOF'
import sys, numpy as np
import calibrate_critic as cc
P, d = cc.load(sys.argv[1])
t = cc.separation_table(P, d)
meds = [t[b][0] for b in ("20-30", "30-50") if t[b][0] is not None]
base = float(np.median(meds))
print(",".join(f"{round(base * f, 1)}" for f in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)))
EOF
)
  echo "[Q10] $out edge cutoffs: $E"
  nice -n 10 python -u calibrate_critic.py --traces "$out/trace_s7.pkl" --horizon-m 50 --gate-m 2.0 \
      --edge-cutoffs "$E" --out "$out/calibration.json" > "$out/calibration.txt" 2>&1
}
chosen(){ python -c "import json,sys;c=json.load(open('$1/calibration.json')).get('chosen');print(c if c else '');sys.exit(0 if c else 1)" 2>/dev/null; }
kill_sim(){ for _ in 1 2 3 4 5 6; do pkill -9 -f 'Linux-Shipping' 2>/dev/null; sleep 2; pgrep -f 'Linux-Shippin[g]' >/dev/null || break; done; sleep 45; }

# A. Town02 calibration in the background
[ -f "$T02/trace_s7.pkl.dist.npy" ] || { say "FATAL: no Town02 budget trace matrix"; exit 1; }
say "A: Town02 calibration (background)"
calibrate $T02 & CAL02=$!

# B. Town04 collection, same protocol as Town02
if [ ! -f "$T04/trace_s7.pkl.dist.npy" ]; then
  say "B: Town04 stage 1 (10k steps, collision-only, no time limit)"
  kill_sim
  RUN_ENV=$E6N MAD_Z=2 CKPT=$V6N ARM=v6n TOWN=Town04 OUT=$T04 SEEDS='7' COLLECT_TOTAL_STEPS=10000 EPISODES=60 \
    STAGES='1' EXPLORER=policy_explore LANE_RULE=strict ./run_placement_eft.sh > logs/q10_town04_stage1.log 2>&1
fi
[ -f "$T04/trace_s7.pkl.dist.npy" ] && say "B done: $(grep -a 'candidates over' logs/q10_town04_stage1.log | tail -1)" \
  || say "B FAILED: no Town04 matrix (see logs/q10_town04_stage1.log)"

# C. Town04 calibration in the background
if [ -f "$T04/trace_s7.pkl.dist.npy" ]; then say "C: Town04 calibration (background)"; calibrate $T04 & CAL04=$!; fi

# D. Town02 stage 3
say "waiting for Town02 calibration"; wait $CAL02
if c=$(chosen $T02); then
  say "D: Town02 stage 3, calibration chose $c"
  kill_sim
  RUN_ENV=$E6N MAD_Z=2 CKPT=$V6N ARM=v6n TOWN=Town02 OUT=$T02 SEEDS='7' COLLECT_TOTAL_STEPS=10000 EPISODES=60 \
    STAGES='3' EXPLORER=policy_explore LANE_RULE=strict ./run_placement_eft.sh > logs/q10_town02_stage3.log 2>&1
  n=$(ls $T02/v6n__*__20-50m__s7.json 2>/dev/null | wc -l)
  grep -aE "\[Result\]" logs/q10_town02_stage3.log | cut -c1-130
  [ "$n" -ge 6 ] && { date +%H:%M > "$Q/budget10k_town02_placement"; say "D DONE ($n rows)"; } || say "D INCOMPLETE ($n/6 rows)"
else
  say "D SKIPPED: Town02 calibration chose nothing even on the widened grid -- needs a decision"
fi

# E. Town04 stage 3
if [ -n "${CAL04:-}" ]; then
  say "waiting for Town04 calibration"; wait $CAL04
  if c=$(chosen $T04); then
    say "E: Town04 stage 3, calibration chose $c"
    kill_sim
    RUN_ENV=$E6N MAD_Z=2 CKPT=$V6N ARM=v6n TOWN=Town04 OUT=$T04 SEEDS='7' COLLECT_TOTAL_STEPS=10000 EPISODES=60 \
      STAGES='3' EXPLORER=policy_explore LANE_RULE=strict ./run_placement_eft.sh > logs/q10_town04_stage3.log 2>&1
    n=$(ls $T04/v6n__*__20-50m__s7.json 2>/dev/null | wc -l)
    grep -aE "\[Result\]" logs/q10_town04_stage3.log | cut -c1-130
    [ "$n" -ge 6 ] && { date +%H:%M > "$Q/budget10k_town04_placement"; say "E DONE ($n rows)"; } || say "E INCOMPLETE ($n/6 rows)"
  else
    say "E SKIPPED: Town04 calibration chose nothing even on the widened grid -- needs a decision"
  fi
fi
kill_sim
say "QUEUE 10 COMPLETE"
