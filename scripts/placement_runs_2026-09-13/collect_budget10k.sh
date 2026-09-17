#!/bin/bash
# USER 2026-09-12: "COLLECT TRACE JUST ONE SEED - RUN 10,000 step budget remove
# timelimit wrapper for trace collection" and "the collection episode never
# ends unless collision happens or max budget 10,000 limit is reached" --
# crossings, off-road and the 100 m arrival are driven through, not obeyed.
# Town02, seed 7, v6n@250k, explore
# policy, strict lane rule, continue-from-nearest-waypoint (fixed version with
# the spawn check). Stage 1 only: trace + critic matrix. New OUT, so the old
# random-respawn traces are untouched.
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl
cd "$W" || exit 1
source /home/robotlab/anaconda3/etc/profile.d/conda.sh
conda activate carla-rl || { echo FATAL; exit 1; }
export ENV_SKIP_IMAGE_FEATURES=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia
OUT=results/placement_v6n_town02_budget10k
say(){ echo "=== [budget10k $(date +%H:%M:%S)] $* ==="; }

say "kill all CARLA before collection"
for _ in 1 2 3 4 5 6; do pkill -9 -f 'CarlaUE4-Linux-Shipping' 2>/dev/null; sleep 2; ps -eo args | grep -qi '[C]arlaUE4' || break; done
sleep 45

say "START stage 1: Town02 seed 7, COLLECT_TOTAL_STEPS=10000, no time limit -> $OUT"
RUN_ENV=results/run_env/carla_gated_her_v6n.env MAD_Z=2 CKPT=checkpoints/carla_gated_her_v6n/checkpoint_250000.pkl \
  ARM=v6n TOWN=Town02 OUT=$OUT SEEDS='7' COLLECT_TOTAL_STEPS=10000 EPISODES=60 \
  STAGES='1' EXPLORER=policy_explore LANE_RULE=strict ./run_placement_eft.sh

say "restart-gap check (rule: next episode starts at the nearest waypoint to where the last ended)"
python - <<'EOF'
import pickle, numpy as np
p = "results/placement_v6n_town02_budget10k/trace_s7.pkl"
tr = pickle.load(open(p, "rb")); P = np.asarray(tr["positions"])[:, :2]; seg = np.asarray(tr["segments"])
ks = np.unique(seg)
g = np.array([np.linalg.norm(P[seg == b][0] - P[seg == a][-1]) for a, b in zip(ks[:-1], ks[1:])])
print(f"steps_done={tr.get('steps_done')} budget={tr.get('collect_total_steps')} no_timelimit={tr.get('no_timelimit')} "
      f"episodes={len(ks)} candidates={len(P)}")
if len(g):
    print(f"restart gap m: median {np.median(g):.1f} p90 {np.percentile(g, 90):.1f} max {g.max():.1f} | <15 m {int((g < 15).sum())}/{len(g)}")
EOF
say "DONE"
