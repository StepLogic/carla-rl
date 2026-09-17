#!/bin/bash
# USER 2026-09-13: "recollect town02 trace - 1) for trace reset in nearest
# waypoint if lane invasion happens 2) goes offroad - only one seed".
# v6n@250k, Town02, seed 7, 10,000-step budget, no time limit. An episode ENDS
# on collision, lane crossing (the explore arm's junction-exempt crossing
# terminal) or off-road; the next one continues from the nearest waypoint.
# Trace + critic matrix only (stage 1), new OUT; no mapping.
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl
cd "$W" || exit 1
source /home/robotlab/anaconda3/etc/profile.d/conda.sh
conda activate carla-rl || { echo FATAL; exit 1; }
export ENV_SKIP_IMAGE_FEATURES=1 PYTHONPATH=src
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia
OUT=results/placement_v6n_town02_budget10k_laneoff; S=7
say(){ echo "=== [laneoff $(date +%H:%M:%S)] $* ==="; }
for _ in 1 2 3 4 5 6; do pkill -9 -f 'Linux-Shippin[g]' 2>/dev/null; sleep 2; pgrep -f 'Linux-Shippin[g]' >/dev/null || break; done
sleep 45

say "collect Town02 seed $S, 10k steps, ends on collision/lane/off_road -> $OUT"
RUN_ENV=results/run_env/carla_gated_her_v6n.env MAD_Z=2 CKPT=checkpoints/carla_gated_her_v6n/checkpoint_250000.pkl \
  ARM=v6n TOWN=Town02 OUT=$OUT SEEDS="$S" COLLECT_TOTAL_STEPS=10000 COLLECT_END_ON="collision,lane,off_road" \
  EPISODES=60 STAGES='1' EXPLORER=policy_explore LANE_RULE=strict ./run_placement_eft.sh > logs/collect_town02_laneoff_stage1.log 2>&1
[ -f "$OUT/trace_s$S.pkl.dist.npy" ] || { say "FAILED: no trace matrix (logs/collect_town02_laneoff_stage1.log)"; exit 1; }

say "trace check"
python - "$OUT/trace_s$S.pkl" <<'EOF'
import sys, pickle, collections, numpy as np
from scipy.spatial import cKDTree
tr = pickle.load(open(sys.argv[1], "rb")); P = np.asarray(tr["positions"])[:, :2]; seg = np.asarray(tr["segments"])
tree = cKDTree(P); n = len(P)
rev = {r: sum(1 for i, nb in enumerate(tree.query_ball_point(P, r)) if any(abs(j - i) > 10 for j in nb)) for r in (2.0, 4.0)}
ks = np.unique(seg)
g = [np.linalg.norm(P[seg == b][0] - P[seg == a][-1]) for a, b in zip(ks[:-1], ks[1:])]
print(f"end_on={tr['meta'].get('collect_end_on')} steps_done={tr.get('steps_done')} episodes={len(ks)} candidates={n}")
print(f"restart gap median {np.median(g):.1f} m, max {max(g):.1f} m" if g else "no restarts")
print(f"revisit within 2 m: {100*rev[2.0]/n:.0f}%  within 4 m: {100*rev[4.0]/n:.0f}%")
EOF
grep -aoE "ended=[a-z_,]+" logs/collect_town02_laneoff_stage1.log | sort | uniq -c
python render_placement_maps.py --run-dir $OUT --town Town02 --no-graph --full-map \
  --out results/figures/exploration_traces_10k_fullmap_laneoff 2>&1 | grep -a Render
for _ in 1 2 3 4 5 6; do pkill -9 -f 'Linux-Shippin[g]' 2>/dev/null; sleep 2; pgrep -f 'Linux-Shippin[g]' >/dev/null || break; done
say "DONE"
