#!/bin/bash
# QUEUE 13, 2026-09-13. USER chose option 1: start the Town04 trace INSIDE the
# NE city grid. Seeds 7 and 8 from random spawns drove the outer highway once
# (3/1200, 4/1230 points revisited within the 2 m SGM gate). Same protocol
# otherwise (v6n@250k, 10,000 steps, no time limit, collision-only ends,
# continue from crash). Seed 9, start (300,-172) -> road 11 lane 1, not a
# junction (checked on the offline xodr). Maps are built only if the trace
# revisits itself (>= 10% of points within 2 m of a non-neighbour).
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl
cd "$W" || exit 1
source /home/robotlab/anaconda3/etc/profile.d/conda.sh
conda activate carla-rl || { echo FATAL; exit 1; }
export ENV_SKIP_IMAGE_FEATURES=1 PYTHONPATH=src
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia
Q="$W/results/queue_markers"; OUT=results/placement_v6n_town04_budget10k_grid; S=9
E6N=results/run_env/carla_gated_her_v6n.env; V6N=checkpoints/carla_gated_her_v6n/checkpoint_250000.pkl
say(){ echo "=== [Q13 $(date +%H:%M:%S)] $* ==="; }
kill_sim(){ for _ in 1 2 3 4 5 6; do pkill -9 -f 'Linux-Shipping' 2>/dev/null; sleep 2; pgrep -f 'Linux-Shippin[g]' >/dev/null || break; done; sleep 45; }
common(){ RUN_ENV=$E6N MAD_Z=2 CKPT=$V6N ARM=v6n TOWN=Town04 OUT=$OUT SEEDS="$S" COLLECT_TOTAL_STEPS=10000 \
          COLLECT_START_XY="300,-172" EPISODES=60 EXPLORER=policy_explore LANE_RULE=strict "$@"; }

say "collect Town04 seed $S from the city grid (300,-172) -> $OUT"
kill_sim
common env STAGES='1' ./run_placement_eft.sh > logs/q13_town04_grid_stage1.log 2>&1
[ -f "$OUT/trace_s$S.pkl.dist.npy" ] || { say "FAILED: no trace matrix (logs/q13_town04_grid_stage1.log)"; kill_sim; exit 1; }
grep -aE "candidates over|ep0: continuing|MISSED" logs/q13_town04_grid_stage1.log | head -3 | cut -c1-160

say "revisit + restart-gap check"
frac=$(python - "$OUT/trace_s$S.pkl" <<'EOF'
import sys, pickle, numpy as np
from scipy.spatial import cKDTree
tr = pickle.load(open(sys.argv[1], "rb")); P = np.asarray(tr["positions"])[:, :2]; seg = np.asarray(tr["segments"])
tree = cKDTree(P); n = len(P)
rev = sum(1 for i, nb in enumerate(tree.query_ball_point(P, 2.0)) if any(abs(j - i) > 3 for j in nb))
ks = np.unique(seg)
g = [np.linalg.norm(P[seg == b][0] - P[seg == a][-1]) for a, b in zip(ks[:-1], ks[1:])]
gap = f"median {np.median(g):.1f} m, max {max(g):.1f} m" if g else "no restarts"
print(f"first position ({P[0,0]:.1f},{P[0,1]:.1f}) start_xy={tr['meta'].get('collect_start_xy')}; episodes={len(ks)} "
      f"candidates={n} steps_done={tr.get('steps_done')} restart gap {gap}; revisit within 2 m: {rev}/{n} ({100*rev/n:.0f}%)", file=sys.stderr)
print(rev / n)
EOF
)
python render_placement_maps.py --run-dir $OUT --town Town04 --no-graph --out results/figures/placement_v6n_town04_budget10k_grid 2>&1 | grep -a "Render"
if python -c "import sys; sys.exit(0 if float('$frac') >= 0.10 else 1)"; then
  say "revisit fraction $frac >= 0.10 -- mapping"
else
  say "STOP: revisit fraction $frac < 0.10 -- needs a decision"; kill_sim; exit 1
fi

say "calibration (widened grid, compiled reachability)"
E=$(python - "$OUT/trace_s$S.pkl" <<'EOF'
import sys, numpy as np
import calibrate_critic as cc
P, d = cc.load(sys.argv[1]); t = cc.separation_table(P, d)
base = float(np.median([t[b][0] for b in ("20-30", "30-50") if t[b][0] is not None]))
print(",".join(f"{round(base * f, 1)}" for f in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)))
EOF
)
echo "[Q13] edge cutoffs: $E"
nice -n 10 python -u calibrate_critic.py --traces "$OUT/trace_s$S.pkl" --horizon-m 50 --gate-m 2.0 \
    --edge-cutoffs "$E" --out "$OUT/calibration.json" > "$OUT/calibration.txt" 2>&1
grep -aE "CHOSEN|NO setting" "$OUT/calibration.txt" | cut -c1-200
c=$(python -c "import json;c=json.load(open('$OUT/calibration.json')).get('chosen');print(c if c else '')" 2>/dev/null)
[ -n "$c" ] || { say "STOP: calibration chose nothing -- needs a decision"; kill_sim; exit 1; }

say "stage 3 (build + navigate), calibration chose $c"
kill_sim
common env STAGES='3' ./run_placement_eft.sh > logs/q13_town04_grid_stage3.log 2>&1
n=$(ls $OUT/v6n__*__20-50m__s$S.json 2>/dev/null | wc -l)
grep -aE "\[Build\] sgm\[|MERGED NOTHING|\[Result\]" logs/q13_town04_grid_stage3.log | cut -c1-150
[ "$n" -ge 6 ] && { date +%H:%M > "$Q/budget10k_town04_grid_placement"; say "DONE ($n rows)"; } || say "INCOMPLETE ($n/6 rows)"
kill_sim
say "QUEUE 13 COMPLETE"
