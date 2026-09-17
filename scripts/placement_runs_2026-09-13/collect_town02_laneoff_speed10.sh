#!/bin/bash
# USER 2026-09-13: recollect Town02 matching the latest (laneoff) trace exactly,
# with trace-collection TARGET_SPEED reduced to 10 km/h (laneoff used 20).
# v6n@250k, seed 7, 10,000 steps, no time limit, episodes end on collision /
# lane crossing / off-road, continue from the nearest waypoint. Trace + matrix.
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl
cd "$W" || exit 1
source /home/robotlab/anaconda3/etc/profile.d/conda.sh
conda activate carla-rl || { echo FATAL; exit 1; }
export ENV_SKIP_IMAGE_FEATURES=1 PYTHONPATH=src
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia
OUT=results/placement_v6n_town02_budget10k_laneoff_speed10; REF=results/placement_v6n_town02_budget10k_laneoff; S=7
say(){ echo "=== [speed10 $(date +%H:%M:%S)] $* ==="; }
for _ in 1 2 3 4 5 6; do pkill -9 -f 'Linux-Shippin[g]' 2>/dev/null; sleep 2; pgrep -f 'Linux-Shippin[g]' >/dev/null || break; done
sleep 45

say "collect Town02 seed $S, 10k steps, ends on collision/lane/off_road, COLLECT_TARGET_SPEED=10 -> $OUT"
RUN_ENV=results/run_env/carla_gated_her_v6n.env MAD_Z=2 CKPT=checkpoints/carla_gated_her_v6n/checkpoint_250000.pkl \
  ARM=v6n TOWN=Town02 OUT=$OUT SEEDS="$S" COLLECT_TOTAL_STEPS=10000 COLLECT_END_ON="collision,lane,off_road" \
  COLLECT_TARGET_SPEED=10 EPISODES=60 STAGES='1' EXPLORER=policy_explore LANE_RULE=strict \
  ./run_placement_eft.sh > logs/collect_town02_laneoff_speed10_stage1.log 2>&1
[ -f "$OUT/trace_s$S.pkl.dist.npy" ] || { say "FAILED: no trace matrix (logs/collect_town02_laneoff_speed10_stage1.log)"; exit 1; }

say "trace check: 10 km/h vs the 20 km/h laneoff trace"
python - "$OUT/trace_s$S.pkl" "$REF/trace_s$S.pkl" \
  logs/collect_town02_laneoff_speed10_stage1.log logs/collect_town02_laneoff_stage1.log <<'EOF'
import sys, re, pickle, collections, numpy as np
from scipy.spatial import cKDTree
def stats(p, log):
    tr = pickle.load(open(p, "rb")); P = np.asarray(tr["positions"])[:, :2]; seg = np.asarray(tr["segments"])
    tree = cKDTree(P); n = len(P)
    rev2 = sum(1 for i, nb in enumerate(tree.query_ball_point(P, 2.0)) if any(abs(j - i) > 10 for j in nb))
    ks = np.unique(seg)
    g = [np.linalg.norm(P[seg == b][0] - P[seg == a][-1]) for a, b in zip(ks[:-1], ks[1:])]
    cells = len({tuple(c) for c in np.floor(P / 10).astype(int)})
    eps = re.findall(r"episode \d+ \(\d+/10000 steps\): (\d+) steps, ([0-9.]+) m driven, ended=([a-z_,]+)", open(log).read())
    steps = sum(int(s) for s, _, _ in eps); dist = sum(float(d) for _, d, _ in eps)
    ends = collections.Counter(e for _, _, e in eps)
    return dict(speed=tr["meta"].get("target_speed_kmh"), end_on=tr["meta"].get("collect_end_on"),
                episodes=len(ks), candidates=n, gap=np.median(g) if g else float("nan"),
                rev2=100 * rev2 / n, cells=cells, m_per_step=dist / max(steps, 1),
                kmh=3.6 * dist / max(steps, 1) / 0.1, ends=dict(ends))
new, ref = stats(sys.argv[1], sys.argv[3]), stats(sys.argv[2], sys.argv[4])
for k in ("speed", "end_on", "episodes", "candidates", "gap", "rev2", "cells", "m_per_step", "kmh", "ends"):
    f = (lambda v: f"{v:.2f}") if k in ("gap", "rev2", "m_per_step", "kmh") else str
    print(f"{k:>11}: 10 km/h trace {f(new[k])}  |  20 km/h trace {f(ref[k])}")
EOF
python render_placement_maps.py --run-dir $OUT --town Town02 --no-graph --full-map \
  --out results/figures/exploration_traces_10k_fullmap_laneoff_speed10 2>&1 | grep -a Render
for _ in 1 2 3 4 5 6; do pkill -9 -f 'Linux-Shippin[g]' 2>/dev/null; sleep 2; pgrep -f 'Linux-Shippin[g]' >/dev/null || break; done
say "DONE"
