#!/bin/bash
# QUEUE 16, 2026-09-13. USER: "run it whatever it takes" -- SGM on Town04.
# At the 2 m metric gate SGM merges nothing on any Town04 10k trace: candidates
# are 2.07 m apart, so a 2 m gate never reaches the previous kept node, and the
# trace never revisits itself (0% within 2 m). Measured on the city-grid trace
# (seed 9): gate 2 m keeps 1160/1160; 3 m 582-618; 4 m 555-609; 6 m 366-428.
# GATE 4 m = the placement arrival radius (4.0). This CHANGES SGM's gate for
# Town04 only (Town02 rows used 2 m) -- caption must say so.
# madent at z>=1.5 (user's Town04 threshold; z=2 gives 1 node here).
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl
cd "$W" || exit 1
source /home/robotlab/anaconda3/etc/profile.d/conda.sh
conda activate carla-rl || { echo FATAL; exit 1; }
set +u; eval "$(grep -E '^export ' run_gated.sh)"
E6N=results/run_env/carla_gated_her_v6n.env
while IFS='=' read -r k v; do case "$k" in ''|'#'*) continue;; esac; export "$k=$v"; done < "$E6N"
export CARLA_START_SERVER=1 CARLA_QUALITY=Epic CARLA_PORT=2000 CARLA_PORTS=2000
export EXPERT_PROB=0 RESUME=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export ENV_SKIP_IMAGE_FEATURES=1 PYTHONPATH=src
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia

Q="$W/results/queue_markers"; SRC=results/placement_v6n_town04_budget10k_grid; S=9; GATE=4.0; Z=1.5
OUT=results/placement_v6n_town04_grid_gate4
V6N=checkpoints/carla_gated_her_v6n/checkpoint_250000.pkl
say(){ echo "=== [Q16 $(date +%H:%M:%S)] $* ==="; }
kill_sim(){ for _ in 1 2 3 4 5 6; do pkill -9 -f 'Linux-Shippin[g]' 2>/dev/null; sleep 2; pgrep -f 'Linux-Shippin[g]' >/dev/null || break; done; sleep 45; }

mkdir -p "$OUT"; cp -n "$SRC/trace_s$S.pkl" "$SRC/trace_s$S.pkl.dist.npy" "$OUT/"
if [ ! -f "$OUT/calibration.json" ]; then
  E=$(python - "$OUT/trace_s$S.pkl" <<'EOF'
import sys, numpy as np
import calibrate_critic as cc
P, d = cc.load(sys.argv[1]); t = cc.separation_table(P, d)
base = float(np.median([t[b][0] for b in ("20-30", "30-50") if t[b][0] is not None]))
print(",".join(f"{round(base * f, 1)}" for f in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)))
EOF
)
  say "calibration, gate $GATE m, edge cutoffs $E"
  nice -n 10 python -u calibrate_critic.py --traces "$OUT/trace_s$S.pkl" --horizon-m 50 --gate-m "$GATE" \
      --edge-cutoffs "$E" --out "$OUT/calibration.json" > "$OUT/calibration.txt" 2>&1
fi
grep -aE "CHOSEN|NO setting" "$OUT/calibration.txt" | cut -c1-220
c=$(python -c "import json;c=json.load(open('$OUT/calibration.json')).get('chosen');print(c if c else '')" 2>/dev/null)
[ -n "$c" ] || { say "STOP: calibration chose nothing at gate $GATE m -- needs a decision"; exit 1; }

for ATTEMPT in 1 2 3; do
  say "build + navigate (sgm gate $GATE m, madent z>=$Z, matched controls), attempt $ATTEMPT"
  kill_sim
  python -u eval_placement.py --ckpt "$V6N" --arm v6n --town Town04 --episodes 60 \
      --goal-min 20 --goal-max 50 --port 2000 --step-cost 1.0 --min-sep-m 2.0 --sgm-gate-m "$GATE" \
      --mad-z "$Z" --out-dir "$OUT" --collect-seed "$S" --collect-total-steps 10000 \
      --trace-cache "$OUT/trace_s$S.pkl" --calibration "$OUT/calibration.json" \
      --methods sgm,madent,uniform,random --match-to sgm,madent --episodes-from-trace \
      --tag "__s$S" >> logs/q16_town04_gate4.log 2>&1
  n=$(ls "$OUT"/v6n__*__20-50m__s$S.json 2>/dev/null | wc -l)
  [ "$n" -ge 6 ] && break
  say "$n/6 rows after attempt $ATTEMPT"
done
grep -aE "\[Build\] sgm\[|MERGED NOTHING|\[Build\] madent \(|\[Result\]" logs/q16_town04_gate4.log | tail -8 | cut -c1-160
n=$(ls "$OUT"/v6n__*__20-50m__s$S.json 2>/dev/null | wc -l)
[ "$n" -ge 6 ] && { date +%H:%M > "$Q/town04_grid_sgm_gate4"; say "DONE ($n rows)"; } || say "INCOMPLETE ($n/6 rows)"
kill_sim
say "QUEUE 16 COMPLETE"
