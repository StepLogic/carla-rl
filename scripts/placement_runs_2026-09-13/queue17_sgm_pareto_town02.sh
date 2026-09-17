#!/bin/bash
# QUEUE 17, 2026-09-13. USER: "Just show that SGM's Pareto optimal node count".
# SGM ONLY, Town02 10k continued trace (v6n@250k, seed 7, gate 2 m), edge cutoff
# fixed at the calibrated 61.6; the consistency threshold sets the node count.
# Measured in the calibration sweep at edge 61.6 (reach >= 0.991 everywhere):
#   cons 5.1 -> 1147 | 10.2 -> 972 | 15.4 -> 845 | 25.7 -> 748 | 41.1 -> 681
#   61.6 -> 644 | 123.2 -> 622 (already run: 12/60, reused)
# SGM cannot go below ~620 nodes on this trace at any cutoff (the 2 m gate never
# merges a once-driven stretch). Same 60 episodes (seed 0, 20-50 m, arrival 4.0).
# Waits for queue16 (Town04 SGM gate 4 m) so the two never share CARLA.
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

Q="$W/results/queue_markers"; SRC=results/placement_v6n_town02_budget10k; S=7; EDGE=61.6
ROOT=results/pareto_sgm_town02_budget10k
V6N=checkpoints/carla_gated_her_v6n/checkpoint_250000.pkl
say(){ echo "=== [Q17 $(date +%H:%M:%S)] $* ==="; }
kill_sim(){ for _ in 1 2 3 4 5 6; do pkill -9 -f 'Linux-Shippin[g]' 2>/dev/null; sleep 2; pgrep -f 'Linux-Shippin[g]' >/dev/null || break; done; sleep 45; }
busy(){ pgrep -f 'queue16_town04_sgm_gate[4]' >/dev/null || pgrep -f '[e]val_placemen[t]' >/dev/null || pgrep -f '[c]alibrate_criti[c]' >/dev/null; }

say "armed; waiting for queue16 (Town04 SGM) to finish"
while busy; do sleep 60; done
mkdir -p "$ROOT/c123.2"
cp -n "$SRC/v6n__sgm__20-50m__s$S.json" "$ROOT/c123.2/" 2>/dev/null   # the calibrated point, already run

for C in 5.1 10.2 15.4 25.7 41.1 61.6; do
  M="pareto_sgm_town02_c$C"
  [ -f "$Q/$M" ] && { say "skip $M"; continue; }
  OUT="$ROOT/c$C"; mkdir -p "$OUT"
  cp -n "$SRC/trace_s$S.pkl" "$SRC/trace_s$S.pkl.dist.npy" "$OUT/"
  for ATTEMPT in 1 2 3; do
    say "START sgm[$C/$EDGE], attempt $ATTEMPT -> logs/q17_c$C.log"
    kill_sim
    python -u eval_placement.py --ckpt "$V6N" --arm v6n --town Town02 --episodes 60 \
        --goal-min 20 --goal-max 50 --port 2000 --step-cost 1.0 --min-sep-m 2.0 --sgm-gate-m 2.0 \
        --mad-z 2 --out-dir "$OUT" --collect-seed "$S" --collect-total-steps 10000 \
        --trace-cache "$OUT/trace_s$S.pkl" --sgm-edge-cutoff "$EDGE" --sgm-consistency "$C" \
        --methods sgm --match-to sgm --episodes-from-trace --tag "__s$S" >> "logs/q17_c$C.log" 2>&1
    [ -f "$OUT/v6n__sgm__20-50m__s$S.json" ] && break
    say "sgm[$C/$EDGE]: no row after attempt $ATTEMPT"
  done
  grep -aE "\[Build\] sgm\[|\[Result\]" "logs/q17_c$C.log" | tail -2 | cut -c1-150
  [ -f "$OUT/v6n__sgm__20-50m__s$S.json" ] && { date +%H:%M > "$Q/$M"; say "DONE sgm[$C/$EDGE]"; } || say "FAILED sgm[$C/$EDGE]"
done
kill_sim
python -u plot_sgm_pareto.py --root "$ROOT" --seed "$S" --town Town02 2>&1 | grep -avE "^WARNING"
say "QUEUE 17 COMPLETE"
