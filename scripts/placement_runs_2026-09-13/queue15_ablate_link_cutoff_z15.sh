#!/bin/bash
# QUEUE 15, 2026-09-13. USER: "ablate link cutoff" on Town04, madent at
# z>=1.5 (57 nodes). queue14 (z=2) was stopped at 01:58 after madent placed
# ONE node on this trace, making every arm a 1-node graph. Town04 city-grid
# trace (v6n@250k, seed 9, 10k continued). SGM cannot thin it (0% revisit
# within 2 m), so the arms are madent and its count-matched uniform/random.
# madent's nodes come from actor entropy, not the critic, so the node count --
# and the matched counts -- are FIXED across cutoffs: only edges change.
# Cutoffs = band median 53.2 steps (20-50 m) x 0.5, 1, 1.5, 2, 3.
# NOTE: Town02 madent rows use z=2; this Town04 ablation uses z=1.5 (user).
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

Q="$W/results/queue_markers"; SRC=results/placement_v6n_town04_budget10k_grid; S=9; Z=1.5
ROOT=results/ablate_linkcutoff_town04_grid_z1.5
V6N=checkpoints/carla_gated_her_v6n/checkpoint_250000.pkl
say(){ echo "=== [Q15 $(date +%H:%M:%S)] $* ==="; }
kill_sim(){ for _ in 1 2 3 4 5 6; do pkill -9 -f 'Linux-Shippin[g]' 2>/dev/null; sleep 2; pgrep -f 'Linux-Shippin[g]' >/dev/null || break; done; sleep 45; }

[ -f "$SRC/trace_s$S.pkl.dist.npy" ] || { say "FATAL: no source trace/matrix in $SRC"; exit 1; }
for E in 26.6 53.2 79.8 106.5 159.7; do
  M="ablate_linkcutoff_town04_z${Z}_e$E"
  [ -f "$Q/$M" ] && { say "skip $M"; continue; }
  OUT="$ROOT/e$E"; mkdir -p "$OUT"
  cp -n "$SRC/trace_s$S.pkl" "$SRC/trace_s$S.pkl.dist.npy" "$OUT/"
  for ATTEMPT in 1 2 3; do
    say "START cutoff $E steps (madent z>=$Z), attempt $ATTEMPT -> logs/q15_e$E.log"
    kill_sim
    python -u eval_placement.py --ckpt "$V6N" --arm v6n --town Town04 --episodes 60 \
        --goal-min 20 --goal-max 50 --port 2000 --step-cost 1.0 --min-sep-m 2.0 --sgm-gate-m 2.0 \
        --mad-z "$Z" --out-dir "$OUT" --collect-seed "$S" --collect-total-steps 10000 \
        --trace-cache "$OUT/trace_s$S.pkl" --max-edge-steps "$E" \
        --methods madent,uniform,random --match-to madent --episodes-from-trace \
        --tag "__s$S" >> "logs/q15_e$E.log" 2>&1
    n=$(ls "$OUT"/v6n__*__20-50m__s$S.json 2>/dev/null | wc -l)
    [ "$n" -ge 3 ] && break
    say "cutoff $E: $n/3 rows after attempt $ATTEMPT"
  done
  grep -aE "\[Build\] madent \(MAD|\[Build\] (madent|uniform@madent|random@madent):|\[Result\]" "logs/q15_e$E.log" | tail -7 | cut -c1-160
  n=$(ls "$OUT"/v6n__*__20-50m__s$S.json 2>/dev/null | wc -l)
  [ "$n" -ge 3 ] && { date +%H:%M > "$Q/$M"; say "DONE cutoff $E ($n rows)"; } || say "FAILED cutoff $E ($n/3 rows)"
done
kill_sim
say "QUEUE 15 COMPLETE"
