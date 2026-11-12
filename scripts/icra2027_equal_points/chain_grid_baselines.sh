#!/bin/bash
# user 2026-09-20 "add to baselines and ablate SR": the reference arms the grid is judged against, driven
# under the SAME localiser as the grid (DINOv2), so the ablation table never compares across localisers.
#   Dense (every trace sample) and the three SEED-ONLY maps (30 nodes each, no filler) -- the latter isolate
#   the signal with no spacing or SGM help at all, which is the cleanest read of the signal itself.
# WAITS for the signal-grid chain (and any peer trainer) to release the simulator.
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl; cd "$W" || exit 1
S=$W/scripts/icra2027_equal_points
say(){ echo "=== [base $(date '+%m-%d %H:%M:%S')] $* ==="; }
die(){ say "FAILED: $*"; exit 1; }
count(){ python3 -c "import json,sys,collections; c=collections.Counter(json.loads(l)['map'] for l in open(sys.argv[1]) if l.strip()); print(min([c.get(m,0) for m in sys.argv[2:]]))" "$@" 2>/dev/null || echo 0; }

while pgrep -f "chain_signal_grid" >/dev/null || pgrep -f "chain_dinov2_followup" >/dev/null \
   || pgrep -f "train_carla_tqc" >/dev/null; do
  say "holding: an earlier chain or a peer trainer still has the simulator"; sleep 300
done
sleep 120
pgrep -f "train_carla_tqc" >/dev/null && die "a trainer came back -- not taking the simulator"
say "simulator free"

MAPS="raw_trace entropy_nosmooth_high_thin4 epistemic_thin4_n30 aleatoric_thin4_n30"
OUT=results/drive_plan_c/signal_grid/dinov2_baselines_run1.jsonl
if [ "$(count $OUT $MAPS)" -ne 40 ]; then
  say "baselines under DINOv2: $MAPS"
  bash $S/run_drive_plan_c.sh $OUT logs/dinov2/signal_grid_baselines.log --maps $MAPS --localiser dinov2_bank \
    >> logs/dinov2/launcher_signal_grid.log 2>&1
fi
n=$(count $OUT $MAPS); [ "$n" -eq 40 ] || die "baselines min $n/40"
say "GRID BASELINES DONE"
