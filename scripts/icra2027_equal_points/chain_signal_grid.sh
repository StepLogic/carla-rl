#!/bin/bash
# user 2026-09-20: "queue epistemic vs aleatoric vs policy entropy + sgm or plus uniform".
# The 3 x 2 grid (uncertainty signal x filler) at four matched budgets, Town02, all under the DINOv2
# localiser so the six arms are directly comparable. Policy-entropy + SGM is already driven under DINOv2
# (results/drive_plan_c/localiser/dinov2_bank_run1.jsonl), so it is not repeated here.
# Seeds are COUNT-MATCHED at 30 per signal -- damage tracks the seed share of the budget, so an unmatched
# seed count would measure the count, not the signal. The three seed sets share 0 of 30 trace samples.
# WAITS for the follow-up chain (and any peer trainer) to release the simulator.
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl; cd "$W" || exit 1
S=$W/scripts/icra2027_equal_points
say(){ echo "=== [grid $(date '+%m-%d %H:%M:%S')] $* ==="; }
die(){ say "FAILED: $*"; exit 1; }
count(){ python3 -c "import json,sys,collections; c=collections.Counter(json.loads(l)['map'] for l in open(sys.argv[1]) if l.strip()); print(min([c.get(m,0) for m in sys.argv[2:]]))" "$@" 2>/dev/null || echo 0; }

while pgrep -f "chain_dinov2_followup" >/dev/null || pgrep -f "train_carla_tqc" >/dev/null; do
  say "holding: follow-up chain or a peer trainer still has the simulator"; sleep 300
done
sleep 120
pgrep -f "train_carla_tqc" >/dev/null && die "a trainer came back -- not taking the simulator"
say "simulator free"

UNIFORM="hybrid_uniform15_entropy_thin4 hybrid_uniform8_entropy_thin4 hybrid_uniform6_entropy_thin4 hybrid_uniform2_entropy_thin4 \
hybrid_uniform15_epistemic_thin4 hybrid_uniform8_epistemic_thin4 hybrid_uniform6_epistemic_thin4 hybrid_uniform2_epistemic_thin4 \
hybrid_uniform15_aleatoric_thin4 hybrid_uniform8_aleatoric_thin4 hybrid_uniform6_aleatoric_thin4 hybrid_uniform2_aleatoric_thin4"
SGM="hybrid_sgm_g44_epistemic_thin4 hybrid_sgm_g13_epistemic_thin4 hybrid_sgm_g9_epistemic_thin4 hybrid_sgm_g2_epistemic_thin4 \
hybrid_sgm_g44_aleatoric_thin4 hybrid_sgm_g13_aleatoric_thin4 hybrid_sgm_g9_aleatoric_thin4 hybrid_sgm_g2_aleatoric_thin4"

OUT=results/drive_plan_c/signal_grid/dinov2_run1.jsonl
for STAGE in "uniform:$UNIFORM" "sgm:$SGM"; do
  NAME=${STAGE%%:*}; MAPS=${STAGE#*:}
  if [ "$(count $OUT $MAPS)" -ne 40 ]; then
    say "$NAME filler: $(echo $MAPS | wc -w) maps x 40 episodes"
    bash $S/run_drive_plan_c.sh $OUT logs/dinov2/signal_grid_${NAME}.log --maps $MAPS --localiser dinov2_bank \
      >> logs/dinov2/launcher_signal_grid.log 2>&1
  fi
  n=$(count $OUT $MAPS); [ "$n" -eq 40 ] || die "$NAME filler min $n/40"
  say "$NAME filler DONE"
done
say "SIGNAL GRID DONE"
