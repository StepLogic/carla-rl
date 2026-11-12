#!/bin/bash
# user 2026-09-19 "replace localization with dinov2", scope chosen: 3 arms x 4 budgets x 2 localiser modes.
# Spacing / SGM / Entropy+SGM at 60, 115, 150, 550 nodes, driven twice: once with the SAME MobileNet encoder
# reading the teleported RGB bank (isolates the capture change) and once with DINOv2 ViT-S/14 patch-mean.
# The shipped rows (mobilenet_trace) are untouched -- each mode writes its own jsonl.
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl; cd "$W" || exit 1
S=$W/scripts/icra2027_equal_points
say(){ echo "=== [loc $(date '+%m-%d %H:%M:%S')] $* ==="; }
die(){ say "FAILED: $*"; exit 1; }
count(){ python3 -c "import json,sys,collections; c=collections.Counter(json.loads(l)['map'] for l in open(sys.argv[1]) if l.strip()); print(min([c.get(m,0) for m in sys.argv[2:]]))" "$@" 2>/dev/null || echo 0; }

MAPS="uniform15_trace uniform8_trace uniform6_trace uniform2_trace \
sgm_g24_e55_c55 sgm_g10_e55_c55 sgm_g7_e55_c55 sgm_g2_e55_c55 \
hybrid_sgm_g44_entropy_thin4 hybrid_sgm_g14_entropy_thin4 hybrid_sgm_g9_entropy_thin4 hybrid_sgm_g2_entropy_thin4"

for MODE in mobilenet_bank dinov2_bank; do
  OUT=results/drive_plan_c/localiser/${MODE}_run1.jsonl
  if [ "$(count $OUT $MAPS)" -ne 40 ]; then
    say "$MODE: 12 maps x 40 episodes -> $OUT"
    bash $S/run_drive_plan_c.sh $OUT logs/dinov2/${MODE}_run1.log --maps $MAPS --localiser $MODE \
      >> logs/dinov2/launcher_localiser.log 2>&1
  fi
  n=$(count $OUT $MAPS); [ "$n" -eq 40 ] || die "$MODE min $n/40 per map"
  say "$MODE DONE (40/40 on all 12 maps)"
done
say "LOCALISER SWEEP DONE"
