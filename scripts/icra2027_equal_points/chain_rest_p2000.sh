#!/bin/bash
# user 2026-09-20: finish the grid with a launcher that only kills its OWN port-2000 CARLA, so the peer's
# image-goal training (now on port 2010) is never touched. Waits for the in-flight drive to finish first.
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl; cd "$W" || exit 1
S=$W/scripts/icra2027_equal_points
P=$S/run_drive_plan_c_p2000.sh
say(){ echo "=== [rest $(date '+%m-%d %H:%M:%S')] $* ==="; }
die(){ say "FAILED: $*"; exit 1; }
count(){ python3 -c "import json,sys,collections; c=collections.Counter(json.loads(l)['map'] for l in open(sys.argv[1]) if l.strip()); print(min([c.get(m,0) for m in sys.argv[2:]]))" "$@" 2>/dev/null || echo 0; }

while pgrep -f "compat_v6n/drive_plan_c\.py" >/dev/null; do say "waiting for the in-flight drive"; sleep 60; done
say "in-flight drive finished"

SGM="hybrid_sgm_g44_epistemic_thin4 hybrid_sgm_g13_epistemic_thin4 hybrid_sgm_g9_epistemic_thin4 hybrid_sgm_g2_epistemic_thin4 \
hybrid_sgm_g44_aleatoric_thin4 hybrid_sgm_g13_aleatoric_thin4 hybrid_sgm_g9_aleatoric_thin4 hybrid_sgm_g2_aleatoric_thin4"
GOUT=results/drive_plan_c/signal_grid/dinov2_run1.jsonl
if [ "$(count $GOUT $SGM)" -ne 40 ]; then
  say "signal grid, sgm filler (resume)"
  bash $P $GOUT logs/dinov2/signal_grid_sgm_pinned.log --maps $SGM --localiser dinov2_bank \
    >> logs/dinov2/launcher_p2000.log 2>&1
fi
n=$(count $GOUT $SGM); [ "$n" -eq 40 ] || die "signal grid sgm min $n/40"
say "signal grid sgm DONE"

BMAPS="raw_trace entropy_nosmooth_high_thin4 epistemic_thin4_n30 aleatoric_thin4_n30"
BOUT=results/drive_plan_c/signal_grid/dinov2_baselines_run1.jsonl
if [ "$(count $BOUT $BMAPS)" -ne 40 ]; then
  say "grid baselines under DINOv2"
  bash $P $BOUT logs/dinov2/signal_grid_baselines_pinned.log --maps $BMAPS --localiser dinov2_bank \
    >> logs/dinov2/launcher_p2000.log 2>&1
fi
n=$(count $BOUT $BMAPS); [ "$n" -eq 40 ] || die "baselines min $n/40"
say "grid baselines DONE"
say "REST CHAIN DONE"
