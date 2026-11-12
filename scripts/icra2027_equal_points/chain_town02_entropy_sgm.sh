#!/bin/bash
# user 2026-09-19: "we have madent + uniform spacing, how about madent + sgm".
# Town02, RUN 1 ONLY: v6n@250k on the four Entropy + SGM maps (madent seeds pinned, SGM frame for the rest),
# count-matched to the existing blocks at 59 / 115 / 150 / 587 nodes.
# Waits for the OTHER session's image-goal eval to release CARLA port 2000 before touching the simulator.
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl; cd "$W" || exit 1
S=$W/scripts/icra2027_equal_points
say(){ echo "=== [ent-sgm $(date '+%m-%d %H:%M:%S')] $* ==="; }
die(){ say "FAILED: $*"; exit 1; }
count(){ python3 -c "import json,sys,collections; c=collections.Counter(json.loads(l)['map'] for l in open(sys.argv[1]) if l.strip()); print(min([c.get(m,0) for m in sys.argv[2:]]))" "$@" 2>/dev/null || echo 0; }

while pgrep -f "image_goal_video_eval" >/dev/null; do say "waiting: another session's image-goal eval holds port 2000"; sleep 60; done
say "port 2000 free"

OUT=results/drive_plan_c/equal_points/run1.jsonl
MAPS="hybrid_sgm_g44_entropy_thin4 hybrid_sgm_g14_entropy_thin4 hybrid_sgm_g9_entropy_thin4 hybrid_sgm_g2_entropy_thin4"
if [ "$(count $OUT $MAPS)" -ne 40 ]; then
  say "Town02 v6n run 1: $MAPS"
  bash $S/run_drive_plan_c.sh $OUT logs/equal_points/run1_town02_entropy_sgm.log --maps $MAPS \
    >> logs/equal_points/launcher_entropy_sgm.log 2>&1
fi
n=$(count $OUT $MAPS); [ "$n" -eq 40 ] || die "min $n/40 per map"
say "TOWN02 ENTROPY+SGM RUN 1 DONE (40/40 per map)"
