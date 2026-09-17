#!/bin/bash
# user 2026-09-16 22:45 "we do not have the time - stick to one town": replaces chain_equal_points.sh.
# Town02 only, run 1 only: v6n on the new Entropy + Spacing[2 m] and SGM g24/g10/g7 maps (already driving when this
# was written), then NoMaD on Spacing[15/8/6 m]. Waits for the orphaned v6n launcher of the old chain to exit first.
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl; cd "$W" || exit 1
S=$W/scripts/icra2027_equal_points
say(){ echo "=== [town02 $(date '+%m-%d %H:%M:%S')] $* ==="; }
die(){ say "FAILED: $*"; exit 1; }
count(){ python3 -c "import json,sys,collections; c=collections.Counter(json.loads(l)['map'] for l in open(sys.argv[1]) if l.strip()); print(min([c.get(m,0) for m in sys.argv[2:]]))" "$@" 2>/dev/null || echo 0; }

while pgrep -f "icra2027_equal_points/run_drive_plan_c.sh" >/dev/null; do sleep 20; done
say "old v6n launcher gone"

OUT=results/drive_plan_c/equal_points/run1.jsonl
NEW02="hybrid_uniform2_entropy_thin4 sgm_g24_e55_c55 sgm_g10_e55_c55 sgm_g7_e55_c55"
if [ "$(count $OUT $NEW02)" -ne 40 ]; then
  say "Town02 v6n resume: $NEW02"
  bash $S/run_drive_plan_c.sh $OUT logs/equal_points/run1_town02.log --maps $NEW02 >> logs/equal_points/launcher.log 2>&1
fi
n=$(count $OUT $NEW02); [ "$n" -eq 40 ] || die "Town02 v6n min $n/40"
say "Town02 v6n done (40/40 per map)"

N2=results/nomad/equal_points
for G in 15 8 6; do
  NAME=nomad_uniform${G}_trace_v6ncap_20kmh; NOUT=$N2/${NAME}_run1.jsonl
  case $G in 6) CAP=results/drive_plan_c/sequence_undirected.jsonl;; *) CAP=results/drive_plan_c/uniform_sweep.jsonl;; esac
  if [ "$(count $NOUT $NAME)" -ne 40 ]; then
    say "Town02 NoMaD on Spacing[$G m] -> $NOUT"
    bash $S/run_drive_nomad.sh logs/equal_points/${NAME}_run1_Town02.log --town Town02 --map-name $NAME --out $NOUT \
      --node-poses $N2/uniform${G}_node_poses.json --seq-map results/maps/sequence/uniform${G}_trace_sequence.json \
      --episodes results/nomad/episodes_short20_long20_seed0.json --image-dir $N2/images_uniform$G \
      --target-speed-kmh 20 --cap-from $CAP --cap-map uniform${G}_trace --seed 1 >> logs/equal_points/launcher.log 2>&1
  fi
  n=$(count $NOUT $NAME); [ "$n" -eq 40 ] || die "Town02 NoMaD[$G m] $n/40"
  say "Town02 NoMaD[$G m] done (40/40)"
done
say "TOWN02 RUN 1 ALL DONE"
