#!/bin/bash
# user 2026-09-17 "Run town07": fill the Town07 matched-node grid, run 1 only (same design as chain_town02_run1.sh).
# v6n on Entropy + Spacing[2 m] (804) and SGM g20 / g9.5 / g6.5 (95 / 189 / 244), then NoMaD on Spacing[15/8/6 m].
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl; cd "$W" || exit 1
S=$W/scripts/icra2027_equal_points
say(){ echo "=== [town07 $(date '+%m-%d %H:%M:%S')] $* ==="; }
die(){ say "FAILED: $*"; exit 1; }
count(){ python3 -c "import json,sys,collections; c=collections.Counter(json.loads(l)['map'] for l in open(sys.argv[1]) if l.strip()); print(min([c.get(m,0) for m in sys.argv[2:]]))" "$@" 2>/dev/null || echo 0; }

T7=results/placement_v6n_town07_budget10k_laneoff_speed10/trace_s7.pkl
mkdir -p results/town07/equal_points logs/equal_points
OUT=results/town07/equal_points/run1.jsonl
MAPS="hybrid_uniform2_entropy_thin4 sgm_g20_e55_c55 sgm_g9.5_e55_c55 sgm_g6.5_e55_c55"
if [ "$(count $OUT $MAPS)" -ne 40 ]; then
  say "Town07 v6n: $MAPS"
  bash $S/run_drive_plan_c.sh $OUT logs/equal_points/run1_town07.log --town Town07 --trace $T7 \
    --seq-dir results/town07/maps/sequence \
    --episodes results/town07/episodes/town07_speed10_s7_episodes_short20_long20_seed0.pkl \
    --views results/town07/episodes/town07_speed10_s7_episode_views_seed0.pkl \
    --maps $MAPS >> logs/equal_points/launcher_town07.log 2>&1
fi
n=$(count $OUT $MAPS); [ "$n" -eq 40 ] || die "Town07 v6n min $n/40"
say "Town07 v6n done (40/40 per map)"

N7=results/town07/nomad/equal_points
for G in 15 8 6; do
  NAME=nomad_uniform${G}_trace_v6ncap_20kmh; NOUT=$N7/${NAME}_run1.jsonl
  case $G in 6) CAP=results/town07/drive_plan_c.jsonl;; *) CAP=results/town07/uniform_sweep.jsonl;; esac
  if [ "$(count $NOUT $NAME)" -ne 40 ]; then
    say "Town07 NoMaD on Spacing[$G m] -> $NOUT"
    bash $S/run_drive_nomad.sh logs/equal_points/${NAME}_run1_Town07.log --town Town07 --map-name $NAME --out $NOUT \
      --node-poses $N7/uniform${G}_node_poses.json --seq-map results/town07/maps/sequence/uniform${G}_trace_sequence.json \
      --episodes results/town07/nomad/episodes_short20_long20_seed0.json --image-dir $N7/images_uniform$G \
      --target-speed-kmh 20 --cap-from $CAP --cap-map uniform${G}_trace --seed 1 >> logs/equal_points/launcher_town07.log 2>&1
  fi
  n=$(count $NOUT $NAME); [ "$n" -eq 40 ] || die "Town07 NoMaD[$G m] $n/40"
  say "Town07 NoMaD[$G m] done (40/40)"
done
say "TOWN07 RUN 1 ALL DONE"
