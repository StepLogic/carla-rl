#!/bin/bash
# user 2026-09-17 "repeat seeds": runs 2 and 3 of every matched-node CARLA point that has a single run, so every cell of
# table_matched_nodes_all.tex has mean +/- sd over 3 runs of the same 40 episodes (as Table I). Town02 first, then Town07.
# v6n reruns the same maps (CARLA non-determinism is the variation); NoMaD uses --seed = run number.
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl; cd "$W" || exit 1
S=$W/scripts/icra2027_equal_points
say(){ echo "=== [rep23 $(date '+%m-%d %H:%M:%S')] $* ==="; }
die(){ say "FAILED: $*"; exit 1; }
count(){ python3 -c "import json,sys,collections; c=collections.Counter(json.loads(l)['map'] for l in open(sys.argv[1]) if l.strip()); print(min([c.get(m,0) for m in sys.argv[2:]]))" "$@" 2>/dev/null || echo 0; }
T7=results/placement_v6n_town07_budget10k_laneoff_speed10/trace_s7.pkl
TOWN07_ARGS="--town Town07 --trace $T7 --seq-dir results/town07/maps/sequence
  --episodes results/town07/episodes/town07_speed10_s7_episodes_short20_long20_seed0.pkl
  --views results/town07/episodes/town07_speed10_s7_episode_views_seed0.pkl"
LOG=logs/equal_points/launcher_rep23.log

v6n(){ # town out maps...
  local TOWN=$1 OUT=$2; shift 2
  [ "$(count $OUT "$@")" -eq 40 ] && { say "skip (done): $TOWN $OUT"; return 0; }
  say "$TOWN v6n: $* -> $OUT"
  if [ $TOWN = Town02 ]; then
    bash $S/run_drive_plan_c.sh $OUT logs/equal_points/$(basename $OUT .jsonl)_town02.log --maps "$@" >> $LOG 2>&1
  else
    bash $S/run_drive_plan_c.sh $OUT logs/equal_points/$(basename $OUT .jsonl)_town07.log $TOWN07_ARGS --maps "$@" >> $LOG 2>&1
  fi
  n=$(count $OUT "$@"); [ "$n" -eq 40 ] || die "$TOWN v6n $OUT min $n/40"
  say "$TOWN v6n $OUT done (40/40 per map)"
}
nomad(){ # town G run
  local TOWN=$1 G=$2 R=$3 NAME=nomad_uniform${2}_trace_v6ncap_20kmh
  if [ $TOWN = Town02 ]; then ND=results/nomad/equal_points; SEQ=results/maps/sequence; EP=results/nomad/episodes_short20_long20_seed0.json
    case $G in 6) CAP=results/drive_plan_c/sequence_undirected.jsonl;; *) CAP=results/drive_plan_c/uniform_sweep.jsonl;; esac
  else ND=results/town07/nomad/equal_points; SEQ=results/town07/maps/sequence; EP=results/town07/nomad/episodes_short20_long20_seed0.json
    case $G in 6) CAP=results/town07/drive_plan_c.jsonl;; *) CAP=results/town07/uniform_sweep.jsonl;; esac; fi
  local OUT=$ND/${NAME}_run$R.jsonl
  [ "$(count $OUT $NAME)" -eq 40 ] && { say "skip (done): $TOWN NoMaD[$G m] run $R"; return 0; }
  say "$TOWN NoMaD on Spacing[$G m], run $R -> $OUT"
  bash $S/run_drive_nomad.sh logs/equal_points/${NAME}_run${R}_$TOWN.log --town $TOWN --map-name $NAME --out $OUT \
    --node-poses $ND/uniform${G}_node_poses.json --seq-map $SEQ/uniform${G}_trace_sequence.json --episodes $EP \
    --image-dir $ND/images_uniform$G --target-speed-kmh 20 --cap-from $CAP --cap-map uniform${G}_trace --seed $R >> $LOG 2>&1
  n=$(count $OUT $NAME); [ "$n" -eq 40 ] || die "$TOWN NoMaD[$G m] run $R $n/40"
  say "$TOWN NoMaD[$G m] run $R done (40/40)"
}

T02="uniform15_trace hybrid_uniform15_entropy_thin4 hybrid_uniform8_entropy_thin4 hybrid_uniform2_entropy_thin4 sgm_g24_e55_c55 sgm_g10_e55_c55 sgm_g7_e55_c55"
T07="uniform15_trace uniform8_trace hybrid_uniform15_entropy_thin4 hybrid_uniform8_entropy_thin4 hybrid_uniform2_entropy_thin4 sgm_g20_e55_c55 sgm_g9.5_e55_c55 sgm_g6.5_e55_c55"

for R in 2 3; do
  M="$T02"; [ $R = 3 ] && M="$M uniform8_trace"   # Town02 Spacing[8 m] already has run 2 (uniform8_run2.jsonl)
  v6n Town02 results/drive_plan_c/equal_points/run$R.jsonl $M
  for G in 15 8 6; do nomad Town02 $G $R; done
done
say "TOWN02 RUNS 2-3 DONE"
for R in 2 3; do
  v6n Town07 results/town07/equal_points/run$R.jsonl $T07
  for G in 15 8 6; do nomad Town07 $G $R; done
done
say "REPEAT RUNS 2-3 ALL DONE"
