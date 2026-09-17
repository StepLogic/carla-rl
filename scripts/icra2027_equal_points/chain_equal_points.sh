#!/bin/bash
# user 2026-09-17 "i need equal number of datapoints - run experiments and add it"; "dense as a line - entropy as a line".
# Four node budgets per town, set by the Spacing sweep (15 / 8 / 6 / 2 m), for Spacing, Entropy + Spacing, SGM, NoMaD:
#   Town02 Spacing 58/116/156/547 | E+S 63/113/149/528 | SGM g24/g10/g7/g2 = 57/122/149/570 | NoMaD on the Spacing maps
#   Town07 Spacing 97/191/253/813 | E+S 101/189/250/804 | SGM g20/g9.5/g6.5/g2 = 95/189/244/836 | NoMaD on the Spacing maps
# SGM keeps edge 55 / consistency 55 (the Table I setting) and varies only the metric gate to hit the node count.
# Same 40 episodes, option C, v6n@250k, NoMaD with the v6n map's step caps at 20 km/h, seed = run number.
# Stage A = run 1 of every new point; stage B = runs 2 and 3 of every grid point that has fewer than 3.
# Every stage checks 40 episodes per map in its output before the next starts; rerunning resumes (drivers skip done rows).
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl; cd "$W" || exit 1
S=$W/scripts/icra2027_equal_points
say(){ echo "=== [equal $(date '+%m-%d %H:%M:%S')] $* ==="; }
die(){ say "FAILED: $*"; exit 1; }
count(){ python3 -c "import json,sys,collections; c=collections.Counter(json.loads(l)['map'] for l in open(sys.argv[1]) if l.strip()); print(min([c.get(m,0) for m in sys.argv[2:]]))" "$@" 2>/dev/null || echo 0; }

T7=results/placement_v6n_town07_budget10k_laneoff_speed10/trace_s7.pkl
TOWN07_ARGS="--town Town07 --trace $T7 --seq-dir results/town07/maps/sequence
  --episodes results/town07/episodes/town07_speed10_s7_episodes_short20_long20_seed0.pkl
  --views results/town07/episodes/town07_speed10_s7_episode_views_seed0.pkl"
O2=results/drive_plan_c/equal_points; O7=results/town07/equal_points
N2=results/nomad/equal_points;        N7=results/town07/nomad/equal_points
mkdir -p $O2 $O7 $N2 $N7 logs/equal_points

# ---- NoMaD node poses for the Spacing[15/8/6 m] maps (offline, from each sequence map + trace yaw) -------------------
poses(){ # seq_map trace out
  python3 - "$@" <<'EOF'
import json, pickle, sys, numpy as np
seq, trace, out = sys.argv[1:4]
m = json.load(open(seq)); yaw = np.asarray(pickle.load(open(trace, "rb"))["yaws"])
nodes = [{k: n[k] for k in ("id", "trace_index", "x_m", "y_m")} | {"yaw_rad": float(yaw[int(n["trace_index"])])} for n in m["nodes"]]
json.dump({"source_map": seq, "nodes": nodes}, open(out, "w"), indent=1); print("[nomad] node poses", len(nodes), "->", out)
EOF
}
T2=results/placement_v6n_town02_budget10k_laneoff_speed10/trace_s7.pkl
for G in 15 8 6; do
  [ -f $N2/uniform${G}_node_poses.json ] || poses results/maps/sequence/uniform${G}_trace_sequence.json $T2 $N2/uniform${G}_node_poses.json
  [ -f $N7/uniform${G}_node_poses.json ] || poses results/town07/maps/sequence/uniform${G}_trace_sequence.json $T7 $N7/uniform${G}_node_poses.json
  # goal images are per episode, identical for every map: reuse the Spacing[2 m] captures
  for D in $N2/images_uniform$G:results/nomad/images $N7/images_uniform$G:results/town07/nomad/images; do
    mkdir -p ${D%%:*}/nodes; [ -d ${D%%:*}/goals ] || cp -r ${D##*:}/goals ${D%%:*}/goals
  done
done

# where run 1 of each Spacing map was driven by v6n (NoMaD reads its per-episode step caps from there)
cap2(){ case $1 in 6) echo results/drive_plan_c/sequence_undirected.jsonl;; *) echo results/drive_plan_c/uniform_sweep.jsonl;; esac; }
cap7(){ case $1 in 6) echo results/town07/drive_plan_c.jsonl;; *) echo results/town07/uniform_sweep.jsonl;; esac; }

v6n02(){ # out maps...
  local OUT=$1; shift
  [ "$(count $OUT "$@")" -eq 40 ] && { say "skip (done): Town02 v6n $* -> $OUT"; return 0; }
  say "Town02 v6n: $* -> $OUT"
  bash $S/run_drive_plan_c.sh $OUT logs/equal_points/$(basename $OUT .jsonl)_town02.log --maps "$@" \
    >> logs/equal_points/launcher.log 2>&1
  n=$(count $OUT "$@"); [ "$n" -eq 40 ] || die "Town02 v6n $OUT min $n/40"
}
v6n07(){
  local OUT=$1; shift
  [ "$(count $OUT "$@")" -eq 40 ] && { say "skip (done): Town07 v6n $* -> $OUT"; return 0; }
  say "Town07 v6n: $* -> $OUT"
  bash $S/run_drive_plan_c.sh $OUT logs/equal_points/$(basename $OUT .jsonl)_town07.log $TOWN07_ARGS --maps "$@" \
    >> logs/equal_points/launcher.log 2>&1
  n=$(count $OUT "$@"); [ "$n" -eq 40 ] || die "Town07 v6n $OUT min $n/40"
}
nomad(){ # town G run
  local TOWN=$1 G=$2 R=$3 NAME=nomad_uniform${2}_trace_v6ncap_20kmh
  if [ $TOWN = Town02 ]; then ND=$N2; SEQ=results/maps/sequence; EP=results/nomad/episodes_short20_long20_seed0.json; CAP=$(cap2 $G)
  else ND=$N7; SEQ=results/town07/maps/sequence; EP=results/town07/nomad/episodes_short20_long20_seed0.json; CAP=$(cap7 $G); fi
  local OUT=$ND/${NAME}_run$R.jsonl
  [ "$(count $OUT $NAME)" -eq 40 ] && { say "skip (done): $TOWN NoMaD[$G m] run $R"; return 0; }
  say "$TOWN NoMaD on Spacing[$G m], run $R -> $OUT"
  bash $S/run_drive_nomad.sh logs/equal_points/$(basename $OUT .jsonl)_$TOWN.log --town $TOWN --map-name $NAME --out $OUT \
    --node-poses $ND/uniform${G}_node_poses.json --seq-map $SEQ/uniform${G}_trace_sequence.json --episodes $EP \
    --image-dir $ND/images_uniform$G --target-speed-kmh 20 --cap-from $CAP --cap-map uniform${G}_trace --seed $R \
    >> logs/equal_points/launcher.log 2>&1
  n=$(count $OUT $NAME); [ "$n" -eq 40 ] || die "$TOWN NoMaD[$G m] run $R $n/40"
}

NEW02="hybrid_uniform2_entropy_thin4 sgm_g24_e55_c55 sgm_g10_e55_c55 sgm_g7_e55_c55"
NEW07="hybrid_uniform2_entropy_thin4 sgm_g20_e55_c55 sgm_g9.5_e55_c55 sgm_g6.5_e55_c55"
OLD07="uniform15_trace uniform8_trace hybrid_uniform15_entropy_thin4 hybrid_uniform8_entropy_thin4"   # 1 run each so far

# ---- stage A: run 1 of every new point ----------------------------------------------------------------------------
say "STAGE A (run 1 of new points)"
v6n02 $O2/run1.jsonl $NEW02
v6n07 $O7/run1.jsonl $NEW07
for G in 15 8 6; do nomad Town02 $G 1; done
for G in 15 8 6; do nomad Town07 $G 1; done
say "STAGE A DONE"

# ---- stage B: runs 2 and 3 ----------------------------------------------------------------------------------------
for R in 2 3; do
  say "STAGE B run $R"
  OLD02="uniform15_trace hybrid_uniform15_entropy_thin4 hybrid_uniform8_entropy_thin4"
  [ $R = 3 ] && OLD02="$OLD02 uniform8_trace"   # Town02 Spacing[8 m] already has a second run (uniform8_run2.jsonl)
  v6n02 $O2/run$R.jsonl $NEW02 $OLD02
  v6n07 $O7/run$R.jsonl $NEW07 $OLD07
  for G in 15 8 6; do nomad Town02 $G $R; done
  for G in 15 8 6; do nomad Town07 $G $R; done
  say "STAGE B run $R DONE"
done
say "EQUAL POINTS ALL DONE"
