#!/bin/bash
# user 2026-09-20: re-queue of everything the peer's 53b285a broke, now against the PINNED source tree
# (compat_v6n/src @163e6f9 -- the last commit before this sweep's first row was driven).
#   1. Town07 DINOv2, 12 maps          (the stage that failed 0/40 at 06:43)
#   2. Signal grid, 20 maps            (epistemic/aleatoric x uniform/SGM, + policy-entropy + uniform)
#   3. Grid baselines, 4 maps          (Dense + the three seed-only maps, under DINOv2)
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl; cd "$W" || exit 1
S=$W/scripts/icra2027_equal_points
P=$S/run_drive_plan_c_pinned.sh
say(){ echo "=== [pinned $(date '+%m-%d %H:%M:%S')] $* ==="; }
die(){ say "FAILED: $*"; exit 1; }
count(){ python3 -c "import json,sys,collections; c=collections.Counter(json.loads(l)['map'] for l in open(sys.argv[1]) if l.strip()); print(min([c.get(m,0) for m in sys.argv[2:]]))" "$@" 2>/dev/null || echo 0; }
# precise pattern: only the peer's trainer matches, not a passing shell that mentions it
busy(){ pgrep -f "bin/python -u train_carla_tqc\.py" >/dev/null; }

while busy; do say "holding: the peer's trainer has the simulator"; sleep 300; done
sleep 60
busy && die "a trainer came back -- not taking the simulator"
say "simulator free; pinned tree @163e6f9"

TR7=results/placement_v6n_town07_budget10k_laneoff_speed10/trace_s7.pkl
EP7="--episodes results/town07/episodes/town07_speed10_s7_episodes_short20_long20_seed0.pkl \
--views results/town07/episodes/town07_speed10_s7_episode_views_seed0.pkl"
FE7=results/town07/episodes/town07_speed10_s7_dinov2.npz
T7="uniform15_trace uniform8_trace uniform6_trace uniform2_trace \
sgm_g20_e55_c55 sgm_g9.5_e55_c55 sgm_g6.5_e55_c55 sgm_g2_e55_c55 \
hybrid_sgm_g24_entropy_thin4 hybrid_sgm_g10_entropy_thin4 hybrid_sgm_g7_entropy_thin4 hybrid_sgm_g2_entropy_thin4"

OUT=results/town07/localiser/dinov2_bank_run1.jsonl
if [ "$(count $OUT $T7)" -ne 40 ]; then
  say "1/3 Town07 dinov2, 12 maps"
  bash $P $OUT logs/dinov2/town07_dinov2_run1_pinned.log --town Town07 --trace $TR7 \
    --seq-dir results/town07/maps/sequence $EP7 --maps $T7 --localiser dinov2_bank --feats $FE7 \
    >> logs/dinov2/launcher_pinned.log 2>&1
fi
n=$(count $OUT $T7); [ "$n" -eq 40 ] || die "Town07 dinov2 min $n/40"
say "1/3 Town07 dinov2 DONE"

UNIFORM="hybrid_uniform15_entropy_thin4 hybrid_uniform8_entropy_thin4 hybrid_uniform6_entropy_thin4 hybrid_uniform2_entropy_thin4 \
hybrid_uniform15_epistemic_thin4 hybrid_uniform8_epistemic_thin4 hybrid_uniform6_epistemic_thin4 hybrid_uniform2_epistemic_thin4 \
hybrid_uniform15_aleatoric_thin4 hybrid_uniform8_aleatoric_thin4 hybrid_uniform6_aleatoric_thin4 hybrid_uniform2_aleatoric_thin4"
SGM="hybrid_sgm_g44_epistemic_thin4 hybrid_sgm_g13_epistemic_thin4 hybrid_sgm_g9_epistemic_thin4 hybrid_sgm_g2_epistemic_thin4 \
hybrid_sgm_g44_aleatoric_thin4 hybrid_sgm_g13_aleatoric_thin4 hybrid_sgm_g9_aleatoric_thin4 hybrid_sgm_g2_aleatoric_thin4"
GOUT=results/drive_plan_c/signal_grid/dinov2_run1.jsonl
for STAGE in "uniform:$UNIFORM" "sgm:$SGM"; do
  NAME=${STAGE%%:*}; MAPS=${STAGE#*:}
  if [ "$(count $GOUT $MAPS)" -ne 40 ]; then
    say "2/3 signal grid, $NAME filler ($(echo $MAPS | wc -w) maps)"
    bash $P $GOUT logs/dinov2/signal_grid_${NAME}_pinned.log --maps $MAPS --localiser dinov2_bank \
      >> logs/dinov2/launcher_pinned.log 2>&1
  fi
  n=$(count $GOUT $MAPS); [ "$n" -eq 40 ] || die "signal grid $NAME min $n/40"
  say "2/3 signal grid $NAME DONE"
done

BMAPS="raw_trace entropy_nosmooth_high_thin4 epistemic_thin4_n30 aleatoric_thin4_n30"
BOUT=results/drive_plan_c/signal_grid/dinov2_baselines_run1.jsonl
if [ "$(count $BOUT $BMAPS)" -ne 40 ]; then
  say "3/3 grid baselines under DINOv2"
  bash $P $BOUT logs/dinov2/signal_grid_baselines_pinned.log --maps $BMAPS --localiser dinov2_bank \
    >> logs/dinov2/launcher_pinned.log 2>&1
fi
n=$(count $BOUT $BMAPS); [ "$n" -eq 40 ] || die "baselines min $n/40"
say "3/3 grid baselines DONE"
say "PINNED CHAIN DONE"
