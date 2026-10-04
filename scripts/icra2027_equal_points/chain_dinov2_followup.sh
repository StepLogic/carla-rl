#!/bin/bash
# user 2026-09-20 "proceed": error bars for the DINOv2 result, then the Town07 replication.
# HOLDS until the PEER SESSION's image-goal training (train_carla_tqc.py, checkpoints/carla_image_goal_v2)
# exits, because run_drive_plan_c.sh kills every CARLA on this box as pre-flight. Nothing here touches it.
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl; cd "$W" || exit 1
S=$W/scripts/icra2027_equal_points
say(){ echo "=== [dino2 $(date '+%m-%d %H:%M:%S')] $* ==="; }
die(){ say "FAILED: $*"; exit 1; }
count(){ python3 -c "import json,sys,collections; c=collections.Counter(json.loads(l)['map'] for l in open(sys.argv[1]) if l.strip()); print(min([c.get(m,0) for m in sys.argv[2:]]))" "$@" 2>/dev/null || echo 0; }

T2="uniform15_trace uniform8_trace uniform6_trace uniform2_trace \
sgm_g24_e55_c55 sgm_g10_e55_c55 sgm_g7_e55_c55 sgm_g2_e55_c55 \
hybrid_sgm_g44_entropy_thin4 hybrid_sgm_g14_entropy_thin4 hybrid_sgm_g9_entropy_thin4 hybrid_sgm_g2_entropy_thin4"
T7="uniform15_trace uniform8_trace uniform6_trace uniform2_trace \
sgm_g20_e55_c55 sgm_g9.5_e55_c55 sgm_g6.5_e55_c55 sgm_g2_e55_c55 \
hybrid_sgm_g24_entropy_thin4 hybrid_sgm_g10_entropy_thin4 hybrid_sgm_g7_entropy_thin4 hybrid_sgm_g2_entropy_thin4"
T7NEW="hybrid_sgm_g24_entropy_thin4 hybrid_sgm_g10_entropy_thin4 hybrid_sgm_g7_entropy_thin4 hybrid_sgm_g2_entropy_thin4"
TR7=results/placement_v6n_town07_budget10k_laneoff_speed10/trace_s7.pkl
EP7="--episodes results/town07/episodes/town07_speed10_s7_episodes_short20_long20_seed0.pkl \
--views results/town07/episodes/town07_speed10_s7_episode_views_seed0.pkl"

while pgrep -f "train_carla_tqc" >/dev/null; do say "holding: a peer session is training (image-goal v2)"; sleep 300; done
sleep 180
pgrep -f "train_carla_tqc" >/dev/null && die "a trainer came back while we waited -- not taking the simulator"
say "simulator free"

# --- 1-2. Town02 DINOv2 runs 2 and 3: error bars on the 9-0 result -------------------------------------
for R in 2 3; do
  OUT=results/drive_plan_c/localiser/dinov2_bank_run${R}.jsonl
  if [ "$(count $OUT $T2)" -ne 40 ]; then
    say "Town02 dinov2 run $R"
    bash $S/run_drive_plan_c.sh $OUT logs/dinov2/dinov2_bank_run${R}.log --maps $T2 --localiser dinov2_bank \
      >> logs/dinov2/launcher_followup.log 2>&1
  fi
  n=$(count $OUT $T2); [ "$n" -eq 40 ] || die "Town02 dinov2 run $R min $n/40"
  say "Town02 dinov2 run $R DONE"
done

# --- 3. Town07 RGB bank + DINOv2 features (the bank is per town) --------------------------------------
BANK7=results/town07/episodes/town07_speed10_s7_rgb_bank.pkl
FE7=results/town07/episodes/town07_speed10_s7_dinov2.npz
if [ ! -f "$BANK7" ]; then
  say "Town07 RGB bank capture"
  bash "$W/scripts/icra2027_equal_points/run_bank.sh" --town Town07 --trace $TR7 \
    --episodes results/town07/episodes/town07_speed10_s7_episodes_short20_long20_seed0.pkl \
    --out $BANK7 >> logs/dinov2/bank_town07.log 2>&1
fi
[ -f "$BANK7" ] || die "Town07 bank was not written"
say "Town07 bank ready"
if [ ! -f "$FE7" ]; then
  say "Town07 DINOv2 embedding"
  /home/robotlab/scratch/carla-rl/dependencies/vc1-venv/bin/python embed_dinov2.py --bank $BANK7 --out $FE7 \
    >> logs/dinov2/embed_town07.log 2>&1
fi
[ -f "$FE7" ] || die "Town07 features were not written"
say "Town07 features ready"

# --- 4. Town07: the new Entropy+SGM maps under the SHIPPED localiser (they have never been driven) -----
OUT=results/town07/localiser/mobilenet_trace_entsgm_run1.jsonl
if [ "$(count $OUT $T7NEW)" -ne 40 ]; then
  say "Town07 shipped localiser on the 4 Entropy+SGM maps"
  bash $S/run_drive_plan_c.sh $OUT logs/dinov2/town07_shipped_entsgm.log --town Town07 --trace $TR7 \
    --seq-dir results/town07/maps/sequence $EP7 --maps $T7NEW >> logs/dinov2/launcher_followup.log 2>&1
fi
n=$(count $OUT $T7NEW); [ "$n" -eq 40 ] || die "Town07 shipped Entropy+SGM min $n/40"
say "Town07 shipped Entropy+SGM DONE"

# --- 5. Town07 under DINOv2, all 12 maps --------------------------------------------------------------
OUT=results/town07/localiser/dinov2_bank_run1.jsonl
if [ "$(count $OUT $T7)" -ne 40 ]; then
  say "Town07 dinov2 run 1, 12 maps"
  bash $S/run_drive_plan_c.sh $OUT logs/dinov2/town07_dinov2_run1.log --town Town07 --trace $TR7 \
    --seq-dir results/town07/maps/sequence $EP7 --maps $T7 --localiser dinov2_bank --feats $FE7 \
    >> logs/dinov2/launcher_followup.log 2>&1
fi
n=$(count $OUT $T7); [ "$n" -eq 40 ] || die "Town07 dinov2 min $n/40"
say "Town07 dinov2 DONE"
say "FOLLOW-UP CHAIN DONE"
