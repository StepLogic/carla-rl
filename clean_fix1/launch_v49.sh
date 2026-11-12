#!/bin/bash
# v49 launch wrapper (user 2026-10-04, chose option 1 after the GPU filled up).
#
# v49 = v48 EXACTLY + ONE TOWN:
#   ENV_TOWNS   Town01,Town05 -> Town05      the eval town; one env, one server
#   CARLA_PORTS 2000,2010     -> 2000
#
# WHY. Another session's CARLA (a different install, /home/carla/...) took
# 5679 MiB of the 12282 MiB card. v48's second server is Epic/windowed and needs
# ~5.7 GB to come up; with the trainer's 2.6 GB there was ~3 GB left, so it was
# killed every time it grew and the trainer blocked forever on a dead env. v45
# completed 1M steps on the two-server config, so this is contention, not a
# broken setup.
#
# TWO DEVIATIONS, both forced by the same 12 GB and both worth knowing:
#   * ONE TOWN. v45-v48 trained on Town01 + Town05, so v49's numbers are NOT
#     directly comparable to that lineage -- it sees half the training
#     distribution.
#   * LOW / OFFSCREEN instead of Epic / windowed, which breaks the standing
#     "show the render while training locally" rule. Measured on this card: an
#     Epic windowed server costs ~5.7 GB and a Low offscreen one ~0.84 GB. At
#     Epic this run would not fit beside the other job at all. Note also that
#     CarlaCore has been measured rendering at Low whatever the setting says.
#
# Revert both the moment the card frees up.
set -u
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl/clean_fix1
cd "$W" || exit 1

_CALLER_RUN_NAME="${RUN_NAME:-}"; _CALLER_MAX_STEPS="${MAX_STEPS:-}"
set -a; . results/run_env/carla_gated_v36_segvae_nobc_noexpert.env; set +a
[ -n "$_CALLER_RUN_NAME" ]  && export RUN_NAME="$_CALLER_RUN_NAME"
[ -n "$_CALLER_MAX_STEPS" ] && export MAX_STEPS="$_CALLER_MAX_STEPS"

# ---- v37 reward changes (verbatim from launch_v37.sh) --------------------------
export GOAL_SPEED_BONUS=0.0
export GOAL_LANE_PENALTY_NONE=5.0
export EXPLORE_LANE_PENALTY_NONE=5.0
export EXPLORE_COLLISION_FATAL=1
export LANE_COLLISION_PENALTY=10.0
export GOAL_COLLISION_TRUNCATE_STEPS=1
export GOAL_OFFROAD_TRUNCATE_STEPS=1
export EXPLORE_ARRIVAL_BONUS=10.0
export REPLAY_CAPACITY=1000000
export DISCOUNT=1.0
export EXPERT_PROB=0 BC_WEIGHT=0

# ---- GPU fit (verbatim from launch_v37.sh) --------------------------------------
export GPU_RESERVE_BEFORE_CARLA=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_ALLOCATOR=bfc
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2

# ---- v42 deltas: applied LAST so nothing above can clobber them ------------------
# v42 = SORB-FAITHFUL (user 2026-10-01: "add fines for lane crossings and offroad
# and proceed"). SORB never asks the policy to reach long-horizon goals -- a graph
# planner chains short hops -- so the critic only has to be right for SHORT hops,
# HER supplies exact step counts, and the value support is bounded.
#   r = -1/step, gamma = 1, arrival terminates with NO bonus: Q = -(steps to goal)
#   goals 5-50 m straight-line (route p95 = 1.79x straight -> ~90 m, ~430 steps at
#     15 km/h); episode cap 500 covers it; START=TARGET=50 pins the trainer's
#     curriculum (it overwrites GOAL_DIST_MAX from them)
#   fault caps keep v37's terminal bill (clock = "never arrives"): off-road 1,
#     collision 1, crossing 20 -- now bounded by the 500-step cap
#   fines KEPT (user): lane graded (solid/unknown 10, broken open road 10, NONE 5),
#     off-road 10, collision 10 -- a few steps of distance, negligible vs 500
#   support [-750, 0]: worst = 19 fined crossing steps (-11 each) + crossing-cap
#     bill (-10 - 481) = -700; ceiling 0 because every return is <= 0
#   HER ON with recost: hindsight goals 5-50 m (HER_MAX_DISP), the distance learner
export EPISODE_HER=1
export HER_RECOST=1
export HER_MIN_DISP=5 HER_MAX_DISP=50
# HER_PAY_ARRIVAL=1: attempt 1 (archived) logged her=6340 paying=0 -- a goal drawn
# >= 5 m away can never be within the 2 m payout radius one 0.28 m step later, so no
# hindsight row ever arrived. This adds, for every drawn goal tp, the row that reaches
# it (tp-1, tp), which pays and terminates like an on-policy arrival.
export HER_PAY_ARRIVAL=1
export GOAL_DIST_MIN=5 GOAL_DIST_MAX=50 GOAL_DIST_START=50 GOAL_DIST_TARGET=50
export GATED_GOAL_MAX_STEPS=500
export GOAL_ARRIVAL_BONUS=0
export DISCOUNT=1.0
export GOAL_LANE_TERMINAL_BILL=clock
export TQC_VALUE_FLOOR=-750 TQC_VALUE_CEILING=0
export GOAL_LANE_PENALTY=10.0 GOAL_LANE_PENALTY_BROKEN=10.0 GOAL_LANE_PENALTY_NONE=5.0
# ---- v43 delta: THE ACTION CAPS REMOVED (user 2026-10-01) -------------------------
# MAX_STEER 0.5 -> 1.0 (full lock), MAX_THROTTLE 0.4 -> 1.0. These define the action
# space box and the action-history obs normalisation, so v43 CANNOT warm-start from
# v42 -- it trains from scratch, and that is the only way to read it.
#   WHY: v42's speed never exceeded 11.5 km/h (p99 11.4) vs TARGET_SPEED=20 -- the
#   0.4 cap, not sluggish learning. Full steer also gives the authority to make the
#   tight turns the off-axis goals need.
#   WATCH: speed obs is clip(speed/TARGET_SPEED,0,1), so above 20 km/h the agent goes
#   BLIND to its own speed; and full lock at speed can spin the car, which could raise
#   off-road rather than lower it. Both are read from the first blocks.
export MAX_STEER=1.0 MAX_THROTTLE=1.0
# ---- v44 delta: THE 5-200 m BAND (user 2026-10-01 "got 5- 200 m v44 run") ----------
# The band is COUPLED to the cap and the floor; moving it alone is what made far goals
# unreachable-by-construction in v37-v41, which is how the policy learned to give up.
#   200 m straight = ~358 m by road at the p95 route ratio (1.79, measured over 280 plans).
#   v43 drives ~29 km/h (inferred: ~50 steps to a ~27 m median goal), so ~920 steps; at
#   20 km/h it would be ~1290. Cap 1250 covers both with headroom.
#   Floor: worst return ~= -(1250 + ~19 crossing fines x 11) = -1459 -> -1500.
#   HER band raised to match, else hindsight would only ever teach <=50 m hops.
# NOTE the goal band is straight-line, and the trainer's curriculum OVERWRITES
# GOAL_DIST_MAX from START/TARGET, so all three move together.
# v45: THE BAND IS REACHED BY CURRICULUM, NOT PINNED (user 2026-10-02 "proceed").
# v44 set START=TARGET=200, which made the curriculum INERT: promotion needs
# curr_max < CURR_TARGET and demotion needs curr_max > CURR_START, so with both at
# 200 NEITHER can ever fire (train_carla_tqc.py:2161,2171). The advertised "demote at
# <=0.15" was unreachable, and v44 sat pinned at the hardest setting -- peak succ 0.40
# at ep610, then down to 0.17-0.18 with off-road unwinding 0.23 -> 0.58, with no way
# to back off. START=50 is where v43 already reaches 0.99, so the ramp begins in the
# regime the policy can already drive.
#   GOAL_DIST_MUL=1.15 makes the ramp GEOMETRIC: 50 -> 57 -> 66 -> ... -> 200 in ~10
#   promotions (~500 episodes minimum), where the default +2.5 m linear step would
#   need 60 promotions (>=3000 episodes) and would barely reach 200 m inside 1M steps.
export GOAL_DIST_MIN=5 GOAL_DIST_MAX=50 GOAL_DIST_START=50 GOAL_DIST_TARGET=200
export GOAL_DIST_MUL=1.15
export GATED_GOAL_MAX_STEPS=1250
export TQC_VALUE_FLOOR=-1500
export HER_MAX_DISP=200
export GOAL_OFF_ROAD_PENALTY=10.0 GOAL_COLLISION_PENALTY=10.0

export RUN_NAME="${_CALLER_RUN_NAME:-carla_gated_v49_segvae_debitpaper_t05}"
export MAX_STEPS="${_CALLER_MAX_STEPS:-1000000}"

echo "=== [v49] RUN_NAME=$RUN_NAME MAX_STEPS=$MAX_STEPS DISCOUNT=$DISCOUNT ==="
echo "=== [v49] EPISODE_HER=$EPISODE_HER HER_PAY_ARRIVAL=$HER_PAY_ARRIVAL GATED_GOAL_MAX_STEPS=$GATED_GOAL_MAX_STEPS GOAL_DIST=$GOAL_DIST_MIN-$GOAL_DIST_TARGET ARRIVAL=$GOAL_ARRIVAL_BONUS FLOOR=$TQC_VALUE_FLOOR MAX_STEER=$MAX_STEER MAX_THROTTLE=$MAX_THROTTLE BAND=$GOAL_DIST_MIN-$GOAL_DIST_START->$GOAL_DIST_TARGET MUL=$GOAL_DIST_MUL CAP=$GATED_GOAL_MAX_STEPS FLOOR=$TQC_VALUE_FLOOR PERCEPTION=$PERCEPTION EXPERT_PROB=$EXPERT_PROB ==="
# THE ONE CHANGE. Set last so nothing downstream can overwrite it.
export LANE_PROGRESS_SCALE=1
# --- the goal arm becomes DEBiT ---
export GATED_GOAL_REWARD=progress
export ROUTE_POTENTIAL=1
export GOAL_PROGRESS_FINES=0
# max((d_prev - d), 0): pay approach, charge nothing for retreat (user 2026-10-04).
# Deliberately NOT the paper's signed term -- see gated_reward.PROGRESS_POSITIVE
# for the telescoping property this gives up.
export GOAL_PROGRESS_POSITIVE=0
# --- single town, single server: see the header ---
export ENV_TOWNS=Town05 EVAL_TOWNS=Town05
export CARLA_PORTS=2000 CARLA_PORT=2000
# Town05 WINDOWED, as it has always been run here. Offscreen was a VRAM saving
# while another job held 5.7 GB; that job has gone. Under -RenderOffScreen the
# server came up, loaded Town05, set the weather and then SEGFAULTED as the
# sensors attached -- three times from a clean, idle card, so it was not thrash.
# Town05 has only ever run windowed (v45 did 1M steps that way); Town01 was the
# offscreen one. This also restores the standing show-the-render rule.
export CARLA_QUALITY=Epic CARLA_QUALITY_PORTS=2000:Epic
export CARLA_OFFSCREEN=0 CARLA_OFFSCREEN_PORTS=

export GATED_GOAL_LANE_TERMINAL=1
export GOAL_COLLISION_TRUNCATE_STEPS=1
export TQC_VALUE_CEILING=300
echo "=== [v49] LANE_PROGRESS_SCALE=$LANE_PROGRESS_SCALE (v45 used 10) ==="
exec ./train_v42.sh
