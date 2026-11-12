#!/bin/bash
# v42 launch wrapper (user 2026-10-01 "add fines for lane crossings and offroad and proceed").
#
# v38 = v37 EXACTLY + two changes:
#   POTENTIAL_K 0 -> 2.0            potential shaping on along-road progress
#   GOAL_OFFROAD_TRUNCATE_STEPS 1 -> 20   off-road recoverable again
#
# WHY. v37's fault terminal (-fine - (1 + remaining)) returns -1011 for EVERY
# failure regardless of when it happens -- the t cancels -- so there was no
# gradient to survive or progress, goal len collapsed 771 -> 89 and success
# 0.20 -> 0.01. Shaping keeps the min-time optimum for arrivals (speed gap stays
# exactly the time difference at any K) and, because GatedReward pays no shaping
# on the terminal step, a failure keeps the progress credit it earned:
# collision at t=300 scores -844 vs -1009 at t=5 (K=2). The 1-step off-road kill
# ended ~85% of v37 goal episodes within ~4 s; its justification (a non-terminal
# fault stops the min-time meter) is answered by the per-step cost that keeps
# running while off-road. v36 ran 20 steps.
#
# The chain below is launch_v37.sh VERBATIM (v36 snapshot as baseline, then v37's
# overrides, then the GPU block), with the v38 deltas applied LAST. After launch,
# diff results/run_env/<v38>.env against the v37 snapshot: those two flags (and
# the run name) must be the only differences.
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
export GOAL_OFF_ROAD_PENALTY=10.0 GOAL_COLLISION_PENALTY=10.0

export RUN_NAME="${_CALLER_RUN_NAME:-carla_gated_v42_segvae_sorb_her}"
export MAX_STEPS="${_CALLER_MAX_STEPS:-1000000}"

echo "=== [v42] RUN_NAME=$RUN_NAME MAX_STEPS=$MAX_STEPS DISCOUNT=$DISCOUNT ==="
echo "=== [v42] EPISODE_HER=$EPISODE_HER HER_PAY_ARRIVAL=$HER_PAY_ARRIVAL GATED_GOAL_MAX_STEPS=$GATED_GOAL_MAX_STEPS GOAL_DIST=$GOAL_DIST_MIN-$GOAL_DIST_TARGET ARRIVAL=$GOAL_ARRIVAL_BONUS FLOOR=$TQC_VALUE_FLOOR CEIL=$TQC_VALUE_CEILING PERCEPTION=$PERCEPTION EXPERT_PROB=$EXPERT_PROB ==="
exec ./train_v42.sh
