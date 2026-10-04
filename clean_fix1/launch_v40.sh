#!/bin/bash
# v40 launch wrapper (user 2026-10-01 "proceed"): v39 with the goal critic DISCOUNTED.
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

# ---- v38 deltas: applied LAST so nothing above can clobber them ------------------
export POTENTIAL_K=2.0
export GOAL_OFFROAD_TRUNCATE_STEPS=20
# v39: per-step slack folded into the step cost, so the fault bill (charged at
# step_cost) stays exact and crash = park + fine still holds. A slack term outside
# the bill would make crashing cheaper than parking by lambda*remaining.
export STEP_COST=1.5
# v40: the gamma test. Every gamma=1.0 goal critic (v37, v38, v39) drifted to ~2x more
# negative than any realised return and the goal arm collapsed; the gamma=0.99 critics
# of v29 and v36 stayed flat. This is the single-variable test of that.
#   DISCOUNT=0.99                    the change under test
#   GATED_ALLOW_DISCOUNTED_SORB=1    gated_config refuses discounted SORB without it
#   GOAL_LANE_TERMINAL_BILL=discounted  the clock bill over-penalised ~10x at 0.99
#                                    (v15/v30 parked); the discounted bill keeps
#                                    crash = idle-out + fine in discounted terms
#   GOAL_SHAPING_DISCOUNTED=1        K*(d_prev - g*d): keeps the shaping Ng-invariant
#                                    at g<1, else it drags in a -(1-g)K*d reward
export DISCOUNT=0.99
export GATED_ALLOW_DISCOUNTED_SORB=1
export GOAL_LANE_TERMINAL_BILL=discounted
export GOAL_SHAPING_DISCOUNTED=1
# The gamma-aware shaping shifts the critic's values by +K*d (V' = V - Phi): up to
# ~+250 at a 200 m start, ~+1060 if the car drives ~480 m away. v36's ceiling of 100
# would CLIP every target more than ~125 m from its goal (attempt 1, archived).
# This only loosens a clip; the reward is unchanged.
export TQC_VALUE_CEILING=1100

export RUN_NAME="${_CALLER_RUN_NAME:-carla_gated_v40_segvae_gamma099}"
export MAX_STEPS="${_CALLER_MAX_STEPS:-1000000}"

echo "=== [v40] RUN_NAME=$RUN_NAME MAX_STEPS=$MAX_STEPS DISCOUNT=$DISCOUNT ==="
echo "=== [v40] POTENTIAL_K=$POTENTIAL_K GOAL_OFFROAD_TRUNCATE_STEPS=$GOAL_OFFROAD_TRUNCATE_STEPS STEP_COST=$STEP_COST GOAL_LANE_TERMINAL_BILL=$GOAL_LANE_TERMINAL_BILL GOAL_SHAPING_DISCOUNTED=$GOAL_SHAPING_DISCOUNTED TQC_VALUE_CEILING=$TQC_VALUE_CEILING PERCEPTION=$PERCEPTION EXPERT_PROB=$EXPERT_PROB ==="
exec ./train_v37.sh
