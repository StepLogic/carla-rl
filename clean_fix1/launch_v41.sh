#!/bin/bash
# v41 launch wrapper (user 2026-10-01 "stop and start v41 - but restore v37 reward").
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

# ---- v41 deltas: applied LAST so nothing above can clobber them ------------------
# v41 = v37's REWARD EXACTLY (user 2026-10-01 "restore v37 reward") with the goal
# critic discounted at 0.999. Nothing from v38-v40 carries over: POTENTIAL_K stays 0
# (train_v37.sh default), off-road cap stays 1 (the v37 block above), STEP_COST
# stays 1.0 (run_gated.sh default), no gamma-aware shaping, ceiling stays 100.
#
# WHY 0.999. gamma=1.0 diverged the goal critic in v37-v39 (q ~2x worse than any
# realised return); gamma=0.99 (v40) fixed that but its ~100-step horizon cannot
# see a 720-step journey -- arriving at a 200 m goal was worth 0.2 more than
# idling, and the goal arm idled (clock 0.65). At 0.999 arrival beats the next-best
# alternative by ~350 while the backup still contracts (factor 0.999).
#
# THE BILL. v37's terminal unification charged a fault "fine + the rest of the
# clock" so that crash = idle-out + fine. At gamma=1 the clock and discounted bills
# are identical; at 0.999 only the discounted bill keeps that equality (the clock
# bill would charge 1001 for an idle-out worth ~632, biasing toward idling).
export DISCOUNT=0.999
export GATED_ALLOW_DISCOUNTED_SORB=1
export GOAL_LANE_TERMINAL_BILL=discounted

export RUN_NAME="${_CALLER_RUN_NAME:-carla_gated_v41_segvae_v37reward_g0999}"
export MAX_STEPS="${_CALLER_MAX_STEPS:-1000000}"

echo "=== [v41] RUN_NAME=$RUN_NAME MAX_STEPS=$MAX_STEPS DISCOUNT=$DISCOUNT ==="
echo "=== [v41] POTENTIAL_K=$POTENTIAL_K GOAL_OFFROAD_TRUNCATE_STEPS=$GOAL_OFFROAD_TRUNCATE_STEPS GOAL_LANE_TERMINAL_BILL=$GOAL_LANE_TERMINAL_BILL PERCEPTION=$PERCEPTION EXPERT_PROB=$EXPERT_PROB ==="
exec ./train_v37.sh
