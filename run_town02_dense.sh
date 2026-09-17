#!/usr/bin/env bash
# Town02 overfitting test at 120x120, with the oracle removed.
#
# Fixes this run depends on, all of which predate it:
#   1. goal_block lateral sign -- CARLA is left-handed, and every earlier run
#      fed the policy a MIRRORED bearing, so it steered away from any goal not
#      straight ahead or behind. That is why no policy ever reached a goal.
#   2. checkpoints now contain network weights, so this run leaves a policy.
#   3. off_road (map.get_waypoint(project_to_road=False) is None) terminates
#      the episode as a lane invasion.
#   4. the VAE corpus is no longer 90% duplicate frames of a stationary hero.
#
# ENV_IMAGE_SIZE MUST MATCH THE VAE'S TRAINING RESOLUTION. The encoder was fit
# on 120x120 frames; handing it the env's old 84x84 would put it out of
# distribution with nothing in the logs to say so. LatentSource cross-checks
# the checkpoint's image_size, so a mismatch fails loudly -- but the variable
# still has to be exported here.
set -euo pipefail
export DISPLAY=:1
export CARLA_PORT=2000 CARLA_PORTS=2000
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
export ENV_TOWNS=Town02 EVAL_TOWNS=Town02          # in-distribution overfit test
export POLICY_FOLDER=carla_tqc_town02_dense
export EXPERT_PROB=0                               # NO ORACLE
# ANY lane crossing terminates, not just solid/kerb/grass. The dashed-line
# exemption existed only so a route-following oracle could turn at a
# junction; EXPERT_PROB=0 here, so it would just hide the signal.
# PER-STEP PROGRESS REWARD: k_dist * (d_prev - d_now) toward the current goal.
# Potential-based, so it telescopes to d_start-d_end and pays 0 on a closed
# loop -- it cannot be farmed and does not change which policy is optimal, only
# how fast the critic finds one. Recomputed identically inside HER.
#
# SCALE: at ~0.5 m/step and ~100 steps, k_dist=1.0 contributes ~50 total against
# k_goal=600, so arrival still dominates. Raising it much further would let
# shaping outweigh the goal itself.
export K_DIST=1.0
export LANE_STRICT=1
# GOAL CURRICULUM: start at 5 m, widen by 2.5 m each time the POLICY
# clears 50% success over 50 episodes, up to 20 m.
# LONG-HORIZON: 5 m -> 113 m, multiplicative (x1.25; additive 2.5 m steps
# would be 43 stages and eat the step budget).
#
# TARGET CAPPED AT 113 m, NOT 200 m. The previous attempt climbed to 113.55 m
# and froze there for 300+ episodes at 0.29-0.44 success -- under the promote
# threshold, over the demote threshold. 113 m is the furthest range this
# observation actually supports, and the reason is structural: the policy has
# NO VELOCITY INPUT. Its state is [fwd, left, sin(yaw), cos(yaw)] plus 0.4 s of
# past (steer, throttle) commands, so it cannot observe its own speed -- only
# what it recently asked for. A car has momentum a Habitat point-mass agent
# does not, and approach control into a 2 m arrival radius depends on current
# speed. Chasing 200 m without that input would spend the step budget
# rediscovering the same wall.
export GOAL_DIST_MIN=5.0 GOAL_DIST_START=5.0 GOAL_DIST_TARGET=113.0
export GOAL_DIST_MUL=1.25 CURRICULUM_SUCCESS=0.5 CURRICULUM_MIN_EPISODES=50
# STEP AND DISTANCE CAPS SIZED FOR THE TARGET.
# At the measured 0.66 m/step a 200 m goal needs ~300 steps, and a 4-goal chain
# at 200 m each needs ~1,200 -- so a 500-step cap would truncate every late
# episode before it could finish. MAX_DISTANCE_M must exceed
# chain_max_goals * target (4 x 200 = 800) or the env's own distance
# terminator ends the episode mid-chain; it defaults to 200, which would have
# capped this curriculum at its own target.
export MAX_EPISODE_STEPS=2000
export MAX_DISTANCE_M=1200
export SHORT_HORIZON=1
# RAISED FROM 500k. The curriculum is now 40x longer in distance and the late
# episodes are ~15x longer in steps, so 500k would very likely end mid-schedule
# -- reporting a policy that never saw 200 m. Revert to 500000 if the shorter
# budget is preferred.
export MAX_STEPS=1000000
export RESUME=0  # fresh run: the reward function changed
# GOAL CHAINING: on arrival, continue to a new goal instead of ending.
# Cap ramps 1 -> 4 goals over the first 100k steps. HER segments at each
# arrival so hindsight goals never cross a goal boundary.
export CHAIN_MAX_GOALS=4 CHAIN_CURRICULUM_STEPS=100000
export CURRICULUM_DEMOTE=0.15
# HER DISABLED. Every transition now goes into the buffer exactly once, as the
# on-policy row the env produced; no hindsight goals are inserted.
#
# WHAT THIS COSTS, stated because it is the reason HER was there. The reward is
# sparse -- +600 only inside a 2 m radius -- so without relabelling the buffer
# contains no positive reward until the policy reaches a goal unaided. The last
# measured relabel statistics were 22,123 hindsight rows over 555 legs with
# 12.4% of them paying k_goal; that supervision is now gone. Expect a long flat
# stretch at zero return, and expect the goal curriculum to matter much more,
# since a 5 m goal is the only thing likely to be reached by accident.
#
# EPISODE_HER=1 restores it.
export EPISODE_HER=0 HER_MAX_DISP=6.0 TEMP_LR=3e-4
export VAE_CKPT=checkpoints/lane_vae_120/best.pkl
export EVAL_VIDEO=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec python -u train_carla_tqc.py
