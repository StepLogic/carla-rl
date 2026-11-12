#!/usr/bin/env bash
# THE GATED ARM WITH HINDSIGHT RELABELLING: explore episodes become goal data.
#
# A THIN WRAPPER, DELIBERATELY. Every setting comes from run_gated.sh; this
# script flips exactly two things -- hindsight on, and a separate checkpoint
# folder. Copying the 170-line parent would let the two arms drift, and then a
# difference in the result would measure the drift rather than the effect of
# relabelling. The ablation is only clean if the scripts differ by one flag.
#
# WHAT RELABELLING BUYS HERE, and it is specific. The goal curriculum is pinned
# at 5-20 m (GOAL_DIST_START == GOAL_DIST_TARGET in the parent), so the goal arm
# never sees a longer goal however long it trains. Measured on the first gated
# run, goal episodes settle around 96 steps while explore episodes run ~370 and
# wander. Relabelling an explore trajectory into goal data therefore supplies
# goals at distances the curriculum CANNOT produce. That is the only route this
# arm has to saying anything about longer-range navigation.
#
# HOW IT STAYS CORRECT, all three enforced in code and pinned by
# tests/test_gated_her.py:
#   - on-policy rows keep their OWN mode, so explore transitions are not filed
#     as goal data and the mode split does not collapse;
#   - every hindsight row is forced to GOAL, so a relabelled explore row cannot
#     train the explore critic on goal-shaped data;
#   - the goal block is rebuilt from the hindsight goal BEFORE the mode is
#     forced, so no row reaches the goal critic carrying explore's zeroed block.
#
# The relabelled reward is minimum time, matching the on-policy goal rows.
# EpisodeRelabelBuffer REFUSES gated=True without it, because sparse hindsight
# rows in a minimum-time buffer are two reward functions over one set of states.
#
# Usage:  ./run_gated_her.sh          (or MAX_STEPS=... ./run_gated_her.sh)
set -uo pipefail
cd "$(dirname "$0")"

export EPISODE_HER=1
export POLICY_FOLDER="${POLICY_FOLDER:-carla_gated_her}"

# n_relabel: hindsight copies per real transition. 1 keeps the buffer's mode
# balance close to the env's, which the mode-split buffer then fixes on sample
# anyway; raising it buys goal diversity at the cost of pushing on-policy
# explore rows out of a finite buffer.
export EPISODE_HER_N="${EPISODE_HER_N:-1}"

# MIN_DISP IS TIED TO THE PAYOUT RADIUS AND MUST NOT BE LOWERED. A hindsight
# goal drawn closer than the radius that pays is a full-reward positive
# requiring no travel. The peer measured 827 of 1294 relabels (63.9%) free at a
# 0.2 m floor under a 0.8 m radius, with Q going +11.7 -> -3.95 and success
# halving. The buffer asserts the tie; this line documents why it exists.
# Leave unset to inherit the goal threshold.

echo "[GatedHER] hindsight ON -- explore episodes are relabelled into GOAL data"
echo "[GatedHER] policy folder: ${POLICY_FOLDER}"
exec ./run_gated.sh
