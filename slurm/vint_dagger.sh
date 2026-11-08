#!/usr/bin/env bash
# ViNT DAgger round (user 2026-11-07, "use DAGGER"): ViNT (EVAL_FROM, a checkpoint
# dir from scripts/vint/export_vint_ckpt.py) DRIVES goal episodes in a training
# town while the expert -- replanning whenever the car is more than 3 m off its
# route -- LABELS every frame: EXPERT_RECORD_AIM stores the lane-centre aim points
# of the expert's current plan (vae_nav.vint_lane_labels.aim_from_plan), ViNT's
# waypoint labels for the states ViNT itself reaches. As c0's slurm/il_dagger.sh
# (same recording overlay, il_record.env), one town per job, from the vint-carla
# tree, into its own root so the expert data is untouched.
#
#   DAGGER_ROUND=1 TOWN=Town01 PORT=3300 \
#   EVAL_FROM=$HOME/carla-rl/checkpoints/vint_carla_lanecentre/epoch_1 \
#   sbatch -p gpu --gres=gpu:1 -c 4 --mem=48G -t 6:00:00 --export=ALL slurm/vint_dagger.sh
#
# Seeds 600000 + 10000*town + 100000*(round-1), even only (EXPERT_GOAL_ONLY): no
# overlap with the expert data (100000-130000), c0's DAgger (300000+), the
# validation (50000+) or the benchmark sets (21000-33110). Town02/Town03 are held
# out: run_record refuses them.
set -u
REPO=${REPO:-$HOME/carla-rl-vint}
cd "$REPO" || exit 1
: "${DAGGER_ROUND:?}" "${EVAL_FROM:?}" "${TOWN:?}" "${PORT:?}"
export REPO EVAL_POLICY=vint EVAL_FROM EXPERT_RECORD=1 EXPERT_DRIVER=policy EXPERT_RECORD_DRY=0
export EXPERT_GOAL_ONLY=1 EXPERT_RECORD_AIM=1
export PYTHONPATH="$HOME/vint/pkgs:$HOME/vint/visualnav/train${PYTHONPATH:+:$PYTHONPATH}"
export EXPERT_RECORD_ROOT=${EXPERT_RECORD_ROOT:-$HOME/carla-rl/checkpoints/il_data_vint_dagger$DAGGER_ROUND}
export EXPERT_RECORD_EPISODES=${EXPERT_RECORD_EPISODES:-300}
case $TOWN in
  Town01) i=0;; Town04) i=1;; Town05) i=2;; Town10HD) i=3;;
  *) echo "=== vint_dagger: $TOWN is not a training town"; exit 1;;
esac
# SEED_OFFSET (even, default 0) splits one town over several jobs (user 2026-11-07,
# "can you make it fast"): e.g. two jobs of 150 episodes with offsets 0 and 300.
SEED_OFFSET=${SEED_OFFSET:-0}
if [ $((SEED_OFFSET % 2)) -ne 0 ]; then echo "=== SEED_OFFSET must be even"; exit 1; fi
seed0=$((600000 + 10000 * i + 100000 * (DAGGER_ROUND - 1) + SEED_OFFSET))
echo "=== VINT_DAGGER round $DAGGER_ROUND $TOWN seeds from $seed0 (even: goal episodes) port $PORT policy=$EVAL_FROM -> $EXPERT_RECORD_ROOT n=$EXPERT_RECORD_EPISODES ==="
REWARD_OVERLAY=slurm/il_record.env ENV_TOWNS=$TOWN EXPERT_SEED0=$seed0 \
  RUN_NAME=vint_dagger_r${DAGGER_ROUND}_${TOWN}_s$SEED_OFFSET CARLA_PORT=$PORT bash slurm/train_sac.sbatch
echo "=== VINT_DAGGER round $DAGGER_ROUND $TOWN rc=$? ==="
