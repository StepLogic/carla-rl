#!/usr/bin/env bash
# FRESH, never-seen benchmark episodes for the frozen map-navigation evaluation
# (user 2026-11-07: develop on the current seeds, then freeze and run fresh seeds).
# The original probe (logs/bench_probe.sh) with NEW seed bases -- easy 31000,
# medium 32000, hard 33000 (the development set used 21000-23110) -- and c0's own
# eval overlay as the base, so each seed draws exactly the episode the eval drives.
# EVAL_PROBE=1: reset each seed once, one zero step, record the planner's route;
# no policy acts. Then scripts/bench_select.py keeps the first 100 per band.
#   TOWN=Town02 PORT=8400 sbatch ... slurm/bench_probe_fresh.sh
set -u
cd ~/carla-rl
: "${TOWN:?}" "${PORT:?}"
C=$HOME/carla-rl/checkpoints/il_bc_c0/step_15000
export GOAL_CAM_MATCH=1 EVAL_PROBE=1 EVAL_EPISODES=${N:-150} BENCH_BASE_OVERLAY=slurm/il_eval_chw.env
for band in easy medium hard; do
  case $band in easy) base=31000;; medium) base=32000;; hard) base=33000;; esac
  echo "=== FRESH PROBE $TOWN $band seeds $base..$((base + EVAL_EPISODES - 1)) ==="
  REWARD_OVERLAY=slurm/bench_$band.env ENV_TOWNS=$TOWN EVAL_SEED=$base \
    RUN_NAME=bench_probe_fresh_${TOWN}_$band CARLA_PORT=$PORT EVAL_FROM=$C bash slurm/train_sac.sbatch
  echo "=== FRESH PROBE $TOWN $band rc=$? ==="
done
echo "=== FRESH PROBES DONE ==="
