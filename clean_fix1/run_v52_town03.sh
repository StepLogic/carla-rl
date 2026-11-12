#!/usr/bin/env bash
# v52 on HELD-OUT Town03 (user 2026-10-06 "evaluate in town03"): goal mode,
# 5-200 m, EVAL_FAULT_ENDS=fault, 100 episodes, seed 0. 1M (final) then 800k
# (best on Town05). Town03 is unseen by both the policy and the seg-VAE.
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl/clean_fix1
C=checkpoints/carla_gated_v52_segvae_samesetup_t05
for S in 1000000 800000; do
  OUT=results/v52_eval/town03_checkpoint_${S}_goal.json
  [ -f "$OUT" ] && { echo "=== skip $S ==="; continue; }
  echo "=== $(date +%H:%M:%S) START Town03 $S ==="
  EPISODES=100 ./eval_v52.sh "$C/checkpoint_${S}.pkl" Town03; rc=$?
  [ -f "$OUT" ] && echo "=== $(date +%H:%M:%S) END $S rc=$rc OK ===" || echo "=== $(date +%H:%M:%S) END $S rc=$rc NO RESULT ==="
  sleep 15
done
echo "=== $(date +%H:%M:%S) ALL DONE ==="
