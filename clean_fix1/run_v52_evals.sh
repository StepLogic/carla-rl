#!/usr/bin/env bash
# v52 checkpoint sweep (user 2026-10-05 "evaluate"): Town05, goal mode, 5-200 m,
# EVAL_FAULT_ENDS=fault, 100 episodes, seed 0 => the SAME episode set for every
# checkpoint. Sequential; a stage counts as done only if its JSON exists.
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl/clean_fix1
C=checkpoints/carla_gated_v52_segvae_samesetup_t05
for S in 1000000 950000 900000 800000; do
  OUT=results/v52_eval/town05_checkpoint_${S}_goal.json
  [ -f "$OUT" ] && { echo "=== skip $S (have $OUT) ==="; continue; }
  echo "=== $(date +%H:%M:%S) START $S ==="
  EPISODES=100 ./eval_v52.sh "$C/checkpoint_${S}.pkl" Town05; rc=$?
  [ -f "$OUT" ] && echo "=== $(date +%H:%M:%S) END $S rc=$rc OK ===" || echo "=== $(date +%H:%M:%S) END $S rc=$rc NO RESULT ==="
  sleep 15
done
echo "=== $(date +%H:%M:%S) ALL DONE ==="
