#!/usr/bin/env bash
# Waits for run_v52_evals.sh to finish, then runs the UNSCORED goal-marker watch
# (one CARLA at a time: two Epic servers + two evals OOMed the GPU at 22:06).
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl/clean_fix1
until grep -q 'ALL DONE' logs/v52_eval_runner.log; do sleep 30; done
sleep 20
EVAL_PORT=2000 EPISODES=20 ./watch_v52.sh checkpoints/carla_gated_v52_segvae_samesetup_t05/checkpoint_1000000.pkl Town05
