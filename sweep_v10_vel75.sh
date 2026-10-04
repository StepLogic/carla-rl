#!/usr/bin/env bash
# SEQUENTIAL SWEEP over the finished v10-vel75 run (user 2026-09-23:
# "proceed" on the three follow-up evals). ONE INVOCATION AT A TIME, each
# holding the port lock; the first starts the server (Town05), later
# invocations attach to the live one (same town, so NO load_world — the
# known crash vector is a mid-sweep world load, which never happens here).
#
#   Phase A  band-restricted evals of the FINAL checkpoint (5-20 / 20-50 /
#            50-200 m) — localizes where the 0.417 lives.
#   Phase B  checkpoint sweep at the DEFAULT band (300k / 750k / 1M / 1.5M;
#            2M already measured) — is the final checkpoint a dip?
#   Phase C  matched-step control: the killed 0.5-run's checkpoint_300000
#            under ITS snapshot (GOAL_SPEED_BONUS=0.5), same seed-0 episode
#            set, same band — the vel75-vs-0.5 verdict at 300k.
set -uo pipefail
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl

# Kill-all-CARLA rule before any sweep (verified loop; my own tool shell is
# never matched thanks to the bracketed pattern).
count=$(pgrep -fc 'Linux-Shippin[g]' || true)
if [ "${count:-0}" -gt 0 ]; then
  echo "[pre-kill] CARLA running (${count}) — killing"
  pkill -f 'Linux-Shippin[g]' 2>/dev/null || true
  pkill -f 'CARLA_0\.9\.1[5]' 2>/dev/null || true
  pkill -f 'CarlaUE4' 2>/dev/null || true
  for _ in $(seq 1 20); do
    pgrep -f 'Linux-Shippin[g]' >/dev/null || break
    sleep 1
  done
  echo "[pre-kill] after loop: $(pgrep -fc 'Linux-Shippin[g]' || echo 0) left"
fi

V75SNAP=results/run_env/carla_gated_her_v10_vel75_vc1.env
V50SNAP=results/run_env/carla_gated_her_v10_2m_vc1.env
V75CK=checkpoints/carla_gated_her_v10_vel75_vc1
V50CK=checkpoints/carla_gated_her_v10_2m_vc1

# -- Phase A: bands on the final checkpoint (same snapshot, one server) ------
GOAL_MIN=5   GOAL_MAX=20  SNAP=$V75SNAP bash eval_v10_vel75.sh \
  "$V75CK/checkpoint_2000000.pkl" Town05 _b5-20   || echo "[A1 failed]"
GOAL_MIN=20  GOAL_MAX=50  SNAP=$V75SNAP bash eval_v10_vel75.sh \
  "$V75CK/checkpoint_2000000.pkl" Town05 _b20-50  || echo "[A2 failed]"
GOAL_MIN=50  GOAL_MAX=200 SNAP=$V75SNAP bash eval_v10_vel75.sh \
  "$V75CK/checkpoint_2000000.pkl" Town05 _b50-200 || echo "[A3 failed]"

# -- Phase B: checkpoint sweep, default (training) band ----------------------
for k in 300000 750000 1000000 1500000; do
  SNAP=$V75SNAP bash eval_v10_vel75.sh "$V75CK/checkpoint_${k}.pkl" Town05 || echo "[B $k failed]"
done

# -- Phase C: 0.5-control at 300k under its own snapshot ---------------------
SNAP=$V50SNAP bash eval_v10_vel75.sh "$V50CK/checkpoint_300000.pkl" Town05 _ctrl05 || echo "[C failed]"

echo "=== sweep complete $(date +%H:%M:%S) ==="