#!/usr/bin/env bash
# Wait for the goal-free policy to reach a usable checkpoint, pause its training,
# measure entropy at junctions, then put training back.
#
# WHY WAIT RATHER THAN USE checkpoint_25000. At 25k the policy drove 214 steps a
# side and was still improving fast. The goal-conditioned version of this
# measurement was badly served by a weak checkpoint: at checkpoint_175000 every
# rollout episode died within ~23 steps and produced ZERO usable junction
# samples, while checkpoint_50000 on the same seeds drove 253. A policy that
# cannot reach a junction cannot tell you anything about entropy at junctions.
#
# KILL ORDER: supervisor, then trainer, then server. run_lanefollow_nogoal.sh
# retries 5x, so killing the trainer first just makes it start another.
set -uo pipefail
cd "$(dirname "$0")"
STAMP="$(date +%Y%m%d_%H%M)"
WAIT_CKPT="${WAIT_CKPT:-checkpoints/carla_lanefollow_nogoal/checkpoint_50000.pkl}"
WAIT_MAX_S="${WAIT_MAX_S:-3600}"
EPISODES="${EPISODES:-60}"
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "waiting up to ${WAIT_MAX_S}s for ${WAIT_CKPT}"
deadline=$(( $(date +%s) + WAIT_MAX_S ))
while [ ! -f "$WAIT_CKPT" ]; do
  if [ "$(date +%s)" -ge "$deadline" ]; then say "timed out; using newest instead"; break; fi
  sleep 30
done
CKPT="${CKPT:-$(ls -t checkpoints/carla_lanefollow_nogoal/checkpoint_*.pkl 2>/dev/null | head -1)}"
say "using checkpoint: ${CKPT:-NONE}"
[ -z "$CKPT" ] && { say "ABORT: no goal-free checkpoint exists yet"; exit 1; }

say "pausing goal-free training"
pgrep -f "run_lanefollow_nog""oal.sh" | xargs -r kill -9 2>/dev/null; sleep 2
pgrep -f "train_carla""_tqc" | xargs -r kill -TERM 2>/dev/null; sleep 5
pgrep -f "train_carla""_tqc" | xargs -r kill -9 2>/dev/null
pgrep -f '[C]arlaUE4-Linux-Shipping' | xargs -r kill -9 2>/dev/null
sleep 8
say "free VRAM now: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"

say "=== STAGE 1: goal-free entropy rollout ==="
CKPT="$CKPT" EPISODES="$EPISODES" OUT=results/entropy_nogoal.json \
  ./run_entropy_junctions_nogoal.sh > "logs/entropy_nogoal_${STAMP}.log" 2>&1
say "stage 1 exit=$?"

if [ -s results/entropy_nogoal.json ]; then
  say "=== STAGE 2: entropy-at-junctions figure ==="
  python -u plot_entropy_junctions.py --in results/entropy_nogoal.json \
      --out results/entropy_junctions_nogoal.png \
      > "logs/entropy_nogoal_plot_${STAMP}.log" 2>&1
  say "stage 2 exit=$?"
else
  say "STAGE 2 SKIPPED: results/entropy_nogoal.json missing or empty"
fi

say "=== STAGE 3: resume goal-free training ==="
RESUME=1 ./run_lanefollow_nogoal.sh > "logs/lanefollow_nogoal_resume_${STAMP}.log" 2>&1
say "stage 3 exit=$? -- done"
