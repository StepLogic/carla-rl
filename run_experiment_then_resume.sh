#!/usr/bin/env bash
# Wait for the next training checkpoint, pause training, run the node-placement
# experiment, then put training back.
#
# WHY WAIT FOR THE CHECKPOINT. Training was at step 173,913 with the next save at
# 175,000 -- about two minutes away. Killing immediately would throw away those
# steps for nothing; the resume can only restart from a checkpoint file.
#
# KILL ORDER IS LOAD-BEARING. run_lanefollow.sh wraps the trainer in a 5-attempt
# retry loop, so killing the python FIRST just makes the supervisor start it
# again. The supervisor must die first, then the trainer, then CARLA.
#
# NO pkill -f ON A PATTERN THIS SCRIPT'S OWN COMMAND LINE CONTAINS. That is how
# an earlier attempt killed its own shell mid-sequence and left the trainer
# running while believing it had stopped it. Patterns here name only
# train_carla_tqc.py / run_lanefollow.sh / the CARLA binary, none of which
# appear in this script's argv.
set -uo pipefail
cd "$(dirname "$0")"

STAMP="$(date +%Y%m%d_%H%M)"
WAIT_CKPT="${WAIT_CKPT:-checkpoints/carla_lanefollow/checkpoint_175000.pkl}"
WAIT_MAX_S="${WAIT_MAX_S:-900}"
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "waiting up to ${WAIT_MAX_S}s for ${WAIT_CKPT}"
deadline=$(( $(date +%s) + WAIT_MAX_S ))
while [ ! -f "$WAIT_CKPT" ]; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    say "checkpoint did not appear in time; proceeding anyway from the latest one"
    break
  fi
  sleep 15
done
# CKPT overrides the newest-file default. NEEDED, not cosmetic: the newest
# checkpoint is not the best one for this measurement. At checkpoint_175000 all
# 60 rollout episodes died on lane_any within ~23 steps at the 40-113 m band,
# against ~253 steps/episode at checkpoint_50000 on the same seeds -- the pinned
# 5-20 m curriculum has specialised the policy onto short hops. A rollout that
# never reaches a junction cannot measure node placement at junctions.
LATEST="${CKPT:-$(ls -t checkpoints/carla_lanefollow/checkpoint_*.pkl 2>/dev/null | head -1)}"
say "using checkpoint: ${LATEST:-NONE}"

say "stopping training (supervisor -> trainer -> server)"
pkill -f "run_lanefollow.sh" 2>/dev/null; sleep 2
pkill -f "train_carla_tqc.py" 2>/dev/null; sleep 5
pkill -f "train_carla_tqc.py" 2>/dev/null   # SIGKILL pass for a wedged trainer
pgrep -f '[C]arlaUE4-Linux-Shipping' | xargs -r kill -9 2>/dev/null
sleep 8
if pgrep -f "train_carla_tqc.py" >/dev/null || pgrep -f '[C]arlaUE4-Linux-Shipping' >/dev/null; then
  say "WARNING: something survived; the port lock may still be held"
fi
say "free GPU now: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"

say "=== STAGE 1: entropy rollout with pose (${LATEST}) ==="
CKPT="$LATEST" EPISODES="${EPISODES:-60}" OUT=results/entropy_nodes.json \
  ./run_entropy_junctions.sh > "logs/entropy_nodes_${STAMP}.log" 2>&1
say "stage 1 exit=$?"

if [ -s results/entropy_nodes.json ]; then
  say "=== STAGE 2: node-placement analysis ==="
  python -u node_placement_entropy.py \
      --in results/entropy_nodes.json \
      --out-json results/node_placement.json \
      --out-png results/node_placement.png \
      > "logs/node_placement_${STAMP}.log" 2>&1
  say "stage 2 exit=$?"
  python -u plot_entropy_junctions.py \
      --in results/entropy_nodes.json \
      --out results/entropy_junctions_latest.png \
      >> "logs/node_placement_${STAMP}.log" 2>&1
  say "figures: results/node_placement.png results/entropy_junctions_latest.png"
else
  say "STAGE 2 SKIPPED: results/entropy_nodes.json missing or empty"
fi

say "=== STAGE 3: resume training ==="
RESUME=1 ./run_lanefollow.sh > "logs/lanefollow_resume_${STAMP}.log" 2>&1
say "stage 3 exit=$? -- done"
