#!/usr/bin/env bash
# Overnight chain: park until the GPU is free, then run the node-placement
# experiment, then resume lane-following training.
#
# WHY THIS ORDER. The experiment is short (~30 min) and the training run is long
# (~4 h for the remaining 200k steps). Running the experiment FIRST means that if
# only one of the two gets a window before morning, it is the one that produces a
# new result rather than the one that adds steps to an already-saturated policy
# (eval has sat at 0.93-0.97 since step 25k).
#
# WHY 7 GB. Peer session robot-learning-tools-c6 reported its two jobs cap at
# about 9 GB combined on this 12.3 GB card: train_image_goal_hpc.py (~5 h left)
# and a queued Cantwell/Tradewinds trace+eval suite (~6-9 h). Its training alone
# plus one eval sits near 8-9 GB, so 7 GB free is not reached until ITS TRAINING
# EXITS -- which is the correct moment for us to start, and is why a lower
# threshold would only earn a second OOM.
#
# THIS SCRIPT NEVER TOUCHES THE PEER'S PROCESSES. It waits; it does not make room.
set -uo pipefail
cd "$(dirname "$0")"

# 9500, not 6000. MEASURED, and the earlier figure was wrong in a way that
# looked safe. CARLA's footprint is NOT the ~2.8 GB recorded in the first
# session note -- that was the LOW end of a range. Observed across this work:
# 2611, 2775, 2808 and 5297 MiB for the same Epic-quality server, and a peer
# session measured 5448 MiB with 9221 MiB for CARLA+jax together. So the real
# requirement is ~9.2 GB, not ~5.1 GB, and a 6000 MiB gate PASSES on a card
# that cannot hold the run.
#
# THAT FAILURE IS DISTINCT FROM THE COMMAND-BUFFER ONE. A 6066 MiB gate let a
# gated run start alongside the Habitat job and jax died with
# RESOURCE_EXHAUSTED allocating 16 MB for the VAE's first convolution. That is
# a genuine shortfall; XLA_FLAGS=--xla_gpu_enable_command_buffer= was already
# set and did not help, because that flag fixes the phantom CUDA-graph
# allocation, not an actual one. Both failure modes exist and need different
# fixes.
#
# CONSEQUENCE: at ~9.2 GB for us and ~4.8 GB for the Habitat peer on one
# 12.3 GB card, our jobs and theirs CANNOT run concurrently. That is a
# scheduling fact, not a parameter to tune.
#
# DO NOT "fix" this by lowering -quality-level. It would cut CARLA's footprint,
# but lane_vae_120 is a FROZEN encoder trained on Epic-quality frames; changing
# render quality shifts the input distribution out from under it and would
# quietly degrade every arm.
NEED_MB="${NEED_MB:-9500}"
POLL_S="${POLL_S:-120}"
MAX_WAIT_H="${MAX_WAIT_H:-16}"
CKPT="${CKPT:-checkpoints/carla_lanefollow/checkpoint_150000.pkl}"
STAMP="$(date +%Y%m%d_%H%M)"

say() { echo "[$(date +%H:%M:%S)] $*"; }

say "waiting for >= ${NEED_MB} MiB free on the GPU (poll ${POLL_S}s, give up after ${MAX_WAIT_H}h)"
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  free=$(( total - used ))
  if [ "$free" -ge "$NEED_MB" ]; then
    say "GPU free: ${free} MiB available -- proceeding"
    break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    say "GAVE UP: ${MAX_WAIT_H}h elapsed and only ${free} MiB free. Nothing was run."
    exit 1
  fi
  sleep "$POLL_S"
done

# A second job could take the card between the check above and our start, so the
# CARLA lock is still the real mutex -- these scripts take it themselves.

say "=== STAGE 1: entropy rollout with pose, for node placement ==="
CKPT="$CKPT" EPISODES="${EPISODES:-60}" OUT=results/entropy_nodes.json \
  ./run_entropy_junctions.sh > "logs/entropy_nodes_${STAMP}.log" 2>&1
say "stage 1 exit=$?"

if [ -f results/entropy_nodes.json ]; then
  say "=== STAGE 2: node-placement analysis ==="
  python -u node_placement_entropy.py \
      --in results/entropy_nodes.json \
      --out-json results/node_placement.json \
      --out-png results/node_placement.png \
      > "logs/node_placement_${STAMP}.log" 2>&1
  say "stage 2 exit=$?"
  # The entropy-vs-junction figure is regenerated from the same, larger rollout
  # so both figures describe one dataset rather than two different ones.
  python -u plot_entropy_junctions.py \
      --in results/entropy_nodes.json \
      --out results/entropy_junctions_150k.png \
      >> "logs/node_placement_${STAMP}.log" 2>&1
else
  say "STAGE 2 SKIPPED: results/entropy_nodes.json was never written."
fi

say "=== STAGE 3: resume lane-following training from ${CKPT} ==="
RESUME=1 ./run_lanefollow.sh > "logs/lanefollow_resume_${STAMP}.log" 2>&1
say "stage 3 exit=$? -- overnight chain done"
