#!/usr/bin/env bash
# Park until the GPU can actually hold this job, then start goal-free training.
#
# 6000 MiB, measured not guessed: the CARLA server takes ~2.8 GB and JAX settled
# at ~2.3 GB in the goal-conditioned run, so ~5.1 GB is the real footprint and
# 6000 leaves headroom for the CUDA-context spike that a free-memory check
# cannot predict. The 22:34 attempt started with 2956 MiB free and died with
# CUDA_ERROR_OUT_OF_MEMORY before the first jitted call.
#
# It NEVER touches the peer's processes -- it waits for them to finish.
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
NEED_MB="${NEED_MB:-9500}"; POLL_S="${POLL_S:-60}"; MAX_WAIT_H="${MAX_WAIT_H:-10}"
say() { echo "[$(date +%H:%M:%S)] $*"; }
say "waiting for >= ${NEED_MB} MiB free (poll ${POLL_S}s, give up after ${MAX_WAIT_H}h)"
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
while :; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -ge "$NEED_MB" ]; then say "GPU free: ${free} MiB -- starting"; break; fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    say "GAVE UP after ${MAX_WAIT_H}h; only ${free} MiB free. Nothing started."; exit 1
  fi
  sleep "$POLL_S"
done
exec ./run_lanefollow_nogoal.sh
