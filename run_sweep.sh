#!/bin/bash
# Thin wrapper: env parity with training, then ONE python process does the sweep.
# No CARLA launcher here -- sweep_eval.py kills all CARLA, starts one from
# Python, builds ONE env and only resets it.
set -uo pipefail
cd "$(dirname "$0")"
set +u; eval "$(grep -E '^export ' run_gated.sh)"; set -u
# THEN OVERLAY THE RUN'S OWN RESOLVED SETTINGS, if given.
# The line above re-reads run_gated.sh's DEFAULTS, not the settings the
# checkpoint was actually trained with. Measured 2026-09-12 against the live
# v6m run: a sweep would have evaluated at DISCOUNT=1.0 while the run trains at
# 0.99, with explore off-road FATAL while the run has it non-terminal, and
# without the per-step goal penalties. Same class of bug the point-goal session
# hit (8 m visit ring offline against 3 m in training).
# RUN_ENV=results/run_env/<policy_folder>.env is written by the launcher.
if [ -n "${RUN_ENV:-}" ]; then
  if [ -f "$RUN_ENV" ]; then
    echo "[Env] overlaying the run's own settings from $RUN_ENV"
    set +u; while IFS='=' read -r k v; do case "$k" in ''|'#'*) continue;; esac; export "$k=$v"; done < "$RUN_ENV"; set -u
  else
    echo "[Env] FATAL: RUN_ENV=$RUN_ENV does not exist" >&2; exit 1
  fi
fi

# QUALITY FOLLOWS THE RUN when its RUN_ENV records one (v7 trained at Low, 2026-09-16). Forcing Epic after the overlay
# would evaluate a Low-trained policy on out-of-distribution frames. Older snapshots never recorded CARLA_QUALITY, so
# their sweeps still get Epic exactly as before.
export CARLA_START_SERVER=1 CARLA_QUALITY="${CARLA_QUALITY:-Epic}"
export CARLA_PORT=2000 CARLA_PORTS=2000
export EVAL_FORCE_MODE=1 EXPERT_PROB=0 RESUME=0 DRAW_GOAL=1
# MEM_FRACTION/ALLOCATOR may be pre-set by the caller (Town12 needs the policy
# process small: the map alone holds 7.7 GB of the 12 GB card).
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}"
[ -n "${XLA_PYTHON_CLIENT_ALLOCATOR:-}" ] && export XLA_PYTHON_CLIENT_ALLOCATOR

CKPT="${CKPT:-checkpoints/carla_gated_her_v4/checkpoint_350000.pkl}"
TOWN="${TOWN:-Town02}"
# ACTION_MODE: "sample" (default, what every earlier sweep used -- the policy is
# stochastic, so this measures the policy as it actually acts) or
# "deterministic" (the mean action). Recorded in every result JSON as
# action_mode, so a row always states which it was. NEVER pool the two.
ACTION_MODE="${ACTION_MODE:-sample}"
STAMP=$(date +%Y%m%d_%H%M)
mkdir -p logs results
# EXIT 75 MEANS "THE SIMULATOR DIED, RERUN ME". sweep_eval.py refuses to
# rebuild an env in-process (the dead env's libcarla threads abort the whole
# interpreter), so it exits and this loop starts a fresh process that skips
# every seed whose JSON already exists (--resume, same --tag). Completed seeds
# are never redone; a seed that died is redone from scratch, never half-kept.
VIDEO_ARGS=()
[ "${VIDEO:-1}" = "1" ] && VIDEO_ARGS=(--video-dir "results/${TOWN,,}_videos_${STAMP}" --max-videos "${MAX_VIDEOS:-5}")
for ATTEMPT in 1 2 3 4 5; do
  echo "=== sweep process $ATTEMPT (tag $STAMP) ==="
  python -u sweep_eval.py --ckpt "$CKPT" --town "$TOWN" \
      --seeds "${SEEDS:-1234,1235,1236,1237}" --episodes "${EPISODES:-25}" \
      --bands "${BANDS:-20-50}" --max-steps 750 \
      --action-mode "${ACTION_MODE:-sample}" \
      "${VIDEO_ARGS[@]}" --tag "$STAMP" --resume
  rc=$?
  [ $rc -eq 75 ] || break
  echo "=== simulator died; rerunning for the remaining seeds ==="
  sleep 5
done
echo "=== sweep exit $rc ==="
