#!/usr/bin/env bash
# Wait for the trace collection to release the CARLA lock, then start the
# dense-reward training run. Both need port 2000 and the flock is non-blocking,
# so they must be serialised rather than raced.
set -uo pipefail
cd "$(dirname "$0")"
echo "[Queue] waiting for trace collection to finish..."
while ps -eo args | grep -q "[e]xplore_trace.py\|[r]un_trace_and_map.sh"; do
  sleep 60
done
echo "[Queue] trace done; starting dense-reward training"
exec ./run_rl_dense.sh
