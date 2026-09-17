#!/usr/bin/env bash
# Supervisor for the dense-reward (K_DIST) training run.
set -uo pipefail
cd "$(dirname "$0")"
LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING TO START: another CARLA job holds $LOCK." >&2
  exit 1
fi
echo "[Lock] acquired $LOCK (pid $$)"
CARLA_SH=/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4.sh
# pgrep -x CANNOT match here: /proc/PID/comm is capped at 15 chars, so
# 'CarlaUE4-Linux-Shipping' (23) is truncated to 'CarlaUE4-Linux-' and an
# exact-name match never fires. That is why servers survived this trap and
# sat holding ~2.8 GB of GPU after their job had exited. -f matches argv.
kill_servers() {
  for _ in 1 2 3 4 5; do
    mapfile -t pids < <(pgrep -f '[C]arlaUE4-Linux-Shipping')
    [ ${#pids[@]} -eq 0 ] && return 0
    for p in "${pids[@]}"; do kill -9 "$p" 2>/dev/null; done
    sleep 3
  done
}
cleanup() { kill_servers; echo "=== dense RL done ==="; }
trap cleanup EXIT
for ATTEMPT in 1 2 3 4 5; do
  echo "=== dense RL attempt $ATTEMPT ==="
  kill_servers
  ( exec 200>&-; "$CARLA_SH" -carla-port=2000 -quality-level=Epic -windowed -ResX=640 -ResY=360 2>&1 \
      | grep -v --line-buffered "Invalid session: no stream available" \
      > logs/carla_dense.log ) &
  sleep 40
  if ./run_town02_dense.sh; then echo "=== dense RL finished cleanly ==="; break; fi
  echo "=== attempt $ATTEMPT died; restarting ==="
  sleep 10
done
