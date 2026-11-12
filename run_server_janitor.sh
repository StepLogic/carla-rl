#!/usr/bin/env bash
# Recycle the CARLA server when it has leaked enough to block the sweep.
#
# WHY. CARLA grows steadily here -- measured 5.2 -> 7.1 GB of VRAM over about
# an hour of use, three times in one session. Once free VRAM drops below the
# evaluation's gate, every point waits out its deadline and the sweep stalls
# with no error. Restarting the server returns the card to ~11 GB.
#
# WHY IT IS SAFE. It recycles ONLY when /tmp/carla_rl_port2000.lock is
# unheld, which means no evaluation is running: the driver takes that lock for
# its whole run. Killing the server under a live evaluation would fail it
# mid-episode, so the lock is the interlock.
#
# It also refuses to act while a driver is merely WAITING on the GPU gate,
# because such a driver holds the lock too -- and a driver that captured
# "no server -> need 9500 MiB" at launch can never pass the gate once a server
# is resident. That deadlock is fixed by killing the orphan, not by recycling
# under it, so this reports it rather than papering over it.
#
#   setsid nohup ./run_server_janitor.sh > logs/janitor.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"

LOCK=/tmp/carla_rl_port2000.lock
MIN_FREE_MB="${MIN_FREE_MB:-5000}"
PERIOD="${PERIOD:-120}"
CARLA_SH=/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4.sh

echo "=== server janitor: recycle below ${MIN_FREE_MB} MiB free, every ${PERIOD}s ==="

while true; do
  sleep "$PERIOD"

  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -z "${free:-}" ] && continue
  [ "$free" -ge "$MIN_FREE_MB" ] && continue

  if fuser "$LOCK" >/dev/null 2>&1; then
    echo "[janitor] ${free} MiB free but the lock is held; not touching the server"
    continue
  fi

  echo "[janitor] ${free} MiB free and no evaluation running -- recycling CARLA"
  date
  pkill -9 -f '[C]arlaUE4-Linux-Shipping' 2>/dev/null
  pkill -9 -f '[C]arlaUE4.sh' 2>/dev/null
  sleep 5
  DISPLAY=:1 setsid nohup "$CARLA_SH" -carla-port=2000 -quality-level=Epic \
      -windowed -ResX=640 -ResY=360 > logs/carla_detached.log 2>&1 < /dev/null &
  # Give it time to come up before the next point tries to connect.
  sleep 45
  echo "[janitor] restarted; $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) free"
done
