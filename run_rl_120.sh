#!/usr/bin/env bash
# RL stage. Single-instance by construction.
#
# WHY THE LOCK. Three separate times this session, two trainers ended up
# running against the SAME CARLA server on port 2000 and silently corrupted
# each other's episodes -- one run reported len=17.2 and a conclusion was drawn
# from it before the duplicate was noticed. The cause was always the same:
# a `pkill -f` that returned non-zero (or was blocked), followed by a launch
# that never verified the kill had worked.
#
# Verifying by hand is the thing that keeps failing, so it is not the fix. The
# lock makes a second instance IMPOSSIBLE rather than merely unlikely: flock is
# held for the lifetime of this script and released by the kernel when it dies,
# including on kill -9, so a crashed run cannot leave a stale lock behind.
set -uo pipefail
cd "$(dirname "$0")"

LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING TO START: another RL run already holds $LOCK." >&2
  echo "Running processes on this port:" >&2
  ps -eo pid,etime,args | grep -E "[t]rain_carla_tqc|[r]un_rl_120" >&2
  echo "Stop it first, then rerun. Do NOT launch a second trainer against" >&2
  echo "one CARLA server -- the episodes interleave and the metrics are junk." >&2
  exit 1
fi
echo "[Lock] acquired $LOCK (pid $$)"

CARLA_SH=/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4.sh

# Kill only OUR server, and confirm it is gone before starting a new one. The
# old code fired pkill and slept, which is what let a survivor coexist with a
# fresh launch.
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
  if pgrep -f '[C]arlaUE4-Linux-Shipping' >/dev/null; then
    echo "WARNING: a CARLA server survived 5 kill attempts" >&2
  fi
}

cleanup() { kill_servers; echo "=== RL stage done ==="; }
trap cleanup EXIT

for ATTEMPT in 1 2 3 4 5; do
  echo "=== RL attempt $ATTEMPT ==="
  kill_servers
  # SERVER LOG: OVERWRITE, AND FILTER THE SPAM.
  #
  # This was `>> logs/carla_server_2000.log`, appending across every restart
  # forever. CARLA emits "ERROR: Invalid session: no stream available with id
  # N" continuously -- one line per dropped sensor frame -- and the file
  # reached 87 GB, filled the root filesystem to 100%, and killed the training
  # run it was supposed to be diagnosing.
  #
  # `>` truncates per attempt so it cannot grow across restarts, and the grep
  # drops the one line that produced essentially all of the volume while
  # keeping any other server error worth reading. grep is line-buffered so a
  # crash message appears immediately rather than sitting in a pipe buffer.
  # 200>&- closes the lock fd in the CHILD. Without it the background CARLA
  # server inherits the flock and keeps holding it after this script exits --
  # MEASURED: an orphaned server from a finished mapping run held the lock for
  # 8.5 hours and blocked every subsequent job with "another CARLA job holds".
  # The SUBSHELL drops fd 200 before anything in the pipeline starts.
  # Closing it only on CarlaUE4 was not enough -- the `grep` stage
  # inherits it too, and an orphaned grep held the lock for 30 minutes
  # after its parent script had exited.
  ( exec 200>&-; "$CARLA_SH" -carla-port=2000 -quality-level=Epic -windowed -ResX=640 -ResY=360 2>&1 \
      | grep -v --line-buffered "Invalid session: no stream available" \
      > logs/carla_server_2000.log ) &
  sleep 40
  if ./run_town02_fixed.sh; then
    echo "=== RL finished cleanly ==="; break
  fi
  echo "=== attempt $ATTEMPT died; restarting server ==="
  sleep 10
done
