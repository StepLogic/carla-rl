#!/usr/bin/env bash
# Run the placement sweep repeatedly until it stops making progress.
#
# WHY ROUNDS. run_sweep_detached.sh skips any point already on disk, so running
# it again costs nothing for finished work and retries only what is missing.
# Points fail for transient reasons -- a supervisor kill, a CARLA hiccup -- and
# a second pass usually lands them.
#
# WHY IT WAITS FIRST, AND THIS IS THE POINT OF THE FILE. Two sweeps running at
# once destroy each other: the first takes /tmp/carla_rl_port2000.lock and every
# attempt by the second is refused instantly, burning all its retries without
# running anything. That happened once here and cost 33 points. So this waits
# for any existing sweep to exit before starting, and runs its own rounds
# strictly sequentially.
#
# Launch it the same way as the sweep, out of the task supervisor's reach:
#   setsid nohup ./run_sweep_rounds.sh > logs/sweep_rounds.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"

ROUNDS="${ROUNDS:-4}"
OUT_DIR="${OUT_DIR:-results/placement_town02}"

echo "=== sweep rounds: waiting for any running sweep to finish ==="
while pgrep -f '[r]un_sweep_detached.sh' >/dev/null 2>&1; do
  echo "[wait] a sweep is still running; $(ls "$OUT_DIR"/*.json 2>/dev/null | wc -l) results so far"
  sleep 60
done
echo "[wait] clear"

prev=0
for round in $(seq 1 "$ROUNDS"); do
  echo "=== round $round of $ROUNDS ==="
  date
  ./run_sweep_detached.sh
  have=$(ls "$OUT_DIR"/*.json 2>/dev/null | wc -l)
  echo "=== round $round finished with $have results ==="
  # STOP WHEN A WHOLE ROUND ADDS NOTHING. Everything still missing is failing
  # for a reason a further identical pass will not fix, and spinning would just
  # hide that behind an endless log.
  if [ "$have" -le "$prev" ]; then
    echo "=== round $round added nothing (was $prev, now $have); stopping ==="
    break
  fi
  prev="$have"
done

echo "=== sweep rounds done: $(ls "$OUT_DIR"/*.json 2>/dev/null | wc -l) results ==="
date
