#!/usr/bin/env bash
# Topological mapping: teleport over every road waypoint, capture frames, build
# a SIFT+VLAD graph with FAISS retrieval.
#
# LOCALIZATION IS GRAYSCALE, and already was. SIFTVLAD.preprocess converts each
# frame with cv2.cvtColor(..., COLOR_*2GRAY) before detectAndCompute, and SIFT
# is a single-channel descriptor by construction -- colour never reaches the
# matcher. The RGB camera is kept because the stored node images are also used
# for viewing and for any later comparison against the VAE latent; only the
# localization path is grayscale.
#
# Towns: Town02 first (the map the policy was trained and evaluated on, so map
# quality and policy quality are separable), then the two HELD-OUT towns from
# the frozen split. Mapping a held-out town is not leakage -- the map is built
# at test time from observations, which is what a topological-memory method is
# entitled to do; the POLICY still never trained there.
set -uo pipefail
cd "$(dirname "$0")"

LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING TO START: another CARLA job holds $LOCK." >&2
  ps -eo pid,etime,args | grep -E "[t]rain_carla_tqc|[r]un_rl_120|[r]un_mapping" >&2
  exit 1
fi
echo "[Lock] acquired $LOCK (pid $$)"

CARLA_SH=/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4.sh
export DISPLAY=:1
# INSERT the loop-closure edges that detect_similar_nodes finds. Without this
# the graph is a bare chain -- the first Town02 map reported 1517 loop closures
# and added none, giving 755 nodes / 754 edges. Off by default in
# eval_graph_builder_vlad so the CORL 2026 numbers are unaffected.
export MAP_LOOP_CLOSURES=1
mkdir -p logs corl-2026-results

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
cleanup() { kill_servers; echo "=== mapping done ==="; }
trap cleanup EXIT

TOWNS="${MAP_TOWNS:-Town02 Town03 Town10HD}"

for TOWN in $TOWNS; do
  OUT="corl-2026-results/${TOWN,,}_map"
  if [ -f "$OUT/graph.pkl" ] || [ -f "$OUT/topological_graph.pkl" ]; then
    echo "== $TOWN already mapped ($OUT) -- skipping"
    continue
  fi
  for ATTEMPT in 1 2; do
    echo "== mapping $TOWN (attempt $ATTEMPT) =="
    kill_servers
    # Server log truncated per attempt and the sensor-stream spam filtered:
    # appending it unfiltered previously reached 87 GB and filled the disk.
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
        > "logs/carla_map_${TOWN}.log" ) &
    sleep 40
    if timeout 7200 python -u corl_2026_map_town02.py \
          --town "$TOWN" --output-dir "$OUT" \
          --sampling-resolution 2.0 --image-size 256 256 \
          >> "logs/map_${TOWN}.log" 2>&1; then
      echo "   ok $TOWN"; break
    fi
    echo "   attempt $ATTEMPT failed for $TOWN (see logs/map_${TOWN}.log)"
    sleep 10
  done
done
