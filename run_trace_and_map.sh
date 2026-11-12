#!/usr/bin/env bash
# Self-built topological maps, matching the peer Habitat construction:
#   explore (trained policy, no road graph) -> trace -> thin -> trace edges
#   -> certified shortcuts from the depth ring -> components / routable.
set -uo pipefail
cd "$(dirname "$0")"

LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING TO START: another CARLA job holds $LOCK." >&2
  ps -eo pid,etime,args | grep -E "[t]rain_carla_tqc|[r]un_rl_120|[r]un_mapping|[e]val_policy|[e]xplore_trace" >&2
  exit 1
fi
echo "[Lock] acquired $LOCK (pid $$)"

export DISPLAY=:1
export CARLA_PORT=2000
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
export VAE_CKPT=checkpoints/lane_vae_120/best.pkl
export MAX_EPISODE_STEPS=2000 LANE_STRICT=1 SHORT_HORIZON=1
export CHAIN_MAX_GOALS=1 EXPERT_PROB=0
# Lane departures do NOT sever the trace: this is a survey, not the driving
# task. Collisions still do. See sparse_reward.EXPLORE_MODE.
export EXPLORE_MODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# CAP OUR JAX FOOTPRINT. This box has one 12 GB card shared with other
# sessions; a Habitat pytest run holding 6.3 GB plus our own CARLA server at
# 4.8 GB left ~1.2 GB and JAX failed with "no supported devices found for
# platform CUDA" -- which reads like a driver fault and is actually contention.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.15

# WAIT FOR HEADROOM rather than race for it. Another session's job is not ours
# to kill, and retrying immediately just burns attempts against a full card.
wait_for_gpu() {
  local need_mib=${1:-2500} tries=${2:-60}
  for ((i=0; i<tries; i++)); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "$free" ] && return 0                 # no nvidia-smi: do not block
    [ "$free" -ge "$need_mib" ] && return 0
    echo "   waiting for GPU: ${free} MiB free, need ${need_mib} MiB"
    sleep 60
  done
  echo "   WARNING: proceeding with only ${free} MiB free"
}
export TRACE_STEPS="${TRACE_STEPS:-15000}"      # their budget, copied

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
cleanup() { kill_servers; echo "=== trace+map done ==="; }
trap cleanup EXIT

mkdir -p traces logs results
TOWNS="${TRACE_TOWNS:-Town02 Town03 Town10HD}"

for TOWN in $TOWNS; do
  OUT="traces/${TOWN,,}_trace.pkl"
  if [ -f "$OUT" ]; then echo "== $TOWN trace cached"; continue; fi
  for ATTEMPT in 1 2; do
    echo "== exploring $TOWN (attempt $ATTEMPT) =="
    kill_servers
    wait_for_gpu 2500 60
    # 200>&- so the server does not inherit and hold the lock after we exit.
    # The SUBSHELL drops fd 200 before anything in the pipeline starts.
    # Closing it only on CarlaUE4 was not enough -- the `grep` stage
    # inherits it too, and an orphaned grep held the lock for 30 minutes
    # after its parent script had exited.
    ( exec 200>&-; "$CARLA_SH" -carla-port=2000 -quality-level=Epic -windowed -ResX=640 -ResY=360 2>&1 \
        | grep -v --line-buffered "Invalid session: no stream available" \
        > "logs/carla_trace_${TOWN}.log" ) &
    sleep 40
    if timeout 5400 python -u explore_trace.py --town "$TOWN" --out "$OUT" \
          >> "logs/trace_${TOWN}.log" 2>&1; then
      echo "   ok $TOWN"; break
    fi
    echo "   attempt $ATTEMPT failed (see logs/trace_${TOWN}.log)"
    sleep 10
  done
done

kill_servers
echo "== building maps (no simulator needed) =="
python -u build_habitat_map.py --traces traces --out results/habitat_maps.json 2>&1 | tail -40
