#!/usr/bin/env bash
# Evaluate the trained policy: in-distribution + held-out, stratified by goal
# distance. Env matches TRAINING exactly -- 120x120 frames, the same VAE, the
# same episode limit -- because any difference here would be measuring the
# mismatch rather than the policy.
set -uo pipefail
cd "$(dirname "$0")"

LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING TO START: another CARLA job holds $LOCK." >&2
  ps -eo pid,etime,args | grep -E "[t]rain_carla_tqc|[r]un_rl_120|[r]un_mapping|[e]val_policy" >&2
  exit 1
fi
echo "[Lock] acquired $LOCK (pid $$)"

export DISPLAY=:1
export CARLA_PORT=2000
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
export VAE_CKPT=checkpoints/lane_vae_120/best.pkl
export MAX_EPISODE_STEPS=2000
export LANE_STRICT=1
export SHORT_HORIZON=1
export CHAIN_MAX_GOALS=1          # ONE goal per episode: this measures point-to-point
export EXPERT_PROB=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

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
cleanup() { kill_servers; echo "=== eval done ==="; }
trap cleanup EXIT

CKPT="${CKPT:-checkpoints/carla_tqc_town02_200m/checkpoint_1000000.pkl}"

for ATTEMPT in 1 2 3; do
  echo "=== eval attempt $ATTEMPT ==="
  kill_servers
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
      > logs/carla_eval.log ) &
  sleep 40
  if python -u eval_policy.py --ckpt "$CKPT" \
        --towns Town02,Town03,Town10HD --episodes 25 \
        --out results/policy_eval.json; then
    echo "=== eval finished cleanly ==="; break
  fi
  echo "=== attempt $ATTEMPT died; retrying ==="
  sleep 10
done
