#!/usr/bin/env bash
# Stratified evaluation of the sparse+per-step policy: five distance bands x
# three towns, with the driving-quality metrics. The distance bands go well
# beyond the 5-20 m the policy trained on, deliberately -- 20-40, 40-70 and
# 70-113 m are all extrapolation and are expected to be weak.
set -uo pipefail
cd "$(dirname "$0")"
LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING TO START: another CARLA job holds $LOCK." >&2; exit 1
fi
echo "[Lock] acquired $LOCK (pid $$)"
export DISPLAY=:1 CARLA_PORT=2000
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
export VAE_CKPT=checkpoints/lane_vae_120/best.pkl
export MAX_EPISODE_STEPS=500
# EVAL MUST MATCH TRAINING EXACTLY, or the numbers measure the mismatch.
export LANE_FOLLOW=0 K_DIST=1.0 DELTA_MODE=positive
export K_DIST_GATE_MODE=positive K_DIST_TARGET_SPEED=20.0
export K_SPEED=0.02 SPEED_LIMIT_KMH=30.0 K_LIVING=0.005
export INCREMENTAL_ACTIONS=1 MAX_STEER=0.5 MAX_THROTTLE=0.6
export STEER_STEP=0.01 THROTTLE_STEP=0.01
export OBS_WITH_VECTOR=1 TARGET_SPEED=20.0
export EPISODE_HER=0 CHAIN_MAX_GOALS=1 EXPERT_PROB=0 LANE_STRICT=1 SHORT_HORIZON=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
CARLA_SH=/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4.sh
# pgrep -x CANNOT match here: /proc/PID/comm is capped at 15 chars, so
# 'CarlaUE4-Linux-Shipping' (23) is truncated to 'CarlaUE4-Linux-' and an
# exact-name match never fires. That is why servers survived this trap and
# sat holding ~2.8 GB of GPU after their job had exited. -f matches argv.
kill_servers() {
  for _ in 1 2 3 4 5; do
    mapfile -t p < <(pgrep -f '[C]arlaUE4-Linux-Shipping'); [ ${#p[@]} -eq 0 ] && return 0
    for q in "${p[@]}"; do kill -9 "$q" 2>/dev/null; done; sleep 3
  done
}
cleanup() { kill_servers; echo "=== eval done ==="; }
trap cleanup EXIT
CKPT="${CKPT:-checkpoints/carla_sparse_perstep/checkpoint_150000.pkl}"
for ATTEMPT in 1 2 3; do
  echo "=== eval attempt $ATTEMPT ==="
  kill_servers
  ( exec 200>&-; "$CARLA_SH" -carla-port=2000 -quality-level=Epic -windowed \
      -ResX=640 -ResY=360 2>&1 | grep -v --line-buffered "Invalid session" \
      > logs/carla_evalcmp.log ) &
  sleep 40
  if python -u eval_policy.py --ckpt "$CKPT" --towns Town02,Town03,Town10HD \
        --episodes 25 --out results/eval_sparse_perstep.json; then
    echo "=== eval finished cleanly ==="; break
  fi
  echo "=== attempt $ATTEMPT died; retrying ==="; sleep 10
done
