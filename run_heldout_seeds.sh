#!/usr/bin/env bash
# Held-out evaluation, 10 episodes per cell across THREE SEEDS, both agents.
#
# Seeds matter here because every held-out number so far is single-seed at
# n=25, which cannot separate "this policy generalises" from "this draw of
# episodes was easy". Three seeds give a spread to report alongside the mean.
#
# EACH AGENT NEEDS ITS OWN ENV CONFIG. They were trained against different
# observation and action spaces -- the old one has no 'vector' key and uses
# absolute control; the new one has both. policy_io.load_agent_state compares
# the loaded pytree against the built agent and exits loudly on a mismatch, so
# a shared config would fail rather than silently mis-evaluate, but it would
# still fail.
set -uo pipefail
cd "$(dirname "$0")"

LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING TO START: another CARLA job holds $LOCK." >&2
  ps -eo pid,etime,args | grep -E "[t]rain_carla|[r]un_" >&2
  exit 1
fi
echo "[Lock] acquired $LOCK (pid $$)"

export DISPLAY=:1 CARLA_PORT=2000
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
export VAE_CKPT=checkpoints/lane_vae_120/best.pkl
export SHORT_HORIZON=1 LANE_STRICT=1 EPISODE_HER=0 CHAIN_MAX_GOALS=1 EXPERT_PROB=0
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
cleanup() { kill_servers; echo "=== held-out seed sweep done ==="; }
trap cleanup EXIT

start_server() {
  kill_servers
  ( exec 200>&-; "$CARLA_SH" -carla-port=2000 -quality-level=Epic -windowed \
      -ResX=640 -ResY=360 2>&1 | grep -v --line-buffered "Invalid session" \
      > logs/carla_heldout.log ) &
  sleep 40
}

# Per-agent env. Must match how each was TRAINED or the numbers measure the
# mismatch rather than the policy.
cfg_old() {
  export INCREMENTAL_ACTIONS=0 OBS_WITH_VECTOR=0
  export LANE_FOLLOW=0 K_DIST=0.0 K_SPEED=0.0 K_LIVING=0.0
  export MAX_EPISODE_STEPS=2000
  unset TARGET_SPEED MAX_STEER MAX_THROTTLE 2>/dev/null || true
}
cfg_new() {
  export INCREMENTAL_ACTIONS=1 OBS_WITH_VECTOR=1
  export MAX_STEER=0.5 MAX_THROTTLE=0.6 STEER_STEP=0.01 THROTTLE_STEP=0.01
  export LANE_FOLLOW=0 K_DIST=1.0 DELTA_MODE=positive
  export K_DIST_GATE_MODE=positive K_DIST_TARGET_SPEED=20.0
  export K_SPEED=0.02 SPEED_LIMIT_KMH=30.0 K_LIVING=0.005
  export TARGET_SPEED=20.0 MAX_EPISODE_STEPS=500
}

mkdir -p results/heldout_seeds logs
for AGENT in old new; do
  case "$AGENT" in
    old) cfg_old; CKPT=checkpoints/carla_tqc_town02_200m/checkpoint_1000000.pkl ;;
    new) cfg_new; CKPT=checkpoints/carla_sparse_perstep/checkpoint_150000.pkl ;;
  esac
  for SEED in 1 2 3; do
    OUT="results/heldout_seeds/${AGENT}_seed${SEED}.json"
    [ -f "$OUT" ] && { echo "== $AGENT seed $SEED cached"; continue; }
    for ATTEMPT in 1 2; do
      echo "== $AGENT seed $SEED (attempt $ATTEMPT) =="
      start_server
      if python -u eval_policy.py --ckpt "$CKPT" \
            --towns Town03,Town10HD --episodes 10 \
            --seed $((1000 * SEED)) --out "$OUT" \
            >> "logs/heldout_${AGENT}_s${SEED}.log" 2>&1; then
        echo "   ok $AGENT seed $SEED"; break
      fi
      echo "   attempt $ATTEMPT failed (see logs/heldout_${AGENT}_s${SEED}.log)"
      sleep 10
    done
  done
done
echo "== all cells written to results/heldout_seeds/ =="
