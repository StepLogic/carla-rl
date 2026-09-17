#!/usr/bin/env bash
# ATTRIBUTION EVAL: was the collapse the harder goal band, or HER?
#
# Three changes landed together at step 75k -- EPISODE_HER=1, GOAL_DIST_MIN
# 5->20, and a restart that discarded the replay buffer and optimiser state --
# and success then fell 0.80 -> 0.45 while idle went 0.10 -> 0.50. Nothing in
# the training log can separate those three, because no configuration ran
# alone.
#
# This does separate them. The SAME band (20-46.51 m), the SAME episode count
# and the SAME seeds, against weights from before the change and after it:
#
#   checkpoint_75000   pre-HER, pre-floor  -- trained on 5-46.51 m
#   checkpoint_100000  post, mid-collapse
#   checkpoint_125000  post, fully collapsed
#
# READING IT. If 75000 also scores ~0.45 here, the band explains everything and
# HER is exonerated -- the policy never could do 20-46 m goals, and the old
# 0.80 was carried by the 5-20 m goals the floor removed. If 75000 scores ~0.80
# while 125000 scores ~0.45, the band is innocent and the regression is HER or
# the restart.
set -uo pipefail
cd "$(dirname "$0")"

LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING TO START: another CARLA job holds $LOCK." >&2
  ps -eo pid,etime,args | grep -E "[t]rain_carla|[r]un_|[w]atch_policy" >&2
  exit 1
fi
echo "[Lock] acquired $LOCK (pid $$)"

export DISPLAY=:1
export CARLA_PORT=2000 CARLA_PORTS=2000

# EVERY ONE OF THESE MUST MATCH run_timetogoal.sh. The observation space is
# built from them, so a mismatch does not degrade the score -- it makes
# load_agent_state raise on a pytree mismatch and the eval measures nothing.
export TIME_TO_GOAL=1 STEP_COST=0.1 TTG_MAX_STEPS=1000
export MAX_EPISODE_STEPS=1000
export K_DIST=0.0 K_SPEED=0.0 K_LIVING=0.0
export INCREMENTAL_ACTIONS=0 MAX_STEER=0.5 MAX_THROTTLE=0.6
export OBS_WITH_VECTOR=0
export ENV_TOWNS=Town02 EVAL_TOWNS=Town02
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
export VAE_CKPT=checkpoints/lane_vae_120/best.pkl
export LANE_WEDGE=1 LANE_STRICT=1
export JUNCTION_GOAL_FRAC=0.6
export SHORT_HORIZON=1 MAX_DISTANCE_M=1200
export TARGET_SPEED=20.0
export EPISODE_HER=0 CHAIN_MAX_GOALS=1 EXPERT_PROB=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

BAND="${BAND:-20-46.51}"
EPISODES="${EPISODES:-25}"

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
cleanup() { kill_servers; echo "=== isolation eval done ==="; }
trap cleanup EXIT

kill_servers
# fd 200 dropped FIRST: an orphaned grep inheriting this lock once held it for
# 8.5 hours after the job it belonged to had exited.
( exec 200>&-; "$CARLA_SH" -carla-port=2000 -quality-level=Epic \
    -windowed -ResX=640 -ResY=360 2>&1 \
  | grep -v --line-buffered "Invalid session: no stream available" \
  > logs/carla_isolation_eval.log ) &
sleep 40

mkdir -p results
for CK in 75000 100000 125000; do
  echo
  echo "################ checkpoint_${CK}  band ${BAND}  n=${EPISODES} ################"
  python -u eval_policy.py \
    --ckpt "checkpoints/carla_timetogoal/checkpoint_${CK}.pkl" \
    --towns Town02 --bands "$BAND" --episodes "$EPISODES" \
    --max-steps 1000 --port 2000 --seed 1234 \
    --out "results/isolation_${CK}_${BAND}.json" \
    || echo "!!! checkpoint_${CK} eval failed"
done
