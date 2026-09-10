#!/usr/bin/env bash
# NODE-PLACEMENT ABLATION: uniform, random, policy (entropy peaks), SGM.
#
# SR and SPL for ONE gated policy under four ways of choosing graph nodes.
# Everything except placement is held identical -- one policy, ONE TRACE, one
# edge rule, one navigation harness, one episode set -- so a difference in the
# table is attributable to placement. eval_placement.py runs all four methods
# in ONE process for exactly that reason: a process per method would give each
# its own rollouts and quietly reintroduce the variable being controlled.
#
# THE BUDGET IS SGM'S OWN NODE COUNT. SGM is the only arm that derives its
# count, so the other three are given that number. An arm with more nodes
# trivially covers more of the world; this is node_placement_entropy.py's
# equal-budget convention and the only fair comparison.
#
# WHY THE GATED ARM AND NOT THE BASELINE. Two of the four methods read the
# critic as a DISTANCE -- SGM's merge test and the graph's edge weights -- and
# that reading is only defined for an arm trained on minimum time. Measured on
# these two arms, the critic correlates 0.98 with true time-to-goal with
# hindsight relabelling and 0.48 without, so the hindsight arm is the default.
set -uo pipefail
cd "$(dirname "$0")"

LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING TO START: another CARLA job holds $LOCK." >&2
  exit 1
fi
echo "[Lock] acquired $LOCK (pid $$)"

# Same measured requirement as training: ~9.2 GB. See run_gated.sh.
NEED_MB="${NEED_MB:-9500}"
MAX_WAIT_H="${MAX_WAIT_H:-6}"
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "${free:-0}" -ge "$NEED_MB" ] && { echo "[GPU] ${free} MiB free -- starting"; break; }
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[GPU] GAVE UP; ${free} MiB free." >&2; exit 1; }
  echo "[GPU] ${free} MiB free, need ${NEED_MB}; waiting"
  sleep 60
done

export DISPLAY=:1
export CARLA_PORT=2000 CARLA_PORTS=2000

# ---- identical to the four-way evaluation, so rows are comparable -----------
# See run_four_way_eval.sh for the town note: Town03/Town10HD are the frozen
# held-out set, Town01 is a train-split town this policy never saw.
TOWN="${TOWN:-Town02}"
export ENV_TOWNS="$TOWN" EVAL_TOWNS="$TOWN"
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
export PERCEPTION=vae
export VAE_CKPT=checkpoints/lane_vae_120/best.pkl
export INCREMENTAL_ACTIONS=0 MAX_STEER=0.5 MAX_THROTTLE=0.6
export TARGET_SPEED=20.0 SHORT_HORIZON=1
export GOAL_DIST_MIN=5.0 GOAL_DIST_START=20.0 GOAL_DIST_TARGET=20.0
export MAX_EPISODE_STEPS=2000 MAX_DISTANCE_M=200
export EXPERT_PROB=0 EPISODE_HER=0 CHAIN_MAX_GOALS=1
export EVAL_VIDEO=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

# ---- the gated arm's own configuration, copied from run_four_way_eval.sh ----
export GATED=1 OBS_WITH_MODE=1 OBS_WITH_GOAL=1 OBS_WITH_VECTOR=1
export LANE_FOLLOW=1 LANE_EXPLORE=0
export GATED_GOAL_REWARD=sorb STEP_COST=1.0
export GATED_GOAL_MAX_STEPS=1000 GATED_EXPLORE_MAX_STEPS=500
export LANE_INVASION_NOOP=1 DISCOUNT=1.0 EXPLORE_DISCOUNT=0.99
export EXPLORE_PROB=0.5 GOAL_BATCH_FRAC=0.5 TQC_KAPPA=1.0

ARM="${ARM:-gated_her}"
CKPT="${CKPT:-checkpoints/carla_gated_her/checkpoint_350000.pkl}"
METHODS="${METHODS:-uniform,random,policy,sgm}"
EPISODES="${EPISODES:-60}"
EVAL_SEED="${EVAL_SEED:-0}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-1000}"
BANDS="${BANDS:-5-20,20-40,40-70}"
COLLECT_EPISODES="${COLLECT_EPISODES:-20}"
COLLECT_STEPS="${COLLECT_STEPS:-500}"
TOWN_LC="$(echo "$TOWN" | tr '[:upper:]' '[:lower:]')"
OUT_DIR="${OUT_DIR:-results/placement_${TOWN_LC}}"
# ONE TRACE FOR EVERY BAND, AND ONE PER TOWN. The bands are separate processes
# because the goal sampler reads its band at import, but the TRACE does not
# depend on the band -- recollecting per band would compare four methods on
# three different maps.
#
# THE TOWN IS IN THE FILENAME, and that is not cosmetic: a Town02 trace reused
# in Town01 would build the map of one town and navigate another. Nothing
# downstream checks, and every number would look ordinary.
TRACE="${TRACE:-results/placement_${TOWN_LC}/trace_${ARM}.pkl}"
mkdir -p "$OUT_DIR" logs

if [ ! -f "$CKPT" ]; then
  echo "REFUSING TO START: no checkpoint at $CKPT" >&2
  exit 1
fi

CARLA_SH=/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4.sh
kill_servers() {
  for _ in 1 2 3 4 5; do
    mapfile -t pids < <(pgrep -f '[C]arlaUE4-Linux-Shipping')
    [ ${#pids[@]} -eq 0 ] && return 0
    for p in "${pids[@]}"; do kill -9 "$p" 2>/dev/null; done
    sleep 3
  done
}
cleanup() { kill_servers; echo "=== placement eval done ==="; }
trap cleanup EXIT

kill_servers
( exec 200>&-; "$CARLA_SH" -carla-port=2000 -quality-level=Epic \
      -windowed -ResX=640 -ResY=360 2>&1 \
    | grep -v --line-buffered "Invalid session: no stream available" \
    > logs/carla_placement_eval.log ) &
sleep 40

for band in ${BANDS//,/ }; do
  lo="${band%%-*}"; hi="${band##*-}"
  echo "=== placement: $ARM  band ${lo}-${hi}m  methods $METHODS ==="
  python -u eval_placement.py --arm "$ARM" --ckpt "$CKPT" --town "$TOWN" \
      --methods "$METHODS" --episodes "$EPISODES" --seed "$EVAL_SEED" \
      --max-steps "$EVAL_MAX_STEPS" --goal-min "$lo" --goal-max "$hi" \
      --collect-episodes "$COLLECT_EPISODES" --collect-steps "$COLLECT_STEPS" \
      --trace-cache "$TRACE" --out-dir "$OUT_DIR" \
    || echo "[Placement $ARM band ${lo}-${hi}m] FAILED" >&2
done

echo
echo "=== PLACEMENT COMPARISON ==="
python summarize_placement.py "$OUT_DIR"
