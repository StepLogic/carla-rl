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

# THE GATE DEPENDS ON WHETHER CARLA IS ALREADY RESIDENT, and getting this wrong
# deadlocks the queue rather than failing loudly.
#
# The 9500 MiB figure is CARLA + JAX together (~9.2 GB measured), and it assumed
# the driver boots CARLA ITSELF, after gating, on an empty card. With a server
# already running that server is holding ~5.5 GB of the card, so a 9500 MiB gate
# can NEVER pass and every job waits out its deadline and exits 0 having done
# nothing -- which is exactly what happened once here, silently.
#
# So the gate asks for what this job still has to ALLOCATE: the full pair when
# it must boot a server, or roughly the JAX half when it is adopting one.
# Detection therefore has to happen BEFORE the gate, not after it.
if pgrep -f '[C]arlaUE4-Linux-Shipping' >/dev/null 2>&1; then
  SERVER_RUNNING=1
  NEED_MB="${NEED_MB:-4000}"
else
  SERVER_RUNNING=0
  NEED_MB="${NEED_MB:-9500}"
fi
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
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
# PLATFORM ALLOCATOR AND NO AUTOTUNING, both to survive the all-pairs critic
# pass. The first call of that pass is a new shape, and XLA's autotuner logged
# "All configs were filtered out ... using the full hints set instead" and then
# took the process down on HOST memory -- the run died between collecting a
# trace and producing a single distance. Autotuning buys kernel speed we do not
# need for one batched forward pass per node.
#
# corl_2026_evaluate.py sets the same allocator and a low mem fraction at import,
# which is where these values come from rather than from guesswork.
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"

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
# METRES BETWEEN CANDIDATES. The critic pass is QUADRATIC in this count, and
# consecutive simulator steps are only ~0.28 m apart, so a fine spacing buys
# duplicates at quadratic cost. At 4 m a 2800 m trace gives ~700 candidates,
# which is far more than the ~113-node budget needs.
CANDIDATE_SPACING="${CANDIDATE_SPACING:-4.0}"
# HOW FAR A POSE MAY BE FROM THE NEAREST NODE AND STILL ENTER THE GRAPH.
#
# MEASURED, and the reason this is not left at 15 m: with a 20-episode trace,
# 34-46 of 60 evaluation episodes failed to localise their START pose, so the
# graph offered no route and they fell back to driving straight at the goal.
# SR and SPL then measured the POLICY, not the map, for most of the episodes --
# a placement comparison in which placement barely participates.
#
# 25 m is bounded by something real rather than tuned: the policy's own
# training band is 5-20 m, so a node within ~25 m is a leg it is competent to
# drive. Anything much larger would be asserting entry points the policy cannot
# actually reach.
LOCALIZE_RADIUS="${LOCALIZE_RADIUS:-25.0}"
# THE SWEEP POINT. There is NO equal budget any more: the question is the
# PARETO FRONTIER of node count against SR/SPL, so each method is run at
# several densities and the curves are compared, not single points.
#
# For uniform, random and policy the node count IS the independent variable and
# DENSITY sets it. For SGM the independent variable is its consistency cutoff
# and the node count is whatever falls out -- which is the honest way to sweep
# a method that derives its own sparsity, and is what "no budget" means for it.
# Pass SGM_CUTOFF to drive that arm; leave it unset and SGM is fitted to DENSITY
# instead, which is only useful when a matched-count point is wanted.
DENSITY="${DENSITY:-4.0}"
SGM_CUTOFF="${SGM_CUTOFF:-}"
# Distinguishes sweep points in the filename. Without it the second density
# overwrites the first and every file still looks complete.
TAG="${TAG:-}"
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
# REUSE A RUNNING SERVER, AND ONLY TEAR DOWN ONE WE STARTED.
#
# WHY: the task supervisor has been killing these jobs at LAUNCH -- one left a
# two-line log, dead within seconds, before any evaluation began -- on a machine
# with 108 GB available and no kernel OOM. Booting CARLA at Epic is the largest
# and fastest allocation a job makes, and every job was doing it. Reusing a live
# server removes that spike from all but the first job, and removes 40 s of boot
# from each as a side benefit.
#
# ONLY THE STARTER KILLS IT. A job that adopted an existing server leaves it
# running for the next one; killing a server this job did not start would take
# down a sibling evaluation mid-episode.
STARTED_SERVER=0
cleanup() {
  if [ "$STARTED_SERVER" = "1" ]; then
    kill_servers
    echo "=== placement eval done (server stopped) ==="
  else
    echo "=== placement eval done (server left running for the next job) ==="
  fi
}
trap cleanup EXIT

if [ "$SERVER_RUNNING" = "1" ]; then
  echo "[CARLA] reusing the server already running on port $CARLA_PORT"
else
  echo "[CARLA] no server running; starting one"
  STARTED_SERVER=1
  kill_servers
  ( exec 200>&-; "$CARLA_SH" -carla-port=2000 -quality-level=Epic \
        -windowed -ResX=640 -ResY=360 2>&1 \
      | grep -v --line-buffered "Invalid session: no stream available" \
      > logs/carla_placement_eval.log ) &
  sleep 40
fi

# ONE METHOD PER PROCESS, and one CARLA server for all of them.
#
# WHY, MEASURED: a single process running all four methods at 60 episodes was
# killed twice by the task supervisor for "low memory", early in the first
# method, while the four-way driver ran six 60-episode evaluations in one job
# without trouble. The difference is that eval_arm.py takes ONE arm per process
# and this took four. The machine was never actually short -- 108 GB available,
# swap untouched, no kernel OOM -- so this is about process lifetime rather
# than a leak worth hunting.
#
# THE TRACE AND THE DISTANCE MATRIX ARE CACHED ON DISK, so a fresh process per
# method re-reads them instead of re-collecting or re-querying the critic. The
# comparison still rests on ONE trace, which is the property that matters.
for band in ${BANDS//,/ }; do
  lo="${band%%-*}"; hi="${band##*-}"
  for method in ${METHODS//,/ }; do
  echo "=== placement: $ARM  band ${lo}-${hi}m  method $method ==="
  python -u eval_placement.py --arm "$ARM" --ckpt "$CKPT" --town "$TOWN" \
      --methods "$method" --episodes "$EPISODES" --seed "$EVAL_SEED" \
      --max-steps "$EVAL_MAX_STEPS" --goal-min "$lo" --goal-max "$hi" \
      --collect-episodes "$COLLECT_EPISODES" --collect-steps "$COLLECT_STEPS" \
      --candidate-spacing "$CANDIDATE_SPACING" \
      --localize-radius "$LOCALIZE_RADIUS" \
      --nodes-per-100m "$DENSITY" --tag "$TAG" \
      ${SGM_CUTOFF:+--consistency-cutoff "$SGM_CUTOFF"} \
      --trace-cache "$TRACE" --out-dir "$OUT_DIR" \
    || echo "[Placement $ARM band ${lo}-${hi}m $method] FAILED" >&2
  done
done

echo
echo "=== PLACEMENT COMPARISON ==="
python summarize_placement.py "$OUT_DIR"
