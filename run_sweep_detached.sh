#!/usr/bin/env bash
# THE PARETO SWEEP, RUN OUT OF REACH OF THE TASK SUPERVISOR.
#
# WHY THIS EXISTS. Background jobs on this machine are killed at random with a
# spurious "low memory" message -- proven workload-independent when the same
# kill landed on a peer session's bash sleep loop while 101 GB was free, swap
# was untouched and dmesg showed no OOM. Roughly two thirds of launches died
# before doing any work, so the sweep was advancing one evaluation per several
# manual relaunches.
#
# The CARLA server survives all of this because it was started with setsid,
# outside any job's process group. This script is meant to be launched the same
# way:
#
#   setsid nohup ./run_sweep_detached.sh > logs/sweep.log 2>&1 < /dev/null &
#
# It then runs every point serially, unattended, and the caller polls the
# result files rather than holding a job open.
#
# EACH POINT IS ONE PROCESS AND IS RETRIED, because a kill can still arrive
# from elsewhere and because a single CARLA hiccup should not cost the sweep.
# Results are written atomically by eval_placement.py, so an interrupted point
# leaves the previous result intact rather than a truncated one.
#
# ALREADY-COMPLETE POINTS ARE SKIPPED, so this can be re-launched after any
# interruption and will only do the work still missing.
set -uo pipefail
cd "$(dirname "$0")"

TOWN="${TOWN:-Town02}"
ARM="${ARM:-gated_her}"
EPISODES="${EPISODES:-20}"
TRIES="${TRIES:-4}"
OUT_DIR="${OUT_DIR:-results/placement_town02}"
TRACE="${TRACE:-results/placement_town02/trace_v2.pkl}"

# THE SWEEP. Bands beyond 20 m are where a map can actually help: the policy
# already solves 5-20 m alone at SR 1.000 in 48 steps, so a graph there can
# only add detours. Densities span the range either side of the 4/100 m point
# already measured.
BANDS="${BANDS:-20-40 40-70}"

# NO NODE CAP. The question is the Pareto frontier of node count against
# SR/SPL, so nothing is forced onto a shared budget.
#
# THE TWO KINDS OF METHOD ARE SWEPT DIFFERENTLY, because they differ in what
# their free parameter IS:
#
#   uniform, random, policy   have NO intrinsic node count -- they must be told
#                             how many to place, so density IS their free
#                             parameter and sweeping it traces their curve.
#   sgm                       DERIVES its own count from its consistency
#                             cutoff. Fitting that cutoff to a density target
#                             would be capping it at someone else's number and
#                             would throw away the one property that makes it
#                             interesting. Its cutoff is swept instead and
#                             whatever node count falls out is the datapoint.
DENSITIES="${DENSITIES:-1 2 4 8 16}"
BUDGETED="${BUDGETED:-uniform random policy}"
# In STEPS, against a 106-step horizon. Low cutoff -> strict merge test -> many
# nodes; high cutoff -> permissive -> few. Spans dense to very sparse.
SGM_CUTOFFS="${SGM_CUTOFFS:-15 30 50 70 90}"

echo "=== detached sweep: $TOWN $ARM  bands=[$BANDS] ==="
echo "    budgeted=[$BUDGETED] over densities=[$DENSITIES]"
echo "    sgm over consistency cutoffs=[$SGM_CUTOFFS] (node count is an OUTPUT)"
date

for band in $BANDS; do
  lo="${band%%-*}"; hi="${band##*-}"
  # The budgeted arms: node count is the free parameter.
  for d in $DENSITIES; do
    for m in $BUDGETED; do
      out="$OUT_DIR/${ARM}__${m}__${lo}-${hi}m__d${d}.json"
      if [ -f "$out" ]; then
        echo "[skip] $out already exists"
        continue
      fi
      ok=0
      for try in $(seq 1 "$TRIES"); do
        echo "[run] band ${lo}-${hi}m density ${d} method ${m} (attempt $try)"
        TOWN="$TOWN" EPISODES="$EPISODES" BANDS="$band" METHODS="$m" \
          DENSITY="$d" TAG="__d${d}" CANDIDATE_SPACING=5.0 \
          EVAL_MAX_STEPS=1000 TRACE="$TRACE" OUT_DIR="$OUT_DIR" \
          ./run_placement_eval.sh >> "logs/sweep_${m}_${lo}-${hi}_d${d}.log" 2>&1
        if [ -f "$out" ]; then ok=1; echo "[done] $out"; break; fi
        sleep 5
      done
      [ "$ok" = "1" ] || echo "[FAILED after $TRIES tries] band ${lo}-${hi}m d${d} $m" >&2
    done
  done

  # SGM: its CUTOFF is the free parameter and the node count is an OUTPUT.
  for c in $SGM_CUTOFFS; do
    out="$OUT_DIR/${ARM}__sgm__${lo}-${hi}m__c${c}.json"
    if [ -f "$out" ]; then
      echo "[skip] $out already exists"
      continue
    fi
    ok=0
    for try in $(seq 1 "$TRIES"); do
      echo "[run] band ${lo}-${hi}m sgm consistency_cutoff ${c} (attempt $try)"
      TOWN="$TOWN" EPISODES="$EPISODES" BANDS="$band" METHODS=sgm \
        SGM_CUTOFF="$c" TAG="__c${c}" CANDIDATE_SPACING=5.0 \
        EVAL_MAX_STEPS=1000 TRACE="$TRACE" OUT_DIR="$OUT_DIR" \
        ./run_placement_eval.sh >> "logs/sweep_sgm_${lo}-${hi}_c${c}.log" 2>&1
      if [ -f "$out" ]; then ok=1; echo "[done] $out"; break; fi
      sleep 5
    done
    [ "$ok" = "1" ] || echo "[FAILED after $TRIES tries] band ${lo}-${hi}m sgm c${c}" >&2
  done
done

echo "=== detached sweep finished ==="
date
