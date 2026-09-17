#!/bin/bash
# MAPPING BASELINES on the gated HER v4 critic -- protocol s2-s4, staged.
#
#   stage 1  COLLECT + MATRIX   per trace seed: drive explore mode, cache the
#                               trace and the all-pairs critic matrix
#   stage 2  CALIBRATE          once, over ALL traces: separation-vs-critic
#                               table, cutoff sweep with directed reachability,
#                               one setting justified against the 20-50 m band
#   stage 3  BUILD + NAVIGATE   per trace seed: sgm[c/e], madent, uniform@n,
#                               random@n; 60 fixed episodes each; per-row JSON
#
# One python process per stage-seed; the env inside is built once and reset.
# CARLA is started from Python (ensure_carla), after killing every CARLA on
# the machine and proving none survived. Every stage is cached, so a killed
# job is resumed by rerunning this script.
set -uo pipefail
cd "$(dirname "$0")"
set +u; eval "$(grep -E '^export ' run_gated.sh)"; set -u
# THEN OVERLAY THE RUN'S OWN RESOLVED SETTINGS, if given.
# The line above re-reads run_gated.sh's DEFAULTS, not the settings the
# checkpoint was actually trained with. Measured 2026-09-12 against the live
# v6m run: a sweep would have evaluated at DISCOUNT=1.0 while the run trains at
# 0.99, with explore off-road FATAL while the run has it non-terminal, and
# without the per-step goal penalties. Same class of bug the point-goal session
# hit (8 m visit ring offline against 3 m in training).
# RUN_ENV=results/run_env/<policy_folder>.env is written by the launcher.
if [ -n "${RUN_ENV:-}" ]; then
  if [ -f "$RUN_ENV" ]; then
    echo "[Env] overlaying the run's own settings from $RUN_ENV"
    set +u; while IFS='=' read -r k v; do case "$k" in ''|'#'*) continue;; esac; export "$k=$v"; done < "$RUN_ENV"; set -u
  else
    echo "[Env] FATAL: RUN_ENV=$RUN_ENV does not exist" >&2; exit 1
  fi
fi
export CARLA_START_SERVER=1 CARLA_QUALITY=Epic CARLA_PORT=2000 CARLA_PORTS=2000
export EXPERT_PROB=0 RESUME=0
# MEM_FRACTION/ALLOCATOR may be pre-set by the caller (Town12 needs the policy
# process small: the map alone holds 7.7 GB of the 12 GB card).
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}"
[ -n "${XLA_PYTHON_CLIENT_ALLOCATOR:-}" ] && export XLA_PYTHON_CLIENT_ALLOCATOR

CKPT="${CKPT:-checkpoints/carla_gated_her_v5/checkpoint_200000.pkl}"
# THE ARM LABEL NAMES EVERY RESULT FILE; it must say which policy is inside.
ARM="${ARM:-gated_her_v5_200k}"
TOWN="${TOWN:-Town02}"
# MAD_Z: robust-z threshold for the madent (entropy-spike) arm. The protocol's
# ported detector uses 3.0 and CAN return nothing -- on the v5 gated checkpoint
# it returned 0 nodes (entropy 0.68-2.37 nats, no candidate 3 MADs above the
# median), which emptied the entropy arm and both of its count-matched
# controls. User chose 2.0 on 2026-09-11 so the arm is populated; the threshold
# is recorded in every result JSON as mad_z, so a row always states its own.
MAD_Z="${MAD_Z:-3.0}"
OUT="${OUT:-results/placement_v4_${TOWN,,}}"
SEEDS="${SEEDS:-7 8 9 10}"
EPISODES="${EPISODES:-60}"
COLLECT_EPISODES="${COLLECT_EPISODES:-700}"
COLLECT_STEPS="${COLLECT_STEPS:-200}"
GOAL_MIN="${GOAL_MIN:-20}"; GOAL_MAX="${GOAL_MAX:-50}"
# USER DECISION 2026-09-11: the trace is driven by the POLICY IN EXPLORE MODE
# (docs sF), under the arm's own strict lane rule. Yield measured at ~1.5
# candidates per ~2 m episode on v4, so COLLECT_EPISODES is large and
# COLLECT_STEPS is a cap the episodes never reach.
EXPLORER="${EXPLORER:-policy_explore}"; LANE_RULE="${LANE_RULE:-strict}"
STAGES="${STAGES:-1 2 3}"
mkdir -p "$OUT" logs
say() { echo "=== [placement $(date +%H:%M:%S)] $* ==="; }

kill_all_carla() {
  for _ in 1 2 3 4 5 6; do
    pkill -9 -f 'CarlaUE4-Linux-Shipping' 2>/dev/null; pkill -9 -f 'CarlaUE4' 2>/dev/null
    pkill -9 -f 'CARLA_0.9.15' 2>/dev/null; sleep 2
    ps -eo args | grep -qi '[C]arlaUE4' || return 0
  done
  say "WARNING: a CARLA process survived six kill rounds"; return 1
}

COMMON=(--ckpt "$CKPT" --arm "$ARM" --town "$TOWN" --episodes "$EPISODES"
        --goal-min "$GOAL_MIN" --goal-max "$GOAL_MAX" --port 2000
        --collect-episodes "$COLLECT_EPISODES" --collect-steps "$COLLECT_STEPS"
        --step-cost 1.0 --min-sep-m 2.0 --sgm-gate-m 2.0 --mad-z "$MAD_Z"
        --out-dir "$OUT")

if [[ " $STAGES " == *" 1 "* ]]; then
  for S in $SEEDS; do
    TR="$OUT/trace_s${S}.pkl"
    if [ -f "$TR.dist.npy" ]; then say "seed $S: trace + matrix cached"; continue; fi
    for ATTEMPT in 1 2 3; do
      say "stage 1 seed $S attempt $ATTEMPT"
      kill_all_carla || exit 1
      # COLLECT_TOTAL_STEPS>0: one step budget, no time limit (user decision
      # 2026-09-12). NO_TIMELIMIT is set for THIS command only, so stage 3's
      # navigation keeps its caps.
      # COLLECT_TARGET_SPEED (km/h): trace collection only, applied AFTER the
      # RUN_ENV overlay (which sets the run's TARGET_SPEED). User 2026-09-13:
      # "reduce target speed for trace collection to 10Km/h". Stage 3 keeps
      # the run's value. The policy sees speed/TARGET_SPEED in its observation.
      NO_TIMELIMIT=$([ "${COLLECT_TOTAL_STEPS:-0}" -gt 0 ] && echo 1 || echo 0) \
      TARGET_SPEED="${COLLECT_TARGET_SPEED:-${TARGET_SPEED:-}}" \
      MAX_DISTANCE_M=$([ "${COLLECT_TOTAL_STEPS:-0}" -gt 0 ] && echo 1000000000 || echo "${MAX_DISTANCE_M:-200}") \
      python -u eval_placement.py "${COMMON[@]}" --collect-seed "$S" \
          --explorer "$EXPLORER" --collect-lane-rule "$LANE_RULE" \
          --collect-total-steps "${COLLECT_TOTAL_STEPS:-0}" \
          --collect-start-xy "${COLLECT_START_XY:-}" \
          --collect-end-on "${COLLECT_END_ON:-collision}" \
          --collect-action-mode "${COLLECT_ACTION_MODE:-eval}" \
          --trace-cache "$TR" --stop-after matrix
      [ -f "$TR.dist.npy" ] && break
      say "seed $S: no matrix yet -- rerunning (trace resumes from its cache)"
    done
  done
fi

if [[ " $STAGES " == *" 2 "* ]]; then
  TRACES=(); for S in $SEEDS; do [ -f "$OUT/trace_s${S}.pkl.dist.npy" ] && TRACES+=("$OUT/trace_s${S}.pkl"); done
  say "stage 2: calibrating on ${#TRACES[@]} traces"
  PYTHONPATH=src python -u calibrate_critic.py --traces "${TRACES[@]}" \
      --horizon-m "$GOAL_MAX" --gate-m 2.0 --out "$OUT/calibration.json" \
      2>&1 | tee "$OUT/calibration.txt"
fi

if [[ " $STAGES " == *" 3 "* ]]; then
  [ -f "$OUT/calibration.json" ] || { say "no calibration.json -- run stage 2"; exit 1; }
  for S in $SEEDS; do
    TR="$OUT/trace_s${S}.pkl"
    DONE=$(ls "$OUT"/${ARM}__*__${GOAL_MIN}-${GOAL_MAX}m__s${S}.json 2>/dev/null | wc -l)
    if [ "$DONE" -ge 6 ]; then say "seed $S: $DONE rows already on disk"; continue; fi
    for ATTEMPT in 1 2 3; do
      say "stage 3 seed $S attempt $ATTEMPT"
      kill_all_carla || exit 1
      # --collect-total-steps too: without it a step-budget trace (9 episodes)
      # reads as incomplete against COLLECT_EPISODES and collection RESUMES.
      python -u eval_placement.py "${COMMON[@]}" --collect-seed "$S" \
          --collect-total-steps "${COLLECT_TOTAL_STEPS:-0}" \
          --trace-cache "$TR" --calibration "$OUT/calibration.json" \
          --methods sgm,madent,uniform,random --match-to sgm,madent --episodes-from-trace \
          --tag "__s${S}"
      rc=$?; [ $rc -eq 0 ] && break
      say "seed $S: exit $rc -- rerunning"
    done
  done
  say "table:"; PYTHONPATH=src python summarize_placement.py "$OUT" 2>&1 | tee "$OUT/table.txt"
fi
kill_all_carla || true
say "done"
