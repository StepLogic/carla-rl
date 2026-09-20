#!/usr/bin/env bash
# THE 2-MODE IMAGE-GOAL ARM (Phase 3): ONE policy serving BOTH the image-goal
# task (reach the viewpoint the goal image was taken from and stop) and the
# goal-free explore task, selected by the mode bit drawn at reset -- the gated
# arm's pairing, with the single-mode image-goal reward serving BOTH modes.
#
# ONE CRITIC, NOT THE FROZEN GATED STACK. The unified TwoModeImageGoalReward
# puts both modes on one all-negative scale, so the two-critic
# GatedTQCLearner + ModeSplitBuffer split has nothing left to justify: under
# GATED_ONE_CRITIC=1 the trainer routes to the PLAIN TQCLearner and a single
# plain ReplayBuffer (the mode bit rides inside the observation dict). The
# critic's targets are clipped by the GLOBAL TQC_TARGET_MIN/MAX window --
# REQUIRED here, read at import in jaxrl2's critic_updater and therefore
# exported at shell launch, before any python process exists.
#
# A NEW LINEAGE. This script SOURCES NOTHING -- not run_gated.sh, not
# run_image_goal.sh -- and is the single authority for every value below. It
# copies only STRUCTURE from run_image_goal.sh (the lock, the GPU poll, the
# ig-venv-or-refuse, the kill/cleanup, the retry loop, the snapshot pattern);
# the env values are this arm's own, because a copied env value is an
# inherited experiment decision -- run_image_goal.sh's OBS_WITH_COLLISION=1,
# GOAL_DIST band 5-100 m and unclipped targets are THAT arm's record, and
# this arm's spec (plan 2026-09-20-two-mode-image-goal, Task 8) differs from
# it on purpose.
#
# GATED_GOAL_REWARD=shaped is SET HERE although the pairing never builds a
# GatedReward: check_gated_run still runs on the pairing (train_carla_tqc
# calls BOTH guards) and its legacy branch reads the flag. Its default
# "sorb" would REFUSE this run at DISCOUNT=0.99 ("sorb requires DISCOUNT=1.0"
# -- a check about the SORB minimum-time goal reward, which this arm does not
# use), and GATED_ALLOW_DISCOUNTED_SORB=1 would keep the run claiming a SORB
# goal reward it does not train. "shaped" tells the guard the truth: the goal
# reward is not the -1/step minimum-time one. It changes NO reward on this
# arm -- the reward is TwoModeImageGoalReward regardless, and the flag's only
# consumers under the pairing are the guard's message branch.
#
# NO_TIMELIMIT stays UNSET (training run, not trace collection) and is
# scrubbed below. OBS_WITH_COLLISION, TQC_VALUE_FLOOR and TQC_VALUE_CEILING
# are scrubbed too: the pairing REFUSES the single-mode `coll` key (the
# 10-d `state` block already carries has_collided + speed), and the
# one-critic route takes NO per-critic value bounds -- TQC_TARGET_MIN/MAX are
# the only bounds, so a stray TQC_VALUE_FLOOR/CEILING must not reach the
# plain learner or the snapshot.
#
# RUNS ON THE CLUSTER (memory rule); this script ships the launch, not the run.
set -uo pipefail
cd "$(dirname "$0")"

# ---- one CARLA job at a time ------------------------------------------------
# Held for the script's lifetime. Non-blocking on purpose: a run that WAITED on
# the lock would sit holding its VRAM reservation while another job ran, which
# is the deadlock this convention exists to avoid. Refuse and exit instead.
# (Structure from run_image_goal.sh, including the fd number another script's
# cleanup may close.)
LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "REFUSING TO START: another CARLA job holds $LOCK." >&2
  ps -eo pid,etime,args | grep -E "[t]rain_carla_tqc|[r]un_" >&2
  exit 1
fi
echo "[Lock] acquired $LOCK (pid $$)"

# ---- wait for the shared card ------------------------------------------------
# Same poll as run_image_goal.sh, gate on FREE MEMORY not on a process name
# (/proc/PID/comm is capped at 15 chars, so the 23-char CARLA binary name is
# truncated and an exact-name match can never fire). Same floor as the
# single-mode image-goal arm, and the accounting behind it is that arm's:
#   CARLA server, Low quality, town loaded   ~4.3-4.9 GB  (v7's live measurement)
#   jax learner, preallocate off, 0.4 share      ~0.4-2.3 GB
#   VC-1 ViT-B (torch, frozen)                   ~0.4 GB   (398 MiB measured)
# so ~6.5 GB covers one server plus the learner with headroom. A caller can
# raise it (NEED_MB=9500) for an Epic server, which costs ~1 GB more.
NEED_MB="${NEED_MB:-6500}"
MAX_WAIT_H="${MAX_WAIT_H:-6}"
POLL_S="${POLL_S:-60}"
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${free:-0}" -ge "$NEED_MB" ]; then
    echo "[GPU] ${free} MiB free -- starting"; break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "[GPU] GAVE UP after ${MAX_WAIT_H}h; only ${free} MiB free." >&2; exit 1
  fi
  echo "[GPU] ${free} MiB free, need ${NEED_MB}; waiting ${POLL_S}s"
  sleep "$POLL_S"
done

# ---- the interpreter is the ig-venv, or nothing -------------------------------
# Task 3's venv carries torch-cu121 (the system torch's cu130 build does not
# run on driver 535) AND jax on GPU in ONE interpreter, plus vc_models for
# PERCEPTION=vc1, with the cudnn pin (nvidia-cudnn-cu12==9.24.0.43) baked in.
# A missing venv must REFUSE, not fall back to system python: a silent
# fallback runs the whole run on a torch that either fails at the first VC-1
# forward or, worse, half-works.
PYTHON="${PYTHON:-dependencies/ig-venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  echo "FATAL: $PYTHON is not executable. Build the venv (Task 3) or pass" >&2
  echo "       PYTHON=/path/to/python explicitly." >&2
  exit 1
fi

# ---- anti-inheritance scrub ---------------------------------------------------
# The values this arm trains under are set below and NOWHERE ELSE. An ambient
# shell that ran a gated, single-mode image-goal or SORB job can still carry
# the variables that would silently change THIS run's meaning -- or make the
# guards refuse a run whose meaning they were never part of -- so they are
# cleared rather than hoped away. TQC_TARGET_MIN/MAX are deliberately NOT on
# this list: unlike the single-mode arm, they are REQUIRED here (the one
# bounded critic clips its targets; image_goal_coherence asserts their
# presence, and jaxrl2's critic_updater reads them at import).
#   NO_TIMELIMIT            training run, not trace collection; the pairing
#                           refuses it outright
#   OBS_WITH_COLLISION      the single-mode arm's 1-d `coll` key; the pairing
#                           refuses it beside the 10-d `state` block
#   LANE_FOLLOW/LANE_EXPLORE/TIME_TO_GOAL   stacked reward wrappers; the
#                           pairing refuses all three
#   TQC_VALUE_FLOOR/CEILING the two-critic learner's per-critic bounds; the
#                           one-critic route takes no per-critic bounds, so
#                           these must never reach the plain learner
#   GATED_ALLOW_DISCOUNTED_SORB  the SORB guard-bypass; this arm's goal reward
#                           is not SORB, so a discount warning about it would
#                           be a lie in the log
#   ACTOR_Q_NORM            the two-critic learner's per-mode actor
#                           normalisation; no second critic here
#   EXPLORE_ARRIVAL_BONUS   train_v7.sh's +2.0 explore arrival bonus -- a
#                           historical record of THAT lineage, not a setting
#   EXPLORE_DISCOUNT        the two-critic route's second discount; the
#                           one-critic route has ONE discount (DISCOUNT), and
#                           a leftover EXPLORE_DISCOUNT>=1.0 would make
#                           check_gated_run refuse a flag this run never uses
#   GOAL_BATCH_FRAC         the ModeSplitBuffer's split; the one-critic route
#                           has one buffer, and a degenerate ambient value
#                           would make check_gated_run refuse an inert flag
#   STEP_COST               the SORB per-step cost; inert under GATED_GOAL_
#                           REWARD=shaped and never read by the image-goal
#                           reward, but cleared so the snapshot cannot imply
#                           a -STEP_COST reward
for _scrub in NO_TIMELIMIT OBS_WITH_COLLISION \
              LANE_FOLLOW LANE_EXPLORE TIME_TO_GOAL \
              TQC_VALUE_FLOOR TQC_VALUE_CEILING \
              GATED_ALLOW_DISCOUNTED_SORB ACTOR_Q_NORM \
              EXPLORE_ARRIVAL_BONUS EXPLORE_DISCOUNT GOAL_BATCH_FRAC STEP_COST; do
  if [ -n "${!_scrub+x}" ]; then
    echo "[Scrub] clearing inherited $_scrub=${!_scrub}"
    unset "$_scrub"
  fi
done

# ---- the arm ------------------------------------------------------------------
# THE BRIEF'S ENV BLOCK, VERBATIM (plan 2026-09-20-two-mode-image-goal,
# Task 8). Every value below is this arm's spec, set here and NOWHERE ELSE.
export IMAGE_GOAL_TASK=1 GATED=1 GATED_ONE_CRITIC=1
export PERCEPTION=vc1 VC1_READOUT=grid2
export OBS_WITH_STATE=1 OBS_WITH_MODE=1
export EPISODE_HER=1 HER_MIN_DISP=5.0 IG_GOAL_DISTANCE_THRESHOLD_M=2.0
export IG_SPEED_PREF_K=0.1
export IG_IN_LANE_BONUS=0    # flat in-lane bonus REMOVED (user 2026-09-20);
                             # pinned explicitly so the 49-var snapshot
                             # records it and the banner cannot disagree
export DISCOUNT=0.99 TOP_QUANTILES_DROP=2 TQC_KAPPA=1.0
export TEMP_LR=3e-4                      # the v2 divergence fix (entropy brake)
export TQC_TARGET_MIN=-260 TQC_TARGET_MAX=100   # ReViND window at gamma=0.99
export EXPLORE_PROB=0.5 EXPLORE_GOAL_DIST_M=100.0
export TARGET_SPEED=20.0 ENV_SKIP_IMAGE_FEATURES=1
export MAX_EPISODE_STEPS=1000
export GOAL_DIST_MIN="${GOAL_DIST_MIN:-5.0}" \
       GOAL_DIST_START="${GOAL_DIST_START:-25.0}" \
       GOAL_DIST_TARGET="${GOAL_DIST_TARGET:-25.0}"
export EXPERT_PROB=0.0
export TOWN_LIST=configs/town_split.json ENV_TOWNS=Town01 TOWN_ROTATE_EPISODES=0
export N_EVAL_EPISODES=0 N_EVAL_EXPLORE=0
# ORDERING TRAP: RUN_NAME and POLICY_FOLDER must NOT share one export line.
# In `export A=... B="${B:-$A}"` the words are expanded BEFORE export runs any
# assignment, so with RUN_NAME unset and `set -u` active the $RUN_NAME in the
# second argument dies ("RUN_NAME: unbound variable") before the default is
# applied. Two lines, RUN_NAME first; the smoke only survived because the
# controller happened to pass RUN_NAME in its env.
export RUN_NAME="${RUN_NAME:-carla_image_goal_2mode}"
export POLICY_FOLDER="${POLICY_FOLDER:-$RUN_NAME}"
export MAX_STEPS="${MAX_STEPS:-1000000}" SEED=42 RESUME=1

# ---- the pairing's guard plumbing (NOT experiment values) ---------------------
# The reward itself is TwoModeImageGoalReward on BOTH modes; these two exports
# only carry the flags the shared guards demand, so a run never dies on a
# check about a lineage it does not train. See the header for why.
export GATED_GOAL_REWARD=shaped  # check_gated_run: its "sorb" branch requires
                                 # DISCOUNT=1.0, a SORB-reward constraint this
                                 # arm's ReViND reward does not inherit. Under
                                 # the pairing this flag changes NO reward.
export OBS_WITH_GOAL=1           # check_gated_run + VAEGoalWrapper require the
                                 # goal block under OBS_WITH_MODE=1; pinned so
                                 # an ambient OBS_WITH_GOAL=0 cannot half-apply
export OBS_WITH_VECTOR=0         # the pairing keeps velocity in the `state`
                                 # block (that is WHY OBS_WITH_STATE=1 is
                                 # required); pinned so an ambient =1 cannot
                                 # widen the observation the checkpoints were
                                 # sized for

# ---- CARLA, started by PYTHON (user rule) ------------------------------------
# ensure_carla (train_carla_tqc.py) starts ONE server PER PORT in CARLA_PORTS
# and honours CARLA_QUALITY / CARLA_OFFSCREEN. There is deliberately NO shell
# CARLA launch here, unlike run_gated.sh's own pre-start block: a shell server
# at hardcoded quality is what made Town01 silently train at Epic on the gated
# path (memory: CarlaCore renders at Low quality). The env owns its servers.
export CARLA_START_SERVER=1
export CARLA_ROOT="${CARLA_ROOT:-/home/robotlab/Apps/CARLA_0.9.15}"
export CARLA_PORT="${CARLA_PORT:-2000}" CARLA_PORTS="${CARLA_PORTS:-2000}"
# Low, not Epic: ~1 GB less VRAM per server (measured v7: 4.85 vs 5.9 GB), and
# VC-1 is a web-scale foundation encoder, not a checkpoint fitted to this
# repo's Epic frames, so the input-distribution argument that pins the gated
# VAE to its render quality does not apply here. Overridable for a caller who
# wants Epic (then raise NEED_MB too).
export CARLA_QUALITY="${CARLA_QUALITY:-Low}"
# Offscreen by default: an unattended cluster run must not depend on a live X
# session (windowed CarlaUE4 EXITS if DISPLAY is wrong or the session locks).
# Offscreen costs only ~12 MiB. Caller can set CARLA_OFFSCREEN=0 to mirror the
# gated lineage's windowed server.
export CARLA_OFFSCREEN="${CARLA_OFFSCREEN:-1}"
export DISPLAY="${DISPLAY:-:1}"

# ---- jax memory (structure from run_image_goal.sh / run_gated.sh) ------------
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
# A free-memory check is necessary but NOT sufficient: jax can still die with
# RESOURCE_EXHAUSTED while instantiating a CUDA command buffer, which no byte
# count predicts. This removes that allocation.
export XLA_FLAGS=--xla_gpu_enable_command_buffer=

# ---- provenance snapshot + banner --------------------------------------------
# Every flag this run is ABOUT must be in results/run_env/${RUN_NAME}.env, or
# an eval replaying it silently differs from training (the prefix pattern is
# "[A-Z0-9_]*="-suffixed for the reason train_v7.sh records: an anchored
# "^PREFIX=" alternation matches nothing but a variable named exactly the
# prefix, and every PREFIX_* flag drops out of the snapshot silently).
# EXPLORE and VC1 are ADDED to the run_image_goal.sh alternation: this arm's
# EXPLORE_PROB / EXPLORE_GOAL_DIST_M / VC1_READOUT are spec'd snapshot keys
# and the old pattern had neither prefix, so they would have dropped out
# silently.
mkdir -p results/run_env logs
env | grep -E "^(IMAGE_GOAL|IG_|PERCEPTION|VC1_|OBS_WITH|EPISODE_HER|HER_|DISCOUNT|TOP_QUANTILES|TQC_|NUM_QS|N_QUANTILES|TARGET_|TEMP_|UTD_|MINIBATCH|REPLAY_|GATED|EXPLORE_|ENV_|TOWN|EVAL_|CARLA|N_EVAL|GOAL_|MAX_|POLICY|RESUME|SEED|XLA|START_|EXPERT|INCREAMENTAL|INCREMENTAL|STEER|THROTTLE|COLLISION|LANE|WEDGE|SHORT_|JUNCTION|STRAIGHT|K_|NO_|RUN_|NEED_MB)[A-Z0-9_]*=" \
  | sort > "results/run_env/${RUN_NAME}.env"
_missing=""
for _k in IMAGE_GOAL_TASK GATED GATED_ONE_CRITIC \
          PERCEPTION VC1_READOUT \
          OBS_WITH_STATE OBS_WITH_MODE OBS_WITH_GOAL OBS_WITH_VECTOR \
          EPISODE_HER HER_MIN_DISP IG_GOAL_DISTANCE_THRESHOLD_M \
          IG_SPEED_PREF_K TEMP_LR \
          TQC_TARGET_MIN TQC_TARGET_MAX \
          DISCOUNT TOP_QUANTILES_DROP TQC_KAPPA \
          EXPLORE_PROB EXPLORE_GOAL_DIST_M GATED_GOAL_REWARD \
          GOAL_DIST_MIN GOAL_DIST_START GOAL_DIST_TARGET \
          TARGET_SPEED ENV_SKIP_IMAGE_FEATURES \
          TOWN_LIST ENV_TOWNS TOWN_ROTATE_EPISODES MAX_EPISODE_STEPS \
          N_EVAL_EPISODES N_EVAL_EXPLORE EXPERT_PROB \
          RUN_NAME POLICY_FOLDER MAX_STEPS SEED RESUME \
          CARLA_ROOT CARLA_QUALITY CARLA_OFFSCREEN CARLA_START_SERVER \
          CARLA_PORT CARLA_PORTS \
          XLA_PYTHON_CLIENT_PREALLOCATE XLA_PYTHON_CLIENT_MEM_FRACTION; do
  grep -qE "^${_k}=" "results/run_env/${RUN_NAME}.env" || _missing="$_missing $_k"
done
if [ -n "$_missing" ]; then
  echo "FATAL: these flags are missing from results/run_env/${RUN_NAME}.env:$_missing" >&2
  exit 1
fi
# AND THE SNAPSHOT MUST NOT CONTAIN WHAT THE PAIRING DELIBERATELY LACKS. The
# scrub block above clears each of these from the LAUNCHING shell; this check
# closes the loop on the snapshot itself, so a future edit that re-exports one
# of them fails HERE instead of at image_goal_coherence (OBS_WITH_COLLISION,
# NO_TIMELIMIT, LANE_*), at check_gated_run (EXPLORE_DISCOUNT,
# GATED_ALLOW_DISCOUNTED_SORB, GOAL_BATCH_FRAC, STEP_COST), or in the plain
# learner's value bounds (TQC_VALUE_FLOOR/CEILING) mid-training.
_present=""
for _k in NO_TIMELIMIT OBS_WITH_COLLISION \
          LANE_FOLLOW LANE_EXPLORE TIME_TO_GOAL \
          TQC_VALUE_FLOOR TQC_VALUE_CEILING \
          GATED_ALLOW_DISCOUNTED_SORB ACTOR_Q_NORM \
          EXPLORE_ARRIVAL_BONUS EXPLORE_DISCOUNT GOAL_BATCH_FRAC STEP_COST; do
  grep -qE "^${_k}=" "results/run_env/${RUN_NAME}.env" && _present="$_present $_k"
done
if [ -n "$_present" ]; then
  echo "FATAL: the 2-mode pairing's snapshot carries refused/bound exports:$_present" >&2
  exit 1
fi
# ---- the spec table (the banner) ---------------------------------------------
echo "[Banner] run=${RUN_NAME} policy_folder=${POLICY_FOLDER} interpreter=${PYTHON}"
echo "[Banner] 2-mode pairing: IMAGE_GOAL_TASK=1 GATED=1 GATED_ONE_CRITIC=1"
echo "[Banner] PERCEPTION=vc1 VC1_READOUT=grid2 OBS_WITH_STATE=1 OBS_WITH_MODE=1"
echo "[Banner] EPISODE_HER=1 HER_MIN_DISP=5.0 IG_GOAL_DISTANCE_THRESHOLD_M=2.0"
echo "[Banner] reward: IG_SPEED_PREF_K=0.1 TARGET_SPEED=20.0 km/h EXPLORE_GOAL_DIST_M=100.0 IG_IN_LANE_BONUS=0 (in-lane bonus removed 2026-09-20)"
echo "[Banner] learner: DISCOUNT=0.99 TOP_QUANTILES_DROP=2 TQC_KAPPA=1.0 TEMP_LR=3e-4"
echo "[Banner] bounded critic: TQC_TARGET_MIN=-260 TQC_TARGET_MAX=100 (ReViND window)"
echo "[Banner] modes: EXPLORE_PROB=0.5 | goal band ${GOAL_DIST_MIN}-${GOAL_DIST_TARGET} m"
echo "[Banner] no OBS_WITH_COLLISION, no TQC_VALUE_FLOOR/CEILING, NO_TIMELIMIT unset"
echo "[Banner] snapshot: results/run_env/${RUN_NAME}.env ($(wc -l < "results/run_env/${RUN_NAME}.env") vars)"

# ---- server cleanup + retry (structure from run_image_goal.sh) ---------------
# pgrep -x CANNOT match: /proc/PID/comm is capped at 15 chars, so the 23-char
# 'CarlaUE4-Linux-Shipping' truncates and an exact-name match never fires. The
# bracket stops a concurrent cleanup matching THIS script's argv.
#
# THE KILL IS SCOPED TO THIS RUN'S OWN PORTS BY DEFAULT (KILL_ALL=0). The old
# kill-9-ALL-CARLA sweep is a shared-box hazard: other sessions on this box
# start and stop servers concurrently, and this script kills at LAUNCH and
# again in the EXIT trap, so an unscoped sweep would destroy a peer's live run
# (measured 2026-09-20: a peer server on port 2000 serving an eval while a
# smoke launch on 2010 would have killed it, and its EXIT would have killed
# it again). ensure_carla starts every server with an explicit -carla-port=P
# in its argv, so matching this run's CARLA_PORTS/CARLA_PORT values -- BOTH
# exported above this point -- scopes the kill to servers this run owns.
# KILL_ALL=1 restores the exact old kill-all sweep: the user's standing
# kill-all rule belongs to a SOLO 1M launch, and the controller sets
# KILL_ALL=1 explicitly there. WHO SETS WHAT: the 50k smoke on a shared box
# runs scoped (KILL_ALL unset or 0, its own CARLA_PORT); the 1M launch runs
# KILL_ALL=1.
kill_servers() {
  local _pat _ports pids p
  if [ "${KILL_ALL:-0}" = "1" ]; then
    _pat='[C]arlaUE4-Linux-Shipping'
  else
    # Accept "2000", "2000,2010" or "2000 2010": normalise any separator list
    # to an alternation. The trailing ([[:space:]]|$) stops 200 from matching
    # a 2000 server's argv, not just 2000 matching 20001.
    _ports="$(printf '%s\n' "${CARLA_PORTS:-${CARLA_PORT:-2000}}" \
              | tr -s ' ,;' '\n' | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' \
              | sed '/^$/d' | paste -sd'|' -)"
    _pat="[C]arlaUE4-Linux-Shipping .*-carla-port=(${_ports:?})([[:space:]]|\$)"
  fi
  for _ in 1 2 3 4 5; do
    mapfile -t pids < <(pgrep -f "$_pat")
    [ ${#pids[@]} -eq 0 ] && return 0
    for p in "${pids[@]}"; do kill -9 "$p" 2>/dev/null; done
    sleep 3
  done
  pgrep -f "$_pat" >/dev/null && \
    echo "WARNING: a CARLA server survived 5 kill attempts" >&2
}
cleanup() { kill_servers; echo "=== 2-mode image-goal run done ==="; }
trap cleanup EXIT

CK="checkpoints/$POLICY_FOLDER"
ckpt_step() {
  ls "$CK" 2>/dev/null | sed -n 's/^checkpoint_\([0-9]*\)\.pkl$/\1/p' | sort -n | tail -1
}

# ---- prove-alive (preflight memory rules) ------------------------------------
# THE RUN IS NOT "LAUNCHED" UNTIL IT IS PROVEN RUNNING. From ANOTHER shell,
# within a couple of minutes of starting an attempt, confirm all four:
#
#   1. the process exists and is the ig-venv interpreter:
#        pgrep -af "train_carla_tqc"
#        (the interpreter in its argv must be dependencies/ig-venv/bin/python)
#   2. the log is advancing:
#        tail -f logs/${RUN_NAME}.log
#      expect, in order: the [Banner] spec table above, [Scrub] lines if the
#      calling shell carried anything, [Towns] frozen split, [Latent]
#      backend 'vc1' (grid2 readout), [CARLA] starting/ready on the run's
#      port (CARLA_PORT -- 2000 is the 1M default; the smoke ran 2010),
#      [GatedConfig] OK ... (check_gated_run passing the pairing),
#      [TwoModeIG] unified image-goal reward ... both modes, [VAEGoal]
#      obs={... 'state': (10,), 'mode': ...} WITHOUT a 'coll' key,
#      [ImageGoal] ... HER banner, then [Ep] lines once training starts.
#      NOTE the absence of 'coll': its presence means OBS_WITH_COLLISION
#      leaked past the scrub and the pairing's guards would have refused --
#      if they did not, the snapshot check was bypassed; stop and read.
#   3. the GPU is actually working:
#        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
#      (a CARLA server at Low with a town loaded holds ~2-4.9 GB)
#   4. the run's OWN record agrees with all of the above -- the LIVE process,
#      not this script: read results/run_env/${RUN_NAME}.env and the first
#      lines of logs/${RUN_NAME}.log, and confirm the checkpoint label, the
#      town, the seed (42) and the band (5-25 m) printed there are the ones
#      this banner claims. The first checkpoint appears in
#      checkpoints/${POLICY_FOLDER} at step 25000:
#        ls -t checkpoints/${POLICY_FOLDER} | head
#
# If (1) or (2) fail within ~3 minutes of [CARLA] ready, the attempt died at
# startup: read the log, do not just re-run the script. The likeliest first
# deaths are the two guards -- image_goal_coherence and check_gated_run --
# which refuse BEFORE a server start; their message names the flag.

# ---- the run ------------------------------------------------------------------
mkdir -p "logs"
kill_servers
for ATTEMPT in 1 2 3 4 5; do
  echo "=== 2-mode image-goal attempt $ATTEMPT ($(date +%H:%M:%S)) ==="
  s0=$(ckpt_step); s0=${s0:-0}
  "$PYTHON" -u train_carla_tqc.py >> "logs/${RUN_NAME}.log" 2>&1
  rc=$?
  s1=$(ckpt_step); s1=${s1:-$s0}
  echo "=== attempt $ATTEMPT exited rc=$rc at checkpoint ${s1} (was ${s0}) ==="
  if [ "$s1" -ge "$MAX_STEPS" ]; then
    echo "=== 2-mode image-goal finished cleanly at checkpoint ${s1} ==="; break
  fi
  # ARTEFACT CHECK, NOT EXIT STATUS. A wrapper's rc says nothing about progress
  # (and a wrapper exit of 0 after a failed attempt is this box's recorded
  # behaviour); only the checkpoint counter does. Three attempts with NO
  # PROGRESS is not a restart loop waiting to be lucky -- stop and read the log.
  if [ "$s1" -le "$s0" ] && [ "$ATTEMPT" -ge 3 ]; then
    echo "=== NO PROGRESS over the last attempt (still ${s1}); stopping ===" >&2
    exit 1
  fi
  # CARLA started by python dies WITH python; a fresh attempt needs a clean
  # card, and another session may have started a server in between.
  kill_servers
  sleep 10
done