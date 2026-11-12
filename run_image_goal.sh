#!/usr/bin/env bash
# THE IMAGE-GOAL ARM (Phase 2): reach the viewpoint the goal image was taken
# from and STOP there -- one reward, one mode, VC-1 features, hindsight
# relabelling that does not recost (ImageGoalRelabelBuffer keeps the flown
# reward and sets done only at the hindsight entry).
#
# A NEW LINEAGE, NOT A VARIANT OF THE GATED ARMS. The observation differs
# (768-d VC-1 latent + the coll key), the reward is ImageGoalReward's
# ReViND-style all-negative shape, and the agent is the PLAIN TQCLearner --
# so no existing checkpoint loads into it and none of the gated learner's
# per-critic machinery applies. Every value this run needs is set HERE,
# explicitly. This script SOURCES NOTHING: not run_gated.sh, not train_v7.sh.
# It copies only STRUCTURE from run_gated.sh (the lock, the GPU poll, the
# kill/cleanup, the retry loop); the env values are this arm's own, because a
# copied env value is an inherited experiment decision -- train_v7.sh's
# ARRIVAL_BONUS=2.0, penalty 5.0s and town_split_v7.json are that lineage's
# record, not this arm's settings.
#
# UNCLIPPED BY DESIGN (user 2026-09-19). The gated learner derives
# value_floor/value_ceiling from the SORB reward window per critic; the plain
# agent applies NO bounds. The bound-related exports therefore appear NOWHERE
# below, and the script actively clears them if a caller's shell carries them:
#
#     TQC_TARGET_MIN / TQC_TARGET_MAX   read at import in jaxrl2's
#                                       critic_updater and baked into every
#                                       TD target (run_sorb_agent.sh exports
#                                       TQC_TARGET_MIN=-30 in its lineage)
#     TQC_VALUE_FLOOR / TQC_VALUE_CEILING  the gated learner's per-critic
#                                       bounds (inert at GATED=0, scrubbed
#                                       anyway so the .env snapshot cannot
#                                       imply they applied)
#
# NO_TIMELIMIT stays UNSET: this is a training run, not trace collection.
# (NO_TIMELIMIT=1 would remove TimeLimit and ImageGoalReward's own step cap.)
# It is scrubbed below for the same reason -- the snapshot must not record a
# cap the run did not train under.
#
# RUNS ON THE CLUSTER (memory rule); this script ships the launch, not the run.
set -uo pipefail
cd "$(dirname "$0")"

# ---- one CARLA job at a time ------------------------------------------------
# Held for the script's lifetime. Non-blocking on purpose: a run that WAITED on
# the lock would sit holding its VRAM reservation while another job ran, which
# is the deadlock this convention exists to avoid. Refuse and exit instead.
# (Structure from run_gated.sh, including the fd number another script's
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
# Same poll as run_gated.sh, gate on FREE MEMORY not on a process name
# (/proc/PID/comm is capped at 15 chars, so the 23-char CARLA binary name is
# truncated and an exact-name match can never fire). This arm's floor is LOWER
# than the gated arm's 9500 MiB and the difference is measured, not hopeful:
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
# run on driver 535) plus vc_models for PERCEPTION=vc1. A missing venv must
# REFUSE, not fall back to system python: a silent fallback runs the whole run
# on a torch that either fails at the first VC-1 forward or, worse, half-works.
PYTHON="${PYTHON:-dependencies/ig-venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  echo "FATAL: $PYTHON is not executable. Build the venv (Task 3) or pass" >&2
  echo "       PYTHON=/path/to/python explicitly." >&2
  exit 1
fi

# ---- anti-inheritance scrub ---------------------------------------------------
# The values this arm trains under are set below and NOWHERE ELSE. An ambient
# shell that ran a gated or SORB job can still carry the variables that would
# silently change THIS run's meaning, so they are cleared rather than hoped
# away. Anything on this list is either inert at GATED=0 or actively wrong:
#   NO_TIMELIMIT            training run, not trace collection (see header)
#   TQC_TARGET_MIN/MAX      global target clip, read at import in critic_updater
#   TQC_VALUE_FLOOR/CEILING gated learner's per-critic bounds
#   EXPLORE_ARRIVAL_BONUS   train_v7.sh's +2.0 explore arrival bonus -- a
#                           historical record of THAT lineage, not a setting
#   GATED_ALLOW_DISCOUNTED_SORB  the gated arm's guard-bypass for a discounted
#                           SORB reward; there is no SORB reward on this arm
#   ACTOR_Q_NORM            gated learner's per-mode actor normalisation
for _scrub in NO_TIMELIMIT TQC_TARGET_MIN TQC_TARGET_MAX \
              TQC_VALUE_FLOOR TQC_VALUE_CEILING \
              EXPLORE_ARRIVAL_BONUS GATED_ALLOW_DISCOUNTED_SORB ACTOR_Q_NORM; do
  if [ -n "${!_scrub+x}" ]; then
    echo "[Scrub] clearing inherited $_scrub=${!_scrub}"
    unset "$_scrub"
  fi
done

# ---- the arm ------------------------------------------------------------------
# THE BRIEF'S ENV BLOCK, VERBATIM (plan 2026-09-19-image-goal-phase2, Task 7).
export IMAGE_GOAL_TASK=1
export PERCEPTION=vc1
export OBS_WITH_COLLISION=1
export GATED=0
export EPISODE_HER=1
export DISCOUNT=0.99            # user 2026-09-19: discounted, unclipped
export TOP_QUANTILES_DROP=2     # proven regime: gamma=0.99 + drop 2 + unclipped
                                # worked in v5/v7 (tqc-drop memory: watch item only)
export TQC_KAPPA=1.0            # per-step TDs are O(1); the default derives
                                # kappa from K_GOAL and would land at 60
export IG_GOAL_DISTANCE_THRESHOLD_M=2.0
export HER_MIN_DISP=5.0         # above the 2 m payout radius: no free positives
# NO target bounds of any kind: the arm is unclipped by design (gated learner
# applies bounds per critic; the plain agent applies none). See the scrub block
# above for the four exports that would have reintroduced one.

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

# ---- the environment ----------------------------------------------------------
export TOWN_LIST="${TOWN_LIST:-configs/town_split.json}"  # the FROZEN split, not
                                  # train_v7.sh's town_split_v7.json: this arm
                                  # is a new lineage, not a v7 continuation
export ENV_TOWNS="${ENV_TOWNS:-Town01}"   # EVAL_TOWNS is DELIBERATELY UNSET:
                                  # with it unset the periodic eval would use
                                  # the split's held-out towns -- except it
                                  # never runs at all, see N_EVAL_* below, so
                                  # there is no in-run eval to aim anywhere.
export TOWN_ROTATE_EPISODES=0     # single town: rotation is inert, pinned 0 so
                                  # the snapshot says so (world reloads are the
                                  # CARLA 0.9.15 crash vector)
export MAX_EPISODE_STEPS=1000     # MUST match IG_MAX_STEPS (image_goal_reward's
                                  # own clock, module default 1000): the
                                  # TimeLimit cap and the reward's cap are the
                                  # same number so truncation has one owner.
                                  # IG_MAX_STEPS is left UNSET on purpose --
                                  # setting it here would make two files own
                                  # one number; change BOTH if it ever moves.
# IN-RUN EVAL OFF (v7's measured lesson, 2026-09-16). The eval env is
# make_env(..., eval_mode=True) with no port, so it attaches to CARLA_PORTS[0]
# -- the server the training env is running on -- and reloads the world there.
# Measured on v7: the server segfaulted at the 25k eval, the trainer hung
# reconnecting ~21 min, and because the checkpoint is written AFTER the eval,
# nothing was saved. N_EVAL_EPISODES=0 skips the goal eval entirely;
# N_EVAL_EXPLORE=0 skips the explore eval, which is a gated-arm measurement and
# has no meaning here. Evaluate checkpoints OFFLINE, on a separate server.
export N_EVAL_EPISODES=0 N_EVAL_EXPLORE=0
export TARGET_SPEED="${TARGET_SPEED:-20.0}"  # the config's 5.0 km/h is
                                  # upstream-stale for this vehicle (measured
                                  # 22.4 km/h at throttle 0.6); the env's idle
                                  # and expert paths read it
# ENV_SKIP_IMAGE_FEATURES=1: the env's two MobileNet passes ("state" and "goal"
# keys) are DISCARDED by this stack -- VAEGoalWrapper builds its observation
# from info["rgb"] (encoded by VcSource), the goal fields, the vector and the
# coll key -- and torch on this machine runs them on CPU, measured at 17 of 32
# cores per step. Pinned byte-exact by tests/test_skip_discarded_image_features.py.
export ENV_SKIP_IMAGE_FEATURES=1

# ---- goal placement + expert (user decisions 2026-09-19) ----------------------
# FIXED BAND, NO CURRICULUM (user: goal band "5 - 100"). carla_core samples the
# goal's distance from [GOAL_DIST_MIN, GOAL_DIST_MAX] at every reset; the
# trainer's curriculum (train_carla_tqc.py, the CURR_* block) advances
# GOAL_DIST_MAX from GOAL_DIST_START toward GOAL_DIST_TARGET on success. Pinning
# START=TARGET=100.0 leaves curr_max at 100 from episode 1 with nothing to
# advance to, so EVERY episode draws its goal in the fixed 5-100 m band. These
# are exported here (rather than left to the trainer's runtime os.environ
# writes) so the .env snapshot records the schedule an eval replay needs.
# OVERRIDE WITH ${VAR:-default} so the v2 retrain (user 2026-09-19: band
# 5-25 m, inside gamma=0.99's ~28 m effective horizon) can set
# GOAL_DIST_START=GOAL_DIST_TARGET=25.0 without editing this file; the
# defaults preserve the v1 record exactly.
export GOAL_DIST_MIN="${GOAL_DIST_MIN:-5.0}" \
       GOAL_DIST_START="${GOAL_DIST_START:-100.0}" \
       GOAL_DIST_TARGET="${GOAL_DIST_TARGET:-100.0}"
# NO ORACLE EPISODES (user: expert_prob = 0). Every episode is policy-driven
# from step 1; the expert path stays available for offline analysis later.
export EXPERT_PROB=0.0

# ---- the run ------------------------------------------------------------------
export RUN_NAME="${RUN_NAME:-carla_image_goal}"
export POLICY_FOLDER="${POLICY_FOLDER:-$RUN_NAME}"
export MAX_STEPS="${MAX_STEPS:-1000000}"   # env steps; trainer default 200k is
                                           # short for a from-scratch arm.
                                           # OVERRIDABLE: this default is a
                                           # budget choice, not a tuning claim.
export SEED="${SEED:-42}"
export RESUME="${RESUME:-1}"

# ---- jax memory (structure from run_gated.sh) --------------------------------
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
mkdir -p results/run_env logs
env | grep -E "^(IMAGE_GOAL|IG_|PERCEPTION|OBS_WITH|EPISODE_HER|HER_|DISCOUNT|TOP_QUANTILES|TQC_|NUM_QS|N_QUANTILES|TARGET_|TEMP_|UTD_|MINIBATCH|REPLAY_|GATED|ENV_|TOWN|EVAL_|CARLA|N_EVAL|GOAL_|MAX_|POLICY|RESUME|SEED|XLA|START_|EXPERT|INCREAMENTAL|INCREMENTAL|STEER|THROTTLE|COLLISION|LANE|WEDGE|SHORT_|JUNCTION|STRAIGHT|K_|NO_|RUN_|NEED_MB)[A-Z0-9_]*=" \
  | sort > "results/run_env/${RUN_NAME}.env"
_missing=""
for _k in IMAGE_GOAL_TASK PERCEPTION OBS_WITH_COLLISION GATED EPISODE_HER \
          DISCOUNT TOP_QUANTILES_DROP TQC_KAPPA \
          IG_GOAL_DISTANCE_THRESHOLD_M HER_MIN_DISP \
          CARLA_ROOT CARLA_QUALITY CARLA_OFFSCREEN CARLA_START_SERVER \
          CARLA_PORT CARLA_PORTS TOWN_LIST ENV_TOWNS TOWN_ROTATE_EPISODES \
          MAX_EPISODE_STEPS TARGET_SPEED ENV_SKIP_IMAGE_FEATURES \
          GOAL_DIST_MIN GOAL_DIST_START GOAL_DIST_TARGET EXPERT_PROB \
          N_EVAL_EPISODES N_EVAL_EXPLORE \
          RUN_NAME POLICY_FOLDER MAX_STEPS SEED RESUME \
          XLA_PYTHON_CLIENT_PREALLOCATE XLA_PYTHON_CLIENT_MEM_FRACTION; do
  grep -qE "^${_k}=" "results/run_env/${RUN_NAME}.env" || _missing="$_missing $_k"
done
if [ -n "$_missing" ]; then
  echo "FATAL: these flags are missing from results/run_env/${RUN_NAME}.env:$_missing" >&2
  exit 1
fi
# AND THE SNAPSHOT MUST NOT CONTAIN WHAT THE ARM DELIBERATELY LACKS. A bound
# export here would mean the unclipped-by-design claim is false and the
# critic's targets are being clipped silently.
_present=""
for _k in NO_TIMELIMIT TQC_TARGET_MIN TQC_TARGET_MAX \
          TQC_VALUE_FLOOR TQC_VALUE_CEILING; do
  grep -qE "^${_k}=" "results/run_env/${RUN_NAME}.env" && _present="$_present $_k"
done
if [ -n "$_present" ]; then
  echo "FATAL: the unclipped arm's snapshot carries bound/clock exports:$_present" >&2
  exit 1
fi
echo "[Banner] run=${RUN_NAME} policy_folder=${POLICY_FOLDER} interpreter=${PYTHON}"
echo "[Banner] IMAGE_GOAL_TASK=1 PERCEPTION=vc1 OBS_WITH_COLLISION=1 GATED=0"
echo "[Banner] EPISODE_HER=1 DISCOUNT=0.99 TOP_QUANTILES_DROP=2 TQC_KAPPA=1.0"
echo "[Banner] IG_GOAL_DISTANCE_THRESHOLD_M=2.0 HER_MIN_DISP=5.0 -- unclipped:"
echo "[Banner] no TQC_TARGET_MIN/MAX, no TQC_VALUE_FLOOR/CEILING, NO_TIMELIMIT unset"
echo "[Banner] snapshot: results/run_env/${RUN_NAME}.env ($(wc -l < "results/run_env/${RUN_NAME}.env") vars)"

# ---- server cleanup + retry (structure from run_gated.sh) --------------------
# pgrep -x CANNOT match: /proc/PID/comm is capped at 15 chars, so the 23-char
# 'CarlaUE4-Linux-Shipping' truncates and an exact-name match never fires. The
# bracket stops a concurrent cleanup matching THIS script's argv.
kill_servers() {
  for _ in 1 2 3 4 5; do
    mapfile -t pids < <(pgrep -f '[C]arlaUE4-Linux-Shipping')
    [ ${#pids[@]} -eq 0 ] && return 0
    for p in "${pids[@]}"; do kill -9 "$p" 2>/dev/null; done
    sleep 3
  done
  pgrep -f '[C]arlaUE4-Linux-Shipping' >/dev/null && \
    echo "WARNING: a CARLA server survived 5 kill attempts" >&2
}
cleanup() { kill_servers; echo "=== image-goal run done ==="; }
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
#      expect, in order: [Banner] lines, [Towns] frozen split, [Latent]
#      backend 'vc1_vitb', [CARLA] starting/ready on port 2000, [ImageGoal]
#      reward banner, [VAEGoal] obs={... 'coll': (1,)}, [EpisodeHER] on,
#      then [Ep] lines once training starts.
#   3. the GPU is actually working:
#        nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
#      (a CARLA server at Low with a town loaded holds ~2-4.9 GB)
#   4. the run's OWN record agrees with all of the above -- the LIVE process,
#      not this script: read results/run_env/${RUN_NAME}.env and the first
#      lines of logs/${RUN_NAME}.log, and confirm the checkpoint label, the
#      town and the seeds printed there are the ones this banner claims. The
#      first checkpoint appears in checkpoints/${POLICY_FOLDER} at step 25000:
#        ls -t checkpoints/${POLICY_FOLDER} | head
#
# If (1) or (2) fail within ~3 minutes of [CARLA] ready, the attempt died at
# startup: read the log, do not just re-run the script.

# ---- the run ------------------------------------------------------------------
mkdir -p "logs"
kill_servers
for ATTEMPT in 1 2 3 4 5; do
  echo "=== image-goal attempt $ATTEMPT ($(date +%H:%M:%S)) ==="
  s0=$(ckpt_step); s0=${s0:-0}
  "$PYTHON" -u train_carla_tqc.py >> "logs/${RUN_NAME}.log" 2>&1
  rc=$?
  s1=$(ckpt_step); s1=${s1:-$s0}
  echo "=== attempt $ATTEMPT exited rc=$rc at checkpoint ${s1} (was ${s0}) ==="
  if [ "$s1" -ge "$MAX_STEPS" ]; then
    echo "=== image-goal finished cleanly at checkpoint ${s1} ==="; break
  fi
  # ARTEFACT CHECK, NOT EXIT STATUS. A wrapper's rc says nothing about progress;
  # only the checkpoint counter does. Three attempts with NO PROGRESS is not a
  # restart loop waiting to be lucky -- stop and read the log.
  if [ "$s1" -le "$s0" ] && [ "$ATTEMPT" -ge 3 ]; then
    echo "=== NO PROGRESS over the last attempt (still ${s1}); stopping ===" >&2
    exit 1
  fi
  # CARLA started by python dies WITH python; a fresh attempt needs a clean
  # card, and another session may have started a server in between.
  kill_servers
  sleep 10
done