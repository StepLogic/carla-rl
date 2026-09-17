#!/usr/bin/env bash
# Queue the CARLA VAE+TQC pipeline to start once the GPU is free.
#
# Waits on a PID rather than on nvidia-smi: a dip in reported memory is not the
# same as a run finishing, and starting a CARLA server into a still-occupied
# 12 GB card would fail or thrash both jobs.
#
# Stages, in order, each gated on the previous succeeding:
#   (a) semantic-camera collection across the TRAINING towns   [needs CARLA server]
#   (b) lane+obstacle VAE training                             [GPU, small]
#   (c) TQC (+HER) training with held-out Town03/Town10HD      [needs CARLA server]
#
# Nothing here kills or modifies another user's job. WAIT_PID is polled
# read-only with kill -0.

set -uo pipefail
# JOB CONTROL, so each background job becomes its own PROCESS GROUP with
# pgid == job pid. Without it every child shares the script's group and
# `kill -- -$pid` either hits nothing or would take the script down with it.
# This is what makes the group kill in cleanup() actually reap CarlaUE4.sh
# together with the CarlaUE4-Linux-Shipping binary it spawns.
set -m
cd "$(dirname "$0")"

WAIT_PID="${WAIT_PID:-1890709}"          # the job to wait for; 0 = don't wait
CARLA_ROOT="${CARLA_ROOT:-/home/robotlab/Apps/CARLA_0.9.15}"
CARLA_PORT="${CARLA_PORT:-2000}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8000}"     # refuse to start below this
FRAMES_PER_TOWN="${FRAMES_PER_TOWN:-5000}"
MAX_STEPS="${MAX_STEPS:-200000}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/carla_queue_${STAMP}"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/queue.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN_LOG"; }

CARLA_PID=""
CARLA_PIDS=()
cleanup() {
  # KILL THE PROCESS GROUP, not the PID. CarlaUE4.sh is a wrapper that execs
  # CarlaUE4-Linux-Shipping as a CHILD; killing the wrapper leaves the actual
  # server alive and still holding its port. That is not theoretical -- it
  # happened: the Epic server survived cleanup, the next pass found port 2000
  # open, logged "reusing it", and began collecting EPIC frames while tagging
  # the shards "Low". Silent mislabelling of a whole corpus.
  #
  # With `set -m` each background job is its own process group led by $pid, so
  # `kill -- -$pid` reaps the wrapper and the binary together. Only groups this
  # script created are ever signalled.
  #
  # NOTE: setsid does NOT work here -- it forks and exits, so $! would be the
  # setsid process rather than the group leader, and the kill would hit an
  # already-empty group while CARLA kept running. Measured.
  local pids=("${CARLA_PIDS[@]:-}")
  [ -n "$CARLA_PID" ] && pids+=("$CARLA_PID")
  for p in "${pids[@]:-}"; do
    [ -z "$p" ] && continue
    log "stopping CARLA server group (pgid $p)"
    kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null
  done
  sleep 5
  for p in "${pids[@]:-}"; do
    [ -z "$p" ] && continue
    kill -9 -- -"$p" 2>/dev/null || kill -9 "$p" 2>/dev/null
  done
  CARLA_PIDS=()
}

# Backstop: kill whatever is still LISTENING on a port we started a server on.
# Name-verified, so this can only ever signal a CarlaUE4 process -- never
# someone else's unrelated service that happens to hold the port.
kill_listener_on_port() {
  local port="$1" pid
  pid=$(ss -tlnpH "sport = :$port" 2>/dev/null |
        grep -oP 'pid=\K[0-9]+' | head -1)
  [ -z "${pid:-}" ] && return 0
  local name
  name=$(ps -o comm= -p "$pid" 2>/dev/null)
  case "$name" in
    *Carla*|*carla*)
      log "backstop: killing leftover $name (pid $pid) on port $port"
      kill -9 "$pid" 2>/dev/null ;;
    *)
      log "backstop: port $port held by '$name' (pid $pid) -- NOT ours, leaving it" ;;
  esac
}

# Block until a port is actually closed. A server takes seconds to release it
# after SIGKILL, and starting the next pass against a still-open port is what
# produced the mislabelled-quality bug.
wait_port_closed() {
  local port="$1" tries="${2:-30}"
  for _ in $(seq 1 "$tries"); do
    (exec 3<>/dev/tcp/localhost/"$port") 2>/dev/null || return 0
    sleep 1
  done
  log "WARNING: port $port still open after ${tries}s"
  return 1
}
# Only ever kills the server THIS script started -- never a pre-existing one.
trap cleanup EXIT
# INT/TERM must EXIT after cleaning up. Without the explicit exit the handler
# returns and the script carries on -- which is exactly what happened the first
# time this queue was signalled: it cleaned up and then went back to waiting,
# and had to be SIGKILLed.
trap 'log "interrupted"; cleanup; exit 130' INT TERM

start_carla() {  # optional $1 = map to boot directly into
  local boot_map="${1:-}"
  if (exec 3<>/dev/tcp/localhost/"$CARLA_PORT") 2>/dev/null; then
    # Reuse only when the caller explicitly allows it. A server already on the
    # port may have been launched at a DIFFERENT render quality, and silently
    # adopting it mislabels every frame collected against it.
    if [ "${CARLA_ALLOW_REUSE:-0}" = "1" ]; then
      log "CARLA already listening on $CARLA_PORT; reusing it (reuse allowed)"
      return 0
    fi
    log "FATAL: port $CARLA_PORT already in use, and this pass needs a server at "
    log "       quality=${CARLA_QUALITY:-Epic}. Refusing to adopt a server whose "
    log "       render quality cannot be verified. Set CARLA_ALLOW_REUSE=1 to override."
    return 1
  fi
  # ON SCREEN, by request: a visible window on $DISPLAY rather than
  # -RenderOffScreen. Two consequences worth knowing --
  #   * the render window needs a live X session; if DISPLAY is wrong or the
  #     session is locked, CarlaUE4 exits rather than falling back to headless,
  #     which is why start_carla checks the PID as well as the port.
  #   * an on-screen render costs extra VRAM and some throughput against a
  #     headless one. It is the right trade for watching collection and eval
  #     happen; set CARLA_OFFSCREEN=1 to go back to headless for an unattended
  #     long run.
  local render_flag=""
  if [ "${CARLA_OFFSCREEN:-0}" = "1" ]; then
    render_flag="-RenderOffScreen"
    log "starting CARLA server on port $CARLA_PORT (offscreen, quality=${CARLA_QUALITY:-Epic})"
  else
    export DISPLAY="${DISPLAY:-:1}"
    log "starting CARLA server on port $CARLA_PORT (ON SCREEN, DISPLAY=$DISPLAY, quality=${CARLA_QUALITY:-Epic})"
  fi
  # shellcheck disable=SC2086
  # $boot_map is forwarded to the binary as a positional argument. NOTE:
  # verified NOT to work on 0.9.15 -- the server boots into Town10HD_Opt
  # regardless, so the client still has to call load_world. Kept because it is
  # free and other versions honour it; the real protection against the
  # load_world segfault is the per-town retry loop below.
  # shellcheck disable=SC2086
  "$CARLA_ROOT/CarlaUE4.sh" $boot_map $render_flag -carla-port="$CARLA_PORT" \
      -quality-level="${CARLA_QUALITY:-Epic}" \
      -windowed -ResX="${CARLA_RESX:-1280}" -ResY="${CARLA_RESY:-720}" \
      >>"$LOG_DIR/carla_server_${CARLA_PORT}.log" 2>&1 &
  CARLA_PID=$!
  for i in $(seq 1 90); do
    sleep 2
    if (exec 3<>/dev/tcp/localhost/"$CARLA_PORT") 2>/dev/null; then
      log "CARLA up after $((i*2))s (pid $CARLA_PID)"
      sleep 5   # let the map finish loading before the first client connects
      return 0
    fi
    if ! kill -0 "$CARLA_PID" 2>/dev/null; then
      log "FATAL: CARLA server died on startup; see $LOG_DIR/carla_server_${CARLA_PORT}.log"
      return 1
    fi
  done
  log "FATAL: CARLA did not open port $CARLA_PORT within 180s"
  return 1
}

# Start up to N servers, one per training town, stopping early if the card
# cannot hold another. Sizing the fleet to MEASURED free memory rather than a
# hardcoded count is what keeps this from thrashing or dying halfway: a UE4
# server that fails to start after three succeeded would otherwise abort a run
# that could have proceeded on three towns.
#
# Ports step by 4: CARLA uses port and port+1 (streaming), and leaving a gap
# avoids a second server colliding with the first one's stream port.
start_carla_fleet() {
  local want="$1" quality="$2"
  local port="$CARLA_PORT" started=0
  CARLA_PORTS=""
  for i in $(seq 0 $((want - 1))); do
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    # Reserve headroom for the LEARNER before adding another server. Sizing
    # only against per-server cost filled the card with two servers and left
    # 1206 MiB, and JAX then died allocating 1.3 MB for a critic. A server that
    # starves the trainer is worse than one fewer town.
    local need=$(( ${PER_SERVER_MIB:-6000} + ${RESERVE_FOR_LEARNER_MIB:-3500} ))
    if [ -n "${free:-}" ] && [ "$free" -lt "$need" ]; then
      log "fleet: ${free} MiB free, need ${need} (server ${PER_SERVER_MIB:-6000} + "
      log "       learner reserve ${RESERVE_FOR_LEARNER_MIB:-3500}) -- stopping at $started server(s)"
      break
    fi
    if CARLA_PORT="$port" CARLA_QUALITY="$quality" \
       CARLA_RESX="${FLEET_RESX:-640}" CARLA_RESY="${FLEET_RESY:-360}" start_carla; then
      CARLA_PIDS+=("$CARLA_PID"); CARLA_PID=""
      CARLA_PORTS="${CARLA_PORTS:+$CARLA_PORTS,}$port"
      started=$((started + 1))
    else
      log "fleet: server on port $port failed to start -- continuing with $started"
      break
    fi
    port=$((port + 4))
  done
  if [ "$started" -eq 0 ]; then
    log "FATAL: could not start any CARLA server"
    return 1
  fi
  export CARLA_PORTS
  log "fleet: $started server(s) on ports $CARLA_PORTS (quality=$quality)"
  return 0
}

stage() {  # stage <name> <logfile> <command...>
  local name="$1" logfile="$2"; shift 2
  log "=== BEGIN $name ==="
  # Capture the status explicitly. `$?` read after an `if` reflects the IF, not
  # the command, which is why the first collection failure logged "exit 0" for
  # a process that had aborted on an uncaught C++ exception.
  local rc=0
  "$@" >"$logfile" 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then
    log "=== OK $name ==="
    return 0
  fi
  log "=== FAILED $name (exit $rc) -- see $logfile ==="
  tail -25 "$logfile" | tee -a "$MAIN_LOG"
  return 1
}

log "queue started; log dir $LOG_DIR"

# ---- wait for the GPU to free -------------------------------------------
if [ "$WAIT_PID" != "0" ] && kill -0 "$WAIT_PID" 2>/dev/null; then
  log "waiting for pid $WAIT_PID to exit ($(ps -o args= -p "$WAIT_PID" 2>/dev/null | cut -c1-70))"
  # `sleep` in the BACKGROUND then `wait`, rather than a foreground sleep.
  # Bash does not run a trap while a foreground command is executing, so a
  # plain `sleep 60` here defers Ctrl-C and SIGTERM by up to a minute and the
  # queue looks unkillable. `wait` is interruptible, so signals land at once.
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60 &
    wait $! 2>/dev/null || true
  done
  log "pid $WAIT_PID has exited"
  sleep 30   # let the driver release memory
else
  log "pid $WAIT_PID not running; proceeding immediately"
fi

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
log "GPU free: ${FREE:-unknown} MiB (need >= $MIN_FREE_MIB)"
if [ -n "${FREE:-}" ] && [ "$FREE" -lt "$MIN_FREE_MIB" ]; then
  log "FATAL: not enough free GPU memory; something else took the card. Aborting."
  exit 1
fi

# ---- (a) collection, once per RENDER QUALITY ------------------------------
#
# Quality is a server launch flag, so each pass needs its own server. Collecting
# at both Epic and Low makes render quality a randomisation axis alongside
# weather: a VAE fitted to one renderer degrades on the other, and that
# degradation would be invisible in the RL logs -- it would just look like a
# weaker policy. Shards are quality-tagged and accumulate, so a pass extends the
# corpus instead of replacing it, and re-running skips towns already on disk.
# ONE SERVER PER (quality, town). CARLA 0.9.15 segfaults on repeated
# client.load_world() inside one server session -- measured: Town01 collected
# cleanly, then the server died with Signal 11 on the transition to Town02 and
# the client saw only a 60s timeout. Loading exactly one map per session avoids
# the transition. Server startup is ~10s and finished towns are skipped via the
# shard cache, so the extra restarts are cheap.
TRAIN_TOWNS=$(python -c "import json;print(' '.join(json.load(open('configs/town_split.json'))['train']))")
log "training towns: $TRAIN_TOWNS"

for QUALITY in ${CARLA_QUALITIES:-Epic Low}; do
  for TOWN in $TRAIN_TOWNS; do
    if [ -f "data/lane_vae/shard_${TOWN}_${QUALITY}.npz" ]; then
      log "shard for ${TOWN}/${QUALITY} exists -- skipping"
      continue
    fi
    # KNOWN-IMPOSSIBLE COMBINATIONS, skipped without burning a server crash.
    #
    # CARLA 0.9.15 segfaults on load_world at reduced quality for every town
    # except Town01 and Town04. Not a guess: rlib_integration's own
    # carla_core.init_server() applies -quality-level ONLY for those two towns
    # and strips it otherwise -- this repo's author hit the same bug and
    # encoded the same workaround. Reproduced here five times (Signal 11).
    #
    # Without this the queue re-attempts them EVERY run, because a crashing
    # pair never writes a shard so the cache check above never fires. That cost
    # four segfaults and ~10 minutes per run.
    if [ "$QUALITY" != "Epic" ] && [ "$TOWN" != "Town01" ] && [ "$TOWN" != "Town04" ]; then
      log "skipping ${TOWN}/${QUALITY}: this CARLA build only supports reduced"
      log "         quality for Town01/Town04 (see carla_core.init_server)"
      grep -qxF "${TOWN}/${QUALITY}" data/lane_vae/SKIPPED.txt 2>/dev/null || \
        echo "${TOWN}/${QUALITY}" >> data/lane_vae/SKIPPED.txt
      continue
    fi
    # RETRY ON A FRESH SERVER. The UE4 server segfaults unpredictably (twice so
    # far, in different places), and that is not something this script can fix
    # from outside -- only survive. A retry is cheap because a completed town is
    # already skipped by its shard, so nothing is recollected.
    ok=0
    for attempt in $(seq 1 "${COLLECT_RETRIES:-3}"); do
      log "--- collect ${TOWN} @ ${QUALITY} (attempt $attempt) ---"
      if ! CARLA_QUALITY="$QUALITY" start_carla "$TOWN"; then
        log "server failed to start; retrying"
        cleanup; CARLA_PID=""
        wait_port_closed "$CARLA_PORT" 20 || kill_listener_on_port "$CARLA_PORT"
        continue
      fi
      if stage "collect ${TOWN} (${QUALITY}) try${attempt}" \
           "$LOG_DIR/collect_${TOWN}_${QUALITY}_try${attempt}.log" \
           env CARLA_QUALITY="$QUALITY" python src/vae_nav/collect_lane_data.py \
               --frames-per-town "$FRAMES_PER_TOWN" --out data/lane_vae \
               --tag "$QUALITY" --town "$TOWN"; then
        ok=1
      fi
      cleanup; CARLA_PID=""
      wait_port_closed "$CARLA_PORT" 20 || kill_listener_on_port "$CARLA_PORT"
      wait_port_closed "$CARLA_PORT" 25
      [ "$ok" = "1" ] && break
      log "attempt $attempt for ${TOWN}/${QUALITY} failed; server will be restarted"
    done
    if [ "$ok" != "1" ]; then
      # SKIP, do not abort. Some (town, quality) pairs crash the CARLA server
      # deterministically -- Town02 at Low quality segfaults it on every
      # attempt (Signal 11), while Town02/Epic and Town01/Low both collect
      # 5000 frames cleanly. Retrying a deterministic engine crash just burns
      # time, and the corpus does not need every combination: it needs enough
      # coverage of towns AND of render qualities, which it still has.
      #
      # Recorded loudly and in a file, because a silently missing combination
      # is exactly the kind of gap that later looks like a mysterious
      # generalisation result.
      log "SKIPPING ${TOWN}/${QUALITY}: failed ${COLLECT_RETRIES:-3} attempts "
      log "         (CARLA server crashes on this combination). Continuing."
      echo "${TOWN}/${QUALITY}" >> "$LOG_DIR/skipped_combinations.txt"
      echo "${TOWN}/${QUALITY}" >> data/lane_vae/SKIPPED.txt
    fi
  done
done

# Assemble the corpus from every shard on disk. No server needed.
stage "assemble corpus" "$LOG_DIR/assemble.log" \
  python src/vae_nav/collect_lane_data.py --out data/lane_vae --assemble-only || exit 1

# ---- (b) VAE --------------------------------------------------------------
# The VAE trainer refuses to write a checkpoint whose per-channel IoU is 0, so a
# failure here means the perception module is unusable -- stop rather than train
# a policy on a blind latent.
stage "train lane+obstacle VAE" "$LOG_DIR/train_vae.log" \
  python train_lane_vae.py --data data/lane_vae/lane_dataset.npz \
      --out checkpoints/lane_vae/best.pkl || exit 1

# ---- (c) RL ---------------------------------------------------------------
#
# THE FLEET STARTS HERE, NOT EARLIER. Starting it before the VAE stage left the
# trainer with 110 MiB of VRAM while two CARLA servers held 11.9 GB -- the VAE
# process simply stalled with an empty log. GPU-hungry stages must not overlap:
# the VAE gets the whole card, then the servers come up for RL.
N_ENVS="${N_ENVS:-4}"
start_carla_fleet "$N_ENVS" "${CARLA_QUALITY_RL:-Epic}" || exit 1

# JAX preallocates ~75% of visible VRAM by default, which on a card shared with
# CARLA servers means it either grabs memory the servers need or fails outright.
# On-demand allocation lets the two coexist.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Respect an inbound VAE_CKPT so the zero-latent ablation can run. Setting it
# to the empty string makes LatentSource return a zero vector, giving a policy
# with the goal block and NO perception -- the only honest baseline for "does
# the lane+obstacle latent help". Without that comparison, a failing
# VAE-equipped policy and a latent contributing nothing look identical.
#
# ${VAR-default} (not :-) so an explicitly EMPTY value is preserved.
export VAE_CKPT="${VAE_CKPT-checkpoints/lane_vae/best.pkl}"
export RL_ALGO=tqc
export EPISODE_HER=0   # HER disabled; see run_town02_fixed.sh for what this costs

# REWARD REBALANCE after the 25k evaluation returned success=0.000 on both
# held-out towns while four of six eval episodes ran the FULL 1001-step cap.
# The policy had learned collision avoidance, not navigation: collisions fell
# 0.90 -> 0.81, episode length rose 139 -> 249, `idle` rose 0 -> 0.07, and
# success never left the oracle floor. With a sparse goal, a dense -5 collision
# penalty and no living cost, standing still is locally optimal.
#
#  * K_LIVING makes waiting cost something. Must satisfy the suicide invariant
#    asserted in sparse_reward.py: k_collision > max_episode_steps * k_living,
#    i.e. < 0.005 here. 0.002 costs at most 2.0 over a 1000-step episode
#    against a 600 goal, so it discourages idling without making a crash
#    preferable to driving.
#  * EPISODE_HER_N=2 doubles the hindsight relabels per episode, which is the
#    lever meant for exactly this: real arrivals are rare, so synthesise more
#    goal-reaching signal from the trajectories that exist.
export K_LIVING="${K_LIVING:-0.002}"
export EPISODE_HER_N="${EPISODE_HER_N:-2}"

# EPISODE LENGTH, matched to the task rather than inherited.
#
# I set MAX_EPISODE_STEPS=1000 without deriving it; the peer's config of record
# used 200. Goals here are 4-8 m away and the oracle reaches one in ~40 steps,
# so 1000 is ~25x longer than the task needs. That matters because it is the
# term that makes idling rational:
#     idle a whole episode  costs k_living * T
#     crash                 costs k_collision
# and the suicide invariant REQUIRES k_collision > k_living * T. So no legal
# k_living can ever make idling worse than crashing -- the two constraints are
# mutually exclusive. T is the only term that moves. MEASURED with T=1000:
# idle rose 0.02 -> 0.07 -> 0.10 while episode length rose 205 -> 241 -> 274
# and success fell 0.117 -> 0.087.
#
# T=300 still leaves ~7x the steps an arrival needs, shrinks the pool of cheap
# "safe" experience 3.3x, and triples episode turnover so real outcomes arrive
# in the buffer far more often.
export MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"

# More oracle episodes. Real arrivals are what HER relabels around, and at
# ~0.1 success the buffer is starved of them. 0.2 -> 0.35 raises the density of
# complete successful trajectories without making the buffer mostly expert.
export EXPERT_PROB="${EXPERT_PROB:-0.35}"

# HER GOAL BAND -- the root cause of the missing goal signal.
#
# MEASURED on realistic trajectories (0.22 m/step, 240 steps): only 2.8% of
# hindsight relabels paid k_goal with max_disp=inf. The mechanism --
#   _pick_goal samples uniformly from ALL future points >= min_disp (2 m),
#   i.e. everything out to ~50 m, but _relabelled pays only when pos[i+1]
#   lands within goal_radius (2 m) of that goal, and pos[i+1] is just 0.22 m
#   from pos[i]. So ONLY goals in the annulus [2.0, ~2.2] m can ever pay.
# Of ~28k relabels the buffer held roughly 790 positives, ~1.4% of rows.
#
#   max_disp   paying fraction
#     inf         0.028
#      12         0.043
#       6         0.072   <- chosen
#       4         0.130
#       3         0.237
#
# 6.0 is 3x the goal radius, matching the ratio of the peer's WORKING runs
# (band [0.8, 2.5] against a 0.8 m radius), and keeps hindsight goals spanning
# 2-6 m, overlapping the task's real 4-8 m goals. Tightening to 3 m yields 3x
# more signal but trains on goal distances the task never presents. Free
# positives measured 0.000 at every value, so the min_disp guard still holds.
export HER_MAX_DISP="${HER_MAX_DISP:-6.0}"

# TEMPERATURE LEARNING RATE -- restored to the library default.
#
# MEASURED over ~6000 training steps: actor entropy sat at -4.19 / -4.27 /
# -4.19 against a TARGET_ENTROPY of -2.0 and never moved toward it. The
# SAC/TQC temperature exists to correct exactly that, and at temp_lr=1e-5 it
# adapts far too slowly -- a policy that goes deterministic early stays that
# way. That matches the behaviour precisely: competent collision avoidance,
# no goal-seeking, four held-out evaluations at 0.000.
#
# 1e-5 was the peer's override and their own notes flag it as such ("NOT the
# learner's own default of 3e-4; the config wins"). It suited a task where a
# navmesh oracle drove 20% of episodes toward a 0.8 m goal. Ours needs the
# agent to search, so the library default is restored.
export TEMP_LR="${TEMP_LR:-3e-4}"
export MAX_STEPS
stage "TQC+HER training" "$LOG_DIR/train_rl.log" \
  python train_carla_tqc.py || exit 1

log "queue complete"
