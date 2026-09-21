#!/bin/bash
# TRAINING v9 (2026-09-21, user spec). THE v8 RECIPE (VC-1 pyramid readout,
# 3854-d flat obs, Town01-only) WITH A REWRITTEN GOAL-MODE REWARD:
#
#   COUNTING REGIME (GOAL_TRUNCATE_COUNTS=1): no terminal bills. Arrival
#   (+10.0) is the ONLY terminus; every other end (collision, lane crossing,
#   off-road, clock) is a TRUNCATION -- truncated=True, mask stays 1.0, the
#   critic bootstraps through the cut (user: "truncate not terminate"; "all
#   events except goal reached truncates"). Truncation thresholds are
#   cumulative STEPS within the episode, 20 for all three faults (user:
#   "make it 20"). The truncating step pays its normal per-step reward; the
#   bootstrap carries the cost. Per-step fines 10.0: lane crossing, off-road,
#   collision (user: "increase lane crossing and offroad to 10"; collision
#   matched for consistency). Full details: gated_reward.py's constants block.
#
# EVERYTHING ELSE IS v8's, UNCHANGED: PERCEPTION=vc1 pyramid 3840-d,
# AGENT_ENCODER=none Dense trunk, OBS_WITH_STATE=1 (9-d block, no vector key),
# ACTOR_Q_NORM=1, HER, EXPLORE mode untouched (fines, 100 m arrival, bonus 10),
# REPLAY_CAPACITY=300k, EVAL_INTERVAL=5k checkpoints, Town01-only, windowed.
# FRESH RUN -- NO RESUME from v8 checkpoints: the v8 critic is fitted to the
# 5.0-fine/no-bonus scale; restoring it under the v9 reward is exactly the
# differently-scaled-reward restore the warm-start lesson forbids.
# NO v8 ROW IS COMPARABLE TO A v9 ROW. v8's 55k-step partial (10 checkpoints)
# stays on disk as the pyramid diagnostic that motivated this change.
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl || exit 1
source /home/robotlab/anaconda3/etc/profile.d/conda.sh
conda activate carla-rl || { echo "FATAL: no carla-rl env"; exit 1; }
export ENV_SKIP_IMAGE_FEATURES=1
# EXPORTED HERE, not only inline on the run_gated_her.sh call, so the .env snapshot records them. They were missing
# from carla_gated_her_v7.env, and a sweep replaying it fell back to PERCEPTION=vae (refused by the latent guard).
# THE v8 FLAGS ARE EXPORTED FOR THE SAME REASON (the v7 lesson): the compressor
# config and the coherence ruling's state-block flags are experiment-defining,
# so they must be in the snapshot (and in the required-vars check below), not
# only on the run line.
export PERCEPTION=vc1 ACTOR_Q_NORM=1
# THE COHERENCE RULING'S FLAG PAIR: OBS_WITH_VECTOR=0 (v8 drops the vector key)
# is accepted for a GATED run only with the 9-d state block -- OBS_WITH_STATE=1
# with OBS_STATE_COLLISION=0. Slot 0 of that block is the same clipped speed
# ratio the vector carried. train_v8.sh must export OBS_WITH_STATE=1 explicitly
# because the wrapper default is "0".
# OBS_WITH_VECTOR=0 is the coherence branch's own precondition (controller
# ruling 2026-09-21 made run_gated.sh overridable for exactly this); the v8
# layout is {goal, pixels, mode, state} with NO vector key.
#   POOLED READOUT (user 2026-09-21, "go with option 2"): VC1_READOUT=pyramid
#     replaces the (7,7,768) patch grid + d4pg conv compressor with a 3840-d
#     vector -- global mean + four 14x14-quadrant means (5x768) -- riding the
#     PROVEN v7 FLAT-LATENT path (AGENT_ENCODER=none, Dense, no conv). Keeps
#     global semantics + quadrant layout; buffer drops ~10x (300k x 3840 fp32
#     ~9 GB incl. next_obs mirror), so BUFFER_PIXELS_FP16 is moot (no pixels
#     key) and the 300k capacity stays.
export OBS_WITH_VECTOR=0
export OBS_WITH_STATE=1 OBS_STATE_COLLISION=0 OBS_LATENT_AS_PIXELS=0
export VC1_READOUT=pyramid AGENT_ENCODER=none AGENT_LATENT_DIM=512
export AGENT_AUGMENT_PIXELS=0
# BUFFER CAPS AT 300k TRANSITIONS (user 2026-09-21; MAX_STEPS stays 450k): after 300k the FIFO wrap keeps the
# most recent ~300k rows; pyramid latent ~9 GB fp32 incl. next_obs mirror.
export REPLAY_CAPACITY="${REPLAY_CAPACITY:-300000}"
# CARLA_ROOT IS REQUIRED AND HAS NO DEFAULT: train_carla_tqc.py:319 does a bare os.environ["CARLA_ROOT"] and would
# KeyError on the first env build. No script on this path exports it, and it is absent from every run_env snapshot --
# runs have been working only because an interactive shell happened to carry it. A detached run must not depend on
# ambient shell state.
export CARLA_ROOT=/home/robotlab/Apps/CARLA_0.9.15
# QUALITY=Low (user 2026-09-16). NOT already the case on this path: train_carla_tqc.py:320 and :441 both default to
# "Epic" and override carla_core's own BASE_CORE_CONFIG "Low". Setting it here covers the server launch flag AND the
# CarlaCore config in one place. MEASURED: a Low server holds ~4.85 GB against ~5.9 GB for Epic.
# CONSEQUENCE ON RECORD: v6n trained on EPIC frames, so v7's images differ from v6n's. That is a SECOND difference
# from v6n beside the termination/penalty change -- a v7-vs-v6n gap can no longer be attributed to termination alone.
export CARLA_QUALITY=Low
# CLIENT TIMEOUT 180s (was 50s). Both 2026-09-21 deaths were the same shape:
# a few episodes into Town10HD the server STOPPED ANSWERING for >50s (HD-map
# streaming stall; it was alive -- killed only afterwards by the retry loop's
# cleanup) and the client's C++ TimeoutException core-dumped the trainer,
# unrecoverable in Python. 180s absorbs the stall; CARLA_CLIENT_TIMEOUT is
# the env name collect_lane_data.py already uses.
export CARLA_CLIENT_TIMEOUT=180
# JAX VRAM CAP 0.2 (run_gated default 0.4 = 4.9 GB of a 12.3 GB card). The
# Town10HD wedge is CARLA-side: per-process VRAM shows the trainer at only
# ~1.1 GB while the server streams Town10HD toward the card ceiling. Capping
# jax's reservation leaves CARLA a bigger address pool before its streaming
# wedges the device. preallocate=false means this only caps the ceiling, it
# does not reserve anything up front.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia
# Overridable so a SMOKE RUN uses this identical flag set rather than a copy of it: a duplicated flag list is a list
# that drifts, and the whole point of the smoke is to exercise what the real run will exercise.
#   smoke:  RUN_NAME=carla_gated_her_v8_smoke MAX_STEPS=2000 ./train_v8.sh
RUN_NAME="${RUN_NAME:-carla_gated_her_v9_vc1}"
export MAX_STEPS="${MAX_STEPS:-450000}"
# RESUME FREEZE (user 2026-09-17, extending v7 450k -> 2M). RESUME restores actor, both critics, temps and curriculum
# but the replay buffer and Adam state start FRESH ([[warm-start-needs-an-actor-freeze]]). Critic-only updates first.
# UNITS ARE LEARNER UPDATES, NOT ENV STEPS: with 2 envs the loop does step += 2 per single update, so 12500 updates
# = 25k env steps. Default 0 keeps the original 0-450k run's behaviour.
export ACTOR_FREEZE_STEPS="${ACTOR_FREEZE_STEPS:-0}"
export EXPLORE_GOAL_DIST_M=100 EXPLORE_ARRIVAL_BONUS=2.0
export DISCOUNT=0.99 GATED_ALLOW_DISCOUNTED_SORB=1
# IN-TRAINING EVAL OFF. The eval env is make_env(..., eval_mode=True) with no port, so it attaches to CARLA_PORTS[0]
# -- the server Town01 is TRAINING on -- and reloads the world there. MEASURED 2026-09-16 at step 25k: that server
# segfaulted (Signal 11), the trainer hung reconnecting for ~21 min, and because the checkpoint is written AFTER the
# eval, nothing was saved -- so every restart would die at 25k forever. With 0 episodes run_evaluation builds no env
# (rate=0.0) and the explore eval is skipped. Checkpoints are written every EVAL_INTERVAL (5000 for v8).
# EVALUATE CHECKPOINTS OFFLINE, on a separate server, after the run.
export N_EVAL_EPISODES=0 N_EVAL_EXPLORE=0

# ---- v9 GOAL-MODE REWARD (user 2026-09-21) ----------------------------------
# COUNTING REGIME (GOAL_TRUNCATE_COUNTS=1): no terminal bills. Arrival (+10,
# GOAL_ARRIVAL_BONUS) is the ONLY terminus; everything else truncates
# (truncated=True, the critic bootstraps through the cut -- user: "truncate
# not terminate", "all events except goal reached truncates"), paying its
# NORMAL per-step reward on the way out (user's pick). Truncation thresholds
# are CUMULATIVE STEPS, 20 for all three (user: "make it 20"): off-road,
# collision, lane-crossing. Per-step fines 10.0 for lane/off-road (user:
# "increase ... to 10") and 10.0 for collision steps (consistent; the old
# regime billed collisions terminally so no per-step constant existed).
# COLLISION_TERMINAL / GATED_GOAL_LANE_TERMINAL are INERT under this regime
# (the counting branch returns before they are consulted) and are exported 0
# to say so honestly in the snapshot.
export GOAL_TRUNCATE_COUNTS=1 GOAL_ARRIVAL_BONUS=10.0
export GOAL_OFFROAD_TRUNCATE_STEPS=20 GOAL_COLLISION_TRUNCATE_STEPS=20
export GOAL_CROSSING_TRUNCATE_STEPS=20
export GOAL_LANE_PENALTY=10.0 GOAL_OFF_ROAD_PENALTY=10.0 GOAL_COLLISION_PENALTY=10.0
export COLLISION_TERMINAL=0 GATED_GOAL_LANE_TERMINAL=0 WEDGE_TRUNCATE=0

# ---- EXPLORE mode: UNCHANGED (off-road fatal, 5.0 fines, 100 m arrival) -----
export EXPLORE_OFF_ROAD_FATAL=1 EXPLORE_OFF_ROAD_PENALTY=5.0
export LANE_COLLISION_PENALTY=5.0 LANE_DEPARTURE_PENALTY=5.0

# ---- GOAL SAMPLING ----------------------------------------------------------
export JUNCTION_GOAL_FRAC=0.5 STRAIGHT_GOAL_FRAC=0.3

# ---- TOWN: Town02 ONLY (user 2026-09-21; earlier: Town01+Town10HD one env each) ----
# TOWN_LIST points at a SEPARATE split rather than editing configs/town_split.json, which carries "Do not edit
# without a new experiment name" and underpins every existing v4/v5/v6* result. town_split_v7.json moves Town10HD
# from heldout to train and changes nothing else. Held-out evidence now rests on Town03 ALONE.
export TOWN_LIST=configs/town_split_v7.json
#
# TWO SERVERS FIT. MEASURED LIVE 2026-09-16 on the aborted single-env run at step 24k with learning active:
#   CARLA server with Town10HD loaded  4290 MiB
#   learner (JAX, preallocate off)      428 MiB
# plus a Town01 server (~3.5 GB) -> ~8.3 GB of 12.3 GB. An earlier "does not fit" was WRONG: its probes booted
# servers with no map (default Town10HD_Opt) and took the learner as ~2.3 GB from a comment instead of measuring.
# Ports step by 4: CARLA uses port and port+1 (streaming), as run_carla_queue.sh does.
# ONE TOWN (user 2026-09-21, "relaunch with only town02 training"): the
# Town10HD stints wedged the CARLA server's RPC 5/5 with VRAM free (not the
# documented load_world crash, not memory -- gdb stack capture was never
# completed), so Town02-only removes the wedge trigger outright AND drops
# all world reloads. Town02 is a TRAIN town in the frozen split. Held-out
# evidence still rests on Town03 alone.
# TOWN01 (user 2026-09-21: "run it train in town01" — second town call of the
# day; the Town02-only relaunch then lost its CARLA server to a Signal 11
# segfault during env build, ON Town02, so the Town10HD-only wedge theory is
# dead too). Town01 is v6n/v7's own training town.
export ENV_TOWNS=Town01
# TOWN_ROTATE_EPISODES=0: with one requested town the trainer disables the
# rotation itself; exported 0 to make that a written decision, not an
# accident of the single-town branch.
export TOWN_ROTATE_EPISODES=0
# CHECKPOINT EVERY 5k (was 25k). Pure resilience, not an experiment change:
# N_EVAL_EPISODES=0 already skips the eval, so EVAL_INTERVAL only sets
# checkpoint cadence -- and every death this run has had happened BELOW the
# 25k first checkpoint, i.e. cost the entire attempt. Restarting resumes
# from the newest checkpoint instead.
export EVAL_INTERVAL=5000
export CARLA_PORTS=2000 CARLA_PORT=2000
#
# EVAL_TOWNS=Town01 keeps v6n's in-distribution overfitting test. UNSET IT to evaluate on held-out Town03 instead.
export EVAL_TOWNS=Town01
# Headless: only ~12 MiB, but it removes the live-X-session dependency -- windowed, CarlaUE4 EXITS rather than
# falling back if DISPLAY is wrong or the session locks, which an unattended 450k-step run must not depend on.
# WINDOWED (user 2026-09-16: "the ui is not showing"). Needs the live X session on DISPLAY=:1, which run_gated.sh
# exports; if that session locks, CarlaUE4 EXITS rather than falling back to headless. Offscreen saved only ~12 MiB.
export CARLA_OFFSCREEN=0
# Python starts BOTH servers (one per port, at CARLA_QUALITY/CARLA_OFFSCREEN). Without this, run_gated.sh started its
# own port-2000 server at hardcoded Epic/windowed and python attached to it -- Town01 silently ran at Epic, and the
# Epic+Low pair starved the card ("A sensor took too long to send their data").
export RUN_GATED_START_SERVER=0

say(){ echo "=== [v8 $(date +%H:%M:%S)] $* ==="; }
kill_all_carla(){ for _ in 1 2 3 4 5 6; do pkill -9 -f 'CarlaUE4-Linux-Shipping' 2>/dev/null; sleep 2; ps -eo args | grep -qi '[C]arlaUE4' || return 0; done; return 1; }
# DO NOT DELETE THE PORT LOCK (/tmp/carla_rl_port2000.lock). run_gated.sh takes flock -n on it so two trainers can
# never share port 2000. Removing the file is what put two trainers on one server on 2026-09-11.
mkdir -p results/run_env logs
kill_all_carla
# THE SNAPSHOT PATTERN MUST ALLOW A SUFFIX. The inherited pattern was "^(GATED|EXPLORE_|GOAL_|...)=", which anchors
# the alternation straight to "=" -- so a PREFIX entry can only match a variable named exactly "EXPLORE_", and every
# EXPLORE_*/GOAL_*/LANE_*/GATED_* flag was silently dropped. That is not cosmetic: run_drive_plan_c.sh REPLAYS these
# .env files at evaluation time, so a dropped flag reverts to its default and the eval quietly stops matching the run.
# "[A-Z0-9_]*=" is what makes the prefixes work. Verified below rather than trusted.
# CARLA and TOWN are in the list because THIS run's identity depends on them: CARLA_QUALITY=Low (v6n was Epic) and
# TOWN_LIST=town_split_v7.json (v6n used the frozen split) are as experiment-defining as any reward flag. Omitting
# them would let an eval replaying this .env run at Epic against the ORIGINAL split and look like a matched rerun.
# v8 ADDS SIX PREFIXES for the same reason: OBS_LATENT_ and OBS_STATE_ (the
# coherence ruling's flag pair) and AGENT_/CNN_/BUFFER_/VC1_ (the compressor
# config) match NONE of the inherited alternations, so without them the
# snapshot would silently drop every v8-defining flag. REPLAY (the 300k
# capacity, user 2026-09-21) matches none of them either.
env | grep -E "^(GATED|OBS_WITH|OBS_STATE|OBS_LATENT|PERCEPTION|STEP_COST|DISCOUNT|EXPLORE|GOAL|LANE|COLLISION|WEDGE|POTENTIAL|TARGET_SPEED|MAX_THROTTLE|MAX_STEER|MIN_THROTTLE|ENV_|EXPERT|N_EVAL|JUNCTION|STRAIGHT|EVAL|CARLA|TOWN|RUN_GATED|ACTOR|AGENT|CNN|BUFFER|VC1|REPLAY)[A-Z0-9_]*=" \
  | sort > "results/run_env/${RUN_NAME}.env"
# ARTEFACT CHECK ON THE SNAPSHOT ITSELF: every flag this run is ABOUT must be in the record, or the record is a lie
# and any later eval replaying it silently differs from training. Abort rather than train against a bad provenance.
_missing=""
for _k in COLLISION_TERMINAL GATED_GOAL_LANE_TERMINAL WEDGE_TRUNCATE EXPLORE_OFF_ROAD_FATAL \
          GOAL_TRUNCATE_COUNTS GOAL_ARRIVAL_BONUS GOAL_LANE_PENALTY GOAL_OFF_ROAD_PENALTY \
          GOAL_COLLISION_PENALTY GOAL_OFFROAD_TRUNCATE_STEPS GOAL_COLLISION_TRUNCATE_STEPS \
          GOAL_CROSSING_TRUNCATE_STEPS EXPLORE_OFF_ROAD_PENALTY LANE_COLLISION_PENALTY \
          JUNCTION_GOAL_FRAC STRAIGHT_GOAL_FRAC ENV_TOWNS EVAL_TOWNS \
          CARLA_QUALITY CARLA_ROOT CARLA_OFFSCREEN CARLA_PORTS CARLA_CLIENT_TIMEOUT TOWN_LIST TOWN_ROTATE_EPISODES EVAL_INTERVAL RUN_GATED_START_SERVER N_EVAL_EPISODES N_EVAL_EXPLORE ACTOR_FREEZE_STEPS PERCEPTION ACTOR_Q_NORM \
          OBS_WITH_VECTOR OBS_WITH_STATE OBS_STATE_COLLISION OBS_LATENT_AS_PIXELS \
          AGENT_ENCODER AGENT_LATENT_DIM AGENT_AUGMENT_PIXELS \
          VC1_READOUT REPLAY_CAPACITY; do
  grep -qE "^${_k}=" "results/run_env/${RUN_NAME}.env" || _missing="$_missing $_k"
done
if [ -n "$_missing" ]; then
  say "FATAL: these flags are missing from results/run_env/${RUN_NAME}.env:$_missing"
  exit 1
fi
say "settings snapshotted to results/run_env/${RUN_NAME}.env ($(wc -l < "results/run_env/${RUN_NAME}.env") vars, all v8 flags present)"
CK="checkpoints/$RUN_NAME"
step(){ ls $CK 2>/dev/null | sed -n 's/^checkpoint_\([0-9]*\)\.pkl$/\1/p' | sort -n | tail -1; }
for ATTEMPT in $(seq 1 40); do
  s=$(step); s=${s:-0}
  if [ "$s" -ge "$MAX_STEPS" ]; then say "TARGET REACHED at step $s"; break; fi
  say "attempt $ATTEMPT (newest checkpoint: ${s} / ${MAX_STEPS}); python=$(which python)"
  POLICY_FOLDER="$RUN_NAME" PERCEPTION=vc1 ACTOR_Q_NORM=1 RESUME=1 \
    ./run_gated_her.sh >> "logs/${RUN_NAME}.log" 2>&1
  rc=$?
  s2=$(step); s2=${s2:-0}
  say "attempt $ATTEMPT exited rc=$rc at step $s2"
  if [ "$s2" -ge "$MAX_STEPS" ]; then say "TARGET REACHED at step $s2"; break; fi
  # ARTEFACT CHECK, NOT EXIT STATUS. The v6n_2m run printed "gated run done" having died at 150000 of 2000000,
  # because the wrapper's rc says nothing about progress. Only the checkpoint counter does.
  if [ "$s2" -le "$s" ] && [ "$ATTEMPT" -gt 3 ]; then
    say "NO PROGRESS over the last attempt (still $s2); stopping so it is not an infinite restart loop"; exit 1
  fi
  kill_all_carla; sleep 30
done
say "done; newest checkpoint $(step) of ${MAX_STEPS}"
