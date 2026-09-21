#!/bin/bash
# TRAINING v8 (2026-09-21). THE v7 RECIPE WITH THE VC-1 CONV-COMPRESSOR OBSERVATION.
# Everything that made v7 v7 is INHERITED BYTE-IDENTICAL below -- termination
# (COLLISION_TERMINAL=1, EXPLORE_OFF_ROAD_FATAL=1), 5.0 penalties in both modes,
# arrival bonus 2.0, DISCOUNT=0.99 with GATED_ALLOW_DISCOUNTED_SORB=1,
# ACTOR_Q_NORM=1, N_EVAL_EPISODES=0/N_EVAL_EXPLORE=0, towns, ports, MAX_STEPS.
# WHAT CHANGES IS THE OBSERVATION ONLY:
#
#   COMPRESSOR (Tasks 1-3): PERCEPTION=vc1 with the VC-1 ViT-B grid2 readout
#     published as a (7, 7, 768) `pixels` key (OBS_LATENT_AS_PIXELS=1,
#     VC1_READOUT=grid2), compressed by the d4pg conv encoder
#     AGENT_ENCODER=d4pg CNN_FEATURES=64,128,512 CNN_FILTERS=3,3,1
#     CNN_STRIDES=2,2,1 -> AGENT_LATENT_DIM=512. AUGMENTATION OFF
#     (AGENT_AUGMENT_PIXELS=0 -- drq-style image aug on patch tokens is not
#     meaningful). BUFFER_PIXELS_FP16=1: the 37632-float-per-step pixel rows
#     are stored fp16 -- at the 300k capacity below that is ~45 GB including
#     the next_observations mirror (fp32 would be ~90 GB and does not fit
#     125 GB RAM beside the run's other residents).
#
#   COHERENCE RULING (ledger ruling on the Tasks 1+2 review): run_gated.sh
#     hardcodes OBS_WITH_VECTOR=1 and both gated coherence checks demanded it,
#     with only the IMAGE_GOAL_TASK escape -- the wrong arm for v8. The
#     user-approved v8 layout has NO vector key: the 9-d state block is the
#     velocity carrier (OBS_WITH_STATE=1, OBS_STATE_COLLISION=0). Its slot 0 IS
#     the same clipped speed/TARGET_SPEED ratio the 4-d vector block carried,
#     i.e. the observable speed the progress reward assumes. Both coherence
#     checks (vae_goal_wrapper.check_gated_coherence, gated_config.check_gated_run)
#     now accept exactly that configuration, which is what this launcher exports.
#
# NO v7 ROW IS COMPARABLE TO A v8 ROW: encoder, observation keys and widths all
# change. v8 is a new lineage the way v7 was.
#
# INHERITED v7 HEADER, STILL TRUE (termination/penalties/sampling):
#
# WHAT CHANGED FROM v6n (the recipe behind checkpoint_250000 that every current result uses):
#   goal mode   arrival AND collision terminate; lane departure NO LONGER does.
#                 COLLISION_TERMINAL=1        (v6n: 0 -- v6n was arrival-only)
#                 GATED_GOAL_LANE_TERMINAL=0  (run_gated.sh default: 1)
#                 WEDGE_TRUNCATE=0            (inert either way: LANE_WEDGE defaults off and is exported nowhere,
#                                              so info["wedged"] is never set. Pinned for the record.)
#   explore     off-road ENDS the episode.
#                 EXPLORE_OFF_ROAD_FATAL=1    (v6n: 0)
#   penalties   every driving fault costs 5.0, in both modes (user: "should be 5 same as COLLISION and OFFROAD").
#                 GOAL_LANE_PENALTY=5.0       (v6n: 1.0)
#                 GOAL_OFF_ROAD_PENALTY=5.0   (v6n: 1.0)
#                 EXPLORE_OFF_ROAD_PENALTY=5.0(v6n: 1.0)
#                 LANE_COLLISION_PENALTY=5.0  (default 3.0)
#                 LANE_DEPARTURE_PENALTY      already 5.0, left alone
#   sampling    junction-to-junction STRAIGHT segments are sampled as well as through-junction routes.
#                 STRAIGHT_GOAL_FRAC=0.3 / JUNCTION_GOAL_FRAC=0.5 / 0.2 uniform
#
# KNOWN CONSEQUENCE, ON RECORD. With per-step fines at 5.0, -V is no longer time-to-goal: it is steps PLUS fines.
# gated_reward.py's own comment warns this "would make V mean 'steps to goal PLUS a fine' and corrupt every graph
# built from it". Every SGM cutoff (the ~76 calibration) and the value-vs-distance ordering must be recalibrated
# against this scale before any map built from this policy is compared to an older one.
#
# STEPS: 450000, matching train_v6n.sh / train_v6m_ext.sh ("extended with v6m so the single-change pair stays
# comparable"). v6m reached it; v6n died at 254876, which is why checkpoint_250000 is the one in use -- a 250k
# checkpoint appears here too, so both comparisons stay available.
#
# INHERITED, STATED PLAINLY: DISCOUNT=0.99 with GATED_ALLOW_DISCOUNTED_SORB=1, so the goal critic's Q is
# an ORDERING, not steps-to-goal; ACTOR_Q_NORM=1; HER on; EXPERT_PROB=0.4; PERCEPTION=vc1.
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
export OBS_WITH_VECTOR=0
export OBS_WITH_STATE=1 OBS_STATE_COLLISION=0 OBS_LATENT_AS_PIXELS=1
# THE CONV COMPRESSOR: d4pg conv encoder 64,128,512 -> 512-d latent, 3x3x1
# filters at strides 2,2,1; patch-token augmentation off; fp16 pixel buffer.
export AGENT_ENCODER=d4pg CNN_FEATURES=64,128,512 CNN_FILTERS=3,3,1
export CNN_STRIDES=2,2,1 AGENT_LATENT_DIM=512 AGENT_AUGMENT_PIXELS=0
export BUFFER_PIXELS_FP16=1 VC1_READOUT=grid2
# BUFFER CAPS AT 300k TRANSITIONS (user 2026-09-21; MAX_STEPS stays 450k): after 300k the FIFO wrap keeps the
# most recent ~300k rows; fp16 ~45 GB incl. next_obs mirror (fp32 ~90 GB).
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
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia
# Overridable so a SMOKE RUN uses this identical flag set rather than a copy of it: a duplicated flag list is a list
# that drifts, and the whole point of the smoke is to exercise what the real run will exercise.
#   smoke:  RUN_NAME=carla_gated_her_v8_smoke MAX_STEPS=2000 ./train_v8.sh
RUN_NAME="${RUN_NAME:-carla_gated_her_v8_vc1}"
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
# (rate=0.0) and the explore eval is skipped. Checkpoints are still written every EVAL_INTERVAL=25000.
# EVALUATE CHECKPOINTS OFFLINE, on a separate server, after the run.
export N_EVAL_EPISODES=0 N_EVAL_EXPLORE=0

# ---- TERMINATION (user 2026-09-16) ------------------------------------------
export COLLISION_TERMINAL=1 GATED_GOAL_LANE_TERMINAL=0 WEDGE_TRUNCATE=0
export EXPLORE_OFF_ROAD_FATAL=1

# ---- PENALTIES: 5.0 for every driving fault, both modes ----------------------
export GOAL_LANE_PENALTY=5.0 GOAL_OFF_ROAD_PENALTY=5.0
export EXPLORE_OFF_ROAD_PENALTY=5.0 LANE_COLLISION_PENALTY=5.0

# ---- GOAL SAMPLING ----------------------------------------------------------
export JUNCTION_GOAL_FRAC=0.5 STRAIGHT_GOAL_FRAC=0.3

# ---- TOWNS: Town01 + Town10HD, SIMULTANEOUSLY, one env each (user 2026-09-16) ----
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
# ONE ENV (user 2026-09-21): two windowed CARLA servers peak at ~11.9 GB of a
# 12.3 GB card, leaving no room for torch VC-1 + jax on GPU (measured deaths:
# cudnn conv probe, then jax CUDA init). The single server leaves ~6 GB. The
# trainer's town rotation cycles the one env through the REQUESTED towns, so
# both towns still train.
export CARLA_PORTS=2000 CARLA_PORT=2000
export ENV_TOWNS=Town01,Town10HD
# ROTATION ON, every 20 episodes (the trainer's designed default). v7 disabled
# it because 2 parallel envs were already town-pinned; with 1 env the rotation
# IS the second town. Rebuilds reload the world every switch (~15s, 22 times
# over 450k).
export TOWN_ROTATE_EPISODES=20
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
env | grep -E "^(GATED|OBS_WITH|OBS_STATE|OBS_LATENT|PERCEPTION|STEP_COST|DISCOUNT|EXPLORE|GOAL|LANE|COLLISION|WEDGE|POTENTIAL|TARGET_SPEED|MAX_THROTTLE|MAX_STEER|MIN_THROTTLE|ENV_|EXPERT|N_EVAL|JUNCTION|STRAIGHT|EVAL_TOWNS|CARLA|TOWN|RUN_GATED|ACTOR|AGENT|CNN|BUFFER|VC1|REPLAY)[A-Z0-9_]*=" \
  | sort > "results/run_env/${RUN_NAME}.env"
# ARTEFACT CHECK ON THE SNAPSHOT ITSELF: every flag this run is ABOUT must be in the record, or the record is a lie
# and any later eval replaying it silently differs from training. Abort rather than train against a bad provenance.
_missing=""
for _k in COLLISION_TERMINAL GATED_GOAL_LANE_TERMINAL WEDGE_TRUNCATE EXPLORE_OFF_ROAD_FATAL \
          GOAL_LANE_PENALTY GOAL_OFF_ROAD_PENALTY EXPLORE_OFF_ROAD_PENALTY LANE_COLLISION_PENALTY \
          JUNCTION_GOAL_FRAC STRAIGHT_GOAL_FRAC ENV_TOWNS EVAL_TOWNS \
          CARLA_QUALITY CARLA_ROOT CARLA_OFFSCREEN CARLA_PORTS TOWN_LIST TOWN_ROTATE_EPISODES RUN_GATED_START_SERVER N_EVAL_EPISODES N_EVAL_EXPLORE ACTOR_FREEZE_STEPS PERCEPTION ACTOR_Q_NORM \
          OBS_WITH_VECTOR OBS_WITH_STATE OBS_STATE_COLLISION OBS_LATENT_AS_PIXELS \
          AGENT_ENCODER CNN_FEATURES CNN_FILTERS CNN_STRIDES AGENT_LATENT_DIM AGENT_AUGMENT_PIXELS \
          BUFFER_PIXELS_FP16 VC1_READOUT REPLAY_CAPACITY; do
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
