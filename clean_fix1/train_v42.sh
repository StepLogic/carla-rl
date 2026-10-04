#!/bin/bash
# v18 (2026-09-27): copy of train_v17.sh. ENV_TOWNS and TOWN_ROTATE_EPISODES were
# UNCONDITIONAL exports (Town05, 0); both are now overridable defaults, so
# ENV_TOWNS=Town05,Town01 TOWN_ROTATE_EPISODES=50 actually reaches the trainer.
# One server rotating between towns, NOT two servers: the 2026-09-21 two-server
# Town01+Town05 squeeze OOM'd four times (see below). The artefact check also
# requires GOAL_LANE_GRADED and ENV_WEATHER_RANDOM in the snapshot.
# TRAINING v10 (2026-09-22, user: "add a term to the goal reward that favors
# moving at target velocity in the lane", then "Make it positive" and "make
# it max out at 0.5"). THE v9mb/v9mbr RECIPE PLUS A TARGET-VELOCITY BONUS,
# warm-started from v9mb's final checkpoint:
#
#   TARGET-VELOCITY BONUS (GOAL_SPEED_BONUS=0.5): on a CLEAN step -- on-road,
#   no lane crossing, no collision THIS step -- the goal reward ADDS
#   +0.5 * max(0, 1 - |speed_kmh - target_speed_kmh| / target_speed_kmh):
#   peak 0.5 at the target speed, linear taper to 0 at half/double, floored
#   at 0. A fault step pays its fines and NO bonus. A missing or
#   non-positive target disables the term. Default weight 0.0 is inert (all
#   pre-v10 runs are byte-identical). The HER recost mirrors it from the
#   staged per-step speeds on the SAME clean gate, so hindsight supervision
#   carries the bonus too (HER_RECOST=1, below).
#
#   HER_RECOST=1: the relabelled rows mirror THIS run's counting
#   regime -- -1/step, the 10.0 fines on fault steps, +10 arrival, leg cut
#   at the 20-count thresholds, and now the velocity bonus on clean steps.
#
#   WARM START: INIT_FROM loads v9mb's checkpoint_450000 weights into a
#   FRESH folder (step counter, replay buffer, optimiser AND curriculum
#   start fresh; the fresh folder makes RESUME find nothing). ACTOR_FREEZE_
#   STEPS=12500 (learner updates; 1 env -> 12.5k env steps) while the fresh
#   buffer refits the critics: v9mb's goal critic saw NO bonus and 100.0
#   fines, this run trains under the bonus and 10.0 -- a reward-scale
#   change, the exact shape of [[warm-start-needs-an-actor-freeze]].
#   (v9mbr -- the same recipe with HER_RECOST but NO bonus -- converged to
#   an agent-only goal SR plateau ~0.19-0.24 vs v9mb's ~0.30 and was killed
#   at ~290k when the user directed the bonus; its checkpoints are archived
#   at checkpoints/carla_gated_her_v9mbr_vc1_partial_290k/.)
#
#   MIXED BANDS (v9mb, unchanged): uniform over 5-200 m via the pinned
#   curriculum (GOAL_DIST_MIN=5 GOAL_DIST_MAX=200,
#   GOAL_DIST_START=GOAL_DIST_TARGET=200), promote/stall/demote all inert,
#   band cannot narrow mid-run. MAX_DISTANCE_M=500 (cap must exceed the
#   200 m goal ceiling; it is a distance-TRAVELLED terminator).
#
#   COUNTING REGIME (GOAL_TRUNCATE_COUNTS=1): no terminal bills. Arrival
#   (+10.0) is the ONLY terminus; every other end (collision, lane crossing,
#   off-road, clock) is a TRUNCATION -- truncated=True, mask stays 1.0, the
#   critic bootstraps through the cut (user: "truncate not terminate"; "all
#   events except goal reached truncates"). Truncation thresholds are
#   cumulative STEPS within the episode, 20 for all three faults (user:
#   "make it 20"). The truncating step pays its normal per-step reward; the
#   bootstrap carries the cost. Per-step fines 10.0 for all three faults
#   (user 2026-09-22: "reduce fines to 10"; v9mb ran 100.0). Full details:
#   gated_reward.py's constants block.
#
# EVERYTHING ELSE IS v9mb's, UNCHANGED: PERCEPTION=vc1 pyramid 3840-d,
# AGENT_ENCODER=none Dense trunk, OBS_WITH_STATE=1 (9-d block, no vector key),
# ACTOR_Q_NORM=1, HER, EXPLORE mode (fines 10.0 -- user 2026-09-22 "use the
# same fine in explore mode" -- off-road fatal, 100 m arrival, bonus 2.0),
# REPLAY_CAPACITY=300k, EVAL_INTERVAL=5k checkpoints, windowed.
# TOWN: Town05 ONLY (v9mb, user 2026-09-21: "use only town05") -- the split's
# highest lane-density map (4.21%) and a TRAIN town in town_split_v7.json.
# Single env, no rotation, no world reloads.
# WARM START: weights from v9mb's final checkpoint_450000 (agent-only goal
# SR 0.30 pooled over the uniform 5-200 m band on Town05); see the header
# block for why INIT_FROM and not RESUME.
# NO v9mb ROW IS COMPARABLE TO A v9mbr ROW: the hindsight reward function
# changed, which reweights ~50% of the goal-side supervision.
cd "$(dirname "$0")" || exit 1   # PINNED SHADOW TREE (clean_fix1): see README
source /home/robotlab/anaconda3/etc/profile.d/conda.sh
conda activate carla-rl || { echo "FATAL: no carla-rl env"; exit 1; }
# THE INTERPRETER IS THE IG-VENV, PREFIXED ONTO PATH -- the same recipe the
# v9 eval driver (eval_v9_town.sh) uses. Measured 2026-09-21 on attempt 1:
# bare conda carla-rl dies at startup with "PERCEPTION=vc1 needs torch,
# hydra-core and vc_models ... No module named 'hydra'" -- v9's wrapper only
# ever worked because its LAUNCHING shell carried the venv PATH; a detached
# launch must not depend on ambient shell state.
V=$PWD/dependencies/ig-venv
export PATH="$V/bin:$PATH"
export VIRTUAL_ENV="$V"
hash -r
# v29: the VC-1 import check applies only when PERCEPTION=vc1
if [ "${PERCEPTION:-vc1}" = "vc1" ]; then
python -c "import hydra, vc_models, torch" \
  || { echo "FATAL: this python lacks hydra/vc_models/torch for PERCEPTION=vc1"; exit 1; }
fi
export ENV_SKIP_IMAGE_FEATURES=1
# EXPORTED HERE, not only inline on the run_gated_her.sh call, so the .env snapshot records them. They were missing
# from carla_gated_her_v7.env, and a sweep replaying it fell back to PERCEPTION=vae (refused by the latent guard).
# THE v8 FLAGS ARE EXPORTED FOR THE SAME REASON (the v7 lesson): the compressor
# config and the coherence ruling's state-block flags are experiment-defining,
# so they must be in the snapshot (and in the required-vars check below), not
# only on the run line.
# v29 (user 2026-09-28 "VC1 is overkill"; "run probe and let start RL training"):
# PERCEPTION overridable -- segvae = the frozen seg-VAE latent (SEGVAE_CKPT).
export PERCEPTION="${PERCEPTION:-vc1}" ACTOR_Q_NORM=1
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
export OBS_WITH_STATE=1 OBS_STATE_COLLISION=0 OBS_LATENT_AS_PIXELS="${OBS_LATENT_AS_PIXELS:-0}"
# v28 (user 2026-09-28 "proceed with recommendations"): the perception probe
# found the 7x7 VC-1 patch grid + conv recovers distance from the lane centre at
# R2 0.65 vs 0.41 for the pooled pyramid. These become overridable so v28 can
# feed grid2 through the conv compressor; defaults are v27's.
export VC1_READOUT="${VC1_READOUT:-pyramid}" AGENT_ENCODER="${AGENT_ENCODER:-none}" AGENT_LATENT_DIM="${AGENT_LATENT_DIM:-512}"
export CNN_FEATURES="${CNN_FEATURES:-64,128,512}" CNN_FILTERS="${CNN_FILTERS:-3,3,1}" CNN_STRIDES="${CNN_STRIDES:-2,2,1}"
export BUFFER_PIXELS_FP16="${BUFFER_PIXELS_FP16:-0}"
export AGENT_AUGMENT_PIXELS=0
# BUFFER CAPS AT 300k TRANSITIONS (user 2026-09-21; MAX_STEPS stays 450k): after 300k the FIFO wrap keeps the
# most recent ~300k rows; pyramid latent ~9 GB fp32 incl. next_obs mirror.
export REPLAY_CAPACITY="${REPLAY_CAPACITY:-300000}"
# CARLA_ROOT IS REQUIRED AND HAS NO DEFAULT: train_carla_tqc.py:319 does a bare os.environ["CARLA_ROOT"] and would
# KeyError on the first env build. No script on this path exports it, and it is absent from every run_env snapshot --
# runs have been working only because an interactive shell happened to carry it. A detached run must not depend on
# ambient shell state.
export CARLA_ROOT=/home/robotlab/Apps/CARLA_0.9.15
# QUALITY: EPIC FOR TOWN05 (user 2026-09-21, "Town05 at Epic"). Town05 at
# Low SEGFAULTED this CARLA build 3/3 today (Signal 11 mid-run and during
# load_world, zero episodes completed); Town05 has historically trained
# whole runs at Epic (v4/v5 era), and one Epic server (~5.9 GB) + the
# trainer fits the card on its own. Overridable so a different-quality run
# can reuse this script without a copy.
# NOTE ON RECORD: the Town01 v9 stint trained 140k steps on LOW frames, so a
# resumed policy sees Epic frames for Town05 -- a cross-quality confound
# within v9 (the known v6n-Epic/v7-Low kind).
# MEASURED: a Low server holds ~4.85 GB against ~5.9 GB for Epic.
export CARLA_QUALITY="${CARLA_QUALITY:-Epic}"
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
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
# FRACTION CAP DROPPED (user 2026-09-21): with PREALLOCATE=false the cap only
# stops jax from grabbing memory that is actually free -- the 0.15 ceiling
# OOM'd on a 768 KB alloc while the card still had headroom. jax now takes
# what it needs, bounded only by what the two CARLA servers leave.
# History: 0.2 = 2.46 GB OOM'd the two-env resume (RESOURCE_EXHAUSTED) --
# two windowed Low servers (~7.8-8.5 GB combined) + the torch VC-1 encoder
# leave less than the old ceiling.
# ALLOCATORS (two-town squeeze): the card is over budget by <1 GB with VC-1
# ON GPU (user 2026-09-21). platform = cuda_malloc_async, no BFC arena
# overhead; expandable_segments stops torch's caching allocator from
# holding freed blocks. Both are pure memory-management changes -- no
# computation, observation or result changes.
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia
# Overridable so a SMOKE RUN uses this identical flag set rather than a copy of it: a duplicated flag list is a list
# that drifts, and the whole point of the smoke is to exercise what the real run will exercise.
#   smoke:  RUN_NAME=carla_gated_her_v10_smoke MAX_STEPS=2000 ./train_v10.sh
RUN_NAME="${RUN_NAME:-carla_gated_her_v10_vc1}"
export MAX_STEPS="${MAX_STEPS:-450000}"
# RESUME FREEZE. A warm start into a fresh buffer must not update the actor
# before the critics re-fit ([[warm-start-needs-an-actor-freeze]]): v5's
# goal steering collapsed (corr 0.57 -> 0.04 by 50k) when it did. Here the
# distribution ALSO widens (5-200 m goal vectors), so the freeze applies.
# UNITS ARE LEARNER UPDATES, NOT ENV STEPS; single env -> 12500 updates =
# 12.5k env steps (v9's own Town05 resume ran freeze 0, but it stayed on the
# restored 20-50 m band -- same distribution, so the lesson did not bite).
export ACTOR_FREEZE_STEPS="${ACTOR_FREEZE_STEPS:-12500}"
export EXPLORE_GOAL_DIST_M="${EXPLORE_GOAL_DIST_M:-100}" EXPLORE_ARRIVAL_BONUS="${EXPLORE_ARRIVAL_BONUS:-2.0}"  # v37: overridable (was hardcoded 2.0, superseded by the 2026-09-18 decision to raise it to 10)
# v35 (user 2026-09-29 goal freeze "with 0.995"): the goal discount is overridable;
# EXPLORE_DISCOUNT stays 0.99 (run_gated.sh default).
export DISCOUNT="${DISCOUNT:-0.99}" GATED_ALLOW_DISCOUNTED_SORB=1
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
export GOAL_TRUNCATE_COUNTS=1 GOAL_ARRIVAL_BONUS="${GOAL_ARRIVAL_BONUS:-10.0}"
# TRUNCATE THRESHOLDS caller-overridable (user 2026-09-24, "terminate
# collision, lane invasion and offroad and bill 101 per step"): the launch
# sets all three to 1, so the FIRST faulting step of each type ends the
# episode (in goal mode the crossing counter is every lane_any step, so a
# lane invasion IS the counted crossing). Default stays 20 for old recipes.
export GOAL_OFFROAD_TRUNCATE_STEPS="${GOAL_OFFROAD_TRUNCATE_STEPS-20}"
export GOAL_COLLISION_TRUNCATE_STEPS="${GOAL_COLLISION_TRUNCATE_STEPS-20}"
export GOAL_CROSSING_TRUNCATE_STEPS="${GOAL_CROSSING_TRUNCATE_STEPS-20}"
# FAULT FINES 10.0 (user 2026-09-22, "reduce fines to 10"; the run LAUNCHED
# earlier this morning at 100.0 was killed and archived to
# checkpoints/carla_gated_her_v9mbr_vc1_fines100_aborted/). Truncate
# thresholds STAY at 20 steps; only the per-step fine magnitude changes. The
# warm-started goal critic (v9mb checkpoint_450000) was fitted under 100.0
# fines -- a reward-scale change -- so per
# [[warm-start-needs-an-actor-freeze]] the freeze (12500 updates, already
# this wrapper's default) gives it time to re-fit before the actor moves.
export GOAL_LANE_PENALTY="${GOAL_LANE_PENALTY-10.0}"
export GOAL_OFF_ROAD_PENALTY="${GOAL_OFF_ROAD_PENALTY-10.0}"
export GOAL_COLLISION_PENALTY="${GOAL_COLLISION_PENALTY-10.0}"
# THE GOAL-CRITIC TARGET CLIP MUST CONTAIN THE RETURN RANGE. The trainer
# derives its default floor as -step_cost*goal_cap = -1000 and IGNORES fines;
# with 10.0 fines a fault-heavy episode reaches ~-1190 (1000 step costs +
# 19 fault steps x 10), which the default clip would flatten to -1000.
# TQC_VALUE_FLOOR is trainer code's own escape hatch (scoped to the goal
# critic only: it is applied inside the GATED two-critic branch,
# train_carla_tqc.py:260-282). -3500 is KEPT from the 100.0-fine launch: it
# still contains the smaller range with margin, and a floor below the
# reachable range changes nothing (it only clamps what cannot occur).
export TQC_VALUE_FLOOR="${TQC_VALUE_FLOOR--3500}"
# HER_RECOST=1 (user 2026-09-22, "ensure it matches the reward of the current
# goal mode"): hindsight rows are recosted under THIS run's counting regime --
# -1/step, the 10.0 fines on fault steps, +10 arrival, truncation cut at the
# 20-count thresholds. Without it the relabelled rows carried the old min-time
# reward (0 arrival, no fines), which measured ~50% of the goal side of the
# buffer on the completed v9 stint and diluted the fines ~2x. The constants
# are read from gated_reward by the buffer itself, so they cannot drift from
# the wrapper's exports above.
# HER_RECOST=1 (user 2026-09-22, "ensure it matches the reward of the current
# goal mode"): hindsight rows are recosted under THIS run's counting regime --
# -1/step, the 10.0 fines on fault steps, +10 arrival, truncation cut at the
# 20-count thresholds, and the TARGET-VELOCITY BONUS on clean steps (the
# buffer reads GOAL_SPEED_BONUS from gated_reward, the single source of
# truth, so it cannot drift from this export). Without it the relabelled
# rows carried the old min-time reward (0 arrival, no fines), which measured
# ~50% of the goal side of the buffer on the completed v9 stint and diluted
# the fines ~2x.
export HER_RECOST=1
# v18nh (user 2026-09-27: "disable HER for v18"): NO hindsight relabelling.
# run_gated_her.sh hard-sets EPISODE_HER=1, so this calls run_gated.sh, whose
# default is EPISODE_HER=0; exported here so the snapshot records it.
# HER_RECOST is then inert (read only when EPISODE_HER=1).
export EPISODE_HER="${EPISODE_HER:-0}"   # v42: OVERRIDABLE (was unconditional, silently voiding a launcher's HER)
# v18ap (user 2026-09-27): ASYMMETRIC GOAL CRITIC. A privileged goal critic
# sees the 13-d lane block (vae_nav.lane_priv) and drives the actor on goal
# rows, trained on the same minimum-time reward; the image-only goal critic
# trains on the same batches and is the map-edge distance. The actor never
# sees the block. Otherwise identical to v18nh (HER off, graded fines,
# weather, Town05+Town01 rotation).
# v27 (user 2026-09-28 "remove priveledge critic now and restore goal critic"):
# OVERRIDABLE, default OFF -- the image-only goal critic drives the actor again.
export OBS_PRIV="${OBS_PRIV:-0}"
# BEHAVIOUR CLONING on expert-driven rows (user 2026-09-28 "i prefer 2").
export BC_WEIGHT="${BC_WEIGHT:-0}"
# v19 (user 2026-09-28): v18ap + "remove route progress shaping" (POTENTIAL_K=0,
# exported here so the snapshot records it; run_gated.sh now honours it),
# "pay 10 for broken line crossed" (GOAL_LANE_PENALTY_BROKEN=10 at launch) and
# "add 100 for arrival" (GOAL_ARRIVAL_BONUS=100 at launch, TQC_VALUE_CEILING=100
# so arrival targets are not clipped).
export POTENTIAL_K="${POTENTIAL_K:-0}"
# TARGET-VELOCITY BONUS WEIGHT (user 2026-09-22: "Make it positive", "make it
# max out at 0.5"). See the header block for the full design.
# NO-COLON expansion (user 2026-09-22 "redo velocity constant to 0.75"): only
# an UNSET variable takes the 0.5 default -- the colon form would ALSO
# substitute on an empty string and hide the caller's intent.
export GOAL_SPEED_BONUS="${GOAL_SPEED_BONUS-0.5}"
# v16 (2026-09-26): copy of train_v15.sh; the snapshot regex also takes ROUTE_
# and the artefact check requires GATED_GOAL_REWARD and ROUTE_REMAINING_MODE,
# so a progress-reward run cannot record less than it ran.
# GATED_GOAL_LANE_TERMINAL IS OVERRIDABLE HERE (v15, user 2026-09-26: "make
# lane invasion terminal and bill remaining steps in goal mode"). In
# train_v10.sh this line hardcoded 0 with an unconditional export, so a
# launch-time GATED_GOAL_LANE_TERMINAL=1 was silently discarded and the
# snapshot recorded 0. Since commit 948a89d the counting branch honours it
# (it used to be inert here). Default stays 0; COLLISION_TERMINAL and
# WEDGE_TRUNCATE stay pinned as before.
export COLLISION_TERMINAL=0 WEDGE_TRUNCATE=0
export GATED_GOAL_LANE_TERMINAL="${GATED_GOAL_LANE_TERMINAL:-0}"

# ---- EXPLORE mode: SAME FINE (user 2026-09-22, "use the same fine in ------
# explore mode"): per-step fines 10.0 here too. Off-road stays FATAL
# (EXPLORE_OFF_ROAD_FATAL=1) and the 100 m arrival + bonus 2.0 stay.
export EXPLORE_OFF_ROAD_FATAL=1 EXPLORE_OFF_ROAD_PENALTY="${EXPLORE_OFF_ROAD_PENALTY-10.0}"
export LANE_COLLISION_PENALTY="${LANE_COLLISION_PENALTY-10.0}" LANE_DEPARTURE_PENALTY="${LANE_DEPARTURE_PENALTY-10.0}"

# ---- GOAL SAMPLING ----------------------------------------------------------
export JUNCTION_GOAL_FRAC=0.5 STRAIGHT_GOAL_FRAC=0.3
# MIXED BANDS (this run's whole point; user choice 2026-09-21: uniform over
# 5-200 m). reset_hero rebuilds its 2.5 m grid from GOAL_DIST_MIN/MAX at every
# reset, so a flat 5-200 band is enough -- no rlib change. The trainer's
# curriculum is PINNED at the ceiling: CURR_START=CURR_TARGET=200 makes
# promote and stall impossible (they need curr_max < CURR_TARGET) and demote
# impossible (it needs curr_max > CURR_START), verified against
# train_carla_tqc.py:1735/:1745/:1790 -- the band cannot narrow mid-run.
# GOAL_DIST_START/TARGET are exported because the snapshot must record them:
# an eval replaying this .env without them would run the default 5->20 m
# curriculum instead of the fixed 5-200 band.
export GOAL_DIST_MIN="${GOAL_DIST_MIN:-5}" GOAL_DIST_MAX="${GOAL_DIST_MAX:-200}"   # v42: OVERRIDABLE
export GOAL_DIST_START="${GOAL_DIST_START:-200}" GOAL_DIST_TARGET="${GOAL_DIST_TARGET:-200}"   # v42: OVERRIDABLE
# HARD DISTANCE-TRAVELLED CAP must exceed the max goal distance (a 200 m
# straight-line goal needs a longer route than 200 m); the env-config comment
# says so. 500 gives ~2.5x headroom over the 200 m ceiling.
export MAX_DISTANCE_M=500
# GOAL EPISODE CLOCK 1000 (user 2026-09-22: "increase timeout to 1000 steps";
# run_gated.sh's default was 750, now caller-overridable). The goal-critic
# target clip [-750, 0] moves to [-1000, 0] with it -- a widening, so the
# resumed critic can express more-negative values; it is NOT a reward-scale
# change (per-step cost and fines unchanged). Recorded in the snapshot.
export GATED_GOAL_MAX_STEPS="${GATED_GOAL_MAX_STEPS:-1000}"   # v42: OVERRIDABLE
# WARM START FROM v9mb'S FINAL CHECKPOINT (weights only). The fresh folder
# means RESUME finds nothing (step stays 0), so INIT_FROM applies and the
# step counter, buffer, optimiser and CURRICULUM all start fresh -- a plain
# RESUME would restore v9mb's curriculum state on top of an identical band
# (harmless here, but INIT_FROM is the clean warm-start path). On later
# attempts this run's own checkpoints exist, RESUME wins, INIT_FROM is
# skipped automatically (it only fires at step == 0).
# CALLER-OVERRIDABLE (from-scratch 2M rerun, user 2026-09-22 "start from
# scratch"): pass INIT_FROM= (empty) and the trainer's `if init_from and
# step == 0` gate sees the empty string as unset -- no warm start. NOTE THE
# MISSING COLON: "${INIT_FROM-default}" defaults only when UNSET; the
# colon form "${INIT_FROM:-default}" ALSO substitutes on the empty string,
# which would have silently re-armed the warm start on a from-scratch
# launch (caught in the snapshot check before it bit). The default stays
# v9mb's checkpoint for every earlier caller.
export INIT_FROM="${INIT_FROM-checkpoints/carla_gated_her_v9mb_vc1/checkpoint_450000.pkl}"

# ---- TOWN: Town05 ONLY (user 2026-09-21, "use only town05") ----
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
# TOWN05 ONLY (user 2026-09-21: "use only town05", after the Town01+Town05
# two-server squeeze OOM'd four times and Town05's server segfaulted at
# startup). One env, one server, Town05@2000 -- the split's highest
# lane-density map (4.21%), a TRAIN town in town_split_v7.json.
# No rotation: with one requested town the rotation list is length 1 and the
# reload loop can never fire; TOWN_ROTATE_EPISODES=0 makes that a written
# decision.
export ENV_TOWNS="${ENV_TOWNS:-Town05}"
# TEXTURE STREAMING POOL CAP: TRIED AND REVERTED (2026-09-21). The
# -ExecCmds="r.Streaming.PoolSize 2000" flag SEGFAULTED every Town05 world
# load (Signal 11 during load_world, 3/3 servers), while a bare Town05 load
# with the identical command MINUS the flag succeeded in 2 s -- a
# discriminating probe, not a guess. The cap was approved for the
# two-server squeeze, which is abandoned; single-server Town05 does not need
# it. The ensure_carla wiring stays, inert unless CARLA_STREAMING_POOL is
# set.
# TOWN_ROTATE_EPISODES=0: single town, no world reloads by construction.
export TOWN_ROTATE_EPISODES="${TOWN_ROTATE_EPISODES:-0}"
# CHECKPOINT EVERY 5k (was 25k). Pure resilience, not an experiment change:
# N_EVAL_EPISODES=0 already skips the eval, so EVAL_INTERVAL only sets
# checkpoint cadence -- and every death this run has had happened BELOW the
# 25k first checkpoint, i.e. cost the entire attempt. Restarting resumes
# from the newest checkpoint instead.
# v21b (2026-09-28): overridable -- the root disk had 12 GB free and a 1M run
# at 5k checkpoints (78 MB each) would nearly fill it.
export EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
# v21 (user 2026-09-28 "Town01 + Town05 low"): TWO servers, one env per town.
# The trainer makes min(#ports, #towns) envs, so both ports and both towns
# must be given. CARLA binds port, port+1 and port+2: 2000 holds 2000-2002, so the second server is 2010.
export CARLA_PORTS="${CARLA_PORTS:-2000}" CARLA_PORT="${CARLA_PORT:-2000}"
#
# EVAL_TOWNS matches the training towns. In-training eval is OFF
# (N_EVAL_EPISODES=0), so this is provenance only.
export EVAL_TOWNS=Town05
# Headless: only ~12 MiB, but it removes the live-X-session dependency -- windowed, CarlaUE4 EXITS rather than
# falling back if DISPLAY is wrong or the session locks, which an unattended 450k-step run must not depend on.
# WINDOWED (user 2026-09-16: "the ui is not showing"). Needs the live X session on DISPLAY=:1, which run_gated.sh
# exports; if that session locks, CarlaUE4 EXITS rather than falling back to headless. Offscreen saved only ~12 MiB.
export CARLA_OFFSCREEN=0
# Python starts BOTH servers (one per port, at CARLA_QUALITY/CARLA_OFFSCREEN). Without this, run_gated.sh started its
# own port-2000 server at hardcoded Epic/windowed and python attached to it -- Town01 silently ran at Epic, and the
# Epic+Low pair starved the card ("A sensor took too long to send their data").
export RUN_GATED_START_SERVER=0

say(){ echo "=== [v10 $(date +%H:%M:%S)] $* ==="; }
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
env | grep -E "^(GATED|OBS_WITH|OBS_STATE|OBS_LATENT|OBS_PRIV|BC_|SENSOR_|REPLAY_|SEGVAE_|PERCEPTION|STEP_COST|DISCOUNT|EXPLORE|GOAL|LANE|COLLISION|WEDGE|POTENTIAL|TARGET_|ROUTE_|MAX_|MIN_THROTTLE|ENV_|EXPERT|N_EVAL|JUNCTION|STRAIGHT|EVAL|CARLA|TOWN|RUN_GATED|ACTOR|AGENT|CNN|BUFFER|VC1|REPLAY|INIT_|TQC_|HER_|EPISODE_HER)[A-Z0-9_]*=" \
  | sort > "results/run_env/${RUN_NAME}.env"
# ARTEFACT CHECK ON THE SNAPSHOT ITSELF: every flag this run is ABOUT must be in the record, or the record is a lie
# and any later eval replaying it silently differs from training. Abort rather than train against a bad provenance.
_missing=""
for _k in COLLISION_TERMINAL GATED_GOAL_LANE_TERMINAL WEDGE_TRUNCATE EXPLORE_OFF_ROAD_FATAL \
          GOAL_TRUNCATE_COUNTS GOAL_ARRIVAL_BONUS GOAL_LANE_PENALTY GOAL_OFF_ROAD_PENALTY \
          GOAL_COLLISION_PENALTY GOAL_OFFROAD_TRUNCATE_STEPS GOAL_COLLISION_TRUNCATE_STEPS \
          GOAL_CROSSING_TRUNCATE_STEPS EXPLORE_OFF_ROAD_PENALTY LANE_COLLISION_PENALTY \
          JUNCTION_GOAL_FRAC STRAIGHT_GOAL_FRAC ENV_TOWNS EVAL_TOWNS \
          GOAL_DIST_MIN GOAL_DIST_MAX GOAL_DIST_START GOAL_DIST_TARGET \
          GATED_GOAL_MAX_STEPS TQC_VALUE_FLOOR HER_RECOST GOAL_SPEED_BONUS \
          GATED_GOAL_REWARD ROUTE_REMAINING_MODE GOAL_LANE_GRADED ENV_WEATHER_RANDOM OBS_PRIV BC_WEIGHT SENSOR_FRAME_SYNC EPISODE_HER POTENTIAL_K GOAL_LANE_PENALTY_BROKEN TQC_VALUE_CEILING \
          MAX_DISTANCE_M INIT_FROM \
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
  POLICY_FOLDER="$RUN_NAME" PERCEPTION="$PERCEPTION" ACTOR_Q_NORM=1 RESUME=1 \
    ./run_gated.sh >> "logs/${RUN_NAME}.log" 2>&1
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
