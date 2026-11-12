#!/bin/bash
# TRAINING v11 (2026-09-22, user: "launch v11 with N_STEP=3"). THE v10 RECIPE
# PLUS N-STEP RETURNS -- exactly one flag differs from train_v10.sh:
#
#   N-STEP RETURNS (N_STEP=3): the replay buffer merges each sampled row with
#   its up-to-2 same-SEGMENT successors -- one segment = a maximal run of rows
#   sharing a segment id, cut at episode end, leg end (arrival), and each
#   hindsight-goal change (the HER assigns ids per relabelled MDP, env index
#   in the high bits so two envs never alias). Merged rewards are the
#   discounted sum sum_k gamma^k r_{t+k}, next_observations is s_{t+n'}, and
#   the learner bootstraps with gamma**n' per row. THE TD FIXED POINT IS
#   UNCHANGED (the gamma-discounted return); what changes is the PATH: the
#   +10 arrival bonus propagates 3 steps per update instead of one and TQC's
#   per-step pessimism compounds 3x fewer times. It does NOT fix the goal
#   critic's far-field saturation under gamma=0.99 -- the user chose it
#   knowing that. Two documented simplifications, both deliberate:
#     - the entropy correction -alpha*log pi is applied at the BOOTSTRAP
#       state only, not per intermediate step (the buffer cannot store
#       per-step next_log_probs; measured bias ~0.01/step vs -1/step reward);
#     - N_STEP=1 stays byte-identical to every earlier run (no n_steps field,
#       raw rewards, the plain jaxrl2 sample path), and N_STEP>1 is REFUSED
#       without EPISODE_HER=1 (segments are a HER-pipeline feature).
#   Implementation: mode_split_buffer.py (pointer+generation chaining),
#   episode_her.py (segment ids), gated_tqc_learner.py (per-row bootstrap),
#   commit eed64fd; tests/test_n_step_returns.py 9/9 + neighbor suites 98/98.
#
# EVERYTHING ELSE IS v10's, UNCHANGED (see train_v10.sh's header for the full
# recipe): TARGET-VELOCITY BONUS 0.5 on clean steps mirrored in the HER
# recost, HER_RECOST=1 counting regime (-1/step, 10.0 fines both modes,
# +10 arrival, truncate@20), WARM START from v9mb's checkpoint_450000 with
# ACTOR_FREEZE_STEPS=12500 -- THE SAME WARM-START POINT AS v9mbr AND v10, so
# the three runs differ only in their reward/n-step changes, not in
# provenance. The restored v9mb critic was fitted to a differently-scaled
# reward ([[warm-start-needs-an-actor-freeze]]); the freeze lets the fresh
# buffer re-fit the critics before the actor moves.
#
# MIXED BANDS 5-200 m, single env Town05 Epic windowed, REPLAY 300k,
# MAX_STEPS 450k, checkpoints every 5k, in-training eval OFF.
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl || exit 1
source /home/robotlab/anaconda3/etc/profile.d/conda.sh
conda activate carla-rl || { echo "FATAL: no carla-rl env"; exit 1; }
# THE INTERPRETER IS THE IG-VENV, PREFIXED ONTO PATH -- the same recipe the
# v9 eval driver (eval_v9_town.sh) and train_v10.sh use. Bare conda carla-rl
# dies at startup missing hydra/vc_models/torch for PERCEPTION=vc1.
V=$PWD/dependencies/ig-venv
export PATH="$V/bin:$PATH"
export VIRTUAL_ENV="$V"
hash -r
python -c "import hydra, vc_models, torch" \
  || { echo "FATAL: this python lacks hydra/vc_models/torch for PERCEPTION=vc1"; exit 1; }
export ENV_SKIP_IMAGE_FEATURES=1
export PERCEPTION=vc1 ACTOR_Q_NORM=1
# THE COHERENCE RULING'S FLAG PAIR, v8 LAYOUT: {goal, pixels, mode, state},
# no vector key; 9-d state block without collision.
export OBS_WITH_VECTOR=0
export OBS_WITH_STATE=1 OBS_STATE_COLLISION=0 OBS_LATENT_AS_PIXELS=0
export VC1_READOUT=pyramid AGENT_ENCODER=none AGENT_LATENT_DIM=512
export AGENT_AUGMENT_PIXELS=0
export REPLAY_CAPACITY="${REPLAY_CAPACITY:-300000}"
export CARLA_ROOT=/home/robotlab/Apps/CARLA_0.9.15
# EPIC FOR TOWN05 (Low segfaults Town05 3/3 on this build).
export CARLA_QUALITY="${CARLA_QUALITY:-Epic}"
export CARLA_CLIENT_TIMEOUT=180
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia
#   smoke:  RUN_NAME=carla_gated_her_v11_smoke MAX_STEPS=2000 ./train_v11.sh
RUN_NAME="${RUN_NAME:-carla_gated_her_v11_vc1_nstep3}"
export MAX_STEPS="${MAX_STEPS:-450000}"
export ACTOR_FREEZE_STEPS="${ACTOR_FREEZE_STEPS:-12500}"
export EXPLORE_GOAL_DIST_M=100 EXPLORE_ARRIVAL_BONUS=2.0
export DISCOUNT=0.99 GATED_ALLOW_DISCOUNTED_SORB=1
export N_EVAL_EPISODES=0 N_EVAL_EXPLORE=0
# ---- v10 GOAL-MODE REWARD, unchanged -----------------------------------------
export GOAL_TRUNCATE_COUNTS=1 GOAL_ARRIVAL_BONUS=10.0
export GOAL_OFFROAD_TRUNCATE_STEPS=20 GOAL_COLLISION_TRUNCATE_STEPS=20
export GOAL_CROSSING_TRUNCATE_STEPS=20
export GOAL_LANE_PENALTY=10.0 GOAL_OFF_ROAD_PENALTY=10.0 GOAL_COLLISION_PENALTY=10.0
export TQC_VALUE_FLOOR=-3500
export HER_RECOST=1
export GOAL_SPEED_BONUS=0.5
export COLLISION_TERMINAL=0 GATED_GOAL_LANE_TERMINAL=0 WEDGE_TRUNCATE=0
# ---- THE ONE NEW FLAG: N-STEP RETURNS ----------------------------------------
# N_STEP=3 (user 2026-09-22). Exported here (not only on the run line) so the
# .env snapshot records it and any later eval replaying this snapshot knows
# the buffer semantics. train_carla_tqc.py refuses N_STEP>1 without
# EPISODE_HER=1, which run_gated_her.sh sets; the merged-reward gamma frame
# comes from the AGENT's discounts (DISCOUNT=0.99 for both critics), so the
# buffer cannot drift from the learner's bootstrap.
export N_STEP="${N_STEP:-3}"
# ---- EXPLORE mode ------------------------------------------------------------
export EXPLORE_OFF_ROAD_FATAL=1 EXPLORE_OFF_ROAD_PENALTY=10.0
export LANE_COLLISION_PENALTY=10.0 LANE_DEPARTURE_PENALTY=10.0
# ---- GOAL SAMPLING -----------------------------------------------------------
export JUNCTION_GOAL_FRAC=0.5 STRAIGHT_GOAL_FRAC=0.3
export GOAL_DIST_MIN=5 GOAL_DIST_MAX=200
export GOAL_DIST_START=200 GOAL_DIST_TARGET=200
export MAX_DISTANCE_M=500
export GATED_GOAL_MAX_STEPS=1000
# WARM START: v9mb's final checkpoint, THE SAME POINT v9mbr AND v10 started
# from (INIT_FROM fires only at step == 0; on later attempts this run's own
# checkpoints exist and RESUME wins).
export INIT_FROM=checkpoints/carla_gated_her_v9mb_vc1/checkpoint_450000.pkl
# ---- TOWN: Town05 ONLY --------------------------------------------------------
export TOWN_LIST=configs/town_split_v7.json
export ENV_TOWNS=Town05
export TOWN_ROTATE_EPISODES=0
export EVAL_INTERVAL=5000
export CARLA_PORTS=2000 CARLA_PORT=2000
export EVAL_TOWNS=Town05
export CARLA_OFFSCREEN=0
export RUN_GATED_START_SERVER=0

say(){ echo "=== [v11 $(date +%H:%M:%S)] $* ==="; }
kill_all_carla(){ for _ in 1 2 3 4 5 6; do pkill -9 -f 'CarlaUE4-Linux-Shipping' 2>/dev/null; sleep 2; ps -eo args | grep -qi '[C]arlaUE4' || return 0; done; return 1; }
mkdir -p results/run_env logs
# N_STEP ADDED to both the snapshot grep and the required-vars check: it is
# experiment-defining (it changes what a replayed .env's buffer does), so it
# must be in the record -- the v7/v8 lesson that a dropped flag reverts an
# eval replaying the snapshot to its default.
env | grep -E "^(GATED|OBS_WITH|OBS_STATE|OBS_LATENT|PERCEPTION|STEP_COST|DISCOUNT|EXPLORE|GOAL|LANE|COLLISION|WEDGE|POTENTIAL|TARGET_SPEED|MAX_|MIN_THROTTLE|ENV_|EXPERT|N_EVAL|N_STEP|JUNCTION|STRAIGHT|EVAL|CARLA|TOWN|RUN_GATED|ACTOR|AGENT|CNN|BUFFER|VC1|REPLAY|INIT_|TQC_|HER_)[A-Z0-9_]*=" \
  | sort > "results/run_env/${RUN_NAME}.env"
_missing=""
for _k in COLLISION_TERMINAL GATED_GOAL_LANE_TERMINAL WEDGE_TRUNCATE EXPLORE_OFF_ROAD_FATAL \
          GOAL_TRUNCATE_COUNTS GOAL_ARRIVAL_BONUS GOAL_LANE_PENALTY GOAL_OFF_ROAD_PENALTY \
          GOAL_COLLISION_PENALTY GOAL_OFFROAD_TRUNCATE_STEPS GOAL_COLLISION_TRUNCATE_STEPS \
          GOAL_CROSSING_TRUNCATE_STEPS EXPLORE_OFF_ROAD_PENALTY LANE_COLLISION_PENALTY \
          JUNCTION_GOAL_FRAC STRAIGHT_GOAL_FRAC ENV_TOWNS EVAL_TOWNS \
          GOAL_DIST_MIN GOAL_DIST_MAX GOAL_DIST_START GOAL_DIST_TARGET \
          GATED_GOAL_MAX_STEPS TQC_VALUE_FLOOR HER_RECOST GOAL_SPEED_BONUS N_STEP \
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
say "settings snapshotted to results/run_env/${RUN_NAME}.env ($(wc -l < "results/run_env/${RUN_NAME}.env") vars, all v8 flags present + N_STEP)"
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
  if [ "$s2" -le "$s" ] && [ "$ATTEMPT" -gt 3 ]; then
    say "NO PROGRESS over the last attempt (still $s2); stopping so it is not an infinite restart loop"; exit 1
  fi
  kill_all_carla; sleep 30
done
say "done; newest checkpoint $(step) of ${MAX_STEPS}"