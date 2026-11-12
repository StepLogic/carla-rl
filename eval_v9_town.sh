#!/usr/bin/env bash
# OFFLINE GENERALIZATION EVAL of the final v9 checkpoint (user 2026-09-21:
# "Evaluate final checkpoint in town02 and town03 - I want to check
# generalization"; "run the evals once training finishes" — confirmed go).
#
# ONE TOWN PER INVOCATION, ONE SERVER PER INVOCATION: eval_arm.py's make_env
# -> ensure_carla attaches to whatever already answers on --port, so a second
# town on a live server would be a load_world mid-sweep -- the known crash
# vector. The caller kills CARLA between towns; this script refuses to run
# while the port lock is held so it can never share the training stack's port.
#
# EVERY experiment-defining flag is replayed from the v9 training snapshot
# (results/run_env/carla_gated_her_v9_vc1.env) so the observation stack
# (PERCEPTION=vc1, VC1_READOUT=pyramid, 3854-d obs) and the reward regime
# (counting regime, +10 arrival, 20-fault-step truncates, 10.0 fines) match
# training exactly; the checkpoint's embedded identity record refuses any
# mismatch before driving. GOAL_DIST_MIN/MAX and MIN_THROTTLE are deliberately
# NOT set: the snapshot carries neither, so eval samples the same code
# defaults training sampled (5-20 m band; code-default throttle floor).
#
# FRAME QUALITY: EPIC for BOTH towns (snapshot CARLA_QUALITY=Epic) -- it
# matches the 310k-step Town05-Epic training distribution, and Town02 at Low
# segfaulted on this CARLA build on 2026-09-21. Town02 (train-split, but
# untrained in v9's Town05-only stint) and Town03 (town_split_v7.json's sole
# HELD-OUT town) therefore differ ONLY in the map, so the Town02-vs-Town03
# comparison stays attributable.
#
# Server self-starts via ensure_carla (windowed, -prefernvidia, RPC-polled),
# logs to logs/carla_py_2000_<ts>.log.
set -euo pipefail
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl

TOWN="${1:?usage: eval_v9_town.sh Town0X}"
CK="checkpoints/carla_gated_her_v9_vc1/checkpoint_450000.pkl"
[ -f "$CK" ] || { echo "FATAL: no $CK"; exit 1; }

# Port lock, held for the whole eval (same lock the trainers use).
LOCK=/tmp/carla_rl_port2000.lock
exec 200>"$LOCK"
flock -n 200 || { echo "REFUSING TO START: another CARLA job holds $LOCK." >&2; exit 1; }
echo "[Lock] acquired $LOCK (pid $$)"

# GPU gate: server (Epic ~3.6-5.9 GB) + eval torch VC-1 + jax.
NEED_MB="${NEED_MB:-7000}"
MAX_WAIT_H="${MAX_WAIT_H:-2}"
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "${free:-0}" -ge "$NEED_MB" ] && { echo "[GPU] ${free} MiB free -- starting"; break; }
  [ "$(date +%s)" -ge "$deadline" ] && { echo "[GPU] gave up; ${free} MiB free." >&2; exit 1; }
  echo "[GPU] ${free} MiB free, need ${NEED_MB}; waiting"; sleep 60
done

source /home/robotlab/anaconda3/etc/profile.d/conda.sh
conda activate carla-rl || { echo "FATAL: no carla-rl env"; exit 1; }
# THE INTERPRETER IS THE IG-VENV, NOT BARE CONDA. Training was LAUNCHED with
# the venv's bin prefixed onto PATH (the train_v9.sh launch recipe), and only
# that interpreter has hydra + vc_models importable for PERCEPTION=vc1.
# Measured 2026-09-21: conda carla-rl alone dies in latent_from_env with
# "No module named 'hydra'" -- the exact failure this block prevents.
V=$PWD/dependencies/ig-venv
export PATH="$V/bin:$PATH"
export VIRTUAL_ENV="$V"
hash -r
python -c "import hydra, vc_models, torch" \
  || { echo "FATAL: this python lacks hydra/vc_models/torch for PERCEPTION=vc1"; exit 1; }
set -a
# shellcheck disable=SC1091
source results/run_env/carla_gated_her_v9_vc1.env
set +a
# THE GATE LAYER'S OWN FLAGS, ABSENT FROM THE SNAPSHOT: run_gated.sh exports
# GATED=1 / OBS_WITH_MODE=1 (:81-83) only AFTER train_v9.sh has snapshotted,
# so the snapshot silently carries neither. Measured 2026-09-21: replaying the
# bare snapshot built a 3853-d obs (no 'mode' key) and the checkpoint loader
# REFUSED it ("checkpoint 'actor' does not match the agent built here") --
# exactly what the loader is for. Training's own banner is the evidence the
# mode key was present: "[VAEGoal] obs={goal 4, latent 3840, mode 1, state 9)}
# = 3854-d [GATED: goal block zeroed in explore mode]".
export GATED=1 OBS_WITH_MODE=1 OBS_WITH_GOAL=1
# IMAGE SIZE PINNED TO TRAINING (found 2026-09-26). run_gated.sh exports
# ENV_IMAGE_SIZE=120 AFTER train_v10.sh snapshots, so the .env this script
# replays never carries it and the env fell back to its default of 84. Every
# earlier eval from this script ran the policy on 84x84 frames it never saw.
export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
# THE ACTION BOX, ALSO MISSING FROM THE SNAPSHOT: run_gated.sh:264-265 exports
# INCREMENTAL_ACTIONS=0 MAX_STEER=0.5 MAX_THROTTLE=0.4 AFTER the snapshot.
# Measured 2026-09-21: replaying the bare snapshot left the code defaults
# (absolute mode MAX_STEER=1.0 MAX_THROTTLE=1.0), so the policy's trained
# action values were interpreted on a 2x-steer / 2.5x-throttle box -- arrivals
# in ~51 steps where training took ~170 (the policy was driving ~55 km/h
# instead of ~22). The action space itself is Box([-max_steer,0],
# [max_steer,max_throttle]), so this is an OOD control mapping, not a nuance.
# TARGET_SPEED=20.0 and INCREMENTAL_ACTIONS=0 match their code defaults and
# are exported anyway so provenance does not lean on luck.
export INCREMENTAL_ACTIONS=0 MAX_STEER=0.5 MAX_THROTTLE=0.4 TARGET_SPEED=20.0
export ENV_TOWNS="$TOWN" EVAL_TOWNS="$TOWN"   # provenance; --towns is what eval_arm reads
export CARLA_PORT=2000 CARLA_PORTS=2000
export DISPLAY=:1

mkdir -p results/v9_generalization logs
# OPTIONAL GOAL BAND (second pass, 2026-09-21): the curriculum had advanced to
# 20-50 m by the end of training (log: "[Resume] curriculum restored to
# 20.0-50.0 m"), so the training-end distribution is 20-50 m and the code
# default 5-20 m (first pass) is the curriculum's START band. GOAL_MIN/GOAL_MAX
# select the band and SUFFIX separates the out/log names; unset = default band.
BAND_ARGS=()
if [ -n "${GOAL_MIN:-}" ] && [ -n "${GOAL_MAX:-}" ]; then
  BAND_ARGS=(--goal-min "$GOAL_MIN" --goal-max "$GOAL_MAX")
fi
SUFFIX="${SUFFIX:-}"
echo "=== [v9-eval $(date +%H:%M:%S)] $TOWN: server self-starts via ensure_carla (Epic, windowed) ==="
python eval_arm.py --arm gated_v9 --ckpt "$CK" \
  --episodes 60 --towns "$TOWN" --seed 0 --max-steps 1000 --goal-conditioned 1 \
  "${BAND_ARGS[@]+"${BAND_ARGS[@]}"}" \
  --out "results/v9_generalization/$(echo "$TOWN" | tr '[:upper:]' '[:lower:]')${SUFFIX}.json" \
  --port 2000 2>&1 | tee "logs/v9_eval_${TOWN}${SUFFIX}.log"
echo "=== [v9-eval $(date +%H:%M:%S)] $TOWN${SUFFIX} done ==="