#!/usr/bin/env bash
# UNATTENDED CHAIN, written for a break: finish the lane-terminating run, put a
# proper evaluation on its best checkpoint, then retrain the gated policy.
#
#   1. wait for carla_ttg_laneterm to finish
#   2. eval_policy.py on its BEST checkpoint (by recorded eval_success)
#   3. run_gated.sh in the gated-impl worktree (goal + explore modes)
#
# STRICTLY SEQUENTIAL, because only one CARLA fits on this card: a run needs
# ~9.2 GB measured of a 12.3 GB card, so two servers do not coexist. Each stage
# waits for the previous one's processes to be gone and clears any leftover
# server before starting.
#
# THE EVAL MUST INHERIT THE TRAINING ENVIRONMENT. MAX_THROTTLE, the action
# space, the VAE checkpoint and the goal band all change how the policy behaves
# or whether it loads at all; evaluating under different values measures a
# different thing and would do it silently. They are repeated explicitly below
# rather than sourced, so a later edit to run_timetogoal.sh cannot retroactively
# change what this evaluation meant.
set -uo pipefail
cd /home/robotlab/scratch/carla-rl

STAMP=$(date +%Y%m%d_%H%M)
CKPT_DIR=checkpoints/carla_ttg_laneterm
CHAIN_LOG=logs/break_chain_${STAMP}.log
exec > >(tee -a "$CHAIN_LOG") 2>&1

say() { echo "=== [chain $(date +%H:%M:%S)] $* ==="; }

wait_gone() {   # $1 = pgrep pattern, $2 = label
  while pgrep -f "$1" >/dev/null 2>&1; do sleep 60; done
  say "$2 is gone"
}

clear_servers() {
  pkill -9 -f '[C]arlaUE4-Linux-Shipping' 2>/dev/null
  pkill -9 -f '[C]arlaUE4.sh' 2>/dev/null
  sleep 8
}

# ---- 1. wait for training ----------------------------------------------------
say "waiting for the lane-terminating run"
wait_gone "[t]rain_carla_tqc.py" "trainer"
wait_gone "[r]un_timetogoal.sh" "run script"
clear_servers

# ---- 2. evaluate the best checkpoint ----------------------------------------
# BEST BY RECORDED eval_success, not latest. The trainer stamps each checkpoint
# with the eval that produced it, and the last checkpoint is not reliably the
# best -- the 500k run's final two evals were 0.567 then 0.417.
BEST=$(python3 - <<'PY'
import glob, pickle, re
best, best_s = None, -1.0
for f in glob.glob("checkpoints/carla_ttg_laneterm/checkpoint_*.pkl"):
    try:
        with open(f, "rb") as fh:
            s = float(pickle.load(fh).get("eval_success") or -1.0)
    except Exception:
        continue
    step = int(re.search(r"checkpoint_(\d+)", f).group(1))
    # Ties break toward the LATER checkpoint: more training at equal measured
    # success is the better bet, and early ties are usually small-n noise.
    if s > best_s or (s == best_s and best and step > int(re.search(r"checkpoint_(\d+)", best).group(1))):
        best, best_s = f, s
print(best or "")
PY
)
if [ -z "$BEST" ]; then
  say "NO CHECKPOINT FOUND in $CKPT_DIR -- skipping evaluation"
else
  say "evaluating $BEST"
  # Towns: Town01 is what training evaluated on, Town02 is what it trained on,
  # Town03 is GENUINELY held out by the frozen split. All three, so the
  # in-distribution / cross-town / held-out gap is visible in one table.
  env DISPLAY=:1 \
      CARLA_START_SERVER=1 CARLA_ROOT=/home/robotlab/Apps/CARLA_0.9.15 \
      CARLA_QUALITY=Epic \
      ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120 \
      VAE_CKPT=checkpoints/lane_vae_120/best.pkl \
      INCREMENTAL_ACTIONS=0 MAX_STEER=0.5 MAX_THROTTLE=0.4 MIN_THROTTLE=0.05 \
      OBS_WITH_VECTOR=0 OBS_WITH_GOAL=1 \
      TIME_TO_GOAL=1 STEP_COST=1.0 TTG_MAX_STEPS=750 MAX_EPISODE_STEPS=750 \
      LANE_WEDGE=1 LANE_STRICT=0 WEDGE_ON_OFF_ROAD=1 STALL_STEPS=0 \
      COLLISION_TERMINAL=1 ROUTE_POTENTIAL=1 POTENTIAL_K=1.0 \
      GOAL_DIST_MIN=20.0 GOAL_DIST_MAX=50.0 \
      SHORT_HORIZON=1 JUNCTION_GOAL_FRAC=0.7 TARGET_SPEED=20.0 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
      python -u eval_policy.py \
        --ckpt "$BEST" \
        --towns Town01,Town02,Town03 \
        --episodes 25 \
        --out "results/laneterm_eval_${STAMP}.json"
  say "evaluation exit=$? -> results/laneterm_eval_${STAMP}.json"
fi
clear_servers

# ---- 3. retrain the gated policy --------------------------------------------
# IN THE WORKTREE, WITH THE WORKTREE'S OWN CODE. The gated wiring lives only in
# .claude/worktrees/gated-impl (the main checkout's train_carla_tqc.py has no
# gated branch), and its config is a separate copy that today's reward changes
# were NOT applied to. That is deliberate: it reproduces the gated experiment as
# it was rather than silently mixing two designs.
#
# One run trains BOTH modes -- gated_reward.py draws the mode per episode at
# EXPLORE_PROB and keeps one critic per mode.
say "starting the gated retrain"
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl
nohup ./run_gated.sh > "logs/gated_retrain_${STAMP}.log" 2>&1
say "gated retrain exit=$? -> gated-impl/logs/gated_retrain_${STAMP}.log"
say "chain complete"
