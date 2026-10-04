#!/bin/bash
# As run_drive_plan_c_pinned.sh, but it only ever kills the CARLA it owns (port 2000).
# the last commit before this sweep's first row was driven (user 2026-09-20).
#
# WHY: a peer session develops the image-goal arm in this same worktree and committed 53b285a at 06:35,
# which shrinks the goal block from 12-d to 4-d UNCONDITIONALLY ("agents built before this change cannot
# load checkpoints trained after it" -- its own words). v6n's checkpoint expects 12-d, so every evaluation
# of the paper's rows crashes against post-53b285a code. Pinning keeps the evaluated code equal to the code
# the existing rows were produced with; it does not touch the peer's files.
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl
OUT="$1"; LOG="$2"; shift 2
cd "$W" || exit 1
source /home/robotlab/anaconda3/etc/profile.d/conda.sh
conda activate carla-rl || { echo FATAL; exit 1; }
set +u; eval "$(grep -E '^export ' run_gated.sh)"
while IFS='=' read -r k v; do case "$k" in ''|'#'*) continue;; esac; export "$k=$v"; done < results/run_env/carla_gated_her_v6n.env
export CARLA_START_SERVER=1 CARLA_QUALITY=Epic CARLA_PORT=2000 CARLA_PORTS=2000
export EXPERT_PROB=0 RESUME=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export ENV_SKIP_IMAGE_FEATURES=1 PYTHONPATH="$W/compat_v6n/src"
# drive_plan_c.py inserts ITS OWN directory's src at sys.path[0], so PYTHONPATH alone is ignored.
# Running the symlinked copy inside compat_v6n makes that insert point at the PINNED src, while the
# working directory stays here so every results/ path resolves as usual.
export NO_TIMELIMIT=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia
say(){ echo "=== [drive-pinned $(date +%H:%M:%S)] $* ==="; }
# PORT-SCOPED KILL (user 2026-09-20). The blanket kill in the other launchers takes down EVERY CARLA on
# the box, including a peer session's -- which is how this run nearly broke the image-goal training a second
# time, after it politely moved itself to port 2010. Only our own port 2000 server is ever killed here.
kill_my_carla() {
  for _ in 1 2 3 4 5 6; do
    pkill -9 -f 'carla-port=2000' 2>/dev/null
    sleep 2
    pgrep -f 'carla-port=2000' >/dev/null || return 0
  done
  say "WARNING: a port-2000 CARLA survived six kill rounds:"
  pgrep -af 'carla-port=2000' | cut -c1-120
  return 1
}
kill_all_carla() { kill_my_carla; }
say "pinned source: compat_v6n/src @163e6f9"
kill_all_carla || exit 1
sleep 20
for ATTEMPT in 1 2 3 4 5 6 7 8; do
  say "attempt $ATTEMPT -> $OUT (log $LOG)"
  python -u compat_v6n/drive_plan_c.py --out "$OUT" "$@" >> "$LOG" 2>&1
  RC=$?
  if tail -3 "$LOG" | grep -aq "^\[drive\] done:\|^\[drive\] 0 episodes to drive"; then
    say "DONE (rc $RC)"; kill_all_carla; exit 0
  fi
  say "attempt $ATTEMPT ended rc $RC without done; killing CARLA and resuming"
  kill_all_carla; sleep 30
done
say "FAILED after 8 attempts"; exit 1
