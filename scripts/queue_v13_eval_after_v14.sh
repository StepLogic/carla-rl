#!/usr/bin/env bash
# Run the 120-px v13 re-eval ALONE on the GPU, as soon as v14 finishes.
# User 2026-09-26: "run v13 with changes locally".
#
# WHY QUEUED, NOT CONCURRENT. Measured 2026-09-26 with v14 training:
#   - windowed Epic eval server: 5.4 GB; free VRAM fell to 793 MiB and v14's
#     server shrank 3480 -> 2704 MiB alongside it;
#   - offscreen Epic eval server: its load peak still took free VRAM from
#     8487 to 1510 MiB, and the watchdog killed it.
# An Epic eval next to an Epic training run does not fit this 12 GB card with a
# safe margin, and dropping the eval to Low quality would feed v13 frames it
# never trained on -- the exact artefact this re-eval exists to remove.
#
# WAITS ON v14's WRAPPER PID, not on a process pattern (a pattern would match
# this script's own command line). The wrapper owns the retry loop, so its exit
# means v14 is finished or has exhausted its attempts -- either way the GPU is
# free.
cd /home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl || exit 1
V14_WRAPPER_PID="${V14_WRAPPER_PID:-3963809}"
say(){ echo "=== [queue $(date '+%m-%d %H:%M:%S')] $* ==="; }

say "waiting for v14 wrapper PID $V14_WRAPPER_PID to exit"
while kill -0 "$V14_WRAPPER_PID" 2>/dev/null; do sleep 60; done
say "v14 wrapper gone; its last word:"
tail -3 logs/train_v14_lat512_wrapper.log
if ls checkpoints/carla_gated_her_v14_lat512_rgb_vc1/checkpoint_450000.pkl >/dev/null 2>&1; then
  say "v14 reached 450000"
else
  say "WARNING: v14 did NOT reach 450000 -- running the v13 eval anyway, it does not depend on v14"
fi
# Nothing of ours should hold the GPU now; give the driver a moment to release.
sleep 30
nvidia-smi --query-gpu=memory.free --format=csv,noheader
bash scripts/eval_v13_sweep_120.sh
say "queue finished rc=$?"
