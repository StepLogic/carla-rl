#!/bin/bash
#SBATCH --job-name=frl_t05t01_2m_l40
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=8-00:00:00
#SBATCH --output=/home/egyaase/carla-rl-frl/logs/%x_%j.log
# MOVE A LEARNER-C RUN TO AN L40 WHEN ONE FREES (user 2026-11-11: "if an l40 become available - lets switch to
# it"). Queued on an L40; it does nothing until Slurm starts it. Learner C cannot resume a run, so the L40 run
# starts over -- which only pays while the A100 run is below BREAK_EVEN steps (A100 ~4.6 steps/s, L40 ~12:
# finishing 2M on the A100 from 1.2M takes as long as 2M on the L40). On start:
#   - the A100 run finished ([Done] in its log)      -> exit, nothing to do;
#   - the A100 run is RUNNING at or past BREAK_EVEN  -> exit, it stays where it is;
#   - otherwise                                      -> scancel the A100 job and run the same run here, from step 0,
#     under its own RUN name, with slurm/frl_towns_job.sbatch (evals in the job).
#   A100_JOB=<id> A100_LOG=<its log> REPO=... RUN=... OVERLAY=... EVAL_STEPS=... [BREAK_EVEN=1200000] \
#     sbatch --export=ALL $REPO/slurm/frl_switch_to_l40.sbatch
set -u
: "${A100_JOB:?}" "${A100_LOG:?}" "${REPO:?}" "${RUN:?}"
BREAK_EVEN=${BREAK_EVEN:-1200000}
last=$(grep -oE "\[FRL\] step=[0-9]+" "$A100_LOG" 2>/dev/null | tail -1 | cut -d= -f2)
last=${last:-0}
state=$(squeue -j "$A100_JOB" -h -o %T 2>/dev/null)
echo "=== $(date +%T) L40 on $(hostname): A100 job $A100_JOB state=${state:-not-queued} at step $last (break-even $BREAK_EVEN)"
if grep -q "^\[Done\]" "$A100_LOG" 2>/dev/null; then
  echo "=== the A100 run finished; nothing to move"; exit 0
fi
if [ "$state" = RUNNING ] && [ "$last" -ge "$BREAK_EVEN" ]; then
  echo "=== the A100 run is past the break-even; it stays on the A100 and this job exits"; exit 0
fi
if [ -n "$state" ]; then
  scancel "$A100_JOB" && echo "=== $(date +%T) cancelled A100 job $A100_JOB at step $last"
  sleep 30
fi
echo "=== $(date +%T) running $RUN here from step 0"
exec bash "$REPO/slurm/frl_towns_job.sbatch"
