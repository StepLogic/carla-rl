#!/usr/bin/env bash
# CPU-only benchmark job (plan docs/superpowers/plans/2026-11-10-learner-validation.md).
# Requests no GPU, so the one-GPU (L40) queue is untouched. VENV picks the interpreter:
#   bench   ~/bench-venv   CleanRL control arm: gymnasium 0.29.1 + numpy 1.26.4 as upstream pins
#   learner ~/learner-venv our learners: carla310's torch 2.5.1 / gymnasium 1.4 / numpy 2.2
# Both inherit carla310's torch; PYTHONNOUSERSITE=1 keeps ~/.local (torch 2.3.1, gymnasium
# 0.29.1) from shadowing it. Everything after the script name is passed through.
#   VENV=bench sbatch -p haswell -c 4 --mem=8G -t 6:00:00 -J t1A \
#     -o logs/bench_t1A_%j.log slurm/bench_cpu.sh scripts/bench/cleanrl_sac_continuous.py \
#     --env-id Pendulum-v1 --total-timesteps 50000 --out-json checkpoints/bench/t1_A.json
set -u
cd ~/carla-rl
case ${VENV:-bench} in
  bench) PY=$HOME/bench-venv/bin/python ;;
  learner) PY=$HOME/learner-venv/bin/python ;;
  *) echo "VENV must be bench or learner, got ${VENV}"; exit 2 ;;
esac
n=${SLURM_CPUS_PER_TASK:-4}
export PYTHONNOUSERSITE=1 MUJOCO_GL=${MUJOCO_GL:-egl} OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n
echo "=== BENCH $(date +%T) node $(hostname) venv ${VENV:-bench} python $PY cpus $n MUJOCO_GL=$MUJOCO_GL"
"$PY" "$@"
rc=$?
echo "=== BENCH rc=$rc $(date +%T)"
exit $rc
