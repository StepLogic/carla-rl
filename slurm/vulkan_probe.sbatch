#!/bin/bash
# Why did CARLA render on the CPU? (smoke job 453151, node-g102)
#
# Evidence to explain: nvidia-smi showed NO compute process for CarlaUE4, the
# container log said "lavapipe is not a conformant vulkan implementation" -- that
# is the SOFTWARE rasteriser announcing itself -- and the engine then died with
# "GameThread timed out waiting for RenderThread after 60.00 secs" and signal 11.
# So Vulkan picked lavapipe over the L40 and Epic quality was too much for it.
#
# This contradicts what we recorded in October ("official image + apptainer run
# --nv -e works, holds 1591 MiB"), so the question is which of these differs:
#   (a) the GPU is not reaching the container at all
#   (b) the NVIDIA Vulkan ICD is missing or losing to lavapipe inside it
#   (c) only EPIC is too heavy for the fallback, and Low was what we tested
# Each is checked below, cheaply, in one job.
#
#SBATCH --job-name=vk_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=00:25:00
#SBATCH --output=%x_%j.log
set -uo pipefail
SIF=${CARLA_SIF:-$HOME/carla-cluster/carla0916.sif}
module load apptainer 2>/dev/null

echo "=== node $(hostname), job ${SLURM_JOB_ID:-none} ==="
echo "--- host sees:"; nvidia-smi -L; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "--- (a) does the GPU reach the container?"
apptainer exec --nv -e "$SIF" nvidia-smi -L 2>&1 | head -5

echo "--- (b) which Vulkan ICDs are inside?"
apptainer exec --nv -e "$SIF" sh -lc '
  for d in /etc/vulkan/icd.d /usr/share/vulkan/icd.d; do
    echo "[$d]"; ls -la $d 2>/dev/null || echo "  (absent)";
  done
  echo "[libGLX/libEGL nvidia present?]"; ls /usr/lib/x86_64-linux-gnu/ 2>/dev/null | grep -i nvidia | head -8
  command -v vulkaninfo >/dev/null && vulkaninfo --summary 2>&1 | grep -iE "deviceName|driverName|apiVersion|GPU id" | head -12 || echo "  (no vulkaninfo in image)"
' 2>&1 | head -40

echo "--- (c) start CARLA at LOW and see whether it holds GPU memory"
PORT=$((3000 + RANDOM % 500))
apptainer run --nv -e "$SIF" /workspace/CarlaUE4.sh \
  -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port="$PORT" \
  > vk_probe_low_${SLURM_JOB_ID:-0}.log 2>&1 &
PID=$!
for i in $(seq 1 150); do
  kill -0 $PID 2>/dev/null || { echo "CARLA(Low) exited after ${i}s"; break; }
  ss -ltn 2>/dev/null | grep -q ":$PORT\b" && { echo "CARLA(Low) listening after ${i}s"; break; }
  command sleep 1
done
command sleep 10
echo "--- GPU compute processes now:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null
echo "--- CARLA(Low) log tail:"; tail -12 vk_probe_low_${SLURM_JOB_ID:-0}.log
kill -9 $PID 2>/dev/null
echo "=== done ==="
