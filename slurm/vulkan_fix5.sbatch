#!/bin/bash
# Fifth attempt, and the first with a root cause rather than a candidate.
#
# ROOT CAUSE (job 453156): the container's Vulkan LOADER is libvulkan.so.1.2.131,
# dated February 2020. The host's driver is 545.23.08 and its ICD manifest
# declares api_version 1.3.260. The loader is five years older than the driver it
# is being asked to drive, which is why:
#   * with no NVIDIA manifest, the only ICD left is lavapipe and everything
#     renders on the CPU (zero GPU memory; Epic times out and segfaults);
#   * with the NVIDIA manifest supplied, lavapipe is correctly excluded and CARLA
#     exits in 3-6 s before printing a single line.
# Both libraries dlopen fine on their own, and every NVIDIA library --nv needs to
# mount is present, so nothing is missing: the loader simply cannot use the ICD.
#
# -opengl IS NOT A WAY OUT: this 0.9.16 build ignores it (0 mentions of opengl in
# the log, lavapipe still chosen), the OpenGL RHI having been dropped from the
# UE4 build CARLA ships.
#
# THE FIX UNDER TEST: bind the HOST's modern Vulkan loader over the image's, the
# way the ICD manifest is bound. One file, one path, and the host's loader is the
# one that matches the host's driver. If this works, the launch line for every
# cluster job is two binds, and nothing in the image needs rebuilding.
#
#SBATCH --job-name=vk_fix5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=00:35:00
#SBATCH --output=%x_%j.log
set -uo pipefail
SIF=${CARLA_SIF:-$HOME/carla-cluster/carla0916.sif}
HOST_ICD=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
module load apptainer 2>/dev/null
echo "=== node $(hostname) job ${SLURM_JOB_ID:-none} ==="
echo "--- the host's Vulkan loader:"
ls -la /usr/lib64/libvulkan.so* 2>/dev/null || echo "  none in /usr/lib64"
HOST_VK=$(readlink -f /usr/lib64/libvulkan.so.1 2>/dev/null || true)
echo "  resolved: ${HOST_VK:-NONE}"
[ -n "$HOST_VK" ] || { echo "FATAL: the host has no libvulkan.so.1 to lend"; exit 1; }

try () {              # $1 label, rest = extra apptainer args
  local label=$1 q=$2; shift 2
  local port=$((8000 + RANDOM % 900))
  local log=vk5_${label}_${SLURM_JOB_ID:-0}.log
  echo "=== [$label] quality=$q port=$port"
  apptainer run --nv -e "$@" "$SIF" /workspace/CarlaUE4.sh \
    -RenderOffScreen -nosound -quality-level="$q" -carla-rpc-port="$port" \
    > "$log" 2>&1 &
  local pid=$! up=0
  for i in $(seq 1 180); do
    kill -0 $pid 2>/dev/null || { echo "[$label] EXITED after ${i}s"; break; }
    ss -ltn 2>/dev/null | grep -q ":$port\b" && { up=$i; break; }
    command sleep 1
  done
  [ "$up" -gt 0 ] && echo "[$label] listening after ${up}s"
  command sleep 12
  local mem
  mem=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head -2 | tr '\n' ';')
  if [ -n "$mem" ]; then echo "[$label] *** ON THE GPU: $mem ***"; else echo "[$label] no GPU memory"; fi
  echo "    lavapipe: $(grep -ciE lavapipe "$log")"
  grep -iE "vulkan|device|L40|error|fatal|signal" "$log" | tail -4 | sed 's/^/    /'
  kill -9 $pid 2>/dev/null; command sleep 5
}

VKBIND=(--bind "$HOST_VK":/usr/lib/x86_64-linux-gnu/libvulkan.so.1
        --bind "$HOST_ICD":/etc/vulkan/icd.d/nvidia_icd.json
        --env VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json)

try loader_low  Low  "${VKBIND[@]}"
try loader_epic Epic "${VKBIND[@]}"
# Control: the same loader bind WITHOUT excluding lavapipe, to see which the
# modern loader picks when both are offered.
try loader_both Low --bind "$HOST_VK":/usr/lib/x86_64-linux-gnu/libvulkan.so.1 \
                    --bind "$HOST_ICD":/usr/share/vulkan/icd.d/nvidia_icd.json
echo "=== done ==="
