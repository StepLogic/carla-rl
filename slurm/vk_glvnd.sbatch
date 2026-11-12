#!/bin/bash
#SBATCH --job-name=vk_glvnd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.log
# THE DIFFERENCE BETWEEN THE TWO IMAGES, found in job 453236: carla0915 is built
# on an NVIDIA GLVND base -- its /etc/ld.so.conf.d still lists
# /usr/local/nvidia/lib and lib64 -- and ships /etc/vulkan/icd.d/nvidia_icd.json.
# carla0916 is plain Ubuntu 20.04 with MESA: libGLdispatch.so.0, libGLX.so.0 and
# libGLX_mesa are Mesa's March-2022 builds, the three Mesa Vulkan drivers are
# present, and there is no NVIDIA ICD at all.
#
# NVIDIA's driver registers itself through GLVND's dispatch library. If it loads
# Mesa's libGLdispatch rather than NVIDIA's, that registration fails and the
# ICD's entry point comes back unusable -- which is exactly the symptom: the
# library loads, and vk_icdGetInstanceProcAddr yields no vkCreateInstance.
#
# So shadow Mesa's GLVND with the host's before rebuilding anything: five minutes
# against a two-hour image build.
set -uo pipefail
SIF=${CARLA_SIF:-$HOME/carla-cluster/carla0916.sif}
module load apptainer 2>/dev/null
echo "=== node $(hostname) job ${SLURM_JOB_ID:-none} ==="
echo "--- host GLVND:"
ls -la /usr/lib64/libGLdispatch.so.0 /usr/lib64/libGLX.so.0 2>/dev/null

ICD=$HOME/carla-cluster/nvidia_icd_0915.json
EMPTY=$HOME/carla-cluster/empty_icd_dir
mkdir -p "$EMPTY"
ARGS=(--bind "$ICD":/etc/vulkan/icd.d/nvidia_icd.json
      --env VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
      --bind "$EMPTY":/usr/share/vulkan/icd.d)
for lib in libGLdispatch.so.0 libGLX.so.0; do
  t=$(readlink -f "/usr/lib64/$lib" 2>/dev/null)
  if [ -n "$t" ] && [ -f "$t" ]; then
    ARGS+=(--bind "$t":"/usr/lib/x86_64-linux-gnu/$lib")
    echo "  shadowing $lib with $t"
  fi
done

PORT=$((15000 + RANDOM % 500))
LOG=vk_glvnd_carla_${SLURM_JOB_ID:-0}.log
setsid apptainer run --nv -e "${ARGS[@]}" "$SIF" /workspace/CarlaUE4.sh \
  -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port="$PORT" \
  > "$LOG" 2>&1 &
PID=$!
for i in $(seq 1 180); do
  kill -0 $PID 2>/dev/null || { echo "EXITED after ${i}s"; break; }
  ss -ltn 2>/dev/null | grep -q ":$PORT\b" && { echo "listening after ${i}s"; break; }
  command sleep 1
done
command sleep 15
mib=$(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader 2>/dev/null \
      | awk -F, 'tolower($1) ~ /carla/ {gsub(/[^0-9]/,"",$2); s+=$2} END {print s+0}')
if [ "${mib:-0}" -gt 100 ]; then echo "*** ON THE GPU: ${mib} MiB ***"
else echo "CPU only (${mib:-0} MiB)"; fi
echo "lavapipe: $(grep -ciE lavapipe "$LOG")"
tail -5 "$LOG"
kill -- -"$PID" 2>/dev/null
pkill -9 -f CarlaUE4-Linux-Shipping 2>/dev/null
echo "=== done ==="
