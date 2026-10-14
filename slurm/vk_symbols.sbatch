#!/bin/bash
# THE ACTUAL FAULT, from job 453232's loader trace:
#   ERROR: loader_scanned_icd_add: Could not get 'vkCreateInstance' via
#          'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0
# The manifest is read and the library is found; the library does not export the
# ICD entry point. So the libGLX_nvidia.so.0 that `apptainer --nv` mounts into
# /.singularity.d/libs is not a Vulkan-capable driver, which is why CARLA dies
# instantly the moment NVIDIA is the only ICD, and why no amount of manifest
# editing or api_version matching ever helped.
#
# (Also settled there: the host's loader cannot be borrowed. libvulkan.so.1.3.250
# needs GLIBC_2.34 and this image is Ubuntu 20.04 with 2.31.)
#
# So: find a libGLX_nvidia.so.0 that DOES export vk_icdGetInstanceProcAddr. The
# host must have one -- its own nvidia_icd.x86_64.json names exactly that library
# and the host renders fine -- so either --nv mounts a different file than the one
# the host's manifest resolves to, or the real driver lives somewhere --nv does
# not look. This job finds out and, if it finds one, binds it and tries CARLA.
#
#SBATCH --job-name=vk_sym
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=00:40:00
#SBATCH --output=%x_%j.log
set -uo pipefail
SIF=${CARLA_SIF:-$HOME/carla-cluster/carla0916.sif}
module load apptainer 2>/dev/null
echo "=== node $(hostname) job ${SLURM_JOB_ID:-none} ==="

echo "### HOST: every libGLX_nvidia and nvidia vulkan library"
ls -la /usr/lib64/libGLX_nvidia* /usr/lib64/nvidia/libGLX_nvidia* 2>/dev/null
find /usr/lib64 /usr/lib -maxdepth 3 -name 'libGLX_nvidia.so*' 2>/dev/null
find /usr/lib64 /usr/lib -maxdepth 3 -name '*nvidia*vulkan*' 2>/dev/null | head
echo "### HOST: which of them export the ICD entry point?"
for f in $(find /usr/lib64 /usr/lib -maxdepth 3 -name 'libGLX_nvidia.so*' 2>/dev/null); do
  r=$(readlink -f "$f")
  n=$(objdump -T "$r" 2>/dev/null | grep -c vk_icdGetInstanceProcAddr || true)
  echo "  $f -> $r : vk_icd symbols = ${n:-0}"
done

echo "### INSIDE: what --nv mounted, and does it export the entry point?"
apptainer exec --nv -e "$SIF" sh -lc '
  f=/.singularity.d/libs/libGLX_nvidia.so.0
  ls -la $f; r=$(readlink -f $f); echo "resolves to: $r"
  md5sum $r
  if command -v objdump >/dev/null; then
    echo "vk_icd symbols: $(objdump -T $r 2>/dev/null | grep -c vk_icdGetInstanceProcAddr)"
  else
    echo "no objdump in image; grepping the binary for the symbol name:"
    grep -c vk_icdGetInstanceProcAddr $r 2>/dev/null || echo "  (not found by grep)"
  fi
' 2>&1 | grep -v gocryptfs
echo "### HOST md5 for comparison"
md5sum "$(readlink -f /usr/lib64/libGLX_nvidia.so.0)" 2>/dev/null

GOOD=""
for f in $(find /usr/lib64 /usr/lib -maxdepth 3 -name 'libGLX_nvidia.so*' 2>/dev/null); do
  r=$(readlink -f "$f")
  if objdump -T "$r" 2>/dev/null | grep -q vk_icdGetInstanceProcAddr; then GOOD=$r; break; fi
done
echo "### a Vulkan-capable driver on the host: ${GOOD:-NONE FOUND}"

if [ -n "$GOOD" ]; then
  ICD=$HOME/carla-cluster/nvidia_icd_bound.json
  printf '{"file_format_version":"1.0.0","ICD":{"library_path":"/opt/nvidia/libGLX_nvidia.so.0","api_version":"1.1.99"}}\n' > "$ICD"
  echo "### CARLA with the real driver bound at /opt/nvidia"
  PORT=$((13000 + RANDOM % 500))
  setsid apptainer run --nv -e \
    --bind "$GOOD":/opt/nvidia/libGLX_nvidia.so.0 \
    --bind "$ICD":/etc/vulkan/icd.d/nvidia_icd.json \
    --env VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    --env LD_LIBRARY_PATH=/opt/nvidia:/.singularity.d/libs \
    "$SIF" /workspace/CarlaUE4.sh -RenderOffScreen -nosound \
    -quality-level=Low -carla-rpc-port="$PORT" > vk_sym_carla_${SLURM_JOB_ID:-0}.log 2>&1 &
  PID=$!
  for i in $(seq 1 180); do
    kill -0 $PID 2>/dev/null || { echo "EXITED after ${i}s"; break; }
    ss -ltn 2>/dev/null | grep -q ":$PORT\b" && { echo "listening after ${i}s"; break; }
    command sleep 1
  done
  command sleep 15
  mib=$(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader 2>/dev/null \
        | awk -F, 'tolower($1) ~ /carla/ {gsub(/[^0-9]/,"",$2); s+=$2} END {print s+0}')
  if [ "${mib:-0}" -gt 100 ]; then echo "*** ON THE GPU: ${mib} MiB ***"; else echo "CPU only (${mib:-0} MiB)"; fi
  echo "lavapipe: $(grep -ciE lavapipe vk_sym_carla_${SLURM_JOB_ID:-0}.log)"
  tail -5 vk_sym_carla_${SLURM_JOB_ID:-0}.log
  kill -- -$PID 2>/dev/null; pkill -9 -f CarlaUE4-Linux-Shipping 2>/dev/null
fi
echo "=== done ==="
