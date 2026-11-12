#!/bin/bash
# User 2026-10-13: use the 0.9.16 container. This is the test that follows from
# the evidence rather than from a guess.
#
# WHAT IS ESTABLISHED:
#  * The driver library --nv mounts into the 0.9.16 image is BYTE-IDENTICAL to
#    the host's (md5 37d51253...) and DOES export vk_icdGetInstanceProcAddr
#    (objdump: 1). So the right driver is inside and the manifest finds it.
#  * The loader then fails at the next step: "Could not get 'vkCreateInstance'
#    via 'vk_icdGetInstanceProcAddr'". The entry point exists and returns NULL
#    for vkCreateInstance, which is the driver declining to initialise.
#  * The SAME host driver, mounted the SAME way, works in the 0.9.15 image.
#
# THE ONE REMAINING DIFFERENCE between the two images on this path is the Vulkan
# LOADER: 0.9.15 ships 1.1.70, 0.9.16 ships 1.2.131. A loader and a driver
# negotiate an ICD interface version, and 1.1.70 is old enough to fall back to
# plain dlsym for vkCreateInstance where 1.2.131 insists on the newer
# vk_icdGetInstanceProcAddr path. That fits every observation, including why a
# NEWER loader made no difference.
#
# Borrowing 0.9.15's loader is the right direction for glibc, which is why this
# can work where borrowing the host's could not: a library built against the
# older glibc (18.04, 2.27) runs on the newer (20.04, 2.31), but not the reverse
# -- the host's 1.3.250 needed GLIBC_2.34 and died on exactly that.
#
#SBATCH --job-name=vk_old
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=00:40:00
#SBATCH --output=%x_%j.log
set -uo pipefail
SIF16=${CARLA_SIF:-$HOME/carla-cluster/carla0916.sif}
SIF15=$HOME/carla-cluster/carla0915.sif
STAGE=$HOME/carla-cluster/vk15
module load apptainer 2>/dev/null
echo "=== node $(hostname) job ${SLURM_JOB_ID:-none} ==="

mkdir -p "$STAGE"
echo "### extract 0.9.15's loader and its own ICD manifest"
apptainer exec -e "$SIF15" sh -lc '
  cp -L /usr/lib/x86_64-linux-gnu/libvulkan.so.1 /tmp/libvulkan.so.1.1.70 2>/dev/null
  cp /etc/vulkan/icd.d/nvidia_icd.json /tmp/nvidia_icd_0915.json 2>/dev/null
  ls -la /tmp/libvulkan.so.1.1.70 /tmp/nvidia_icd_0915.json
' 2>&1 | grep -v gocryptfs
# apptainer gives no copy-out, so re-read them through a bind of the staging dir.
apptainer exec -e --bind "$STAGE":/stage "$SIF15" sh -lc '
  cp -L /usr/lib/x86_64-linux-gnu/libvulkan.so.1 /stage/libvulkan.so.1.1.70
  cp /etc/vulkan/icd.d/nvidia_icd.json /stage/nvidia_icd_0915.json
' 2>&1 | grep -v gocryptfs
ls -la "$STAGE"
md5sum "$STAGE"/* 2>/dev/null

run () {   # $1 label, rest = apptainer args
  local label=$1; shift
  local port=$((14000 + RANDOM % 500))
  local log=vkold_${label}_${SLURM_JOB_ID:-0}.log
  echo ""
  echo "### [$label] port $port"
  setsid apptainer run --nv -e "$@" "$SIF16" /workspace/CarlaUE4.sh \
    -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port="$port" \
    > "$log" 2>&1 &
  local pid=$! up=0
  for i in $(seq 1 180); do
    kill -0 $pid 2>/dev/null || { echo "[$label] EXITED after ${i}s"; break; }
    ss -ltn 2>/dev/null | grep -q ":$port\b" && { up=$i; echo "[$label] listening after ${i}s"; break; }
    command sleep 1
  done
  command sleep 15
  local mib
  mib=$(nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader 2>/dev/null \
        | awk -F, 'tolower($1) ~ /carla/ {gsub(/[^0-9]/,"",$2); s+=$2} END {print s+0}')
  if [ "${mib:-0}" -gt 100 ]; then echo "[$label] *** ON THE GPU: ${mib} MiB ***"
  else echo "[$label] CPU only (${mib:-0} MiB)"; fi
  echo "    lavapipe: $(grep -ciE lavapipe "$log")"
  tail -4 "$log" | sed 's/^/    /'
  kill -- -"$pid" 2>/dev/null; pkill -9 -f CarlaUE4-Linux-Shipping 2>/dev/null
  command sleep 8
}

if [ -s "$STAGE/libvulkan.so.1.1.70" ]; then
  # 0.9.15's loader over 0.9.16's, plus 0.9.15's manifest. Bound onto the
  # RESOLVED file, not the symlink, so the swap really happens.
  run oldloader_icd \
    --bind "$STAGE/libvulkan.so.1.1.70":/usr/lib/x86_64-linux-gnu/libvulkan.so.1.2.131 \
    --bind "$STAGE/nvidia_icd_0915.json":/etc/vulkan/icd.d/nvidia_icd.json \
    --env VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
  # Same, but let the loader see the Mesa ICDs too: if NVIDIA initialises, UE4
  # should still pick the discrete GPU, and this is the configuration closest to
  # a plain image with one file added.
  run oldloader_all \
    --bind "$STAGE/libvulkan.so.1.1.70":/usr/lib/x86_64-linux-gnu/libvulkan.so.1.2.131 \
    --bind "$STAGE/nvidia_icd_0915.json":/etc/vulkan/icd.d/nvidia_icd.json
else
  echo "FATAL: could not stage 0.9.15's loader"
fi
echo "=== done ==="
