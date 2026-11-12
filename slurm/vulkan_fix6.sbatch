#!/bin/bash
# Sixth: is it the IMAGE or the NODE? And what does the failing run actually say?
#
# RULED OUT SO FAR, all on node-g102 with apptainer 1.2.5 (jobs 453151-453157):
#   GPU reaches the container (nvidia-smi works inside) | every NVIDIA library
#   --nv should mount is present and dlopens | supplying the ICD manifest, in
#   three forms, excludes lavapipe but kills CARLA in 3-6 s with no output |
#   binding the host's modern loader (1.3.250 over the image's 1.2.131) changes
#   nothing | -opengl is ignored, this UE4 build is Vulkan-only.
#
# TWO THINGS LEFT TO SEPARATE:
#   1. THE IMAGE. In October we recorded CARLA 0.9.15 under plain `apptainer run
#      --nv -e` holding 1591 MiB on this same node. If carla0915.sif still does
#      that today, the fault is in the 0.9.16 image and not in the node or the
#      site -- which would also mean the memory note is right and only 0.9.16 is
#      affected. If 0.9.15 ALSO renders on the CPU now, that note was measured
#      against a nvidia-smi reading that no longer holds and the node changed
#      under us.
#   2. WHAT THE FAILING RUN SAYS. Every NVIDIA-ICD attempt died having printed
#      two lines. UE4 writes its own log under Saved/Logs inside the container,
#      which is where the real error will be, and a core/crash report may exist.
#
# A separate job (vulkan_fix6_a100.sbatch) runs the same 0.9.16 check on an A100
# node, so image-vs-node is answered from both sides.
#
#SBATCH --job-name=vk_fix6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=00:35:00
#SBATCH --output=%x_%j.log
set -uo pipefail
SIF16=${CARLA_SIF:-$HOME/carla-cluster/carla0916.sif}
SIF15=$HOME/carla-cluster/carla0915.sif
HOST_ICD=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
module load apptainer 2>/dev/null
echo "=== node $(hostname) job ${SLURM_JOB_ID:-none} ==="

probe () {            # $1 label, $2 image, $3 entry script, rest = extra apptainer args
  local label=$1 sif=$2 entry=$3; shift 3
  local port=$((9000 + RANDOM % 900))
  local log=vk6_${label}_${SLURM_JOB_ID:-0}.log
  echo "=== [$label] $(basename "$sif") $entry port $port"
  [ -f "$sif" ] || { echo "[$label] image missing"; return; }
  apptainer run --nv -e "$@" "$sif" "$entry" \
    -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port="$port" \
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
  echo "    lavapipe: $(grep -ciE lavapipe "$log")   lines: $(wc -l < "$log")"
  tail -5 "$log" | sed 's/^/    /'
  kill -9 $pid 2>/dev/null; command sleep 5
}

# 1. The 0.9.15 image exactly as October recorded it: plain --nv, no binds.
probe c0915_plain "$SIF15" /home/carla/CarlaUE4.sh
# 2. The 0.9.16 image the same way, as the control on this node today.
probe c0916_plain "$SIF16" /workspace/CarlaUE4.sh

# 3. What does the NVIDIA-ICD run actually write? Run it with a writable overlay
#    so UE4's own log survives, then read it.
echo "=== [uelog] NVIDIA ICD + writable tmpfs, then read UE4's own log"
PORT=$((9500 + RANDOM % 400))
apptainer run --nv -e --writable-tmpfs \
  --bind "$HOST_ICD":/etc/vulkan/icd.d/nvidia_icd.json \
  --env VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  "$SIF16" sh -lc "
    /workspace/CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low \
      -carla-rpc-port=$PORT > /tmp/carla.out 2>&1
    echo '--- exit:' \$?
    echo '--- stdout:'; cat /tmp/carla.out
    echo '--- UE4 logs:'; find /workspace -name '*.log' -newermt '-5 minutes' 2>/dev/null | head -5
    for f in \$(find /workspace -name '*.log' -newermt '-5 minutes' 2>/dev/null | head -2); do
      echo \"=== \$f\"; tail -40 \"\$f\"; done
  " 2>&1 | grep -v gocryptfs | tail -45
echo "=== done ==="
