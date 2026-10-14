#!/bin/bash
#SBATCH --job-name=vk_ldpath
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=00:35:00
#SBATCH --output=%x_%j.log
# THE CAUSE, from the controlled probe (job 453238). Same node, same apptainer,
# same byte-identical driver file, asked the same question:
#
#   carla0915  LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu:
#                              /usr/local/nvidia/lib:/usr/local/nvidia/lib64:
#                              /.singularity.d/libs
#              vk_icdGetInstanceProcAddr("vkCreateInstance") -> 0x7ff34964a1f0
#
#   carla0916  LD_LIBRARY_PATH=/.singularity.d/libs
#              vk_icdGetInstanceProcAddr("vkCreateInstance") -> NULL
#
# The 0.9.15 image inherits an LD_LIBRARY_PATH from its NVIDIA base image; the
# 0.9.16 image sets none, so apptainer's /.singularity.d/libs is the whole search
# path and the driver cannot resolve something it needs at init. It then declines
# every entry point, the loader reports "Could not get 'vkCreateInstance'", and
# CARLA dies in 3-6 s. That is the entire mystery, and it explains why no
# manifest, loader or GLVND change ever helped: none of them touched the search
# path.
#
# Fix under test: restore the system library directories alongside apptainer's.
set -uo pipefail
SIF=${CARLA_SIF:-$HOME/carla-cluster/carla0916.sif}
REPO=${REPO:-$HOME/carla-rl}
module load apptainer 2>/dev/null
echo "=== node $(hostname) job ${SLURM_JOB_ID:-none} ==="

ICD=$HOME/carla-cluster/nvidia_icd_0915.json
cat > "$ICD" <<'JSON'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.1.99"
    }
}
JSON
LDP=/usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu:/.singularity.d/libs

echo "### does the driver answer now?"
apptainer exec --nv -e --env LD_LIBRARY_PATH="$LDP" \
  --bind "$REPO/slurm/icd_probe.py":/tmp/icd_probe.py \
  "$SIF" python3 /tmp/icd_probe.py 2>&1 | grep -v gocryptfs

run () {   # $1 label, $2 quality, rest = extra apptainer args
  local label=$1 q=$2; shift 2
  local port=$((16000 + RANDOM % 500))
  local log=vk_ldpath_${label}_${SLURM_JOB_ID:-0}.log
  echo ""
  echo "### [$label] quality=$q port=$port"
  setsid apptainer run --nv -e --env LD_LIBRARY_PATH="$LDP" "$@" "$SIF" \
    /workspace/CarlaUE4.sh -RenderOffScreen -nosound \
    -quality-level="$q" -carla-rpc-port="$port" > "$log" 2>&1 &
  local pid=$!
  for i in $(seq 1 180); do
    kill -0 $pid 2>/dev/null || { echo "[$label] EXITED after ${i}s"; break; }
    ss -ltn 2>/dev/null | grep -q ":$port\b" && { echo "[$label] listening after ${i}s"; break; }
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

ICDARGS=(--bind "$ICD":/etc/vulkan/icd.d/nvidia_icd.json
         --env VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json)
run ldpath_icd_low   Low  "${ICDARGS[@]}"
run ldpath_icd_epic  Epic "${ICDARGS[@]}"
# Without the manifest: if the driver now initialises, does the loader still
# prefer lavapipe? This says whether the manifest bind is needed at all.
run ldpath_only_low  Low
echo "=== done ==="
