#!/bin/bash
# Seventh, and the first test built on a working reference rather than a guess.
#
# WHAT THE TWO IMAGES ACTUALLY CONTAIN (job 453160):
#   carla0915 (renders on the GPU, 1668 MiB, zero lavapipe):
#       /etc/vulkan/icd.d/nvidia_icd.json  SHIPS IN THE IMAGE,
#           library_path "libGLX_nvidia.so.0", api_version "1.1.99"
#       /usr/share/vulkan/icd.d            ABSENT -- no Mesa ICDs to compete
#       loader 1.1.70 | glvnd 10_nvidia.json AND 50_mesa.json
#   carla0916 (falls to lavapipe):
#       /etc/vulkan/icd.d                  EMPTY
#       /usr/share/vulkan/icd.d            intel + radeon + lvp (lavapipe)
#       loader 1.2.131 | glvnd 50_mesa.json ONLY, no nvidia entry
#
# So 0.9.16 ships Mesa's software ICDs, no NVIDIA ICD, and no glvnd nvidia
# vendor file. Supplying an NVIDIA ICD did remove lavapipe, but every manifest I
# wrote declared api_version 1.3.0 or 1.3.260 (copied from the host's) against a
# 1.2.131 loader -- and the manifest that is KNOWN to work on this node declares
# 1.1.99. The API version is the one thing never matched to the working case.
#
# Candidates, in order of how closely they reproduce the working image:
#   A  0.9.15's manifest byte for byte (api_version 1.1.99)
#   B  A, plus the glvnd nvidia vendor file 0.9.16 is missing
#   C  A, with the Mesa ICD directory emptied by an overlay, so the loader has
#      exactly what 0.9.15 gives it: one ICD and no alternatives
#
# The GPU check is now per-process and matched on the BINARY name: `apptainer
# run` forks, so the UE4 binary's pid is not the wrapper's, and an earlier job
# credited an orphaned 0.9.15 process's memory to 0.9.16. Cleanup kills the
# process group, not the wrapper.
#
#SBATCH --job-name=vk_fix7
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

# 0.9.15's manifest, reproduced exactly.
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
GLVND=$HOME/carla-cluster/10_nvidia.json
cat > "$GLVND" <<'JSON'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
JSON
EMPTY=$HOME/carla-cluster/empty_icd_dir
mkdir -p "$EMPTY"

carla_gpu_mib () {    # MiB held by any CarlaUE4 process, 0 if none
  nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader 2>/dev/null \
    | awk -F, 'tolower($1) ~ /carla/ {gsub(/[^0-9]/,"",$2); s+=$2} END {print s+0}'
}

try () {              # $1 label, $2 quality, rest = apptainer args
  local label=$1 q=$2; shift 2
  local port=$((10000 + RANDOM % 900))
  local log=vk7_${label}_${SLURM_JOB_ID:-0}.log
  echo "=== [$label] quality=$q port=$port"
  setsid apptainer run --nv -e "$@" "$SIF" /workspace/CarlaUE4.sh \
    -RenderOffScreen -nosound -quality-level="$q" -carla-rpc-port="$port" \
    > "$log" 2>&1 &
  local pid=$! up=0
  for i in $(seq 1 180); do
    kill -0 $pid 2>/dev/null || { echo "[$label] EXITED after ${i}s"; break; }
    ss -ltn 2>/dev/null | grep -q ":$port\b" && { up=$i; break; }
    command sleep 1
  done
  [ "$up" -gt 0 ] && echo "[$label] listening after ${up}s"
  command sleep 15
  local mib; mib=$(carla_gpu_mib)
  if [ "${mib:-0}" -gt 100 ]; then echo "[$label] *** ON THE GPU: ${mib} MiB ***"
  else echo "[$label] CPU ONLY (CarlaUE4 holds ${mib:-0} MiB)"; fi
  echo "    lavapipe: $(grep -ciE lavapipe "$log")"
  tail -3 "$log" | sed 's/^/    /'
  kill -- -"$pid" 2>/dev/null; pkill -9 -f CarlaUE4-Linux-Shipping 2>/dev/null
  command sleep 8
}

ICDARGS=(--bind "$ICD":/etc/vulkan/icd.d/nvidia_icd.json
         --env VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json)

try A_1199_low   Low  "${ICDARGS[@]}"
try A_1199_epic  Epic "${ICDARGS[@]}"
try B_glvnd      Low  "${ICDARGS[@]}" \
                      --bind "$GLVND":/usr/share/glvnd/egl_vendor.d/10_nvidia.json
try C_no_mesa    Low  --bind "$ICD":/etc/vulkan/icd.d/nvidia_icd.json \
                      --bind "$EMPTY":/usr/share/vulkan/icd.d
echo "=== done ==="
