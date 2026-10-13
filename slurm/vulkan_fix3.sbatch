#!/bin/bash
# Third attempt, from two findings in job 453154.
#
# FINDING 1: the host DOES have an NVIDIA Vulkan manifest, named
# nvidia_icd.x86_64.json -- my previous test looked for nvidia_icd.json and so
# skipped the most authoritative candidate there was.
#
# FINDING 2: the two knobs that each half-worked were never combined.
#   vk_fix  : VK_ICD_FILENAMES + a manifest naming the library RELATIVELY
#             -> lavapipe gone (0 mentions), but CARLA exited in 3-10 s having
#                printed nothing: the manifest was honoured, the library was not
#                resolvable as written.
#   vk_fix2 : VK_DRIVER_FILES + a manifest naming the ABSOLUTE path
#             -> lavapipe still won (1 mention), so this image's Vulkan loader is
#                old enough to ignore VK_DRIVER_FILES and only read
#                VK_ICD_FILENAMES.
# So the combination to try is VK_ICD_FILENAMES with an absolute library path.
#
# apptainer is 1.2.5, the same version as in October, so this is not the
# apptainer-1.3 --nv regression. Every NVIDIA library Vulkan needs is inside
# (/.singularity.d/libs has libGLX_nvidia.so.0, libnvidia-glvkspirv and
# libnvidia-rtcore). Only the manifest is missing.
#
#SBATCH --job-name=vk_fix3
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
echo "--- the host's own manifest ($HOST_ICD):"; cat "$HOST_ICD" 2>/dev/null || echo "(absent)"

ABS=$HOME/carla-cluster/nvidia_icd_abs.json
cat > "$ABS" <<'JSON'
{"file_format_version":"1.0.0","ICD":{"library_path":"/.singularity.d/libs/libGLX_nvidia.so.0","api_version":"1.3.0"}}
JSON

try () {              # $1 label, rest = apptainer args before the image
  local label=$1; shift
  local port=$((6000 + RANDOM % 900))
  local log=vk3_${label}_${SLURM_JOB_ID:-0}.log
  echo "=== [$label] port $port"
  apptainer run --nv -e "$@" "$SIF" /workspace/CarlaUE4.sh \
    -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port="$port" \
    > "$log" 2>&1 &
  local pid=$! up=0
  for i in $(seq 1 120); do
    kill -0 $pid 2>/dev/null || { echo "[$label] EXITED after ${i}s"; break; }
    ss -ltn 2>/dev/null | grep -q ":$port\b" && { up=$i; break; }
    command sleep 1
  done
  [ "$up" -gt 0 ] && echo "[$label] listening after ${up}s"
  command sleep 10
  local mem
  mem=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head -2 | tr '\n' ';')
  if [ -n "$mem" ]; then echo "[$label] *** ON THE GPU: $mem ***"; else echo "[$label] no GPU memory"; fi
  echo "    lavapipe mentions: $(grep -ciE lavapipe "$log")"
  grep -iE "vulkan|device|GPU|error|cannot" "$log" | tail -4 | sed 's/^/    /'
  kill -9 $pid 2>/dev/null; command sleep 5
}

# A. THE UNTESTED COMBINATION: absolute library path, read via VK_ICD_FILENAMES.
try abs_icdenv --bind "$ABS":/etc/vulkan/icd.d/nvidia_icd.json \
               --env VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

# B. The host's own manifest, bound in and selected explicitly. It names whatever
#    path this site's driver install uses, which beats any guess of mine -- but it
#    may name a host path that does not exist inside, which is exactly the
#    October mistake in miniature, so it is tried as a candidate and not adopted
#    on faith.
if [ -f "$HOST_ICD" ]; then
  try hosticd --bind "$HOST_ICD":/etc/vulkan/icd.d/nvidia_icd.json \
              --env VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

# C. Host manifest PLUS the host library directory it points at, in case its
#    library_path is an absolute host path.
if [ -f "$HOST_ICD" ] && [ -d /usr/lib64 ]; then
  try hosticd_lib --bind "$HOST_ICD":/etc/vulkan/icd.d/nvidia_icd.json \
                  --bind /usr/lib64/libGLX_nvidia.so.0:/usr/lib64/libGLX_nvidia.so.0 \
                  --env VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi
echo "=== done ==="
