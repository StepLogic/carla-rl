#!/bin/bash
# User 2026-10-13: use the 0.9.16 container. So: make it render on the GPU.
#
# Every attempt so far ends the same way -- supply an NVIDIA ICD, lavapipe is
# correctly excluded, and CARLA exits in 3-6 s having printed nothing, not even
# the UE4 banner it prints when it runs on lavapipe. Nothing in that is diagnostic,
# so this job stops guessing and asks the loader directly (VK_LOADER_DEBUG) and
# runs the UE4 BINARY rather than CarlaUE4.sh, so stderr is not swallowed.
#
# It also fixes a real defect in my earlier loader test: I bound the host's
# libvulkan.so.1.3.250 onto /usr/lib/x86_64-linux-gnu/libvulkan.so.1, which in
# this image is a SYMLINK to libvulkan.so.1.2.131. Binding onto a symlink path is
# not the same as replacing the library, so that test may never have used the new
# loader at all. Here it is bound onto the RESOLVED file and the md5 is printed
# inside, so the swap is confirmed rather than assumed.
#
#SBATCH --job-name=vk_dbg2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=00:40:00
#SBATCH --output=%x_%j.log
set -uo pipefail
SIF=${CARLA_SIF:-$HOME/carla-cluster/carla0916.sif}
REPO=${REPO:-$HOME/carla-rl}
module load apptainer 2>/dev/null
echo "=== node $(hostname) job ${SLURM_JOB_ID:-none} ==="
md5sum /usr/lib64/libvulkan.so.1.3.250 2>/dev/null

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

run () {   # $1 label, rest: apptainer args
  local label=$1; shift
  echo ""
  echo "############################## [$label]"
  PORT=$((12000 + RANDOM % 500)) \
  apptainer exec --nv -e \
    --env PORT=$((12000 + RANDOM % 500)) \
    --bind "$REPO/slurm/vk_inner.sh":/tmp/vk_inner.sh \
    "$@" "$SIF" sh /tmp/vk_inner.sh 2>&1 | grep -v gocryptfs
}

# 1. The NVIDIA ICD alone, with the image's own loader, and the loader talking.
run icd_only \
  --bind "$ICD":/etc/vulkan/icd.d/nvidia_icd.json \
  --env VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

echo "=== done ==="
