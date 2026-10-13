#!/bin/bash
# Make CARLA render on the L40 instead of on the CPU.
#
# DIAGNOSIS (job 453152, node-g102): the GPU reaches the container -- nvidia-smi
# inside lists the L40 -- but /etc/vulkan/icd.d inside is EMPTY and the only
# ICDs present are /usr/share/vulkan/icd.d/{intel,radeon,lvp}_icd.json. lvp is
# lavapipe, the software rasteriser, so Vulkan has nothing else to choose: at Low
# CARLA starts and holds ZERO GPU memory, and at Epic the render thread times out
# after 60 s and the engine takes signal 11.
#
# So `apptainer run --nv -e` does NOT supply the NVIDIA Vulkan ICD here, contrary
# to what we recorded in October. The driver libraries come in (that is why
# nvidia-smi works) but the ICD manifest that points Vulkan at them does not.
#
# THE FIX UNDER TEST: supply that one manifest. This is NOT the mistake made in
# October -- that bound the host's whole /usr/share/vulkan/icd.d directory, whose
# RHEL entries name /usr/lib64/libvulkan_{radeon,lvp,intel}.so, paths that do not
# exist inside, and which DISPLACED the working setup. Here a single generated
# json naming libGLX_nvidia.so.0 is bound to one path, and VK_DRIVER_FILES /
# VK_ICD_FILENAMES point at it alone so lavapipe cannot win the pick.
#
#SBATCH --job-name=vk_fix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.log
set -uo pipefail
SIF=${CARLA_SIF:-$HOME/carla-cluster/carla0916.sif}
module load apptainer 2>/dev/null
echo "=== node $(hostname) job ${SLURM_JOB_ID:-none} ==="

echo "--- where are the nvidia libraries inside?"
apptainer exec --nv -e "$SIF" sh -lc '
  echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  for d in /.singularity.d/libs /usr/lib/x86_64-linux-gnu /usr/lib64; do
    n=$(ls $d 2>/dev/null | grep -ciE "nvidia|GLX" || true); echo "$d: $n nvidia/GLX entries"
  done
  ls /.singularity.d/libs 2>/dev/null | grep -iE "GLX_nvidia|nvidia-glcore|libnvidia-vulkan|EGL_nvidia" | head
' 2>&1 | grep -v gocryptfs

ICD=$HOME/carla-cluster/nvidia_icd.json
cat > "$ICD" <<'JSON'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.3.0"
    }
}
JSON
echo "--- wrote $ICD"

run_carla () {            # $1 = quality, $2 = label, rest = extra apptainer args
  local q=$1 label=$2; shift 2
  local port=$((4000 + RANDOM % 900))
  local log=vk_fix_${label}_${SLURM_JOB_ID:-0}.log
  echo "=== [$label] quality=$q port=$port"
  apptainer run --nv -e \
    --bind "$ICD":/etc/vulkan/icd.d/nvidia_icd.json \
    --env VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    --env VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
    "$@" "$SIF" /workspace/CarlaUE4.sh \
    -RenderOffScreen -nosound -quality-level="$q" -carla-rpc-port="$port" \
    > "$log" 2>&1 &
  local pid=$!
  local up=0
  for i in $(seq 1 180); do
    kill -0 $pid 2>/dev/null || { echo "[$label] EXITED after ${i}s"; break; }
    ss -ltn 2>/dev/null | grep -q ":$port\b" && { up=$i; break; }
    command sleep 1
  done
  [ "$up" -gt 0 ] && echo "[$label] listening after ${up}s"
  command sleep 12
  echo "[$label] GPU compute processes:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
    | sed 's/^/    /' || true
  if nvidia-smi --query-compute-apps=used_memory --format=csv,noheader 2>/dev/null | grep -q MiB; then
    echo "[$label] *** ON THE GPU ***"
  else
    echo "[$label] *** STILL NO GPU MEMORY ***"
  fi
  grep -icE "lavapipe" "$log" | sed 's/^/    lavapipe mentions: /'
  tail -4 "$log" | sed 's/^/    /'
  kill -9 $pid 2>/dev/null
  command sleep 5
}

run_carla Low  icd_low
run_carla Epic icd_epic
echo "=== done ==="
