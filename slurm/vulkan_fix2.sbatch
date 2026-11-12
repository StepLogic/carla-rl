#!/bin/bash
# Second attempt at getting CARLA onto the L40. Four candidate fixes, one job.
#
# WHERE WE ARE. --nv brings the driver libraries in (nvidia-smi works inside, and
# /.singularity.d/libs holds libGLX_nvidia.so.0, libEGL_nvidia.so.0 and
# libnvidia-glcore.so.545.23.08) but NOT the Vulkan ICD manifest that points
# Vulkan at them: /etc/vulkan/icd.d is empty inside and only intel/radeon/lvp
# remain, so the pick falls to lavapipe and renders on the CPU.
#
# Binding a hand-written manifest naming "libGLX_nvidia.so.0" DID remove lavapipe
# (0 mentions) but CARLA then exited in 3-10 s having printed nothing past the
# chmod line -- so the manifest was read and the library behind it was not
# usable as written. The candidates below differ only in how the driver is named
# or supplied, and each is followed by the one measurement that settles it:
# whether the CarlaUE4 process holds GPU memory.
#
# This also re-checks the apptainer version. In October we noted the
# apptainer-1.3 --nv regression (CARLA issue 6374) as real elsewhere but not our
# situation, on 1.2.5. If the site has upgraded since, our situation has become
# that one and the version is the explanation.
#
#SBATCH --job-name=vk_fix2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=00:35:00
#SBATCH --output=%x_%j.log
set -uo pipefail
SIF=${CARLA_SIF:-$HOME/carla-cluster/carla0916.sif}
module load apptainer 2>/dev/null
echo "=== node $(hostname) job ${SLURM_JOB_ID:-none} ==="
apptainer --version
echo "--- host ICDs:"; ls -la /usr/share/vulkan/icd.d/ 2>/dev/null; ls -la /etc/vulkan/icd.d/ 2>/dev/null
for f in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
  [ -f "$f" ] && { echo "--- host $f:"; cat "$f"; }
done
echo "--- inside /.singularity.d/libs:"
apptainer exec --nv -e "$SIF" sh -lc 'ls /.singularity.d/libs' 2>/dev/null | tr '\n' ' '; echo

ABS=$HOME/carla-cluster/nvidia_icd_abs.json
cat > "$ABS" <<'JSON'
{"file_format_version":"1.0.0","ICD":{"library_path":"/.singularity.d/libs/libGLX_nvidia.so.0","api_version":"1.3.0"}}
JSON

try () {              # $1 label, rest = apptainer args before the image
  local label=$1; shift
  local port=$((5000 + RANDOM % 900))
  local log=vk2_${label}_${SLURM_JOB_ID:-0}.log
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
  mem=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader 2>/dev/null | head -1)
  if [ -n "$mem" ]; then echo "[$label] *** ON THE GPU: $mem ***"; else echo "[$label] no GPU memory"; fi
  echo "    lavapipe mentions: $(grep -ciE lavapipe "$log")"
  tail -3 "$log" | sed 's/^/    /'
  kill -9 $pid 2>/dev/null; command sleep 5
}

# 1. absolute library path in the manifest
try abs --bind "$ABS":/etc/vulkan/icd.d/nvidia_icd.json \
        --env VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json

# 2. the host's own manifest, if the host has one (it names whatever path the
#    site's driver install uses, which is more authoritative than a guess)
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  try hosticd --bind /usr/share/vulkan/icd.d/nvidia_icd.json:/etc/vulkan/icd.d/nvidia_icd.json \
              --env VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
else
  echo "=== [hosticd] skipped: the host has no nvidia_icd.json either"
fi

# 3. apptainer's nvidia-container-cli path, which supplies Vulkan properly when
#    the legacy --nv library list does not
try nvccli --nvccli

# 4. the manifest alone, with no VK_DRIVER_FILES override, so the loader scans
#    both directories and may accept the nvidia entry on its own merits
try abs_noenv --bind "$ABS":/etc/vulkan/icd.d/nvidia_icd.json
echo "=== done ==="
