#!/bin/bash
# Fourth attempt: find out WHY the NVIDIA Vulkan driver cannot start, and try the
# renderer that does not need it.
#
# STATE OF THE EVIDENCE (jobs 453151-453155, all node-g102, apptainer 1.2.5):
#   * --nv brings every NVIDIA library in; nvidia-smi works inside the container.
#   * It does NOT bring the Vulkan ICD manifest, so Vulkan's only choice is
#     lavapipe (software). At Low that starts and holds zero GPU memory; at Epic
#     the render thread times out after 60 s and the engine takes signal 11.
#   * Supply the manifest -- mine with a relative path, mine with an absolute
#     path, or the host's own nvidia_icd.x86_64.json -- and lavapipe is correctly
#     excluded, but CARLA then exits in 3-6 s having printed NOTHING, not even the
#     UE4 banner. So the NVIDIA ICD is read and the driver behind it cannot
#     initialise here.
#
# TWO THINGS TO SETTLE, cheaply and in one job:
#   A. WHY. dlopen the library directly and print the loader's own error, and
#      check the image's Vulkan loader version. A missing dependency or a loader
#      too old to accept the ICD both show up here, and neither is visible from
#      CARLA's silence.
#   B. WHETHER IT MATTERS. CARLA's UE4 build still has an OpenGL RHI (-opengl).
#      The OpenGL stack -- libGL, libGLX_nvidia, libnvidia-glcore -- IS fully
#      mounted by --nv, and it needs no ICD manifest at all. If -opengl renders
#      on the GPU, the Vulkan question becomes irrelevant to this arm.
#
#SBATCH --job-name=vk_fix4
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

echo "--- A: can the NVIDIA GLX/Vulkan library even be loaded inside?"
apptainer exec --nv -e "$SIF" sh -lc '
  echo "[libvulkan in image]"; ls -la /usr/lib/x86_64-linux-gnu/libvulkan.so* 2>/dev/null || echo "  none in /usr/lib/x86_64-linux-gnu"
  find / -maxdepth 4 -name "libvulkan.so*" 2>/dev/null | head -5
  echo "[ldd libGLX_nvidia]"; ldd /.singularity.d/libs/libGLX_nvidia.so.0 2>&1 | grep -iE "not found|=>" | head -12
  echo "[dlopen test]"
  python3 - <<PY 2>&1 | tail -6
import ctypes
for lib in ("/.singularity.d/libs/libGLX_nvidia.so.0", "libvulkan.so.1"):
    try:
        ctypes.CDLL(lib); print("dlopen OK  ", lib)
    except OSError as e:
        print("dlopen FAIL", lib, "->", e)
PY
' 2>&1 | grep -v gocryptfs

try () {              # $1 label, rest = CarlaUE4 args
  local label=$1; shift
  local port=$((7000 + RANDOM % 900))
  local log=vk4_${label}_${SLURM_JOB_ID:-0}.log
  echo "=== [$label] port $port  args: $*"
  apptainer run --nv -e "$SIF" /workspace/CarlaUE4.sh \
    -RenderOffScreen -nosound -carla-rpc-port="$port" "$@" > "$log" 2>&1 &
  local pid=$! up=0
  for i in $(seq 1 150); do
    kill -0 $pid 2>/dev/null || { echo "[$label] EXITED after ${i}s"; break; }
    ss -ltn 2>/dev/null | grep -q ":$port\b" && { up=$i; break; }
    command sleep 1
  done
  [ "$up" -gt 0 ] && echo "[$label] listening after ${up}s"
  command sleep 12
  local mem
  mem=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head -2 | tr '\n' ';')
  if [ -n "$mem" ]; then echo "[$label] *** ON THE GPU: $mem ***"; else echo "[$label] no GPU memory"; fi
  echo "    lavapipe: $(grep -ciE lavapipe "$log")  | opengl: $(grep -ciE opengl "$log")"
  grep -iE "vulkan|opengl|rhi|renderer|device|error|fatal" "$log" | tail -5 | sed 's/^/    /'
  kill -9 $pid 2>/dev/null; command sleep 5
}

echo "--- B: the OpenGL renderer, which needs no ICD"
try opengl_low  -opengl -quality-level=Low
try opengl_epic -opengl -quality-level=Epic
echo "=== done ==="
