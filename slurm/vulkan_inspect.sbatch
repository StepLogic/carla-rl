#!/bin/bash
# What does the 0.9.15 image have that the 0.9.16 image lacks?
#
# ESTABLISHED (job 453158, node-g102): the 0.9.15 image renders ON THE GPU under
# plain `apptainer run --nv -e` (1668 MiB, zero lavapipe mentions). The 0.9.16
# image, same node, same flags, same minute, falls to lavapipe. So the fault is
# the IMAGE, not the node, the driver or the site -- and the October note was
# right about 0.9.15 all along.
#
# (The GPU-memory readings in that job need one correction: `apptainer run`
# forks, so the UE4 binary has a different pid from the wrapper, and kill -9 on
# the wrapper leaves the binary running -- our own "killing the nohup parent
# spares the child" lesson. The 4987 MiB attributed to 0.9.16 was in fact the
# orphaned 0.9.15 process still loading its map. Hence the per-pid check below.)
#
# This job reads the Vulkan configuration out of both images so the 0.9.16 fix is
# a copy of something known to work rather than another guess. The login node
# cannot mount a .sif (its squashfuse is broken), so the inspection has to run
# here.
#
#SBATCH --job-name=vk_inspect
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16gb
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.log
set -uo pipefail
module load apptainer 2>/dev/null
echo "=== node $(hostname) job ${SLURM_JOB_ID:-none} ==="
for s in carla0915 carla0916; do
  echo "######################## $s"
  apptainer exec -e "$HOME/carla-cluster/$s.sif" sh -lc '
    echo "[os]"; grep -h PRETTY_NAME /etc/os-release 2>/dev/null
    echo "[/etc/vulkan/icd.d]"; ls -la /etc/vulkan/icd.d/ 2>/dev/null || echo "  absent"
    for f in /etc/vulkan/icd.d/*.json; do [ -f "$f" ] && { echo "--- $f"; cat "$f"; }; done
    echo "[/usr/share/vulkan/icd.d]"; ls /usr/share/vulkan/icd.d/ 2>/dev/null || echo "  absent"
    echo "[loader]"; ls -la /usr/lib/x86_64-linux-gnu/libvulkan.so* 2>/dev/null || echo "  none"
    echo "[glvnd]"; ls /usr/share/glvnd/egl_vendor.d/ 2>/dev/null || echo "  absent"
    echo "[entry]"; ls -la /home/carla/CarlaUE4.sh /workspace/CarlaUE4.sh 2>/dev/null
    echo "[python wheels]"; ls /home/carla/PythonAPI/carla/dist/ 2>/dev/null | head -4
    ls /workspace/PythonAPI/carla/dist/ 2>/dev/null | head -4
  ' 2>&1 | grep -v gocryptfs
done
echo "=== done ==="
