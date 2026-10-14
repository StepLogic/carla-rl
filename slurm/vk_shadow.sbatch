#!/bin/bash
#SBATCH --job-name=vk_shadow
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16gb
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.log
# Does the 0.9.16 image ship NVIDIA/GLX libraries that shadow the real driver?
# The loader dlopens "libGLX_nvidia.so.0" by NAME, and the driver pulls its own
# backing libraries in by name too. If the image carries its own copies, a
# mismatched one wins somewhere in that chain and the driver refuses at
# vkCreateInstance -- which is exactly the symptom, and the 0.9.15 image (which
# works) ships none of them.
set -uo pipefail
module load apptainer 2>/dev/null
echo "=== node $(hostname) job ${SLURM_JOB_ID:-none} ==="
for s in carla0915 carla0916; do
  echo "######## $s : nvidia/GLX/vulkan libraries SHIPPED IN THE IMAGE"
  apptainer exec -e "$HOME/carla-cluster/$s.sif" sh -lc '
    ls -la /usr/lib/x86_64-linux-gnu/ 2>/dev/null | grep -iE "nvidia|GLX|GLdispatch|vulkan" 
    echo "[ld.so.conf.d]"; cat /etc/ld.so.conf.d/*.conf 2>/dev/null | grep -v "^#" | head
  ' 2>&1 | grep -v gocryptfs
done
echo "=== done ==="
