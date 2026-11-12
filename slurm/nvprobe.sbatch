#!/bin/bash
#SBATCH --job-name=nvprobe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16gb
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.log
# Does an NVIDIA-GLVND Ubuntu 20.04 base let the driver answer, where plain
# ubuntu:20.04 does not? 126 MB probe before a 20 GB rebuild.
#
# If vkCreateInstance comes back non-NULL here, rebuilding CARLA 0.9.16 on this
# base is the fix and worth the build. If it is NULL, the base is not the
# difference and the cause lies in the 0.9.16 simulator itself -- in which case
# no rebuild would have helped and we would have found that out the expensive way.
set -uo pipefail
module load apptainer 2>/dev/null
echo "=== node $(hostname) job ${SLURM_JOB_ID:-none} ==="
apptainer exec --nv -e --bind "$HOME/carla-rl/slurm/icd_probe.py":/tmp/icd_probe.py \
  "$HOME/carla-cluster/nvprobe.sif" python3 /tmp/icd_probe.py 2>&1 | grep -v gocryptfs
echo "=== done ==="
