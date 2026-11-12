#!/bin/bash
#SBATCH --job-name=icd_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.log
# The controlled comparison. Same node, same apptainer, same driver file, same
# probe -- run inside the image that WORKS (carla0915) and the image that does
# NOT (carla0916). Whatever differs in the output is the cause.
#
# Everything bind-level has already been neutralised and did not help: 0.9.15's
# ICD manifest byte for byte, 0.9.15's own Vulkan loader, the Mesa ICD directory
# masked empty, and the host's GLVND shadowing Mesa's. The fault therefore is not
# the manifest, the loader generation, the competing ICDs or GLVND, and the
# driver file inside is byte-identical to the host's. This isolates the one step
# that fails.
set -uo pipefail
module load apptainer 2>/dev/null
echo "=== node $(hostname) job ${SLURM_JOB_ID:-none} ==="
P=${REPO:-$HOME/carla-rl}/slurm/icd_probe.py
for s in carla0915 carla0916; do
  echo ""
  echo "######################## $s"
  apptainer exec --nv -e --bind "$P":/tmp/icd_probe.py \
    "$HOME/carla-cluster/$s.sif" python3 /tmp/icd_probe.py 2>&1 | grep -v gocryptfs
  echo "  exit: $?"
done
echo "=== done ==="
