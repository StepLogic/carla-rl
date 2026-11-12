#!/bin/bash
# Capture the RGB bank (smoke first, then full) with the drive launcher's exact env.
W=/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl; cd "$W" || exit 1
source /home/robotlab/anaconda3/etc/profile.d/conda.sh; conda activate carla-rl || exit 1
set +u; eval "$(grep -E '^export ' run_gated.sh)"
while IFS='=' read -r k v; do case "$k" in ''|'#'*) continue;; esac; export "$k=$v"; done < results/run_env/carla_gated_her_v6n.env
export CARLA_START_SERVER=1 CARLA_QUALITY=Epic CARLA_PORT=2000 CARLA_PORTS=2000
export EXPERT_PROB=0 RESUME=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export ENV_SKIP_IMAGE_FEATURES=1 PYTHONPATH=src NO_TIMELIMIT=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia
kill_all_carla() {
  for _ in 1 2 3 4 5 6; do
    pkill -9 -f 'CarlaUE4-Linux-Shipping' 2>/dev/null; pkill -9 -f 'CarlaUE4' 2>/dev/null
    pkill -9 -f 'CARLA_0.9.15' 2>/dev/null; sleep 2
    ps -eo args | grep -i '[C]arlaUE4' >/dev/null || return 0
  done
  return 1
}
echo "=== [bank $(date +%H:%M:%S)] pre-flight kill ==="
kill_all_carla || exit 1
sleep 15
python -u capture_rgb_bank.py "$@"
RC=$?
echo "=== [bank $(date +%H:%M:%S)] python rc $RC ==="
kill_all_carla
exit $RC
