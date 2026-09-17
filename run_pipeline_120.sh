#!/usr/bin/env bash
# Collection (120x120, 4 towns) -> VAE training -> RL training, unattended.
set -uo pipefail
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "[Pipe] waiting for any in-flight collector..."
while ps -eo args | grep -q "[c]ollect_lane_data.py --town"; do sleep 30; done

echo "[Pipe] === stage 1: collection ==="
./run_collect_120.sh 2>&1 | tail -25

CORPUS=data/lane_vae_120/lane_dataset.npz
[ -f "$CORPUS" ] || { echo "[Pipe] ABORT: $CORPUS missing"; exit 1; }

export ENV_IMAGE_SIZE=120 VAE_IMAGE_SIZE=120
python - <<'PY'
import numpy as np
d = np.load('data/lane_vae_120/lane_dataset.npz', allow_pickle=True)
rgb, m = d['rgb'].astype(np.float32), d['mask']
diff = np.abs(rgb[1:] - rgb[:-1]).mean(axis=(1, 2, 3))
print(f"[Pipe] corpus {rgb.shape}  shards {str(d['shards'])}")
print(f"[Pipe]   stuck%={(diff < 1.0).mean() * 100:.1f}  "
      f"lane%={(m[..., 0] > 0.25).mean() * 100:.3f}  "
      f"laneEmpty%={((m[..., 0] > 0.25).mean(axis=(1, 2)) < 1e-6).mean() * 100:.1f}")
PY

echo "[Pipe] === stage 2: VAE at 120x120 ==="
# 120 epochs, not 60: at 64x64 the loss was still falling steadily when
# training stopped, so the previous run was cut short rather than converged.
python -u train_lane_vae.py --data "$CORPUS" \
    --out checkpoints/lane_vae_120/best.pkl --epochs 120 2>&1 | tail -30

echo "[Pipe] === stage 3: per-town predictions ==="
python -u src/vae_nav/dump_vae_predictions.py \
    --ckpt checkpoints/lane_vae_120/best.pkl --data data/lane_vae_120 \
    --out results/vae_predictions_120 --n 12 2>&1 | grep -v -i "warn\|cuda\|torch"

echo "[Pipe] === stage 4: RL (Town02, no oracle, 500k) ==="
for p in $(pgrep -f '[C]arlaUE4-Linux-Shipping'); do kill -9 "$p" 2>/dev/null; done
sleep 8
/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4.sh -carla-port=2000 -quality-level=Epic \
    -windowed -ResX=640 -ResY=360 >> logs/carla_server_2000.log 2>&1 &
sleep 35
./run_town02_fixed.sh 2>&1 | tail -60
echo "[Pipe] done"
