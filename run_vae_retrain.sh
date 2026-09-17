#!/usr/bin/env bash
# Wait for the v2 collection to finish, then retrain the VAE on it and dump
# per-town predictions. Stops there deliberately: the RL run is only worth
# launching once the lane channel is actually good on Town02, and that is a
# judgement to make from the numbers rather than automatically.
set -uo pipefail
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "[Chain] waiting for collection to finish..."
# Match on the process ARGS of the driver or the collector, with the classic
# [b]racket trick so this script's own grep cannot match itself -- the failure
# that made an earlier version of this loop exit on its first iteration.
while true; do
  if ps -eo args | grep -q "[r]un_recollect_vae.sh" \
     || ps -eo args | grep -q "[c]ollect_lane_data.py"; then
    sleep 60
  else
    break
  fi
done
echo "[Chain] collection finished."

CORPUS=data/lane_vae_v2/lane_dataset.npz
if [ ! -f "$CORPUS" ]; then
  echo "[Chain] ABORT: $CORPUS was never assembled. Collection did not finish."
  exit 1
fi

echo "[Chain] corpus stats"
python - <<'PY'
import numpy as np, json
d = np.load('data/lane_vae_v2/lane_dataset.npz', allow_pickle=True)
rgb = d['rgb'].astype(np.float32); m = d['mask']
diff = np.abs(rgb[1:] - rgb[:-1]).mean(axis=(1, 2, 3))
print(f"  frames      {len(rgb)}")
print(f"  shards      {str(d['shards'])}")
print(f"  stuck%      {(diff < 1.0).mean() * 100:.1f}   (was 47-94 in v1)")
print(f"  lane pos%   {(m[..., 0] > 0.25).mean() * 100:.3f}")
print(f"  obst pos%   {(m[..., 1] > 0.25).mean() * 100:.3f}")
print(f"  laneEmpty%  {((m[..., 0] > 0.25).mean(axis=(1, 2)) < 1e-6).mean() * 100:.1f}")
PY

echo "[Chain] retraining VAE"
python -u train_lane_vae.py --data "$CORPUS" \
    --out checkpoints/lane_vae_v2/best.pkl --epochs 60 2>&1 | tail -40

echo "[Chain] per-town predictions"
python -u src/vae_nav/dump_vae_predictions.py \
    --ckpt checkpoints/lane_vae_v2/best.pkl --data data/lane_vae_v2 \
    --out results/vae_predictions_v2 --n 12 2>&1 | grep -v -i "warn\|cuda\|torch"
echo "[Chain] done"
