"""Standalone end-to-end check of the VC-1 encoder backend.

Builds VcSource the way from_env does, encodes N random 120x160 frames and
prints device, mean latency per frame, and peak VRAM when on cuda.

Gates for the run (from the Phase 2 brief): GPU <= 10 ms/frame and
<= 1 GiB peak; CPU <= 100 ms/frame.

    dependencies/vc1-venv/bin/python scripts/vc1_encoder_check.py [N] [device]
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import numpy as np


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    device = sys.argv[2] if len(sys.argv) > 2 else None

    from vae_nav.vc1_source import VcSource

    src = VcSource(device=device)
    print(f"torch device reported by source: {src.device}")

    import torch
    if src.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (n, 120, 160, 3), dtype=np.uint8)

    # Warm-up (first call also triggers any lazy CUDA kernel compile).
    src.latent(frames[0])

    t0 = time.perf_counter()
    for i in range(n):
        z = src.latent(frames[i])
    dt = (time.perf_counter() - t0) / n * 1000.0

    assert z.shape == (768,), z.shape
    assert z.dtype == np.float32
    assert np.isfinite(z).all()

    peak = ""
    if src.device.startswith("cuda"):
        peak = f", peak VRAM {torch.cuda.max_memory_allocated()/2**30:.3f} GiB"
    print(f"frames={n} device={src.device} mean latency {dt:.2f} ms/frame{peak}")
    print(f"latent stats: mean={z.mean():.4f} std={z.std():.4f} "
          f"min={z.min():.4f} max={z.max():.4f}")
    print("identity:", src.identity())


if __name__ == "__main__":
    main()