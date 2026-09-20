"""Probe: verify VC-1 patch-token extraction against the live ViT-B model.

The grid readout (VC1_READOUT=grid2) is built on the assumption that slicing
the pre-pooling token sequence at [1:] gives the 14x14 patch tokens whose
first token IS the CLS embedding VcSource returns today. This script proves
that on the real checkpoint, with no CARLA involved:

  1. loads vc1_vitb through vc_models the way VcSource does (same hydra
     compose + instantiate path), and asserts model.classifier_feature ==
     "use_cls_token" -- anything else and the extraction recipe is wrong for
     this checkpoint;
  2. runs the token recipe (forward_features of vit.py WITHOUT
     handle_outcome, i.e. with an explicit final self.norm) and asserts
     torch.allclose(tokens[:, 0], model(x)) -- if the sliced CLS equals the
     model's own output, the tokens are exactly the model's pre-pooling
     sequence;
  3. pools the (196, 768) tokens to the 37632-d spatial grid, prints shape,
     statistics and per-frame timing, and saves the pooled vectors for a few
     frames.

When VC1_READOUT=grid2 is set, it additionally cross-checks the pooled
vector against VcSource.latent() itself.

Run with the torch/hydra interpreter:

    dependencies/ig-venv/bin/python scripts/vc1_grid_probe.py [n_frames]
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import numpy as np

OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results", "vc1_grid_probe.npz")


def pool_grid(tokens, pool=2):
    """Mirrors vae_nav.vc1_source.pool_grid (same math, kept inline so the
    probe can run before the implementation lands)."""
    t = np.asarray(tokens, np.float32).reshape(14, 14, 768)
    g = t.reshape(14 // pool, pool, 14 // pool, pool, 768).mean(axis=(1, 3))
    return g.reshape(-1).astype(np.float32)


def synthetic_frames(n, seed=0):
    """Noise frames plus a linear gradient, uint8 HWC, 120x160 (the env's
    camera size). No CARLA needed."""
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 256, (n, 120, 160, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:120, 0:160]
    grad = np.stack([(xx * 255 // 159), (yy * 255 // 119),
                     ((xx + yy) * 255 // 278)], axis=-1).astype(np.uint8)
    for i in range(n):
        if i % 2 == 1:                      # alternate noise / gradient
            frames[i] = grad
    return frames


def main():
    n_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 8

    import torch

    # --- load the model exactly as VcSource does -------------------------
    from hydra import compose, initialize
    import hydra
    with initialize(version_base=None):
        cfg = compose(config_name="model/vc1_vitb")
    model, embed_dim, transform, _metadata = hydra.utils.instantiate(
        cfg["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"classifier_feature = {model.classifier_feature!r}")
    if model.classifier_feature != "use_cls_token":
        print("BLOCKED: expected 'use_cls_token' -- the token extraction "
              "path (norm then slice [1:]) is wrong for this checkpoint.")
        sys.exit(2)

    from vae_nav.vc1_source import VcSource
    src = VcSource.__new__(VcSource)
    src._model = model
    src._transform = transform
    src.device = device

    # --- token recipe (vit.py forward_features, without handle_outcome) --
    def tokens_from(x):
        """Normed full token sequence (B, 197, 768); index 0 is the CLS
        token, [1:] the 196 patch tokens."""
        with torch.no_grad():
            B = x.shape[0]
            h = model.patch_embed(x)
            cls_tok = model.cls_token.expand(B, -1, -1) \
                + model.pos_embed[:, :1, :]
            h = torch.cat((cls_tok, h + model.pos_embed[:, 1:, :]), dim=1)
            h = model.blocks(h)
            h = model.norm(h)
            return h

    frames = synthetic_frames(n_frames)

    # --- the load-bearing check: the CLS token of the normed sequence IS
    # the model's own output. NOTE: the check must index the FULL (197,)
    # sequence at 0; slicing [1:] first (as the task brief's snippet does)
    # lands on patch token 0, not the CLS token.
    x = src._prepare(frames[0]).to(device)
    full = tokens_from(x)
    toks = full[:, 1:]
    print(f"tokens shape: {tuple(toks.shape)}")
    with torch.no_grad():
        ref = model(x)
    ok = torch.allclose(full[:, :1].squeeze(1), ref, atol=1e-4)
    diff = (full[:, :1].squeeze(1) - ref).abs().max().item()
    print(f"allclose(full[:, 0] (CLS), model(x), atol=1e-4): {ok} "
          f"(max abs diff {diff:.3e})")
    if not ok:
        print("BLOCKED: token extraction does not reproduce the CLS output.")
        sys.exit(3)

    # --- pool and time ----------------------------------------------------
    pooled = []
    for f in frames:
        full = tokens_from(src._prepare(f).to(device))
        t0 = time.perf_counter()
        g = pool_grid(full[0, 1:].cpu().numpy())
        dt_pool = (time.perf_counter() - t0) * 1000.0
        pooled.append(g)
    pooled = np.stack(pooled)
    print(f"pooled shape: {pooled.shape}  "
          f"(grid 7x7x768 = {pooled.shape[1]} dims)")

    # timing: mean over 50 repeats of (transform + forward + pool), and the
    # pool step alone (target < 10 ms).
    f0 = src._prepare(frames[0]).to(device)
    for _ in range(3):                          # warm-up
        tokens_from(f0)
    t0 = time.perf_counter()
    for _ in range(50):
        full = tokens_from(f0)
    dt_full = (time.perf_counter() - t0) / 50 * 1000.0
    tn = full[0, 1:].cpu().numpy()
    t0 = time.perf_counter()
    for _ in range(50):
        pool_grid(tn)
    dt_pool = (time.perf_counter() - t0) / 50 * 1000.0
    print(f"timing: tokens+pool {dt_full:.2f} ms/frame, pool alone "
          f"{dt_pool:.2f} ms/frame (target < 10 ms)")

    print(f"pooled stats: mean={pooled.mean():.4f} std={pooled.std():.4f} "
          f"min={pooled.min():.4f} max={pooled.max():.4f} "
          f"norm mean={np.linalg.norm(pooled, axis=1).mean():.3f}")
    print(f"per-frame pooled norms: "
          f"{np.round(np.linalg.norm(pooled, axis=1), 2).tolist()}")

    # --- optional cross-check against the wired backend -------------------
    readout = os.environ.get("VC1_READOUT", "cls").lower()
    if readout == "grid2":
        from vae_nav.vc1_source import pool_grid as wired_pool
        z = src.latent(frames[0])
        same = np.allclose(z, wired_pool(
            tokens_from(src._prepare(frames[0]).to(device))[0, 1:]
            .cpu().numpy()), atol=1e-5)
        print(f"cross-check VcSource.latent vs probe pool (READOUT=grid2): "
              f"{same}")
        if not same:
            print("BLOCKED: wired grid latent does not match the probe.")
            sys.exit(4)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(OUT_PATH, pooled=pooled, frames=frames)
    print(f"saved {pooled.shape} pooled vectors + frames -> {OUT_PATH}")


if __name__ == "__main__":
    main()