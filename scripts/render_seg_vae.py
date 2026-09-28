#!/usr/bin/env python3
"""Render seg-VAE predictions on validation frames: RGB | ground truth |
prediction from the unmasked input | prediction from a 75%-masked input.
usage: render_seg_vae.py CKPT DATA_DIR OUT_PNG [n_rows]"""
import glob
import os
import pickle
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
PAL = np.array([[0, 0, 0], [128, 64, 128], [255, 255, 0], [244, 35, 232], [0, 0, 142],
                [220, 20, 60], [70, 70, 70], [107, 142, 35], [70, 130, 180]], np.uint8)


def main():
    ck, data, out = sys.argv[1:4]
    rows = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    import jax
    import jax.numpy as jnp
    from vae_nav import seg_vae as sv
    c = pickle.load(open(ck, "rb"))
    m = sv.SegVAE(latent_dim=c["latent_dim"])
    shards = sorted(glob.glob(os.path.join(data, "shard_*.npz")))
    rng = np.random.default_rng(1)
    picks = []
    for f in rng.choice(shards, rows, replace=len(shards) < rows):
        z = np.load(f)
        n = len(z["rgb"]); i = int(rng.integers(int(n * 0.95), n))   # validation tail
        picks.append((os.path.basename(f)[6:-4], z["rgb"][i], z["label"][i]))
    x = jnp.asarray(np.stack([p[1] for p in picks]).astype(np.float32) / 255.0)
    xm = sv.patch_mask(jax.random.PRNGKey(0), x)

    def pred(inp):
        mu = m.apply(c["params"], inp, method=lambda mm, a: mm.encoder(a))[0]
        return np.asarray(m.apply(c["params"], mu, method=lambda mm, zz: mm.decoder(zz)).argmax(-1))
    p_full, p_mask = pred(x), pred(xm)
    S = 200
    W = Image.new("RGB", (S * 5, S * rows + 24), (30, 30, 30))
    d = ImageDraw.Draw(W)
    for j, t in enumerate(("RGB", "ground truth", "pred (full input)", "masked input", "pred (75% masked)")):
        d.text((j * S + 6, 6), t, fill=(255, 255, 255))
    for r, (name, rgb, lab) in enumerate(picks):
        gt = lab[::2, ::2]
        tiles = [rgb, PAL[gt], PAL[p_full[r]], (np.asarray(xm[r]) * 255).astype(np.uint8), PAL[p_mask[r]]]
        for j, t in enumerate(tiles):
            W.paste(Image.fromarray(t).resize((S, S), Image.NEAREST), (j * S, 24 + r * S))
        d.text((6, 24 + r * S + 4), name, fill=(255, 255, 0))
    W.save(out)
    print(f"step {c['step']} -> {out}")


if __name__ == "__main__":
    main()
