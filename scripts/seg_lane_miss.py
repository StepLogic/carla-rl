#!/usr/bin/env python3
"""How often does a seg-VAE miss lane lines entirely? usage: seg_lane_miss.py CKPT DATA_DIR
Per validation frame (last 5% of each shard) with >0.5% lane pixels in the ground
truth: missed = prediction has zero lane pixels. Split by shard and by lane share."""
import collections, glob, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import jax, jax.numpy as jnp
from vae_nav import seg_vae as sv
c = pickle.load(open(sys.argv[1], "rb"))
m = sv.SegVAE(latent_dim=c["latent_dim"], dec_start=c.get("dec_start", 15))
pred = jax.jit(lambda x: m.apply(c["params"], m.apply(c["params"], x, method=lambda mm, a: mm.encoder(a))[0],
                                 method=lambda mm, z: mm.decoder(z)).argmax(-1))
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[2], "shard_*.npz"))):
    z = np.load(f); n = len(z["rgb"]); s = int(n * 0.95)
    rgb, gt = z["rgb"][s:], z["label"][s:, ::2, ::2]
    P = np.concatenate([np.asarray(pred(jnp.asarray(rgb[i:i + 128].astype(np.float32) / 255)))
                        for i in range(0, len(rgb), 128)])
    for k in range(len(rgb)):
        rows.append((os.path.basename(f)[6:-4], (gt[k] == 2).mean(), (P[k] == 2).mean()))
g = np.array([r[1] for r in rows]); p = np.array([r[2] for r in rows]); has = g > 0.005
print(f"step {c['step']} dec_start {c.get('dec_start', 15)}: lane missed in {((p == 0) & has).sum()}/{has.sum()} "
      f"= {((p == 0) & has).sum() / has.sum():.1%} of lane frames")
d = collections.defaultdict(lambda: [0, 0])
for r, h, pp in zip(rows, has, p):
    if h: d[r[0]][0] += 1; d[r[0]][1] += int(pp == 0)
print("  by shard: " + "  ".join(f"{k} {v[1] / v[0]:.0%}" for k, v in sorted(d.items())))
for lo, hi in ((0.005, 0.01), (0.01, 0.02), (0.02, 0.04), (0.04, 1)):
    sel = has & (g >= lo) & (g < hi)
    print(f"  lane share {lo:.3f}-{hi:.3f}: n={sel.sum()} missed {((p == 0) & sel).sum() / max(sel.sum(), 1):.0%}")
