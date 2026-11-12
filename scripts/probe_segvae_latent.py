#!/usr/bin/env python3
"""Lane-position probe on a seg-VAE's 512-d latent (mu), directly comparable to
the VC-1 results of scripts/perception_probe.py: the same probe frames (the
exact 120x120 RGB the policy saw, v27@225k, Town05), the same row filter
(off-junction, |heading err| < 90 deg), the same targets and probes.

usage: probe_segvae_latent.py CKPT PROBE.npz TRACE.qtrace.jsonl
VC-1 reference (2026-09-28): |lateral| R2 pyramid 0.41 / grid 0.65;
lateral |lat|<1 m 0.38 / 0.56; heading 0.81 / 0.81 (best of linear, nn).
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from perception_probe import linear_probe, nn_probe  # noqa: E402


def main():
    ck, npz, trace = sys.argv[1:4]
    import jax
    import jax.numpy as jnp
    from vae_nav import seg_vae as sv
    c = pickle.load(open(ck, "rb"))
    m = sv.SegVAE(latent_dim=c["latent_dim"], dec_start=c.get("dec_start", 15),
                  skips=c.get("skips", False))
    enc = jax.jit(lambda x: m.apply(c["params"], x, method=sv.SegVAE.encode))
    z = np.load(npz)
    rows = [json.loads(l) for l in open(trace)]
    groups = np.array([r["ep"] for r in rows])
    lane = z["lane"]
    keep = (~np.isnan(lane).any(1)) & (lane[:, 3] == 0) & (np.abs(lane[:, 1]) < np.pi / 2)
    rgb = z["rgb"][keep]
    mu = np.concatenate([np.asarray(enc(jnp.asarray(rgb[i:i + 256].astype(np.float32) / 255.0)))
                         for i in range(0, len(rgb), 256)])
    L = lane[keep].astype(np.float32)
    g = groups[keep]
    lat, head = L[:, 0], L[:, 1]
    inner = np.abs(lat) < 1.0
    tasks = [("|lateral| m", np.abs(lat)[:, None], np.ones(len(L), bool)),
             ("lateral, |lat|<1 m", lat[:, None], inner),
             ("heading err deg", np.degrees(head)[:, None], np.ones(len(L), bool))]
    print(f"{os.path.basename(os.path.dirname(ck))} step {c['step']}: latent {mu.shape[1]}-d, "
          f"{len(mu)} frames, {len(np.unique(g))} episodes")
    res = {}
    for tname, Y, msk in tasks:
        lr2, lmae = linear_probe(mu[msk], Y[msk], g[msk])
        nr2, nmae = nn_probe(mu[msk], Y[msk], g[msk], conv=False)
        res[tname] = dict(linear_r2=float(lr2[0]), nn_r2=float(nr2[0]),
                          linear_mae=float(lmae[0]), nn_mae=float(nmae[0]))
        print(f"   {tname:20s} linear R2 {lr2[0]:6.3f} MAE {lmae[0]:.3f} | "
              f"nn R2 {nr2[0]:6.3f} MAE {nmae[0]:.3f}")
    json.dump(res, open(ck.replace(".pkl", "_lane_probe.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
