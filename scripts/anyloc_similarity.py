#!/usr/bin/env python3
"""Image similarity for the evaluation maps, the Habitat peer's recipe
(rig/rigbig.py, relayed 2026-11-06) on OUR single forward camera
(user 2026-11-06: "stick to current camera"; 2026-11-07: "running panorama
similarity").

DESCRIPTOR (AnyLoc, as theirs): DINOv2 ViT-B/14 (torch hub, cached), fp16,
224x224 ImageNet-normalised; last block, VALUE facet (qkv output, value slice),
all 256 patch tokens (CLS dropped), each L2-normalised -> 768-d. Vocabulary PER
TRACE: 32 clusters, spherical k-means (cosine, 30 iterations, seed 0) on 200k
patches from <= 2,000 seeded random frames. VLAD: hard assignment, residuals
summed per cluster, intra-normalised, flattened, global L2 -> 24,576-d. With ONE
view per frame the "panorama" score is that one view's cosine.

CHECKS (no driving):
  calibration   cosine of SAME-place pairs (< 2 m, heading < 30 deg), DIFFERENT
                places (> 10 m), OPPOSITE-lane pairs (2-8 m, heading > 150 deg);
                threshold = the 2% far-false point, recall of same-place there.
  localisation  each trace frame's top-1 map node by cosine: within 3 m and
                heading-aligned? (their multi-view figure: ~.76)
  coverage      share of trace frames whose best node scores >= the threshold.

    anyloc_similarity.py --trace checkpoints/trace_expert/trace_expert_Town02 \
        --map checkpoints/maps_expert/uniform/Town02.json --out logs/anyloc_Town02.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from vae_nav import image_map as im  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ValueFacet:
    """DINOv2 ViT-B/14, last block's value facet, all patches L2-normalised."""

    def __init__(self, dev):
        local = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
        self.m = (torch.hub.load(local, "dinov2_vitb14", source="local")   # cached: no download
                  if os.path.isdir(local) else
                  torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True))
        self.m = self.m.to(dev).half().eval()
        self.dev, self._v = dev, None
        blk = self.m.blocks[-1].attn

        def hook(_mod, _inp, out):
            B, N, C3 = out.shape
            C = C3 // 3
            self._v = out[:, :, 2 * C:]          # value slice
        blk.qkv.register_forward_hook(hook)

    @torch.no_grad()
    def patches(self, frames_u8):
        x = torch.as_tensor(np.asarray(frames_u8)).to(self.dev).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = ((x - MEAN.to(self.dev)) / STD.to(self.dev)).half()
        self.m(x)
        v = self._v[:, 1:, :].float()            # drop CLS -> (B, 256, 768)
        return F.normalize(v, dim=-1)


def spherical_kmeans(X, k=32, iters=30, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = (X @ C.T).argmax(1)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = F.normalize(X[m].sum(0), dim=0)
    return C


def vlad(P, C):
    """P (B, N, D) patches, C (k, D) -> (B, k*D) intra-normalised, L2."""
    a = (P @ C.T).argmax(-1)                     # (B, N)
    oh = F.one_hot(a, C.shape[0]).float()        # (B, N, k)
    res = torch.einsum("bnk,bnd->bkd", oh, P) - oh.sum(1)[..., None] * C[None]
    res = F.normalize(res, dim=-1)
    return F.normalize(res.flatten(1), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thin-m", type=float, default=1.0, help="trace frames kept every this many metres")
    ap.add_argument("--pairs", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    tr = im.load_trace(a.trace)
    keep = im.place_uniform(tr, a.thin_m)        # directional thinning: every lane kept
    pos, yaw = tr["pos"][keep], tr["yaw"][keep]
    frames = im.node_frames(tr, keep)            # (M, H, W, 3) uint8
    g = im.Graph.load(a.map)
    node_idx = np.array([n["idx"] for n in g.nodes])
    nframes = im.node_frames(tr, node_idx)
    npos = np.array([n["pos"] for n in g.nodes]); nyaw = np.array([n["yaw"] for n in g.nodes])
    print(f"[AnyLoc] trace {len(tr['pos'])} frames -> {len(keep)} kept at {a.thin_m:g} m; map {len(g.nodes)} nodes; {dev}", flush=True)

    enc = ValueFacet(dev)
    # vocabulary: 200k patches from <= 2000 seeded random frames of THIS trace
    vf = rng.choice(len(frames), size=min(2000, len(frames)), replace=False)
    pool = torch.cat([enc.patches(frames[vf[i:i + a.batch]]).reshape(-1, 768)
                      for i in range(0, len(vf), a.batch)])
    pool = pool[torch.randperm(len(pool), generator=torch.Generator().manual_seed(0))[:200000].to(pool.device)]
    C = spherical_kmeans(pool, 32, 30, 0)
    del pool

    def describe(fr):
        return torch.cat([vlad(enc.patches(fr[i:i + a.batch]), C).half()
                          for i in range(0, len(fr), a.batch)])
    D = describe(frames)                         # (M, 24576)
    DN = describe(nframes)
    print(f"[AnyLoc] descriptors {tuple(D.shape)} + nodes {tuple(DN.shape)}", flush=True)

    def wrapd(y1, y2):
        return np.degrees(np.abs((y1 - y2 + np.pi) % (2 * np.pi) - np.pi))

    # ---- calibration
    i = rng.integers(0, len(pos), a.pairs * 20); j = rng.integers(0, len(pos), a.pairs * 20)
    d = np.hypot(*(pos[i] - pos[j]).T); dy = wrapd(yaw[i], yaw[j])
    sets = {"same": (d < 2) & (dy < 30) & (i != j), "different": d > 10,
            "opposite_lane": (d >= 2) & (d <= 8) & (dy > 150)}
    sims = {}
    for k, m in sets.items():
        ii, jj = i[m][:a.pairs], j[m][:a.pairs]
        sims[k] = (D[ii].float() * D[jj].float()).sum(1).cpu().numpy() if len(ii) else np.array([])
    thr = float(np.percentile(sims["different"], 98))
    cal = {k: {"n": int(len(v)), "median": round(float(np.median(v)), 3) if len(v) else None,
               "p10": round(float(np.percentile(v, 10)), 3) if len(v) else None,
               "p90": round(float(np.percentile(v, 90)), 3) if len(v) else None,
               "above_threshold": round(float((v >= thr).mean()), 3) if len(v) else None}
           for k, v in sims.items()}

    # ---- localisation + coverage
    best, bsim = [], []
    for s in range(0, len(D), 1024):
        S = D[s:s + 1024].float() @ DN.float().T
        v, k = S.max(1)
        best.append(k.cpu().numpy()); bsim.append(v.cpu().numpy())
    best, bsim = np.concatenate(best), np.concatenate(bsim)
    dist = np.hypot(*(npos[best] - pos).T); ali = wrapd(nyaw[best], yaw) <= 60
    loc3 = float(((dist <= 3) & ali).mean()); loc8 = float(((dist <= 8) & ali).mean())
    cov = float((bsim >= thr).mean())
    res = {"trace": a.trace, "map": a.map, "frames": int(len(pos)), "nodes": int(len(g.nodes)),
           "threshold_2pct_far_false": round(thr, 3), "calibration": cal,
           "image_localisation_top1_within_3m_aligned": round(loc3, 3),
           "image_localisation_top1_within_8m_aligned": round(loc8, 3),
           "localisation_error_m_median": round(float(np.median(dist)), 2),
           "coverage_frames_with_a_node_above_threshold": round(cov, 3)}
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
