"""AnyLoc image descriptors, the Habitat peer's recipe (rig/rigbig.py, relayed
2026-11-06), for pose-free localisation on the evaluation maps.

DINOv2 ViT-B/14 (torch hub, cached), fp16, 224x224 ImageNet-normalised; last
block, VALUE facet (qkv output, value slice), all 256 patch tokens (CLS dropped),
each L2-normalised -> 768-d. VLAD over a 32-cluster vocabulary: hard assignment,
residuals summed per cluster, intra-normalised, flattened, global L2 ->
24,576-d. The vocabulary is built PER TRACE (spherical k-means, 30 iterations,
seed 0, 200k patches from <= 2,000 seeded random frames) and stored with the map
(scripts/build_map_descriptors.py), so the eval uses the map's own vocabulary.

scripts/anyloc_similarity.py and build_image_sim_maps.py carry the same code;
this module is the importable copy the eval loads.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ValueFacet:
    """DINOv2 ViT-B/14, last block's value facet, all patches L2-normalised."""

    def __init__(self, dev):
        local = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
        self.m = (torch.hub.load(local, "dinov2_vitb14", source="local")
                  if os.path.isdir(local) else
                  torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True))
        self.m = self.m.to(dev).half().eval()
        self.dev, self._v = dev, None

        def hook(_mod, _inp, out):
            c = out.shape[-1] // 3
            self._v = out[:, :, 2 * c:]
        self.m.blocks[-1].attn.qkv.register_forward_hook(hook)

    @torch.no_grad()
    def patches(self, frames_u8):
        x = torch.as_tensor(np.asarray(frames_u8)).to(self.dev).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = ((x - MEAN.to(self.dev)) / STD.to(self.dev)).half()
        self.m(x)
        return F.normalize(self._v[:, 1:, :].float(), dim=-1)


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
    """P (B, N, D) patches, C (k, D) -> (B, k*D), intra-normalised, L2."""
    a = (P @ C.T).argmax(-1)
    oh = F.one_hot(a, C.shape[0]).float()
    res = torch.einsum("bnk,bnd->bkd", oh, P) - oh.sum(1)[..., None] * C[None]
    res = F.normalize(res, dim=-1)
    return F.normalize(res.flatten(1), dim=-1)


@torch.no_grad()
def describe(enc, C, frames_u8, batch=128):
    frames_u8 = np.asarray(frames_u8)
    return torch.cat([vlad(enc.patches(frames_u8[i:i + batch]), C)
                      for i in range(0, len(frames_u8), batch)])
