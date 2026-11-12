#!/usr/bin/env python3
"""A ViNT-style TEMPORAL DISTANCE function for the SGM map baseline (user
2026-11-07: "add sgm baseline - train distance function like in vint"; memory
sgm-vint-distance-baseline).

Data: c0's expert demonstrations (checkpoints/il_data/<Town>/ep_*.npz), the
TRAINING towns only (Town01, Town04, Town05, Town10HD) -- never Town02 / Town03.
Model: FROZEN DINOv2 ViT-B/14 (the class token of each frame) + an MLP head on
the pair [a, b, a*b, |a-b|] that CLASSIFIES the temporal distance, as ViNT's
distance head does: classes 0..MAX frames at a STRIDE-step spacing (0.1 s per
step), plus a FAR class for pairs from different episodes. Distance read out =
the expected class (FAR counted as MAX + 5).
10% of episodes are held out to report accuracy. Writes <out>/head.pt + meta.json.

    train_vint_distance.py --out checkpoints/sgm_distance
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

TRAIN_TOWNS = ("Town01", "Town04", "Town05", "Town10HD")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def dino(dev):
    local = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
    m = (torch.hub.load(local, "dinov2_vitb14", source="local") if os.path.isdir(local)
         else torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True))
    return m.to(dev).half().eval()


@torch.no_grad()
def embed(model, frames_u8, dev, batch=256):
    out = []
    for i in range(0, len(frames_u8), batch):
        x = torch.as_tensor(np.asarray(frames_u8[i:i + batch])).to(dev).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = ((x - MEAN.to(dev)) / STD.to(dev)).half()
        out.append(F.normalize(model(x).float(), dim=-1).cpu())     # class token, L2
    return torch.cat(out)


class DistanceHead(nn.Module):
    def __init__(self, d=768, n_classes=22, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4 * d, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, n_classes))

    def forward(self, a, b):
        return self.net(torch.cat([a, b, a * b, (a - b).abs()], -1))


def expected_distance(logits, max_cls, far_value):
    p = logits.softmax(-1)
    vals = torch.arange(p.shape[-1], device=p.device, dtype=p.dtype)
    vals[-1] = far_value
    return (p * vals).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="checkpoints/il_data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-cls", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--neg-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda"
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    files = sorted(f for t in TRAIN_TOWNS for f in glob.glob(os.path.join(a.data, t, "ep_*.npz")))
    assert files and not any(("Town02" in f or "Town03" in f) for f in files)
    cache = os.path.join(a.out, "features.npz")
    if os.path.exists(cache):
        z = np.load(cache); feats = torch.as_tensor(z["feats"]); ep = z["ep"]; t = z["t"]
    else:
        model = dino(dev); fs, eps, ts = [], [], []
        t0 = time.time()
        for k, f in enumerate(files):
            with np.load(f) as z:
                fr = z["frames"][:: a.stride]
            fs.append(embed(model, fr, dev)); eps.append(np.full(len(fr), k)); ts.append(np.arange(len(fr)))
            if k % 200 == 0:
                print(f"[Dist] featurised {k}/{len(files)} episodes, {time.time() - t0:.0f}s", flush=True)
        feats = torch.cat(fs); ep = np.concatenate(eps); t = np.concatenate(ts)
        np.savez(cache, feats=feats.numpy().astype(np.float16), ep=ep, t=t)
        del model; torch.cuda.empty_cache()
    feats = feats.float()
    n_ep = int(ep.max()) + 1
    val_eps = set(rng.choice(n_ep, max(1, int(a.val_frac * n_ep)), replace=False).tolist())
    is_val = np.array([e in val_eps for e in ep])
    print(f"[Dist] {len(files)} episodes ({len(val_eps)} held out), {len(ep)} frames at stride "
          f"{a.stride}; classes 0..{a.max_cls} + FAR", flush=True)
    starts = {}
    for i, e in enumerate(ep):
        starts.setdefault(int(e), [i, i])[1] = i
    FAR = a.max_cls + 1

    def sample(n, val):
        pool = np.flatnonzero(is_val == val)
        i = rng.choice(pool, n)
        y = rng.integers(0, a.max_cls + 1, n)
        j = i + y
        ok = np.array([jj <= starts[int(ep[ii])][1] for ii, jj in zip(i, j)])
        i, j, y = i[ok], j[ok], y[ok]
        n_neg = int(a.neg_frac * len(i))
        ni = rng.choice(pool, n_neg); nj = rng.choice(pool, n_neg)
        keep = ep[ni] != ep[nj]
        I = np.r_[i, ni[keep]]; J = np.r_[j, nj[keep]]; Y = np.r_[y, np.full(keep.sum(), FAR)]
        sw = rng.random(len(I)) < 0.5                          # order-free: distance is symmetric here
        I2, J2 = np.where(sw, J, I), np.where(sw, I, J)
        return (feats[I2].to(dev), feats[J2].to(dev), torch.as_tensor(Y, device=dev, dtype=torch.long))

    head = DistanceHead(feats.shape[1], FAR + 1).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    for s in range(a.steps + 1):
        A, B, Y = sample(a.batch, False)
        loss = F.cross_entropy(head(A, B), Y)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 2000 == 0:
            with torch.no_grad():
                A, B, Y = sample(8192, True)
                lg = head(A, B); pred = lg.argmax(-1)
                near = Y < FAR
                acc = (pred == Y).float().mean().item()
                within1 = ((pred - Y).abs() <= 1)[near].float().mean().item()
                far_ok = (pred[~near] == FAR).float().mean().item() if (~near).any() else float("nan")
                ed = expected_distance(lg, a.max_cls, a.max_cls + 5)
                mae = (ed[near] - Y[near].float()).abs().mean().item()
            print(f"[Dist] step {s} loss {loss.item():.3f} | VAL acc {acc:.3f}, within +-1 class "
                  f"{within1:.3f}, FAR recognised {far_ok:.3f}, expected-distance MAE {mae:.2f} frames",
                  flush=True)
    torch.save(head.state_dict(), os.path.join(a.out, "head.pt"))
    json.dump({"train_towns": TRAIN_TOWNS, "episodes": len(files), "val_episodes": len(val_eps),
               "stride_steps": a.stride, "max_cls": a.max_cls, "far_value": a.max_cls + 5,
               "encoder": "dinov2_vitb14 class token, L2-normalised, frozen",
               "head": "MLP [a,b,a*b,|a-b|] -> 512 -> 512 -> classes",
               "val_acc": acc, "val_within1": within1, "val_far_ok": far_ok, "val_mae_frames": mae},
              open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"[DistDone] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
