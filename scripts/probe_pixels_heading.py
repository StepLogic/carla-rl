#!/usr/bin/env python3
"""Step 0a control (plan 2026-11-09): is heading error in the PIXELS c0 sees?

c0's camera latent reads lateral offset (R2 0.70 linear / 0.81 MLP) but not heading
error (0.05 / 0.05), on clean-label frames (probe_c0_features.clean_label_mask). Its
GRU state reads heading partly (0.31 / 0.36). Before blaming the encoder: train a
small CNN directly on the same 128 px frames to predict heading error and offset,
(a) from ONE frame, (b) from TWO consecutive frames (k-1, k) stacked. High (a) =
the frame shows heading and c0's encoder drops it; low (a) but high (b) = heading is
only visible as motion, which a memory must integrate.

Same frames, labels and episode split as the probes (Town05; test = fold 0 of the
5-fold episode assignment, seed 0). Diagnostic only.

usage: python scripts/probe_pixels_heading.py CACHE DATA_ROOT OUT [--epochs 12] [--device cuda]
  CACHE = features_Town05.npz from scripts/probe_c0_features.py (labels + frame index)
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "src"), os.path.join(REPO, "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from probe_c0_features import _folds, clean_label_mask, r2  # noqa: E402


class SmallCNN(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        def blk(i, o, k, s):
            return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2), nn.GroupNorm(8, o), nn.ReLU())
        self.net = nn.Sequential(blk(in_ch, 32, 5, 2), blk(32, 64, 3, 2), blk(64, 128, 3, 2),
                                 blk(128, 128, 3, 2), nn.Flatten(),
                                 nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, x):
        return self.net(x)


def load_frames(C, root, town, rows, stack):
    """uint8 (N, C, 128, 128) for frames `rows` (cache frame indices); with stack,
    the previous frame of the same episode is channel-stacked first."""
    out = np.zeros((len(rows), 6 if stack else 3, 128, 128), np.uint8)
    ep_of = C["f_ep"][rows]
    k_of = C["f_k"][rows]
    order = np.argsort(ep_of, kind="stable")
    cur_ep, frames = None, None
    for j in order:
        e = int(ep_of[j])
        if e != cur_ep:
            frames = np.load(os.path.join(root, town, str(C["e_path"][e])))["frames"]
            cur_ep = e
        k = int(k_of[j])
        f = frames[k].transpose(2, 0, 1)
        if stack:
            out[j, :3] = frames[k - 1].transpose(2, 0, 1)
            out[j, 3:] = f
        else:
            out[j] = f
    return out


def run(X, Y, tr, va, te, dev, epochs, seed=0, bs=128):
    torch.manual_seed(seed)
    ym, ys = Y[tr].mean(0), Y[tr].std(0)
    net = SmallCNN(X.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Yn = torch.as_tensor((Y - ym) / ys, dtype=torch.float32)

    def batches(idx, shuffle):
        idx = np.random.permutation(idx) if shuffle else idx
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            yield torch.as_tensor(X[b], device=dev).float() / 255.0, Yn[b].to(dev), b

    def predict(idx):
        net.eval()
        out = np.zeros((len(idx), 2), np.float32)
        with torch.no_grad():
            for i in range(0, len(idx), 512):
                b = idx[i:i + 512]
                out[i:i + len(b)] = net(torch.as_tensor(X[b], device=dev).float() / 255.0).cpu().numpy()
        net.train()
        return out * ys + ym

    best, best_state = -1e9, None
    for ep in range(epochs):
        t0 = time.time()
        for xb, yb, _ in batches(tr, True):
            loss = ((net(xb) - yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        pv = predict(va)
        score = np.mean([r2(Y[va, j], pv[:, j]) for j in range(2)])
        print(f"  epoch {ep + 1}: val R2 heading {r2(Y[va, 0], pv[:, 0]):.3f} offset "
              f"{r2(Y[va, 1], pv[:, 1]):.3f} ({time.time() - t0:.0f} s)", flush=True)
        if score > best:
            best, best_state = score, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    pt = predict(te)
    return {"heading_err": round(r2(Y[te, 0], pt[:, 0]), 3), "offset": round(r2(Y[te, 1], pt[:, 1]), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--town", default="Town05")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    C = dict(np.load(args.cache, allow_pickle=False))
    rows = np.flatnonzero(clean_label_mask(C) & (C["f_k"] >= 1))
    groups = C["f_ep"][rows]
    fold, rng = _folds(groups, 5, 0)
    te = np.flatnonzero(fold == 0)
    rest = np.flatnonzero(fold != 0)
    ug = np.unique(groups[rest])
    vg = set(rng.choice(ug, max(1, len(ug) // 10), replace=False).tolist())
    is_va = np.array([g in vg for g in groups[rest]])
    tr, va = rest[~is_va], rest[is_va]
    Y = np.stack([C["f_heading_err"][rows], C["f_offset"][rows]], -1).astype(np.float64)
    res = {"frames": int(len(rows)), "train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
           "label_sd": {"heading_err": round(float(Y[:, 0].std()), 3),
                        "offset": round(float(Y[:, 1].std()), 3)}}
    for name, stack in (("one_frame", False), ("two_frames", True)):
        t0 = time.time()
        X = load_frames(C, args.root, args.town, rows, stack)
        print(f"[pixels] {name}: {X.shape} loaded ({time.time() - t0:.0f} s)", flush=True)
        res[name] = run(X, Y, tr, va, te, args.device, args.epochs)
        print(f"[pixels] {name}: test R2 {res[name]}", flush=True)
        del X
        with open(os.path.join(args.out, "pixels_heading.json"), "w") as fh:
            json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
