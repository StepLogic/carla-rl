#!/usr/bin/env python3
"""Does DEBiT-TINY carry the goal on CARLA road imagery, as DEBiT-B does?

WHY THIS EXISTS. Seven training runs have failed to learn the image-goal task
while the explore half learned to drive. Every fix so far has been to the
reward. Nobody checked the premise: that the encoder's output tells the policy
where the goal is. The published numbers for this (distance R2 0.50-0.56,
bearing 0.83-0.85 on CARLA pairs) were measured with DEBiT-**B**; the arm runs
DEBiT-**TINY**, chosen later. If tiny is blind here, no reward can fix it.

THE TEST, identical for both encoders so the comparison is clean:
  pairs      (view i, goal j) from the SAME trace segment, j ahead of i by
             5-25 m, which is the band the arm trains on
  labels     distance, and bearing sin/cos in the car's own frame
  probe      ridge regression, SPLIT BY SEGMENT -- adjacent samples are ~2 m
             apart and near-duplicates, so a random split leaks and everything
             scores well
  control    predicting the training mean. R2 <= 0 means the encoder told the
             probe nothing.

usage: probe_debit_tiny_vs_base.py TRACE.pkl [--pairs N]
"""
import argparse
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

GOAL_MIN_M, GOAL_MAX_M = 5.0, 25.0
TINY = "/home/robotlab/scratch/debit/out/ckpt/hab_bl/imgnav/debit_tiny.pth"
BASE = os.path.join(ROOT, "dependencies", "debit_weights", "debit_base.pth")


def build_pairs(pos, yaw, seg, rng, n_want):
    """(i, j) with j ahead of i in the same segment, 5-25 m away, + labels."""
    I, J = [], []
    for s in np.unique(seg):
        idx = np.flatnonzero(seg == s)
        if len(idx) < 3:
            continue
        p = pos[idx]
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        for a in range(len(idx)):
            lo = np.searchsorted(d, d[a] + GOAL_MIN_M)
            hi = np.searchsorted(d, d[a] + GOAL_MAX_M)
            if hi > lo:
                b = int(rng.integers(lo, hi))
                I.append(idx[a])
                J.append(idx[b])
    I, J = np.array(I), np.array(J)
    if len(I) > n_want:                      # subsample, keeping segments mixed
        k = rng.choice(len(I), n_want, replace=False)
        I, J = I[k], J[k]
    rel = pos[J] - pos[I]
    c, s_ = np.cos(-yaw[I]), np.sin(-yaw[I])
    x = rel[:, 0] * c - rel[:, 1] * s_       # into the car's own frame
    y = rel[:, 0] * s_ + rel[:, 1] * c
    dist = np.hypot(x, y)
    return I, J, dict(distance=dist, bearing_sin=y / dist, bearing_cos=x / dist)


def ridge_r2(X, y, seg, rng, lam=10.0):
    """R2 on held-out SEGMENTS, with the predict-the-mean control."""
    segs = np.unique(seg)
    rng.shuffle(segs)
    te_seg = set(segs[: max(1, len(segs) // 3)].tolist())
    te = np.array([s in te_seg for s in seg])
    tr = ~te
    if tr.sum() < 10 or te.sum() < 10:
        return float("nan"), float("nan")
    X1 = np.c_[X, np.ones(len(X))]
    A = X1[tr].T @ X1[tr] + lam * np.eye(X1.shape[1])
    A[-1, -1] -= lam
    w = np.linalg.solve(A, X1[tr].T @ y[tr])
    pred = X1[te] @ w
    ss = float(np.sum((y[te] - pred) ** 2))
    tot = float(np.sum((y[te] - y[tr].mean()) ** 2))
    return (1.0 - ss / tot if tot > 0 else float("nan")), float(np.sqrt(ss / te.sum()))


def encode(ckpt, croco, frames_i, frames_j, device, batch=32):
    from torch_rl.debit_encoder import DebitTinyEncoder
    enc = DebitTinyEncoder(ckpt, device=device, half=False, croco=croco)
    out = []
    for k in range(0, len(frames_i), batch):
        out.append(enc.encode_batch(frames_i[k:k + batch], frames_j[k:k + batch]))
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--pairs", type=int, default=1200)
    a = ap.parse_args()

    import torch
    from torch_rl.debit_encoder import BASE_CROCO, TINY_CROCO
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr = pickle.load(open(a.trace, "rb"))
    pos = np.asarray(tr["positions"], np.float64)[:, :2]
    yaw = np.asarray(tr["yaws"], np.float64)
    seg = np.asarray(tr.get("segments", tr.get("segment_ids")), int)
    rgb = tr["rgb"]
    rng = np.random.default_rng(0)
    I, J, labels = build_pairs(pos, yaw, seg, rng, a.pairs)
    print(f"[{os.path.basename(a.trace)}] {len(I)} pairs, {len(np.unique(seg[I]))} "
          f"segments, device {dev}", flush=True)
    print(f"  distance {labels['distance'].mean():.1f} m "
          f"(sd {labels['distance'].std():.1f})", flush=True)

    fi = np.stack([np.asarray(rgb[i]) for i in I])
    fj = np.stack([np.asarray(rgb[j]) for j in J])

    for name, ckpt, croco in (("DEBiT-TINY (what the arm runs)", TINY, TINY_CROCO),
                              ("DEBiT-B   (what was probed)", BASE, BASE_CROCO)):
        if not os.path.exists(ckpt):
            print(f"\n{name}: checkpoint missing, skipped")
            continue
        X = encode(ckpt, croco, fi, fj, dev)
        print(f"\n{name}  -> {X.shape[1]}-d")
        # Scored by the EARLIER probe's own function (standardised ridge + a
        # small MLP, split by segment), so these numbers sit beside the
        # published DEBiT-B ones without a methods difference between them.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from probe_debit_goal import probe
        Y = np.stack([labels["distance"], labels["bearing_sin"],
                      labels["bearing_cos"]], 1)
        probe(X, Y, seg[I], name.split()[0])


if __name__ == "__main__":
    main()
