#!/usr/bin/env python3
"""Does the VC-1 PATCH GRID carry lane position that the POOLED pyramid loses?

User 2026-09-28: "can we convolve patch tokens? - or what can we fix to get
better success rates". Every run plateaus with ~43% lane crossings and ~30%
off-road; the actor sees VC-1's 'pyramid' readout -- the mean over all 196
patch tokens plus the mean of each 7x7 quadrant -- which averages away where
a lane marking sits inside each quadrant.

Input: an EVAL_PROBE_RGB=1 probe .npz (the exact frames VC-1 encoded + true
lane geometry from the map) and its .qtrace.jsonl (episode ids).
Per frame, one VC-1 forward pass gives the 14x14 patch tokens; both readouts
are pooled from the SAME tokens:
    pyramid  (3840,)      what v25-v27 trained on
    grid2    (7,7,768)    2x2-pooled patch grid (VC1_READOUT=grid2)
Targets: lateral offset from lane centre (m) and heading error to the lane
(rad). Rows inside junctions, or with |heading error| > 90 deg (the map
projected onto an opposite-direction lane), are dropped.
Probes, split by EPISODE so no episode is in both fit and score:
    linear   ridge, 5-fold grouped CV            -> R^2, MAE
    small nn one 80/20 episode split, 300 epochs  -> R^2, MAE
             pyramid: MLP 3840-256-256-2 ; grid2: conv3x3x128 s1 -> conv3x3x64 s2
             -> flatten -> 256 -> 2 (roughly the agent's conv compressor)
usage: perception_probe.py PROBE.npz TRACE.qtrace.jsonl
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.environ.setdefault("VC1_READOUT", "pyramid")


def vc1_tokens(frames, batch=64):
    import torch
    from vae_nav.vc1_source import VcSource
    src = VcSource()
    m = src._model
    out = np.empty((len(frames), 196, 768), np.float16)
    with torch.no_grad():
        for i in range(0, len(frames), batch):
            xs = torch.cat([src._prepare(f) for f in frames[i:i + batch]]).to(src.device)
            h = m.patch_embed(xs)
            cls = m.cls_token.expand(len(xs), -1, -1) + m.pos_embed[:, :1, :]
            h = torch.cat((cls, h + m.pos_embed[:, 1:, :]), dim=1)
            h = m.norm(m.blocks(h))
            out[i:i + len(xs)] = h[:, 1:].cpu().numpy().astype(np.float16)
    return out


def readouts(tok):
    t = tok.astype(np.float32).reshape(len(tok), 14, 14, 768)
    pyr = np.concatenate([t.mean((1, 2)), t[:, :7, :7].mean((1, 2)), t[:, :7, 7:].mean((1, 2)),
                          t[:, 7:, :7].mean((1, 2)), t[:, 7:, 7:].mean((1, 2))], axis=1)
    grid = t.reshape(len(t), 7, 2, 7, 2, 768).mean((2, 4))           # (N,7,7,768)
    return pyr, grid


def r2(y, p):
    return 1.0 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def linear_probe(X, Y, groups):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    pred = np.zeros_like(Y)
    for tr, te in GroupKFold(5).split(X, Y, groups):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=1000.0).fit(sc.transform(X[tr]), Y[tr])
        pred[te] = m.predict(sc.transform(X[te])).reshape(len(te), -1)
    return r2(Y, pred), np.abs(Y - pred).mean(0)


def nn_probe(X, Y, groups, conv):
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eps = np.unique(groups)
    rng = np.random.default_rng(0)
    te_eps = set(rng.choice(eps, max(1, len(eps) // 5), replace=False).tolist())
    te = np.array([g in te_eps for g in groups])
    mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-6
    ym, ys = Y[~te].mean(0), Y[~te].std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev)
    Yt = torch.tensor((Y - ym) / ys, dtype=torch.float32, device=dev)
    if conv:
        Xt = Xt.permute(0, 3, 1, 2)                                    # N,768,7,7
        net = nn.Sequential(nn.Conv2d(768, 128, 3, padding=1), nn.ReLU(),
                            nn.Conv2d(128, 64, 3, stride=2), nn.ReLU(), nn.Flatten(),
                            nn.Linear(64 * 3 * 3, 256), nn.ReLU(), nn.Linear(256, Y.shape[1]))
    else:
        net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.ReLU(),
                            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, Y.shape[1]))
    net = net.to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    tri = np.where(~te)[0]
    for ep in range(300):
        rng.shuffle(tri)
        for b in range(0, len(tri), 256):
            idx = torch.tensor(tri[b:b + 256], device=dev)
            loss = ((net(Xt[idx]) - Yt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = net(Xt[torch.tensor(np.where(te)[0], device=dev)]).cpu().numpy() * ys + ym
    y = Y[te]
    return r2(y, p), np.abs(y - p).mean(0)


def main():
    z = np.load(sys.argv[1])
    rows = [json.loads(l) for l in open(sys.argv[2])]
    groups = np.array([r["ep"] for r in rows])
    lane = z["lane"]
    keep = (~np.isnan(lane).any(1)) & (lane[:, 3] == 0) & (np.abs(lane[:, 1]) < np.pi / 2)
    print(f"frames {len(lane)}  kept {int(keep.sum())} (off-junction, |heading err| < 90 deg)  "
          f"episodes {len(np.unique(groups[keep]))}")
    cache = sys.argv[1].replace(".probe.npz", ".readouts.npz")
    if os.path.exists(cache):
        c = np.load(cache)
        pyr, grid = c["pyr"].astype(np.float32), c["grid"].astype(np.float32)
    else:
        pyr, grid = readouts(vc1_tokens(z["rgb"][keep]))
        np.savez(cache, pyr=pyr.astype(np.float16), grid=grid.astype(np.float16))
    L = lane[keep].astype(np.float32)
    g = groups[keep]
    lat, head = L[:, 0], L[:, 1]
    # THE SIGNED LATERAL LABEL IS AMBIGUOUS AT LANE BOUNDARIES: offset is taken
    # from the NEAREST lane centre, so +1.7 m in lane A and -1.8 m in lane B
    # are nearly the same image with opposite labels. Two unambiguous targets:
    # |lateral| (distance from the lane centre) on all rows, and signed lateral
    # only well inside the lane (|lat| < 1 m).
    inner = np.abs(lat) < 1.0
    tasks = [("|lateral| m", np.abs(lat)[:, None], np.ones(len(L), bool)),
             ("lateral, |lat|<1 m", lat[:, None], inner),
             ("heading err deg", np.degrees(head)[:, None], np.ones(len(L), bool))]
    res = {}
    for tname, Y, m in tasks:
        print(f"== {tname}: rows {int(m.sum())}  target sd {Y[m].std():.3f}")
        for name, X, conv in (("pyramid", pyr, False), ("grid2", grid, True)):
            lr2, lmae = linear_probe(X[m].reshape(int(m.sum()), -1), Y[m], g[m])
            nr2, nmae = nn_probe(X[m], Y[m], g[m], conv)
            res[f"{tname}|{name}"] = dict(linear_r2=float(lr2[0]), linear_mae=float(lmae[0]),
                                         nn_r2=float(nr2[0]), nn_mae=float(nmae[0]))
            print(f"   {name:8s} linear R2 {lr2[0]:6.3f} MAE {lmae[0]:.3f} | "
                  f"nn R2 {nr2[0]:6.3f} MAE {nmae[0]:.3f}")
    json.dump(res, open(sys.argv[1].replace(".probe.npz", ".perception_probe.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
