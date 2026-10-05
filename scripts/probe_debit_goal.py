#!/usr/bin/env python
"""Can a FROZEN DEBiT-B binocular encoder read the goal from a CARLA pair?

THE QUESTION. DEBiT works by finding visual correspondence between the current
view and the goal image; its weights were trained on indoor Habitat scenes with
goals a few metres away. Our goals are 5-25 m along a road. Before building a
training arm on it, measure whether its 3136-d output carries the one thing the
policy needs: where the goal is relative to the car.

THE TEST. From an explore trace (which stores the exact frame the policy saw at
a known true pose), build (observation, goal) pairs from the SAME segment with
the goal 5-25 m AHEAD along the driven path -- so the pair is reachable and
ordered, like a real image goal. Probe the encoder's output for:
    distance m        how far to the goal
    bearing sin/cos   which way, in the car's own frame
A probe that cannot read these means the encoder is blind to the goal on our
imagery, whatever it does indoors.

THE CONTROL IS THE POINT. A binocular encoder must beat simply concatenating
two MONOCULAR embeddings of the same two frames -- that baseline has all the
same pixels and no correspondence mechanism. We use the seg-VAE for it, the
encoder v29-v37 actually trained on. If DEBiT does not beat it, its
cross-attention is buying nothing here.

SPLIT BY SEGMENT, never by pair: adjacent samples in one segment are 2 m apart
and near-duplicates, so a random split leaks the test set into training and
every probe scores well.

usage: probe_debit_goal.py TRACE.pkl [TRACE2.pkl ...]
"""
import os
import pickle
import sys

import numpy as np
import torch

W = os.path.dirname(os.path.abspath(__file__))
DEBIT_CKPT = "/home/robotlab/scratch/carla-rl/dependencies/debit_weights/debit_base.pth"
SEGVAE_CKPT = ("/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl/"
               "results/seg_vae_v1_dec30/seg_vae.pkl")
# croco needs BOTH paths. Its repo root holds models/ directly, so `croco` is a
# namespace package and the repo's PARENT makes `croco.models` importable; but
# croco.py itself does `from models.blocks import ...`, an ABSOLUTE import that
# only resolves with the repo root on the path too. Either alone fails.
sys.path.insert(0, "/home/robotlab/scratch/carla-rl/dependencies")
sys.path.insert(0, "/home/robotlab/scratch/carla-rl/dependencies/croco")
sys.path.insert(0, os.path.join(W, "..", "src"))

GOAL_MIN_M, GOAL_MAX_M = 5.0, 25.0
# DEBiT-B, from debit/configs/debit/{debit_defaults,debit_base}.yaml.
CROCO_B = dict(img_size=112, patch_size=16, enc_embed_dim=384, enc_depth=12,
               enc_num_heads=6, dec_embed_dim=512, dec_depth=8,
               dec_num_heads=16, pos_embed="cosine")
BINOC_REPR = 3136          # 49 patches x 64 projected channels


# ----------------------------------------------------------------- pairs ---
def build_pairs(trace, rng):
    """[(i, j)] with j ahead of i in the same segment, 5-25 m apart, plus the
    relative-pose labels in i's own frame."""
    pos = np.asarray(trace["positions"], np.float64)[:, :2]
    yaw = np.asarray(trace["yaws"], np.float64)
    seg = np.asarray(trace["segments"], int)
    I, J = [], []
    for s in np.unique(seg):
        idx = np.flatnonzero(seg == s)
        if len(idx) < 3:
            continue
        p = pos[idx]
        # cumulative path length so "5-25 m ahead" is along the road, not
        # straight-line through a building.
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        for a in range(len(idx)):
            lo = np.searchsorted(d, d[a] + GOAL_MIN_M)
            hi = np.searchsorted(d, d[a] + GOAL_MAX_M)
            if hi <= lo:
                continue
            b = int(rng.integers(lo, min(hi, len(idx))))
            if b <= a:
                continue
            I.append(idx[a]); J.append(idx[b])
    I, J = np.array(I), np.array(J)
    dxy = pos[J] - pos[I]
    c, s_ = np.cos(-yaw[I]), np.sin(-yaw[I])
    ex = c * dxy[:, 0] - s_ * dxy[:, 1]          # forward in the car's frame
    ey = s_ * dxy[:, 0] + c * dxy[:, 1]          # left
    dist = np.hypot(ex, ey)
    bearing = np.arctan2(ey, ex)
    Y = np.stack([dist, np.sin(bearing), np.cos(bearing)], 1)
    return I, J, Y, seg[I]


# -------------------------------------------------------------- encoders ---
def debit_encoder(device):
    from croco.models.croco_downstream import CroCoDownstreamBinocular
    import torch.nn as nn

    class LinearHead(nn.Module):
        """DEBiT's head, reproduced: project each decoder token and flatten."""
        def __init__(self, ch):
            super().__init__()
            self.ch = ch

        def setup(self, net):
            self.n_cls_token = getattr(net, "n_cls_token", 0)
            self.proj = nn.Linear(net.dec_embed_dim, self.ch)

        def forward(self, dec_out, *a, **k):
            return self.proj(dec_out[:, self.n_cls_token:, :]).flatten(1, -1)

    n_patches = (CROCO_B["img_size"] // CROCO_B["patch_size"]) ** 2
    net = CroCoDownstreamBinocular(LinearHead(BINOC_REPR // n_patches), **CROCO_B)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from inspect_debit_ckpt import load_tensors      # stub-unpickles the ckpt
    sd = load_tensors(DEBIT_CKPT)
    pre = "actor_critic.net.binocular_encoder."
    enc = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    miss, extra = net.load_state_dict(enc, strict=False)
    print(f"[DEBiT] loaded {len(enc)} tensors; missing {len(miss)}, unexpected "
          f"{len(extra)}")
    if extra:
        raise SystemExit(f"unexpected keys, architecture mismatch: {extra[:5]}")
    if miss:
        raise SystemExit(f"missing keys, architecture mismatch: {miss[:5]}")
    return net.eval().to(device)


def debit_features(net, rgb_i, rgb_j, device, bs=48):
    """(N,112*,3) uint8 pairs -> (N, 3136) float32. Preprocessing is DEBiT's
    own: resize, to float, ImageNet normalise."""
    import torchvision.transforms as tf
    pre = tf.Compose([tf.Resize(CROCO_B["img_size"], antialias=True),
                      tf.ConvertImageDtype(torch.float),
                      tf.Normalize((0.485, 0.456, 0.406),
                                   (0.229, 0.224, 0.225))])
    out = []
    with torch.no_grad():
        for k in range(0, len(rgb_i), bs):
            a = torch.from_numpy(rgb_i[k:k + bs]).permute(0, 3, 1, 2).to(device)
            b = torch.from_numpy(rgb_j[k:k + bs]).permute(0, 3, 1, 2).to(device)
            out.append(net(pre(a), pre(b)).float().cpu().numpy())
    return np.concatenate(out)


# The seg-VAE is JAX and DEBiT is torch, and no single venv on this machine has
# both: vc1-venv carries the working GPU torch, ig-venv carries jax/flax. The
# control is therefore computed by re-invoking THIS file under ig-venv, which
# keeps one definition of the pairs and one definition of the features. CPU-only
# and no preallocation for the child, so it cannot disturb anything on the card.
IG_PY = ("/home/robotlab/scratch/carla-rl/.claude/worktrees/gated-impl/"
         "clean_fix1/dependencies/ig-venv/bin/python")


def _segvae_worker(trace_path, idx_npz, out_npz):
    import jax
    from vae_nav import seg_vae as sv
    z = np.load(idx_npz)
    rgb = pickle.load(open(trace_path, "rb"))["rgb"]
    c = pickle.load(open(SEGVAE_CKPT, "rb"))
    m = sv.SegVAE(latent_dim=c["latent_dim"], dec_start=c.get("dec_start", 15),
                  skips=c.get("skips", False))
    enc = jax.jit(lambda x: m.apply(c["params"], x, method=sv.SegVAE.encode))

    def mu(sel):
        r = rgb[sel]
        return np.concatenate([np.asarray(enc(r[k:k + 128].astype(np.float32) / 255.0))
                               for k in range(0, len(r), 128)])
    np.savez(out_npz, X=np.concatenate([mu(z["I"]), mu(z["J"])], 1))


def segvae_features(trace_path, I, J, tmpdir):
    """Control: concatenated MONOCULAR seg-VAE embeddings, no correspondence."""
    import subprocess
    idx = os.path.join(tmpdir, "idx.npz")
    out = os.path.join(tmpdir, "segvae.npz")
    np.savez(idx, I=I, J=J)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
               XLA_PYTHON_CLIENT_PREALLOCATE="false")
    r = subprocess.run([IG_PY, os.path.abspath(__file__), "--segvae-worker",
                        trace_path, idx, out], env=env,
                       capture_output=True, text=True)
    if not os.path.exists(out):
        raise SystemExit(f"seg-VAE control failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return np.load(out)["X"]


# ----------------------------------------------------------------- probe ---
def probe(X, Y, groups, name, seed=0):
    """Ridge (closed form) and a small MLP, split by GROUP. Reports R2."""
    rng = np.random.default_rng(seed)
    g = np.unique(groups)
    rng.shuffle(g)
    te_g = set(g[:max(1, len(g) // 4)].tolist())
    te = np.array([x in te_g for x in groups])
    tr = ~te
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    ym, ys = Y[tr].mean(0), Y[tr].std(0) + 1e-6
    Ytr = (Y[tr] - ym) / ys

    def r2(pred):
        p = pred * ys + ym
        return 1.0 - ((Y[te] - p) ** 2).sum(0) / ((Y[te] - Y[te].mean(0)) ** 2).sum(0)

    A = Xtr.T @ Xtr + 10.0 * np.eye(Xtr.shape[1])
    Wm = np.linalg.solve(A, Xtr.T @ Ytr)
    lin = r2(Xte @ Wm)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xt = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    yt = torch.tensor(Ytr, dtype=torch.float32, device=dev)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(),
                              torch.nn.Linear(256, 128), torch.nn.ReLU(),
                              torch.nn.Linear(128, Y.shape[1])).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(400):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(xt), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        nn_r2 = r2(net(torch.tensor(Xte, dtype=torch.float32,
                                    device=dev)).cpu().numpy())
    print(f"  {name:28s} dim {X.shape[1]:5d} | "
          f"distance R2 lin {lin[0]:6.3f} nn {nn_r2[0]:6.3f} | "
          f"bearing sin lin {lin[1]:6.3f} nn {nn_r2[1]:6.3f} | "
          f"cos lin {lin[2]:6.3f} nn {nn_r2[2]:6.3f}")
    return dict(linear=lin.tolist(), nn=nn_r2.tolist(), dim=int(X.shape[1]))


def main():
    if sys.argv[1:2] == ["--segvae-worker"]:       # re-invoked under ig-venv
        _segvae_worker(*sys.argv[2:5])
        return
    traces = sys.argv[1:]
    tmpdir = os.environ.get("PROBE_TMP", "/home/robotlab/.claude/jobs/af872d1b/tmp")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = debit_encoder(dev)
    results = {}
    for t in traces:
        tr = pickle.load(open(t, "rb"))
        town = tr.get("meta", {}).get("town", "?")
        rng = np.random.default_rng(0)
        I, J, Y, groups = build_pairs(tr, rng)
        rgb = tr["rgb"]
        print(f"\n{town}: {len(I)} pairs from {len(np.unique(groups))} segments, "
              f"distance {Y[:, 0].min():.1f}-{Y[:, 0].max():.1f} m "
              f"(median {np.median(Y[:, 0]):.1f})")
        Xd = debit_features(net, rgb[I], rgb[J], dev)
        results[f"{town}/debit"] = probe(Xd, Y, groups, "DEBiT-B binocular")
        Xs = segvae_features(os.path.abspath(t), I, J, tmpdir)
        results[f"{town}/segvae"] = probe(Xs, Y, groups, "seg-VAE obs+goal (control)")
    import json
    out = os.path.join(W, "..", "results", "debit_goal_probe.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
