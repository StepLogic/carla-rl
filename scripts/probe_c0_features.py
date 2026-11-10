#!/usr/bin/env python3
"""Step 0a of plan 2026-11-09 ("is it the encoder?"): what c0's frozen features carry.

c0 (il_bc_c0 step_15000) drives at 68/80 from these features, yet every SAC critic
on them failed to learn HOW MUCH to steer. Two questions, answered on recorded
demonstrations (no simulator):

1. STATE: can lateral offset and heading error be read from the camera latent
   (compress, 512), the GRU state (512) and the DEBiT binocular features (3136)?
   Ridge with episode-grouped 5-fold CV, a small MLP on one episode split, a
   shuffled-label control, and Town05 -> Town01 transfer.
2. VISIBILITY: does ONE steering step leave a trace in the next features? The
   recordings add DART noise (iid N(0, sigma) on the executed steering, sigma
   0.1 / 0.2) to the expert's action, independent of the state. Its mean over
   steps t .. t+k-1 is predicted from [f_t] (control, ~0) and [f_t, f_{t+k}] for
   k = 1, 2, 4. The CEILING is the same prediction from the TRUE heading and offset
   change over the window (privileged): how much of the noise the car's motion
   shows at all.

Labels (offset, heading) come from the recorded positions and the town's OpenDRIVE
map (torch_rl.lane_labels) -- diagnostic only, never a policy input.

usage (local venv, CPU):
  python scripts/probe_c0_features.py CKPT DATA_ROOT OUT [--towns Town05,Town01] [--threads 12]
writes OUT/features_<Town>.npz (reused by scripts/offline_priv_critic.py) and OUT/probe.json
"""
import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("MONO_CHW", "1")      # c0's frame layout
os.environ.setdefault("SPEED_OBS", "1")     # c0 reads the speed column
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "src")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from torch_rl import lane_labels as ll  # noqa: E402
from torch_rl import pixel_rows as pxr  # noqa: E402

MIN_KMH = 5.0          # frames slower than this are left out (heading undefined)
KS = (1, 2, 4)


def load_agent(ckpt):
    from torch_rl.gated_recurrent_sac import GatedRecurrentSAC, GatedRecurrentSACConfig
    d = json.load(open(os.path.join(ckpt, "config.json")))
    d.update(device="cpu", compile=False, bf16=False)
    agent = GatedRecurrentSAC(GatedRecurrentSACConfig(**d))
    agent.load(ckpt)                                   # strict: every module
    return agent


def load_episode(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


@torch.no_grad()
def episode_features(agent, ep):
    """Per FRAME (T + 1): camera latent, GRU state (zero at the episode start, as
    act() resets it), the state block [mode, speed, binoc], and the actor's
    pre-tanh mean and log-std -- exactly what the critic and actor read."""
    cfg = agent.cfg
    T = len(ep["act_exec"])
    kmh = np.concatenate([[0.0], np.asarray(ep["speed"], np.float32)])
    mode = float(ep["mode"])
    rows = np.stack([pxr.mono_row(ep["frames"][t], mode, ep["binoc"][t].astype(np.float32),
                                  speed_kmh=float(kmh[t])) for t in range(T + 1)])
    obs = torch.from_numpy(rows)
    tokens = obs[:, :cfg.token_dim].reshape(-1, cfg.embed_dim, cfg.grid, cfg.grid)
    visual = agent.compress(tokens).float()
    e = agent._embed(agent.compress, agent.act_embed, obs)
    h, _ = agent.core(e[None], torch.zeros(1, 1, cfg.hidden))
    h = h[0]
    st = agent._state(obs)
    mean_u, log_std = agent._actor_forward(torch.cat([h, st], -1))
    return {"visual": visual.numpy(), "h": h.numpy(), "state": st.numpy(),
            "mean_u": mean_u.numpy(), "log_std": log_std.numpy(), "kmh": kmh}


def build_cache(agent, town, paths, out):
    lanes = ll.OpenDriveLanes(town)
    F = {k: [] for k in ("visual", "h", "binoc", "modespeed", "mean_u", "log_std", "kmh",
                         "offset", "heading_err", "heading", "dheading", "curv", "junction",
                         "width", "valid", "ep", "k")}
    S = {k: [] for k in ("act_exec", "act_label", "reward", "ep", "t")}
    E = {k: [] for k in ("path", "T", "sigma", "driver", "end", "terminal", "truncated",
                         "mode", "f0", "s0")}
    nf = ns = 0
    t0 = time.time()
    for i, p in enumerate(paths):
        ep = load_episode(p)
        T = len(ep["act_exec"])
        if T < 2:
            continue
        f = episode_features(agent, ep)
        lab = ll.episode_labels(ep["pos"], lanes)
        F["visual"].append(f["visual"].astype(np.float16))
        F["h"].append(f["h"].astype(np.float16))
        F["binoc"].append(np.asarray(ep["binoc"], np.float16))
        F["modespeed"].append(f["state"][:, :2].astype(np.float32))
        F["mean_u"].append(f["mean_u"])
        F["log_std"].append(f["log_std"])
        F["kmh"].append(f["kmh"].astype(np.float32))
        for k in ("offset", "heading_err", "heading", "dheading", "curv", "width"):
            F[k].append(lab[k].astype(np.float32))
        F["junction"].append(lab["junction"])
        F["valid"].append(lab["valid"])
        F["ep"].append(np.full(T + 1, len(E["path"]), np.int32))
        F["k"].append(np.arange(T + 1, dtype=np.int32))
        S["act_exec"].append(np.asarray(ep["act_exec"], np.float32))
        S["act_label"].append(np.asarray(ep["act_label"], np.float32))
        S["reward"].append(np.asarray(ep["reward"], np.float32))
        S["ep"].append(np.full(T, len(E["path"]), np.int32))
        S["t"].append(np.arange(T, dtype=np.int32))
        for k, v in (("path", os.path.basename(p)), ("T", T), ("sigma", float(ep["sigma"])),
                     ("driver", str(ep["driver"])), ("end", str(ep["end"])),
                     ("terminal", bool(ep["terminal"])), ("truncated", bool(ep["truncated"])),
                     ("mode", int(ep["mode"])), ("f0", nf), ("s0", ns)):
            E[k].append(v)
        nf += T + 1
        ns += T
        if (i + 1) % 50 == 0:
            print(f"  {town}: {i + 1}/{len(paths)} episodes, {nf} frames ({time.time() - t0:.0f} s)",
                  flush=True)
    arrays = {f"f_{k}": np.concatenate(v) for k, v in F.items()}
    arrays.update({f"s_{k}": np.concatenate(v) for k, v in S.items()})
    arrays.update({f"e_{k}": np.asarray(v) for k, v in E.items()})
    tmp = out + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, out)
    print(f"  {town}: cached {len(E['path'])} episodes, {nf} frames -> {out} "
          f"({time.time() - t0:.0f} s)", flush=True)


# ------------------------------------------------------------------ probes --
def r2(y, p):
    st = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - p) ** 2).sum()) / st if st > 0 else float("nan")


def _std(Xtr, *others):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd < 1e-6] = 1.0
    return [(X - mu) / sd for X in (Xtr,) + others]


class _Ridge:
    """Ridge on standardised features, solved for a lambda grid from one
    eigendecomposition of X'X (lambda scales with the row count)."""
    GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)

    def __init__(self, X):
        self.n = len(X)
        s, V = np.linalg.eigh(X.T @ X)
        self.s, self.V = s, V
        self.X = X

    def weights(self, y):
        ym = float(y.mean())
        vc = self.V.T @ (self.X.T @ (y - ym))
        return [(self.V @ (vc / (self.s + a * self.n)), ym) for a in self.GRID]


def _folds(groups, n_folds, seed):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    rng.shuffle(ug)
    fold_of = {g: i % n_folds for i, g in enumerate(ug)}
    return np.array([fold_of[g] for g in groups]), rng


def ridge_cv(X, ys, groups, n_folds=5, seed=0, shuffle=False):
    """Episode-grouped CV ridge -> {target: pooled held-out R2}. Lambda is chosen
    on an inner split of each training fold (20% of its episodes)."""
    X = np.asarray(X, np.float64)
    fold, rng = _folds(groups, n_folds, seed)
    preds = {name: np.zeros(len(X)) for name in ys}
    for i in range(n_folds):
        tr, te = np.flatnonzero(fold != i), np.flatnonzero(fold == i)
        gtr = groups[tr]
        ug = np.unique(gtr)
        iv_g = set(rng.choice(ug, max(1, len(ug) // 5), replace=False).tolist())
        iv = np.array([g in iv_g for g in gtr])
        A, B = _std(X[tr[~iv]], X[tr[iv]])
        inner = _Ridge(A)
        A2, C = _std(X[tr], X[te])
        full = _Ridge(A2)
        for name, y in ys.items():
            ytr = y[tr].copy()
            if shuffle:
                ytr = rng.permutation(ytr)
            scores = [r2(ytr[iv], B @ w + b) for w, b in inner.weights(ytr[~iv])]
            j = int(np.nanargmax(scores))
            w, b = full.weights(ytr)[j]
            preds[name][te] = C @ w + b
    return {name: round(r2(y, preds[name]), 3) for name, y in ys.items()}


def ridge_transfer(Xa, ya, ga, Xb, yb, seed=0):
    """Fit on town A (lambda from an inner episode split), score R2 on town B."""
    Xa, Xb = np.asarray(Xa, np.float64), np.asarray(Xb, np.float64)
    rng = np.random.default_rng(seed)
    ug = np.unique(ga)
    iv_g = set(rng.choice(ug, max(1, len(ug) // 5), replace=False).tolist())
    iv = np.array([g in iv_g for g in ga])
    A, B = _std(Xa[~iv], Xa[iv])
    scores = [r2(ya[iv], B @ w + b) for w, b in _Ridge(A).weights(ya[~iv])]
    j = int(np.nanargmax(scores))
    A2, C = _std(Xa, Xb)
    w, b = _Ridge(A2).weights(ya)[j]
    return round(r2(yb, C @ w + b), 3)


def mlp_probe(X, y, groups, seed=0, epochs=60, patience=6, threads=12):
    """2x256 ReLU MLP on one episode split (80/20), early-stopped on 10% of the
    training episodes -> held-out R2."""
    torch.set_num_threads(threads)
    fold, rng = _folds(groups, 5, seed)
    tr, te = np.flatnonzero(fold != 0), np.flatnonzero(fold == 0)
    ug = np.unique(groups[tr])
    iv_g = set(rng.choice(ug, max(1, len(ug) // 10), replace=False).tolist())
    iv = np.array([g in iv_g for g in groups[tr]])
    trn, val = tr[~iv], tr[iv]
    Xa, Xv, Xt = _std(np.asarray(X[trn], np.float32), np.asarray(X[val], np.float32),
                      np.asarray(X[te], np.float32))
    ym, ys = float(y[trn].mean()), float(y[trn].std()) or 1.0
    tX = [torch.from_numpy(np.ascontiguousarray(a, np.float32)) for a in (Xa, Xv, Xt)]
    ty = torch.from_numpy(((y[trn] - ym) / ys).astype(np.float32))
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(),
                              torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = -1e9, None, 0
    for _ in range(epochs):
        perm = torch.randperm(len(ty))
        for j in range(0, len(perm), 1024):
            b = perm[j:j + 1024]
            loss = ((net(tX[0][b])[:, 0] - ty[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pv = net(tX[1])[:, 0].numpy() * ys + ym
        s = r2(y[val], pv)
        if s > best + 1e-4:
            best, bad = s, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        pt = net(tX[2])[:, 0].numpy() * ys + ym
    return round(r2(y[te], pt), 3)


def feature_sets(C):
    """Name -> per-frame feature matrix (float32)."""
    return {"camera": C["f_visual"].astype(np.float32),
            "gru": C["f_h"].astype(np.float32),
            "binoc": C["f_binoc"].astype(np.float32)}


def clean_label_mask(C):
    """Frames whose lane labels can be trusted: moving, heading defined, off-junction,
    well inside the lane (|offset| <= 1 m), |heading error| <= 15 deg, and no lane
    jump (|offset change| < 0.3 m to both neighbours in the same episode).

    Why (checked on the Town05 cache before any probe was re-run): the first filter,
    |offset| <= half the lane width, kept frames at lane boundaries -- where the
    nearest lane centre flips to the neighbouring lane and the signed offset jumps by
    a lane width -- and wrong-way / lane-snapping outliers (heading errors up to
    180 deg, mostly in clone-driven episodes). 3% of frames took the heading error's
    SD to 15.4 deg and its correlation with the next offset change to +0.10. Under
    this filter: SD 3.5 deg, correlation +0.93, offset SD 0.21 m (53,237 frames)."""
    off, he, ep = C["f_offset"], C["f_heading_err"], C["f_ep"]
    d = np.diff(off)
    same = ep[1:] == ep[:-1]
    fwd_ok = np.r_[same & (np.abs(d) < 0.3), False]
    bwd_ok = np.r_[False, same & (np.abs(d) < 0.3)]
    with np.errstate(invalid="ignore"):
        return (C["f_valid"] & ~C["f_junction"] & (C["f_kmh"] >= MIN_KMH)
                & (np.abs(off) <= 1.0) & (np.abs(he) <= 15.0) & fwd_ok & bwd_ok)


def state_rows(C):
    """Frames for the state probes: the clean-label frames."""
    return np.flatnonzero(clean_label_mask(C))


def visibility_rows(C, k):
    """(step index, frame t, frame t+k) for expert-driven, DART-noised steps whose
    window fits the episode and whose frame t is moving."""
    out_s, out_a, out_b = [], [], []
    for e in range(len(C["e_T"])):
        if C["e_driver"][e] != "expert" or C["e_sigma"][e] <= 0:
            continue
        T, f0, s0 = int(C["e_T"][e]), int(C["e_f0"][e]), int(C["e_s0"][e])
        t = np.arange(0, T - k + 1)
        t = t[C["f_kmh"][f0 + t] >= MIN_KMH]
        out_s.append(s0 + t)
        out_a.append(f0 + t)
        out_b.append(f0 + t + k)
    return np.concatenate(out_s), np.concatenate(out_a), np.concatenate(out_b)


def run_probes(caches, threads):
    torch.set_num_threads(threads)
    res = {"state": {}, "visibility": {}, "transfer": {}, "counts": {}}
    main = caches["Town05"]
    Fs = feature_sets(main)
    rows = state_rows(main)
    g = main["f_ep"][rows]
    ys = {"offset": main["f_offset"][rows].astype(np.float64),
          "heading_err": main["f_heading_err"][rows].astype(np.float64)}
    res["counts"]["state_frames_Town05"] = int(len(rows))
    res["counts"]["state_episodes_Town05"] = int(len(np.unique(g)))
    res["label_sd"] = {k: round(float(v.std()), 3) for k, v in ys.items()}
    for name, X in Fs.items():
        t0 = time.time()
        r = ridge_cv(X[rows], ys, g)
        sh = ridge_cv(X[rows], ys, g, shuffle=True)
        mlp = {t: mlp_probe(X[rows], y, g, threads=threads) for t, y in ys.items()}
        res["state"][name] = {t: {"ridge": r[t], "mlp": mlp[t], "shuffled": sh[t]} for t in ys}
        print(f"[state] {name}: {res['state'][name]} ({time.time() - t0:.0f} s)", flush=True)
    sp = main["f_kmh"][rows][:, None].astype(np.float64)
    res["state"]["speed_only"] = {t: {"ridge": v} for t, v in ridge_cv(sp, ys, g).items()}
    if "Town01" in caches:
        other = caches["Town01"]
        rows_b = state_rows(other)
        res["counts"]["state_frames_Town01"] = int(len(rows_b))
        Fb = feature_sets(other)
        for name in Fs:
            res["transfer"][name] = {
                t: ridge_transfer(Fs[name][rows], ys[t], g, Fb[name][rows_b],
                                  other[f"f_{t}"][rows_b].astype(np.float64))
                for t in ys}
            # within Town01 (its own CV), for scale
            res["transfer"][name + "_within_Town01"] = ridge_cv(
                Fb[name][rows_b], {t: other[f"f_{t}"][rows_b].astype(np.float64) for t in ys},
                other["f_ep"][rows_b])
            print(f"[transfer] {name}: Town05->Town01 {res['transfer'][name]}, within Town01 "
                  f"{res['transfer'][name + '_within_Town01']}", flush=True)

    exe, lab = main["s_act_exec"], main["s_act_label"]
    for k in KS:
        s_idx, fa, fb = visibility_rows(main, k)
        noise = np.concatenate([ll.window_noise(exe[s0:s0 + T], lab[s0:s0 + T], k)
                                for s0, T in zip(main["e_s0"], main["e_T"])])
        y = noise[s_idx].astype(np.float64)
        gg = main["f_ep"][fa]
        # labelled subset (clean lane labels at both ends): the ceiling's rows
        clean = clean_label_mask(main)
        lab_ok = clean[fa] & clean[fb]
        dpsi = ll.wrap_deg(main["f_heading"][fb] - main["f_heading"][fa])
        doff = main["f_offset"][fb] - main["f_offset"][fa]
        ceil_X = np.stack([dpsi, doff, main["f_kmh"][fa]], -1)[lab_ok]
        out = {"n_all": int(len(y)), "n_labelled": int(lab_ok.sum()),
               "noise_sd": round(float(y.std()), 3),
               "ceiling_true_motion": ridge_cv(ceil_X, {"y": y[lab_ok]}, gg[lab_ok])["y"],
               "ceiling_true_motion_mlp": mlp_probe(ceil_X, y[lab_ok], gg[lab_ok], threads=threads)}
        for name in ("camera", "gru"):
            X = Fs[name]
            out[name] = {"f_t_only": ridge_cv(X[fa], {"y": y}, gg)["y"],
                         "f_t_and_t+k": ridge_cv(np.concatenate([X[fa], X[fb]], 1), {"y": y}, gg)["y"],
                         "f_t_and_t+k_labelled": ridge_cv(np.concatenate([X[fa], X[fb]], 1)[lab_ok],
                                                          {"y": y[lab_ok]}, gg[lab_ok])["y"],
                         "mlp_f_t_and_t+k": mlp_probe(np.concatenate([X[fa], X[fb]], 1), y, gg,
                                                      threads=threads)}
        Xb = Fs["binoc"]
        out["binoc"] = {"f_t_and_t+k": ridge_cv(np.concatenate([Xb[fa], Xb[fb] - Xb[fa]], 1),
                                                {"y": y}, gg, n_folds=3)["y"]}
        crit = np.concatenate([Fs["gru"], Fs["binoc"], main["f_kmh"][:, None]], 1)
        out["critic_input_mlp"] = mlp_probe(np.concatenate([crit[fa], crit[fb]], 1), y, gg,
                                            threads=threads)
        res["visibility"][f"k{k}"] = out
        print(f"[visibility] k={k}: {out}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--towns", default="Town05,Town01")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--max-eps", type=int, default=0)
    ap.add_argument("--cache-only", action="store_true")
    args = ap.parse_args()
    if not (pxr.MONO_CHW and pxr.SPEED_OBS):
        raise SystemExit("c0 needs MONO_CHW=1 SPEED_OBS=1")
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    caches = {}
    agent = None
    for town in args.towns.split(","):
        cache = os.path.join(args.out, f"features_{town}.npz")
        if not os.path.exists(cache):
            paths = sorted(glob.glob(os.path.join(args.root, town, "ep_*.npz")))
            if args.max_eps:
                paths = paths[:args.max_eps]
            if not paths:
                print(f"  {town}: no episodes under {args.root}", flush=True)
                continue
            if agent is None:
                agent = load_agent(args.ckpt)
            build_cache(agent, town, paths, cache)
        caches[town] = dict(np.load(cache, allow_pickle=False))
    if args.cache_only:
        return
    res = run_probes(caches, args.threads)
    res["ckpt"] = args.ckpt
    res["towns"] = list(caches)
    with open(os.path.join(args.out, "probe.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
