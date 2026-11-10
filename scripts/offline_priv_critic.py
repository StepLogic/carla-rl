#!/usr/bin/env python3
"""Step 0b of plan 2026-11-09: the privileged-critic control (offline; DIAGNOSTIC
ONLY -- the user approved it "offline only"; never a result, never the navigator).

Question: every SAC critic on c0's image features failed to learn HOW MUCH to
steer (best steering on a 21-point grid vs the expert's: corr +0.04 with the DrQ
head, +0.10 without, 20k offline updates on c0's demos). Is that the
representation, or the learning signal? Train the SAME critic, the SAME way, on the
same demonstrations, but give it the TRUE lane state instead of image features:
  priv  = [lateral offset, heading error, lane curvature ahead, speed, in-junction,
           heading-known, mode]  (from positions + OpenDRIVE, torch_rl.lane_labels)
  image = [GRU state | mode | speed | binoc]  (c0's frozen trunk, the critic's input)
If the privileged critic learns the expert's steering (corr >= 0.5) and the image
critic does not, the encoder is the bottleneck; if neither does, the TD signal is.

Matched to the earlier offline study (offline_proj_critic.py): executed actions,
IL_PRELOAD_TRANSFORM=no_arrival (reached episodes lose their last 30 steps and end
truncated), rewards x10, transitions sampled uniformly, 10 LayerNorm critics
(256, 256), REDQ min over 2 random targets, gamma 0.99, tau 0.005, Adam 3e-4, batch
320, the target's next action sampled from c0's frozen actor at alpha 0.1 (the
projection run's temperature; recorded data never moves it). The critic trains
alone; the trunk and actor are c0's and frozen (their outputs come from the cache).

Probe: the first 3 Town05 episodes (the earlier study's probe) plus every 10th
episode, all held out of training; the expert's LABEL action with its steering
replaced by each of 21 values in [-1, 1].

Arm frl_embed50 (user 2026-11-10, "Compressed to 50-d, compression trained by the
critic (FastRLAP)"): the same protocol, with learner C's own critic (torch_rl.frl,
the FastRLAP port) in FastRLAP's embeddings mode instead of QuantileCritics: c0's
frozen [GRU | binoc] -> Dense(50) -> LayerNorm -> tanh, trained by the critic loss;
[mode, speed] -> its own Dense(50) -> LayerNorm -> tanh; 10 LayerNorm Q heads
(256, 256); min over 2 target heads; no entropy term in the target (FastRLAP's
backup_entropy=False, so the alpha 0.1 above does not enter). The next action is
still c0's frozen actor's.

usage: python scripts/offline_priv_critic.py CACHE OUT [--arms priv,image] [--updates 20000]
  CACHE = <probe out>/features_Town05.npz from scripts/probe_c0_features.py
"""
import argparse
import copy
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "src"), os.path.join(REPO, "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from torch_rl.nets import QuantileCritics  # noqa: E402
from torch_rl.slope_diag import ARRIVAL_CUT  # noqa: E402

REWARD_SCALE = 10.0
GAMMA, TAU, LR, ALPHA = 0.99, 0.005, 3e-4, 0.1
N_CRITICS, SUBSET = 10, 2
H_DIM, MS_DIM = 512, 2          # image_features rows: [GRU 512 | mode, speed | binoc 3136]
FRL_LATENT = 50


def priv_features(C):
    """Per frame: the true lane state, clipped and scaled to ~unit range, with a
    flag for frames whose labels are trustworthy (probe_c0_features.clean_label_mask:
    off-junction, well inside the lane, no lane-boundary jump); elsewhere the offset
    and heading error are kept but clipped, and the flag tells the critic so."""
    from probe_c0_features import clean_label_mask
    known = np.isfinite(C["f_heading"]) & C["f_valid"]
    clean = clean_label_mask(C)
    herr = np.where(known, np.clip(np.nan_to_num(C["f_heading_err"]), -15.0, 15.0), 0.0) / 5.0
    off = np.clip(np.nan_to_num(C["f_offset"]), -1.75, 1.75)
    curv = np.clip(np.nan_to_num(C["f_curv"]), -90.0, 90.0) / 30.0
    kmh = C["f_kmh"] / 15.0
    mode = C["f_modespeed"][:, 0]
    return np.stack([off, herr, curv, kmh, C["f_junction"].astype(np.float32),
                     known.astype(np.float32), clean.astype(np.float32), mode],
                    -1).astype(np.float32)


def image_features(C):
    return np.concatenate([C["f_h"].astype(np.float16), C["f_modespeed"].astype(np.float16),
                           C["f_binoc"].astype(np.float16)], 1)


def transitions(C, keep_eps):
    """(frame t, frame t+1, step t, reward, mask) for the episodes in `keep_eps`,
    after the no_arrival cut."""
    fa, fb, st, rw, mk = [], [], [], [], []
    for e in keep_eps:
        T, f0, s0 = int(C["e_T"][e]), int(C["e_f0"][e]), int(C["e_s0"][e])
        terminal = bool(C["e_terminal"][e])
        if C["e_end"][e] == "reached":
            if T <= ARRIVAL_CUT:
                continue
            T, terminal = T - ARRIVAL_CUT, False
        t = np.arange(T)
        fa.append(f0 + t)
        fb.append(f0 + t + 1)
        st.append(s0 + t)
        rw.append(C["s_reward"][s0 + t] * REWARD_SCALE)
        m = np.ones(T, np.float32)
        if terminal:
            m[-1] = 0.0
        mk.append(m)
    return tuple(np.concatenate(x) for x in (fa, fb, st, rw, mk))


def probe_rows(C, eps):
    """(frame, step) pairs of the probe episodes: every step with its frame."""
    fr, st = [], []
    for e in eps:
        T, f0, s0 = int(C["e_T"][e]), int(C["e_f0"][e]), int(C["e_s0"][e])
        fr.append(f0 + np.arange(T))
        st.append(s0 + np.arange(T))
    return np.concatenate(fr), np.concatenate(st)


def sample_next_action(mean_u, log_std, gen):
    noise = torch.randn(mean_u.shape, generator=gen, device=mean_u.device)
    pre = mean_u + log_std.exp() * noise
    a = torch.tanh(pre)
    logp = (-0.5 * noise ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
    logp = logp - torch.log(1.0 - a ** 2 + 1e-6).sum(-1)
    return a, logp


@torch.no_grad()
def measure(critics, X, a0, junction, clean, dev, max_states=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(a0)) if len(a0) <= max_states else np.sort(
        rng.choice(len(a0), max_states, replace=False))
    x = torch.as_tensor(np.asarray(X[idx], np.float32), device=dev)
    a = torch.as_tensor(a0[idx], device=dev)
    grid = torch.linspace(-1.0, 1.0, 21, device=dev)
    qs = []
    for g in grid:
        aa = a.clone()
        aa[:, 0] = g
        qs.append(critics(x, aa).float().mean(dim=(0, 2)))
    qs = torch.stack(qs)                                   # (21, N)
    best = grid[qs.argmax(0)].cpu().numpy()
    exp = a0[idx, 0]
    jn = junction[idx]
    cl = clean[idx]

    def stats(m):
        b, e = best[m], exp[m]
        return {"n": int(m.sum()),
                "corr": round(float(np.corrcoef(b, e)[0, 1]), 3) if b.std() > 0 else None,
                "share_full_lock": round(float((np.abs(b) >= 0.99).mean()), 3),
                "mean_abs_best": round(float(np.abs(b).mean()), 3),
                "mean_abs_expert": round(float(np.abs(e).mean()), 3)}
    span = (qs.max(0).values - qs.min(0).values).cpu().numpy()
    return {"all": stats(np.ones(len(idx), bool)), "off_junction": stats(~jn),
            "clean_labels": stats(cl), "q_span_over_grid": round(float(np.median(span)), 4)}


def run_arm(arm, C, X, train, probes, args, dev):
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    dim = X.shape[1]
    proj = 50 if arm == "image_proj50" else 0
    critics = QuantileCritics(dim, 2, n_critics=N_CRITICS, n_quantiles=1, hidden=(256, 256),
                              layer_norm=True, proj_dim=proj).to(dev)
    targ = copy.deepcopy(critics).requires_grad_(False)
    opt = torch.optim.Adam(critics.parameters(), lr=LR)
    fa, fb, st, rw, mk = train
    mean_u = torch.as_tensor(C["f_mean_u"], device=dev)
    log_std = torch.as_tensor(C["f_log_std"], device=dev)
    acts = torch.as_tensor(C["s_act_exec"], device=dev)
    rng = np.random.default_rng(args.seed)
    from probe_c0_features import clean_label_mask
    clean = clean_label_mask(C)
    trace, t0 = [], time.time()
    for u in range(1, args.updates + 1):
        b = rng.integers(0, len(fa), args.batch)
        x = torch.as_tensor(np.asarray(X[fa[b]], np.float32), device=dev)
        xn = torch.as_tensor(np.asarray(X[fb[b]], np.float32), device=dev)
        a = acts[st[b]]
        r = torch.as_tensor(rw[b], device=dev)
        m = torch.as_tensor(mk[b], device=dev)
        with torch.no_grad():
            an, logp = sample_next_action(mean_u[fb[b]], log_std[fb[b]], gen)
            qn = targ(xn, an).float()[..., 0]                       # (n_critics, B)
            idx = torch.randperm(N_CRITICS, generator=gen, device=dev)[:SUBSET]
            y = r + GAMMA * m * (qn.index_select(0, idx).min(0).values - ALPHA * logp)
        q = critics(x, a).float()[..., 0]
        loss = ((q - y[None]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for p, pt in zip(critics.parameters(), targ.parameters()):
                pt.mul_(1 - TAU).add_(p, alpha=TAU)
        if u % args.every == 0 or u == args.updates:
            row = {"arm": arm, "updates": u, "loss": round(float(loss), 4),
                   "q_mean": round(float(q.mean()), 3), "secs": round(time.time() - t0)}
            for name, (fr, sp) in probes.items():
                row[name] = measure(critics, X[fr], C["s_act_label"][sp], C["f_junction"][fr],
                                    clean[fr], dev)
            trace.append(row)
            print("[offline_priv] " + json.dumps(row), flush=True)
    torch.save(critics.state_dict(), os.path.join(args.out, f"critics_{arm}.pt"))
    return trace


def split_image_row(x):
    """An image_features row -> learner C's multiplexer inputs: embeddings = [GRU | binoc]
    (c0's frozen image features), states = [mode, speed]."""
    return {"embeddings": torch.cat([x[:, :H_DIM], x[:, H_DIM + MS_DIM:]], 1),
            "states": x[:, H_DIM:H_DIM + MS_DIM]}


class C0NextActor:
    """c0's frozen actor in place of learner C's: update_critic draws the target's next action
    from actor.sample(next_observations), which here carries c0's cached mean and log-std."""

    def __init__(self, gen):
        self.gen = gen

    def sample(self, obs):
        a, logp = sample_next_action(obs["c0_mean_u"], obs["c0_log_std"], self.gen)
        return a, logp, torch.tanh(obs["c0_mean_u"])


class MeasureView:
    """measure() reads critics(x, a) as (heads, batch, quantiles); learner C's is (heads, batch)."""

    def __init__(self, critic):
        self.critic = critic

    def __call__(self, x, a):
        return self.critic(split_image_row(x), a)[..., None]


def frl_critic(emb_dim, state_dim, device, seed):
    """Learner C (SACLearner, embeddings mode) with FastRLAP's critic settings and this study's
    gamma / tau / learning rate. Only its critic is used; the caller replaces its actor."""
    from torch_rl.frl.learner import SACLearner
    return SACLearner(action_dim=2, embedding_dim=emb_dim, state_dim=state_dim,
                      latent_dim=FRL_LATENT, critic_lr=LR, hidden_dims=(256, 256),
                      discount=GAMMA, tau=TAU, num_qs=N_CRITICS, num_min_qs=SUBSET,
                      critic_layer_norm=True, backup_entropy=False, seed=seed, device=device)


def run_arm_frl(arm, C, X, train, probes, args, dev):
    """The run_arm protocol with learner C's critic and its own update_critic."""
    L = frl_critic(X.shape[1] - MS_DIM, MS_DIM, dev, args.seed)      # [GRU | binoc] = 3648
    L.actor = C0NextActor(torch.Generator(device=dev).manual_seed(args.seed))
    fa, fb, st, rw, mk = train
    mean_u = torch.as_tensor(C["f_mean_u"], device=dev)
    log_std = torch.as_tensor(C["f_log_std"], device=dev)
    acts = torch.as_tensor(C["s_act_exec"], device=dev)
    rng = np.random.default_rng(args.seed)
    from probe_c0_features import clean_label_mask
    clean = clean_label_mask(C)
    view = MeasureView(L.critic)
    trace, t0 = [], time.time()
    for u in range(1, args.updates + 1):
        b = rng.integers(0, len(fa), args.batch)
        x = torch.as_tensor(np.asarray(X[fa[b]], np.float32), device=dev)
        xn = torch.as_tensor(np.asarray(X[fb[b]], np.float32), device=dev)
        nxt = split_image_row(xn)
        nxt["c0_mean_u"], nxt["c0_log_std"] = mean_u[fb[b]], log_std[fb[b]]
        info = L.update_critic({"observations": split_image_row(x), "next_observations": nxt,
                                "actions": acts[st[b]], "rewards": torch.as_tensor(rw[b], device=dev),
                                "masks": torch.as_tensor(mk[b], device=dev)})
        if u % args.every == 0 or u == args.updates:
            row = {"arm": arm, "updates": u, "loss": round(info["critic_loss"], 4),
                   "q_mean": round(info["q"], 3), "secs": round(time.time() - t0)}
            for name, (fr, sp) in probes.items():
                row[name] = measure(view, X[fr], C["s_act_label"][sp], C["f_junction"][fr],
                                    clean[fr], dev)
            trace.append(row)
            print("[offline_priv] " + json.dumps(row), flush=True)
    torch.save(L.critic.state_dict(), os.path.join(args.out, f"critics_{arm}.pt"))
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("out")
    ap.add_argument("--arms", default="priv,image")
    ap.add_argument("--updates", type=int, default=20000)
    ap.add_argument("--every", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=320)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    C = dict(np.load(args.cache, allow_pickle=False))
    n_eps = len(C["e_T"])
    order = np.argsort(C["e_path"])                       # sorted shard names
    first3 = set(order[:3].tolist())
    every10 = set(order[::10].tolist())
    held = first3 | every10
    train_eps = [e for e in range(n_eps) if e not in held]
    train = transitions(C, train_eps)
    probes = {"probe_first3": probe_rows(C, sorted(first3)),
              "probe_every10": probe_rows(C, sorted(every10 - first3))}
    print(f"[offline_priv] {n_eps} episodes: train {len(train_eps)} ({len(train[0])} transitions), "
          f"probe first3 {len(probes['probe_first3'][0])} + every10 {len(probes['probe_every10'][0])} steps",
          flush=True)
    feats = {"priv": priv_features, "image": image_features, "image_proj50": image_features,
             "frl_embed50": image_features}
    result = {"args": vars(args), "train_episodes": len(train_eps), "arms": {}}
    for arm in args.arms.split(","):
        X = feats[arm](C)
        runner = run_arm_frl if arm.startswith("frl_") else run_arm
        result["arms"][arm] = runner(arm, C, X, train, probes, args, args.device)
        with open(os.path.join(args.out, "offline_priv_critic.json"), "w") as fh:
            json.dump(result, fh, indent=1)
    print(f"[offline_priv] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
