#!/usr/bin/env python3
"""Offline critic action-slope diagnosis (user 2026-11-07: "Diagnose the slope first").

Every SAC fine-tune of c0 collapsed once its actor trained, and run I's critic
already scored "steer left + less throttle" above the expert's own action in 100%
of recorded expert states at step 25k, BEFORE the actor moved: it learned that
slope in the warm-up. This re-trains the warm-up critic OFFLINE -- no CARLA --
the way the fine-tune does it, with one candidate cause removed per --variant:

  * the agent is built from the fine-tune's own config (--config, e.g. run I's
    step_25000) and loads the clone (--init, c0) with fresh_critics_on_mismatch,
    so the LayerNorm critics start FRESH and everything else is c0's;
  * the encoder, GRU and actor are frozen (set_trunk_frozen + set_warmup) and
    the critic takes --updates uniform-replay updates (IL_CRITIC_WARMUP_UPDATES,
    5000) of CHUNK_B x CHUNK_T (8 x 40) on the recorded episodes' EXECUTED
    actions and rewards, exactly as il_finetune_prepare preloads them;
  * the variant (torch_rl.slope_diag.VARIANTS) transforms each episode on its
    way into the replay; --roots chooses the data (drop il_data_cover for
    "no coverage data").

Then the slope test of ~/hangprobe/critic_slope.py on the same six recorded
expert shards (first three of il_data/Town05 and il_data/Town01, expert LABEL
actions): mean Q(a0 + d) - Q(a0) per action dimension, the share of states whose
best offset is not 0, per-critic agreement at d = -0.2, and the same split by
mode. Results: <out>/slope.json and the critic checkpoint in <out>/.

  SPEED_OBS=1 MONO_CHW=1 REPLAY_DEVICE=cpu python scripts/offline_critic_slope.py \
    --config checkpoints/il_ft_c0_sac_per/step_25000 \
    --init checkpoints/il_bc_c0/step_15000 \
    --roots checkpoints/il_data checkpoints/il_data_cover \
    --variant baseline --out checkpoints/slope_diag/baseline
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

from torch_rl import pixel_rows as pxr  # noqa: E402
from torch_rl.bc_trainer import _buffer, _episode_rows  # noqa: E402
from torch_rl.episode_buffer import DeviceEpisodeBuffer  # noqa: E402
from torch_rl.expert_data import episode_paths, fill_buffer  # noqa: E402
from torch_rl.gated_recurrent_sac import GatedRecurrentSAC, GatedRecurrentSACConfig  # noqa: E402
from torch_rl.slope_diag import VARIANTS  # noqa: E402

DELTAS = (-0.4, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.4)
TOWNS = ("Town01", "Town04", "Town05", "Town10HD")


def parse_overrides(pairs):
    """--set KEY=VALUE pairs -> {key: value}, VALUE read as JSON when it parses
    (numbers, true/false, lists) and as a string otherwise."""
    out = {}
    for p in pairs or ():
        if "=" not in p:
            raise SystemExit(f"--set {p!r}: expected KEY=VALUE")
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def build_agent(config_dir, init_dir, device, seed, overrides=None):
    d = json.load(open(os.path.join(config_dir, "config.json")))
    unknown = set(overrides or {}) - set(d)
    if unknown:
        raise SystemExit(f"--set {sorted(unknown)}: not fields of {config_dir}/config.json")
    d.update(overrides or {})
    d.update(device=device, compile=False)
    if pxr.PRIV_OBS:
        # PRIVILEGED CRITIC (2026-11-07): the rows carry the goal distance for the
        # critic alone; the views come from the layout, as build_agent sets them.
        # c0's actor and GRU still load: they read the same columns in the same
        # order, the two new ones skipped.
        S = pxr.mono_state_dim(pxr.BINOC_DIM)
        d.update(state_dim=S, core_state_cols=list(pxr.mono_core_state_cols(pxr.BINOC_DIM)),
                 goal_cols=list(range(S)),
                 actor_state_cols=list(pxr.mono_actor_state_cols(pxr.BINOC_DIM)))
    cfg = GatedRecurrentSACConfig(**d)
    rows_state = pxr.mono_state_dim(pxr.BINOC_DIM)
    if cfg.state_dim != rows_state:
        raise SystemExit(f"config state_dim {cfg.state_dim} but rows give {rows_state}: export "
                         f"the run's SPEED_OBS / MONO_CHW before running this")
    torch.manual_seed(seed)
    agent = GatedRecurrentSAC(cfg)
    fresh = {k: v.clone() for k, v in agent.critics_g.state_dict().items()}
    agent.load(init_dir, fresh_critics_on_mismatch=True)
    if any(not torch.equal(fresh[k], v) for k, v in agent.critics_g.state_dict().items()):
        raise SystemExit(f"{init_dir}'s critics loaded into the agent: pass a clone (a "
                         f"checkpoint without LayerNorm critics) as --init, so they start fresh")
    agent.set_trunk_frozen(True)
    agent.set_warmup(True)
    return agent


def warm_up(agent, buf, updates, chunk_b, chunk_t, log_every=500):
    K = int(agent.cfg.burn_in)
    t0, m, trace = time.time(), {}, []
    for i in range(int(updates)):
        m = agent.update(buf.sample_chunks(chunk_b, chunk_t + K, K))
        if (i + 1) % log_every == 0 or i + 1 == updates:
            row = {"update": i + 1, "q_g": float(m.get("q_g", float("nan"))),
                   "critic_loss_g": float(m.get("critic_loss_g", float("nan")))}
            trace.append(row)
            print(f"  update {i + 1}: q_g {row['q_g']:.2f} critic_loss_g "
                  f"{row['critic_loss_g']:.3f} ({time.time() - t0:.0f} s)", flush=True)
    return trace


@torch.no_grad()
def slope(agent, probe_paths, device, n_batches=4, chunk_b=16, length=40):
    """critic_slope.py's measurement, on device, plus per-critic agreement and a
    split by mode (state column 0: 1 = goal)."""
    cfg = agent.cfg
    lbuf, _ = _buffer(probe_paths, 2, device, seed=1)
    K = int(cfg.burn_in)
    gain = {(d, dl): [] for d in (0, 1) for dl in DELTAS}
    best, modes, agree, per_critic = {0: [], 1: []}, [], {0: [], 1: []}, {0: [], 1: []}
    q0s, pi_gap, known = [], [], []
    kcol = pxr.mono_priv_cols()[1] if pxr.PRIV_OBS else None
    for _ in range(n_batches):
        ch = lbuf.sample_chunks(chunk_b, length + K, K)
        obs, keep, valid, actions = (ch[k].to(device=device, dtype=torch.float32)
                                     for k in ("obs", "keep", "valid", "actions"))
        goal = ch["goal"].to(device=device, dtype=torch.float32) if cfg.goal_tokens else None
        B, T = valid.shape
        N = B * T
        s = agent._roll(agent.compress, agent.act_embed, agent.core, obs, keep, goal)[:, :-1]
        s = s.reshape(N, cfg.hidden)
        st = agent._state(obs[:, K:-1]).reshape(N, cfg.state_dim)
        a0 = actions.reshape(N, cfg.action_dim)
        v = valid.reshape(N) > 0

        def qk(a):
            return agent._q(agent.critics_g, s, st, agent.goal_cols, a).float().mean(dim=2)

        q0k = qk(a0)
        q0 = q0k.mean(0)
        q0s.append(q0[v].cpu().numpy())
        modes.append(st[:, 0][v].cpu().numpy())
        if kcol is not None:
            known.append(st[:, kcol][v].cpu().numpy())
        for d in (0, 1):
            per = []
            for dl in DELTAS:
                a = a0.clone()
                a[:, d] = (a[:, d] + dl).clamp(-1.0, 1.0)
                gk = qk(a) - q0k
                g = gk.mean(0)
                gain[(d, dl)].append(g[v].cpu().numpy())
                per.append(g)
                if dl == -0.2:
                    gv = gk[:, v]
                    per_critic[d].append(gv.mean(1).cpu().numpy())
                    agree[d].append(((gv > 0).all(0) | (gv < 0).all(0)).float().cpu().numpy())
            best[d].append(np.asarray(DELTAS)[torch.stack(per).argmax(0)[v].cpu().numpy()])
        mean_u, _ = agent._actor_forward(torch.cat([s, st], -1))
        pi_gap.append((qk(torch.tanh(mean_u)).mean(0) - q0)[v].cpu().numpy())
    mode = np.concatenate(modes)
    out = {"states": int(len(mode)), "goal_states": int((mode > 0.5).sum()),
           "q_expert_mean": float(np.concatenate(q0s).mean()),
           "q_actor_minus_expert": float(np.concatenate(pi_gap).mean())}
    for d, name in ((0, "steer"), (1, "throttle")):
        g = {dl: np.concatenate(gain[(d, dl)]) for dl in DELTAS}
        b = np.concatenate(best[d])
        out[name] = {
            "mean_gain": {f"{dl:+.2f}": float(g[dl].mean()) for dl in DELTAS},
            "mean_gain_goal_mode": {f"{dl:+.2f}": float(g[dl][mode > 0.5].mean())
                                    for dl in DELTAS} if (mode > 0.5).any() else None,
            "mean_gain_explore_mode": {f"{dl:+.2f}": float(g[dl][mode <= 0.5].mean())
                                       for dl in DELTAS} if (mode <= 0.5).any() else None,
            "best_offset_not_zero": float(np.mean(b != 0)),
            "mean_best_offset": float(b.mean()),
            "per_critic_gain_at_-0.20": np.mean(np.stack(per_critic[d]), 0).tolist(),
            "critics_agree_on_sign_at_-0.20": float(np.concatenate(agree[d]).mean()),
        }
        if known:
            k = np.concatenate(known) > 0.5
            for tag, m in (("known_distance", k), ("unknown_distance", ~k)):
                out[name][f"mean_gain_{tag}"] = (
                    {f"{dl:+.2f}": float(g[dl][m].mean()) for dl in DELTAS} if m.any() else None)
    if known:
        out["known_distance_states"] = int((np.concatenate(known) > 0.5).sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="checkpoint dir whose config.json to use")
    ap.add_argument("--init", required=True, help="the clone to load (critics stay fresh)")
    ap.add_argument("--roots", nargs="+", required=True, help="recorded episode roots")
    ap.add_argument("--towns", nargs="+", default=list(TOWNS))
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="baseline")
    ap.add_argument("--updates", type=int, default=5000)
    ap.add_argument("--chunk-b", type=int, default=8)
    ap.add_argument("--chunk-t", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-root", default=None,
                    help="root of the six probe shards (default: the first --roots)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config.json field, e.g. subset_size=1 (REDQ: the "
                         "target is one random critic, no min) or subset_size=10")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    overrides = parse_overrides(args.set)

    agent = build_agent(args.config, args.init, dev, args.seed, overrides)
    paths = episode_paths(os.pathsep.join(args.roots), args.towns)
    cap = sum(_episode_rows(p) for p in paths)          # transforms only shorten or drop
    buf = DeviceEpisodeBuffer(pxr.mono_row_dim(pxr.BINOC_DIM), 2, cap, dev, seed=args.seed,
                              goal_dim=pxr.PIX, goal_batch_frac=0.5, store_device="cpu",
                              u8_cols=pxr.MONO_PIX)
    t0 = time.time()
    n = fill_buffer(buf, paths, which="exec", transform=VARIANTS[args.variant])
    print(f"[slope] variant {args.variant}: {len(paths)} episode files -> {buf.episodes} "
          f"episodes, {n} executed steps in the replay ({time.time() - t0:.0f} s)", flush=True)
    trace = warm_up(agent, buf, args.updates, args.chunk_b, args.chunk_t)
    agent.save(args.out)

    probe_root = args.probe_root or args.roots[0]
    probe = (sorted(glob.glob(os.path.join(probe_root, "Town05", "ep_*.npz")))[:3]
             + sorted(glob.glob(os.path.join(probe_root, "Town01", "ep_*.npz")))[:3])
    res = slope(agent, probe, dev)
    res.update(variant=args.variant, overrides=overrides, roots=args.roots, towns=args.towns,
               updates=args.updates,
               episodes=int(buf.episodes), steps=int(n), seed=args.seed, warmup=trace,
               config=args.config, init=args.init, probe=probe)
    with open(os.path.join(args.out, "slope.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    for name in ("steer", "throttle"):
        r = res[name]
        print(f"[slope] {name:8s} " + "  ".join(f"{k}:{v:+.3f}" for k, v in r["mean_gain"].items()))
        print(f"[slope] {name:8s} best offset not 0 in {r['best_offset_not_zero']:.0%} "
              f"(mean {r['mean_best_offset']:+.3f}); critics agree on the sign at -0.2 in "
              f"{r['critics_agree_on_sign_at_-0.20']:.0%}")
        for tag in ("known_distance", "unknown_distance"):
            gk = r.get(f"mean_gain_{tag}")
            if gk:
                print(f"[slope] {name:8s} {tag:16s} "
                      + "  ".join(f"{k}:{v:+.3f}" for k, v in gk.items()))
    print(f"[slope] Q(expert) {res['q_expert_mean']:.2f}, Q(actor) - Q(expert) "
          f"{res['q_actor_minus_expert']:+.3f} over {res['states']} states "
          f"({res['goal_states']} goal mode)", flush=True)


if __name__ == "__main__":
    main()
