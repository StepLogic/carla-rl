#!/usr/bin/env python3
"""The critic's steering preference at a checkpoint (plan 2026-11-09, C1; promoted
from the job-tmp diag_q_extremes.py that measured every 2026-11-08/09 run).

On recorded EXPERT states (the first Town05 shards, read as bc_loss reads them, the
checkpoint's OWN trunk), the goal critic's Q at the expert's label action with the
steering replaced by each of 21 values in [-1, 1] (throttle kept): the best
steering per state against the expert's (corr, share at full lock, mean when the
expert turns left / right); Q gaps at the extremes; the actor's own mean action.
Gate C reads corr and the full-lock share at 50k decisions.

usage: python scripts/critic_steer_probe.py CKPT PROBE_ROOT [--n-shards 3]
       (SPEED_OBS=1 MONO_CHW=1 for c0-lineage checkpoints; REPLAY_DEVICE=cpu off-GPU)
"""
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "src")]
import numpy as np  # noqa: E402
import torch  # noqa: E402

from torch_rl.bc_trainer import _buffer  # noqa: E402
from torch_rl.gated_recurrent_sac import GatedRecurrentSAC, GatedRecurrentSACConfig  # noqa: E402


def load(ckpt, dev):
    d = json.load(open(os.path.join(ckpt, "config.json")))
    d.update(device=dev, compile=False)
    agent = GatedRecurrentSAC(GatedRecurrentSACConfig(**d))
    agent.load(ckpt)                       # strict: every module
    return agent


ALTS = (("steer -1", 0, -1.0), ("steer -0.5", 0, -0.5), ("steer 0", 0, 0.0),
        ("steer +0.5", 0, 0.5), ("steer +1", 0, 1.0),
        ("throttle 0", 1, -1.0), ("throttle full", 1, 1.0))


@torch.no_grad()
def measure(agent, probe, dev, n_batches=6, chunk_b=16, length=40):
    cfg = agent.cfg
    K = int(cfg.burn_in)
    lbuf, _ = _buffer(probe, 2, dev, seed=1)
    acc = {}
    for _ in range(n_batches):
        ch = lbuf.sample_chunks(chunk_b, length + K, K)
        obs, keep, valid, actions = (ch[k].to(device=dev, dtype=torch.float32)
                                     for k in ("obs", "keep", "valid", "actions"))
        B, T = valid.shape
        N = B * T
        s = agent._roll(agent.compress, agent.act_embed, agent.core, obs, keep, None)[:, :-1]
        s = s.reshape(N, cfg.hidden)
        st = agent._state(obs[:, K:-1]).reshape(N, cfg.state_dim)
        a0 = actions.reshape(N, cfg.action_dim)
        v = valid.reshape(N) > 0

        def q(a):
            return agent._q(agent.critics_g, s, st, agent.goal_cols, a).float().mean(dim=(0, 2))

        q0 = q(a0)
        for name, dim, val in ALTS:
            a = a0.clone()
            a[:, dim] = val
            acc.setdefault(name, []).append((q(a) - q0)[v].cpu().numpy())
        # STATE-DEPENDENCE: the critic's best steering on a grid (throttle kept)
        # per state, against the expert's own steering in that state.
        grid = torch.linspace(-1.0, 1.0, 21, device=a0.device)
        qs = []
        for gv in grid:
            a = a0.clone()
            a[:, 0] = gv
            qs.append(q(a))
        best = grid[torch.stack(qs).argmax(0)]
        acc.setdefault("_best_steer", []).append(best[v].cpu().numpy())
        acc.setdefault("_expert_steer", []).append(a0[:, 0][v].cpu().numpy())
        mean_u, log_std = agent._actor_forward(torch.cat([s, st], -1))
        acc.setdefault("actor mean", []).append((q(torch.tanh(mean_u)) - q0)[v].cpu().numpy())
        acc.setdefault("_q_expert", []).append(q0[v].cpu().numpy())
        acc.setdefault("_actor |u| steer", []).append(mean_u[:, 0].abs()[v].cpu().numpy())
        acc.setdefault("_actor log_std", []).append(log_std.float().mean(-1)[v].cpu().numpy())
        acc.setdefault("_expert |steer|", []).append(a0[:, 0].abs()[v].cpu().numpy())
    out = {}
    best = np.concatenate(acc.pop("_best_steer"))
    exp = np.concatenate(acc.pop("_expert_steer"))
    out["_best_steer_grid"] = {
        "mean": round(float(best.mean()), 3),
        "share_at_+1": round(float((best >= 0.99).mean()), 3),
        "share_at_-1": round(float((best <= -0.99).mean()), 3),
        "corr_with_expert_steer": round(float(np.corrcoef(best, exp)[0, 1]), 3)
        if best.std() > 0 else None,
        # in the states where the expert turns clearly left / right
        "mean_when_expert_left": round(float(best[exp < -0.05].mean()), 3) if (exp < -0.05).any() else None,
        "mean_when_expert_right": round(float(best[exp > 0.05].mean()), 3) if (exp > 0.05).any() else None,
        "states_expert_left_right": [int((exp < -0.05).sum()), int((exp > 0.05).sum())]}
    for k, vals in acc.items():
        x = np.concatenate(vals)
        out[k] = ({"mean": round(float(x.mean()), 3)} if k.startswith("_") else
                  {"gain": round(float(x.mean()), 2), "better": round(float((x > 0).mean()), 3)})
    out["_states"] = int(sum(len(x) for x in acc["_q_expert"]))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("probe_root")
    ap.add_argument("--n-shards", type=int, default=3)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    probe = sorted(glob.glob(os.path.join(args.probe_root, "Town05", "ep_*.npz")))[:args.n_shards]
    res = measure(load(args.ckpt, dev), probe, dev)
    print(json.dumps({"ckpt": args.ckpt, "probe_shards": len(probe), **res}, indent=1))
