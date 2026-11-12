#!/usr/bin/env python3
"""Entropy map's EXTRA cost (user 2026-11-09: "extra costs is what I am interested
in"): the per-frame forward pass c0 makes while riding along the expert drive
(run_expert_trace -> agent.act_with_stats, once per step). Times (a) the call the
trace makes (encoder + GRU + actor + critic Q) and (b) entropy alone (encoder +
GRU + actor), batch 1, as in the drive. Run under the eval overlay's env.
    time_entropy_forward.py --ckpt checkpoints/il_bc_c0/step_15000 --steps 12332 77887
"""
import argparse, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
import train_carla_sac as T  # noqa: E402
from torch_rl import pixel_rows as pxr  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--steps", type=int, nargs="+", required=True)
ap.add_argument("--n", type=int, default=3000)
a = ap.parse_args()
agent = T.build_agent(pxr.BINOC_DIM, 2, critic_layer_norm=T.checkpoint_critic_layer_norm(a.ckpt),
                      critic_proj_dim=T.checkpoint_critic_proj_dim(a.ckpt))
agent.load(a.ckpt)
rng = np.random.default_rng(0)
frame = (rng.random((pxr.MONO_IMG, pxr.MONO_IMG, 3)) * 255).astype(np.uint8)
row = pxr.mono_row(frame, T._net_mode(np.zeros(1, np.float32)), rng.standard_normal(pxr.BINOC_DIM).astype(np.float32),
                   speed_kmh=15.0)

@torch.no_grad()
def entropy_only(row, hidden):
    x = torch.as_tensor(np.asarray(row, np.float32)[None], device=agent.device)
    g = agent._goal_input(None, 1)
    s, hidden = agent.core.step(agent._embed(agent.compress, agent.act_embed, x, g), hidden)
    _m, log_std = agent._actor_forward(torch.cat([s, agent._state(x)], dim=-1))
    return float(log_std.sum()), hidden

def bench(fn):
    h = agent.reset_hidden(1)
    for _ in range(100):
        h = fn(h)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(a.n):
        h = fn(h)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / a.n

full = bench(lambda h: agent.act_with_stats(row, h, deterministic=True, goal=None)[1])
ent = bench(lambda h: entropy_only(row, h)[1])
print(f"GPU {torch.cuda.get_device_name(0)} | per frame: trace call {full*1e3:.2f} ms, entropy only {ent*1e3:.2f} ms")
for n in a.steps:
    print(f"  {n} drive steps -> trace call {n*full:.1f} s ({n*full/60:.2f} min), entropy only {n*ent:.1f} s")
