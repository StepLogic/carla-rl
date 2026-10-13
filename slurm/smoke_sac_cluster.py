#!/usr/bin/env python3
"""Does the whole torch recurrent-SAC arm run on an ACG cluster node?

Run under slurm/sac_smoke.sbatch, which starts CARLA 0.9.16 in apptainer first.
It proves the five things that can only be proved there, in order, and prints a
timing for each so the first training run is not also the first measurement:

  1. the CARLA 0.9.16 python client talks to the containerised server
  2. the requested town LOADS OFFSCREEN. This is the open risk: locally, under
     0.9.15, Town05 SEGFAULTS with -RenderOffScreen and only runs windowed. A
     cluster node has no display at all, so if that reproduces under 0.9.16 the
     arm needs a different town and we need to know now, not in hour three.
  3. the v52 wrapper stack steps -- CoreState, LaneFollow, GatedReward and
     DebitGoalWrapper, with the goal camera actually delivering frames
  4. DEBiT-tiny loads, takes the adapters, and encodes at a usable rate
  5. GatedRecurrentSAC takes an update on this GPU

It writes nothing except its log: no checkpoints, no replay. Exit code 0 means
the arm can be launched.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

TOWN = os.environ.get("ENV_TOWNS", "Town05").split(",")[0]
PORT = int(os.environ.get("CARLA_PORT", 2000))
STEPS = int(os.environ.get("SMOKE_STEPS", 60))
TINY = os.environ.get("DEBIT_WEIGHTS",
                      os.path.join(ROOT, "dependencies", "debit_weights",
                                   "debit_tiny.pth"))


def say(stage, msg):
    print(f"[smoke:{stage}] {msg}", flush=True)


def main():
    t_all = time.time()
    import torch
    say("torch", f"{torch.__version__} cuda={torch.cuda.is_available()} "
                 f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA: the arm is GPU-resident, so this is fatal")
    dev = "cuda"

    # 1 -- the client reaches the containerised server.
    import carla
    t = time.time()
    client = carla.Client("localhost", PORT)
    client.set_timeout(120.0)
    say("client", f"client {client.get_client_version()} "
                  f"server {client.get_server_version()} in {time.time() - t:.1f}s")

    # 2 -- the town loads offscreen. Done through our own builder, so this also
    # proves the town reaches CarlaCore (the bug that put every early result on
    # Town01 regardless of what was asked).
    os.environ.setdefault("GATED", "1")
    os.environ.setdefault("PERCEPTION", "debit")
    os.environ.pop("CARLA_START_SERVER", None)
    from torch_rl.carla_env_build import build_env

    # 4 (before the env, so a weight problem fails fast and cheap) -- DEBiT.
    from torch_rl import debit_adapters as da
    from torch_rl.debit_encoder import DebitTinyEncoder
    t = time.time()
    enc = DebitTinyEncoder(TINY, device=dev, half=False)
    say("debit", f"{enc.describe()} in {time.time() - t:.1f}s")
    n_blocks = da.add_encoder_adapters(
        enc.net, bottleneck=int(os.environ.get("ADAPTER_BOTTLENECK", 64)),
        scale=float(os.environ.get("ADAPTER_SCALE", 0.1)))
    say("debit", f"adapters on {n_blocks} encoder blocks | "
                 f"{da.describe_adapters(enc.net)}")

    t = time.time()
    env = build_env(TOWN, seed=0, encoder=enc, port=PORT)
    say("env", f"built, town={TOWN} in {time.time() - t:.1f}s")
    t = time.time()
    obs, info = env.reset()
    say("env", f"reset in {time.time() - t:.1f}s | obs keys {sorted(obs)} "
               f"| binoc {np.shape(obs['binoc'])} encoded={info.get('binoc_encoded')}")
    loaded = client.get_world().get_map().name
    say("env", f"simulator map is {loaded}")
    if TOWN.split("_")[0].lower() not in loaded.lower():
        raise SystemExit(f"asked for {TOWN} but the simulator loaded {loaded}")
    if "goal_rgb" not in info:
        raise SystemExit("no info['goal_rgb']: the goal camera is not delivering")

    # 3 -- step the stack and time the parts.
    rng = np.random.default_rng(0)
    rows, t_step, t_enc = [], 0.0, 0.0
    for i in range(STEPS):
        a = np.array([rng.uniform(-1, 1), rng.uniform(0, 1)], np.float32)
        t = time.time()
        obs, r, term, trunc, info = env.step(a)
        t_step += time.time() - t
        rows.append((float(r), bool(info.get("lane_any", False)),
                     bool(info.get("off_road", False))))
        if term or trunc:
            say("env", f"episode ended at step {i + 1}: "
                       f"{info.get('goal_end', info.get('explore_end', '?'))}")
            obs, info = env.reset()
    say("env", f"{STEPS} steps in {t_step:.1f}s = {STEPS / max(t_step, 1e-9):.1f} "
               f"steps/s | mean reward {np.mean([x[0] for x in rows]):.3f} "
               f"| crossings {sum(x[1] for x in rows)} off-road {sum(x[2] for x in rows)}")

    # 4b -- encoder throughput at the batch the learner will use.
    import torch as _t
    for bs in (1, 16, 64):
        cur = _t.rand(bs, 3, 112, 112, device=dev)
        goal = _t.rand(bs, 3, 112, 112, device=dev)
        enc.forward_tensors(cur, goal)
        _t.cuda.synchronize()
        t = time.time()
        for _ in range(5):
            out = enc.forward_tensors(cur, goal)
        _t.cuda.synchronize()
        say("debit", f"forward batch {bs:3d}: "
                     f"{(time.time() - t) / 5 * 1000:.1f} ms -> {tuple(out.shape)}")

    # 5 -- one learner update on this GPU.
    from torch_rl.gated_recurrent_sac import (GatedRecurrentSAC,
                                              GatedRecurrentSACConfig)
    state_dim = int(np.shape(obs["state"])[0]) + 1          # + the mode bit
    cfg = GatedRecurrentSACConfig(
        embed_dim=int(os.environ.get("SMOKE_EMBED", 16)), grid=8,
        state_dim=state_dim, mode_index=state_dim - 1,
        goal_cols=tuple(range(min(3, state_dim - 1))), explore_cols=(0,),
        hidden=512, visual_dim=512, mlp=(256, 256), burn_in=8,
        device=dev, compile=False, seed=0, explore_actor_warmup=0,
        value_floor_e=-200.0, value_ceiling_e=150.0)
    agent = GatedRecurrentSAC(cfg)
    say("learner", agent.describe())
    B, K, T = 8, cfg.burn_in, 40
    row = cfg.embed_dim * cfg.grid ** 2 + 2 + state_dim
    chunk = {"obs": _t.rand(B, K + T + 1, row),
             "keep": _t.ones(B, K + 1),
             "actions": _t.rand(B, T, 2) * 2 - 1,
             "rewards": _t.randn(B, T),
             "masks": _t.ones(B, T),
             "valid": _t.ones(B, T),
             "mode": (_t.arange(B) % 2).float()}
    agent.update({k: v.to(dev) for k, v in chunk.items()})     # warm up
    _t.cuda.synchronize()
    t = time.time()
    for _ in range(10):
        out = agent.update({k: v.to(dev) for k, v in chunk.items()})
    _t.cuda.synchronize()
    say("learner", f"update {(time.time() - t) / 10 * 1000:.1f} ms | "
                   f"critic_loss_g {out['critic_loss_g']:.3f} "
                   f"q_g {out['q_g']:.2f} q_e {out['q_e']:.2f} "
                   f"alpha_g {out['alpha_g']:.4f}")
    say("gpu", f"{_t.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB peak allocated")
    env.close()
    say("done", f"ALL STAGES PASSED in {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
