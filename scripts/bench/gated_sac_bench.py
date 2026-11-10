"""Learner B on the benchmark tasks (plan docs/superpowers/plans/2026-11-10-learner-validation.md,
Task 4): our GatedRecurrentSAC exactly as the C1 action-repeat runs configured it.

Glue only -- the learning code is the real class and the real replay:
  * the agent is built from a C1 checkpoint's config.json (sac_scratch_ar1) with only the input
    sizes changed (action_dim, the state block, the critic / GRU column lists), device cpu, the
    seed, and the target entropy by C1's own rule (-action_dim; C1 ran -2.0 for 2-D actions);
    DEBiT's pretrained monocular trunk is loaded into the online and target encoders, as
    train_carla_sac.build_agent does;
  * rows in the C1 layout [frame/255, channels first | mode | speed | task state]
    (MONO_CHW=1, SPEED_OBS=1, as C1); mode is always 1 (goal), speed 0; the replay is
    DeviceEpisodeBuffer with C1's settings (uniform transitions, no mode split, uint8 frames);
  * the loop is train_carla_sac.main()'s: random actions for the first 5000 steps, then
    agent.act with the GRU state reset at every episode start; one update of 8 chunks x 40
    steps (burn-in 8) every 5 env steps.
Task state: Pendulum's 3 numbers with a black frame (T1); for pixels (T2) a single 128 px
render per step (the GRU carries history, as on CARLA) and a constant 0 state.

    MONO_CHW=1 SPEED_OBS=1 python scripts/bench/gated_sac_bench.py --env Pendulum-v1 \
        --env-steps 50000 --c1-config checkpoints/sac_scratch_ar1/step_25000/config.json \
        --out-json checkpoints/bench/t1_B.json
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("MONO_CHW", "1")      # C1's frame layout; pixel_rows reads it at import
os.environ.setdefault("SPEED_OBS", "1")     # C1's speed column

import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "src"), os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)

from torch_rl import pixel_rows as pxr  # noqa: E402

# train_carla_sac.py's defaults, which the C1 runs used (banner: "every 5 steps x1, chunk B=8
# T=40 burn_in=8, starts at 5000").
C1_LOOP = {"learning_starts": 5000, "update_every": 5, "updates_per": 1,
           "chunk_b": 8, "chunk_t": 40, "burn_in": 8}

_CHANGED = ("action_dim", "state_dim", "goal_cols", "core_state_cols", "device", "seed",
            "target_entropy")


def loop_settings(learning_starts=None, update_every=None) -> dict:
    """C1's loop, optionally with another start and update rate -- the budget-matched control
    (learner C trains from step 1000 with updates every step): everything else stays C1's."""
    out = dict(C1_LOOP)
    if learning_starts is not None:
        out["learning_starts"] = int(learning_starts)
    if update_every is not None:
        if int(update_every) < 1:
            raise ValueError(f"update_every must be >= 1, got {update_every}")
        out["update_every"] = int(update_every)
    return out


def bench_config(c1: dict, obs_dim: int, act_dim: int, seed: int, device: str = "cpu") -> dict:
    """C1's learner config with only the input sizes, device, seed and the -action_dim entropy
    target changed. State block = [mode | speed | task state]; the critic reads all of it, the
    GRU everything after the mode bit (as C1: goal_cols = all, core_state_cols = 1..)."""
    cfg = dict(c1)
    state_dim = 2 + int(obs_dim)
    cfg.update(action_dim=int(act_dim), state_dim=state_dim, goal_cols=list(range(state_dim)),
               core_state_cols=list(range(1, state_dim)), device=str(device), seed=int(seed),
               target_entropy=-float(act_dim))
    return cfg


def bench_row(frame, state, strict: bool = False) -> np.ndarray:
    """[frame/255 (C1 layout) | mode 1 | speed 0 | state]; the frame part is computed exactly as
    pixel_rows.mono_row computes it for CARLA."""
    f = np.asarray(frame)
    if strict and tuple(f.shape[:2]) != (pxr.MONO_IMG, pxr.MONO_IMG):
        raise ValueError(f"frame {f.shape} is not {pxr.MONO_IMG}x{pxr.MONO_IMG}")
    img = pxr.to_frame(f, pxr.MONO_IMG) / 255.0
    pix = (img.transpose(2, 0, 1) if pxr.MONO_CHW else img).reshape(-1)
    return np.concatenate([pix, np.asarray([1.0, 0.0], np.float32),
                           np.asarray(state, np.float32).reshape(-1)]).astype(np.float32)


def _split(obs, pixels: bool):
    """-> (frame HWC uint8, task state)."""
    if pixels:
        return np.asarray(obs).transpose(1, 2, 0), np.zeros(1, np.float32)
    return np.zeros((pxr.MONO_IMG, pxr.MONO_IMG, 3), np.uint8), np.asarray(obs, np.float32)


def make_env(env_id: str, pixels: bool):
    import gymnasium as gym
    kwargs = {"frame_stack": 1, "image_size": pxr.MONO_IMG} if pixels else {}
    env = gym.make(env_id, **kwargs)
    lo, hi = env.action_space.low, env.action_space.high
    if not (np.allclose(lo, -1.0) and np.allclose(hi, 1.0)):
        env = gym.wrappers.RescaleAction(env, -1.0, 1.0)   # the tanh actor's own range
    return env


def evaluate(agent, env_id, pixels, goal_blk, episodes, seed0):
    env = make_env(env_id, pixels)
    out = []
    for k in range(episodes):
        o, _ = env.reset(seed=seed0 + k)
        hidden, done, ret = agent.reset_hidden(1), False, 0.0
        while not done:
            row = bench_row(*_split(o, pixels))
            a, hidden = agent.act(row, hidden, deterministic=True, goal=goal_blk)
            o, r, term, trunc, _ = env.step(np.asarray(a, np.float32))
            ret += float(r)
            done = bool(term or trunc)
        out.append(ret)
    env.close()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--pixels", action="store_true", help="a pixel env: single 128 px frames")
    p.add_argument("--env-steps", type=int, required=True)
    p.add_argument("--c1-config", required=True, help="a C1 checkpoint's config.json")
    p.add_argument("--debit-weights",
                   default=os.path.join(ROOT, "dependencies", "debit_weights", "debit_tiny.pth"))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cpu",
                   help="cuda: the learner on the GPU, replay frames stored on cpu (as C1)")
    p.add_argument("--capacity", type=int, default=100000)
    p.add_argument("--learning-starts", type=int, default=None, help="default: C1's 5000")
    p.add_argument("--update-every", type=int, default=None, help="default: C1's 5")
    p.add_argument("--eval-every", type=int, default=0, help="agent steps; 0 = only at the end")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-seed", type=int, default=1000)
    p.add_argument("--out-json", default="")
    args = p.parse_args()

    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.manual_seed(args.seed)
    from torch_rl.debit_mono import load_debit_mono
    from torch_rl.episode_buffer import DeviceEpisodeBuffer
    from torch_rl.gated_recurrent_sac import GatedRecurrentSAC, GatedRecurrentSACConfig

    env = make_env(args.env, args.pixels)
    repeat = int(getattr(env.unwrapped, "action_repeat", 1))
    agent_steps = args.env_steps // repeat
    act_dim = int(np.prod(env.action_space.shape))
    obs_dim = 1 if args.pixels else int(np.prod(env.observation_space.shape))
    with open(args.c1_config) as fh:
        c1 = json.load(fh)
    cfg = bench_config(c1, obs_dim, act_dim, args.seed, device=args.device)
    agent = GatedRecurrentSAC(GatedRecurrentSACConfig(**cfg))
    report = load_debit_mono(agent.compress, args.debit_weights)
    agent.compress_targ.load_state_dict(agent.compress.state_dict())
    row_dim = pxr.MONO_PIX + cfg["state_dim"]
    L = loop_settings(args.learning_starts, args.update_every)
    buf = DeviceEpisodeBuffer(row_dim, act_dim, args.capacity, args.device, seed=args.seed,
                              goal_dim=pxr.PIX, uniform_transitions=True, goal_batch_frac=None,
                              store_device="cpu", u8_cols=pxr.MONO_PIX)
    goal_blk = pxr.goal_block(np.zeros((pxr.IMG, pxr.IMG, 3), np.uint8))
    print(f"=== GATED-SAC (learner B, C1 config {args.c1_config}) device {args.device} | env {args.env} | action repeat "
          f"{repeat} | {args.env_steps} env steps = {agent_steps} agent steps | {L} | "
          f"target entropy {cfg['target_entropy']} | DEBiT trunk {report.get('loaded', report)} "
          f"| {{n_critics {cfg.get('n_critics')}, subset {cfg.get('subset_size')}, LN "
          f"{cfg.get('critic_layer_norm')}, clip [{cfg.get('value_floor')}, "
          f"{cfg.get('value_ceiling')}], init alpha {cfg.get('init_temperature')}}}", flush=True)

    t0 = time.time()
    obs = env.reset(seed=args.seed)[0]
    env.action_space.seed(args.seed)
    row, hidden = bench_row(*_split(obs, args.pixels)), agent.reset_hidden(1)
    ep_ret, train_returns, curve, metrics = 0.0, [], [], {}
    for step in range(1, agent_steps + 1):
        if step <= L["learning_starts"]:
            action = env.action_space.sample().astype(np.float32)
        else:
            action, hidden = agent.act(row, hidden, goal=goal_blk)
            action = np.asarray(action, np.float32)
        nobs, reward, term, trunc, _ = env.step(action)
        nrow = bench_row(*_split(nobs, args.pixels))
        buf.add(row, action, float(reward), bool(term), bool(trunc), nrow, 1, goal=goal_blk)
        ep_ret += float(reward)
        row = nrow
        if term or trunc:
            train_returns.append(ep_ret)
            ep_ret = 0.0
            row = bench_row(*_split(env.reset()[0], args.pixels))
            hidden = agent.reset_hidden(1)
        if (step > L["learning_starts"] and step % L["update_every"] == 0
                and buf.sampleable >= L["chunk_t"] + L["burn_in"] + 1):
            for _ in range(L["updates_per"]):
                metrics = agent.update(buf.sample_chunks(L["chunk_b"], L["chunk_t"] + L["burn_in"],
                                                         L["burn_in"]))
        if step % 1000 == 0:
            m = {k: v for k, v in (metrics or {}).items() if isinstance(v, (int, float))}
            print(f"[GSAC] step {step}/{agent_steps} env_step {step * repeat} | train last10 "
                  f"{np.mean(train_returns[-10:]) if train_returns else float('nan'):.1f} | "
                  + " ".join(f"{k} {v:.4g}" for k, v in sorted(m.items())[:12])
                  + f" | {step / (time.time() - t0):.1f} steps/s", flush=True)
        if args.eval_every and step % args.eval_every == 0 and step < agent_steps:
            rets = evaluate(agent, args.env, args.pixels, goal_blk, args.eval_episodes, args.eval_seed)
            curve.append([step * repeat, float(np.mean(rets))])
            print(f"[GSACEval] env_step {step * repeat} eval mean {np.mean(rets):.1f}", flush=True)

    rets = evaluate(agent, args.env, args.pixels, goal_blk, args.eval_episodes, args.eval_seed)
    curve.append([agent_steps * repeat, float(np.mean(rets))])
    result = {"arm": "B_gated_sac", "c1_config": args.c1_config, "env_id": args.env,
              "env_steps": agent_steps * repeat, "agent_steps": agent_steps, "action_repeat": repeat,
              "seed": args.seed, "loop": L, "target_entropy": cfg["target_entropy"],
              "eval_seeds": [args.eval_seed + k for k in range(args.eval_episodes)],
              "eval_returns": rets, "eval_mean": float(np.mean(rets)), "curve": curve,
              "train_last10_mean": float(np.mean(train_returns[-10:])) if train_returns else None,
              "train_episodes": len(train_returns), "seconds": round(time.time() - t0, 1)}
    print("[BenchResult] " + json.dumps(result), flush=True)
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main()
