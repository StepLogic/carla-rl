"""Learner C's training loop (plan docs/superpowers/plans/2026-11-10-learner-validation.md, Task 3):
a port of fastrlap-mujoco-env/train_online_pixels.py (kylestach/fastrlap-release).

As the reference: random actions for the first `start_training` agent steps; then the learner's
stochastic action, clipped to the action space; mask = 0 only on a true termination (a time
limit keeps bootstrapping); one `update(batch, utd_ratio)` per agent step once training starts,
batch 256, utd 8; the target entropy annealed by -25e-7 every step. A deterministic evaluation
(10 episodes, seeds 1000-1009, the squashed mean action) runs every --eval-every agent steps and
at the end; the result goes to --out-json.

    PYTHONNOUSERSITE=1 ~/learner-venv/bin/python scripts/bench/frl_train.py \
        --env Pendulum-v1 --config redq --env-steps 50000 --out-json checkpoints/bench/t1_C.json
Pixel envs need MUJOCO_GL=egl on the cluster's CPU nodes.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# fastrlap-ros/offroad_learning/config/{redq,drq}_config.py. target_entropy is left to the
# learner's default (-action_dim / 2): the configs' -3.0 / -2.0 are tuned for FastRLAP's 2-D car.
CONFIGS = {
    "redq": {"learner": "sac", "actor_lr": 3e-4, "critic_lr": 3e-4, "temp_lr": 3e-4,
             "hidden_dims": (256, 256), "discount": 0.99, "num_qs": 10, "num_min_qs": 2,
             "critic_layer_norm": True, "tau": 0.005, "init_temperature": 0.1,
             "backup_entropy": False},
    "drq": {"learner": "drq", "actor_lr": 3e-4, "critic_lr": 3e-4, "temp_lr": 1e-4,
            "hidden_dims": (256, 256), "latent_dim": 50,
            "encoder_kwargs": {"features": (32, 32, 32, 32), "filters": (3, 3, 3, 3),
                               "strides": (2, 2, 2, 2)},
            "discount": 0.99, "num_qs": 10, "num_min_qs": 2, "critic_layer_norm": True,
            "tau": 0.005, "init_temperature": 0.5, "backup_entropy": False},
}


def make_env(env_id: str):
    """The env with its action space rescaled to [-1, 1] when it is not already, so the tanh
    actor reaches the whole range (Pendulum's torque is [-2, 2]). The reference loop only clips,
    which is right for its own [-1, 1] envs and would halve Pendulum's torque."""
    import gymnasium as gym
    env = gym.make(env_id)
    lo, hi = env.action_space.low, env.action_space.high
    if not (np.allclose(lo, -1.0) and np.allclose(hi, 1.0)):
        env = gym.wrappers.RescaleAction(env, -1.0, 1.0)
    return env


def mask_for(terminated: bool, truncated: bool) -> float:
    """0 only on a true termination; a time-limit truncation keeps bootstrapping."""
    return 0.0 if terminated else 1.0


class ReplayBuffer:
    """A ring of transitions. Pixel observations stay uint8 and come out as {"pixels": ...}."""

    def __init__(self, obs_shape, obs_dtype, action_dim: int, capacity: int, pixels: bool):
        self.capacity, self.pixels, self.n, self.i = int(capacity), bool(pixels), 0, 0
        self.obs = np.zeros((self.capacity, *obs_shape), obs_dtype)
        self.next_obs = np.zeros((self.capacity, *obs_shape), obs_dtype)
        self.actions = np.zeros((self.capacity, action_dim), np.float32)
        self.rewards = np.zeros(self.capacity, np.float32)
        self.masks = np.zeros(self.capacity, np.float32)

    def __len__(self) -> int:
        return self.n

    def insert(self, obs, action, reward, mask, next_obs) -> None:
        j = self.i
        self.obs[j], self.next_obs[j] = obs, next_obs
        self.actions[j], self.rewards[j], self.masks[j] = action, reward, mask
        self.i = (self.i + 1) % self.capacity
        self.n = min(self.n + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict:
        idx = rng.integers(0, self.n, size=int(batch_size))
        o, no = self.obs[idx], self.next_obs[idx]
        if self.pixels:
            o, no = {"pixels": o}, {"pixels": no}
        return {"observations": o, "actions": self.actions[idx], "rewards": self.rewards[idx],
                "masks": self.masks[idx], "next_observations": no}


def _wrap(obs, pixels: bool):
    """One observation -> a batch of one, in the form the learner takes."""
    return {"pixels": obs[None]} if pixels else obs[None].astype(np.float32)


def evaluate(agent, env_id: str, pixels: bool, episodes: int, seed0: int) -> list:
    env = make_env(env_id)
    returns = []
    for k in range(episodes):
        o, _ = env.reset(seed=seed0 + k)
        done, ret = False, 0.0
        while not done:
            a = agent.eval_actions(_wrap(o, pixels))[0]
            o, r, term, trunc, _ = env.step(a)
            ret += float(r)
            done = bool(term or trunc)
        returns.append(ret)
    env.close()
    return returns


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--device", default="cpu", help="cuda: the learner on the GPU")
    p.add_argument("--config", required=True, choices=sorted(CONFIGS))
    p.add_argument("--env-steps", type=int, required=True, help="environment steps (x action repeat)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--utd-ratio", type=int, default=8)
    p.add_argument("--start-training", type=int, default=1000, help="agent steps of random actions")
    p.add_argument("--replay-size", type=int, default=100000)
    p.add_argument("--no-entropy-anneal", action="store_true")
    p.add_argument("--eval-every", type=int, default=0, help="agent steps; 0 = only at the end")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-seed", type=int, default=1000)
    p.add_argument("--out-json", default="")
    return p


def main():
    args = build_parser().parse_args()

    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    from torch_rl.frl.learner import DrQLearner, SACLearner

    cfg = dict(CONFIGS[args.config])
    kind = cfg.pop("learner")
    env = make_env(args.env)
    repeat = int(getattr(env.unwrapped, "action_repeat", 1))
    agent_steps = args.env_steps // repeat
    pixels = env.observation_space.dtype == np.uint8 and len(env.observation_space.shape) == 3
    act_dim = int(np.prod(env.action_space.shape))
    if kind == "drq":
        if not pixels:
            raise ValueError("the drq config needs a pixel env")
        agent = DrQLearner(pixel_shape=tuple(env.observation_space.shape), action_dim=act_dim,
                           seed=args.seed, device=args.device, **cfg)
    else:
        if pixels:
            raise ValueError("the redq (SAC) config takes a state env")
        agent = SACLearner(obs_dim=int(np.prod(env.observation_space.shape)), action_dim=act_dim,
                           seed=args.seed, device=args.device, **cfg)
    rb = ReplayBuffer(env.observation_space.shape, env.observation_space.dtype, act_dim,
                      args.replay_size, pixels)
    rng = np.random.default_rng(args.seed)
    env.action_space.seed(args.seed)
    print(f"=== FRL {args.config} ({kind}) device {args.device} | env {args.env} | action repeat {repeat} | "
          f"{args.env_steps} env steps = {agent_steps} agent steps | utd {args.utd_ratio} batch "
          f"{args.batch_size} | start {args.start_training} | target entropy {agent.target_entropy} "
          f"(anneal {'off' if args.no_entropy_anneal else '-25e-7/step'}) | {cfg}", flush=True)

    t0, obs = time.time(), env.reset(seed=args.seed)[0]
    ep_ret, train_returns, curve, info = 0.0, [], [], {}
    for i in range(1, agent_steps + 1):
        if not args.no_entropy_anneal:
            agent.target_entropy -= 25e-7
        if i < args.start_training:
            action = env.action_space.sample()
        else:
            action = np.clip(agent.sample_actions(_wrap(obs, pixels))[0],
                             env.action_space.low, env.action_space.high)
        next_obs, reward, term, trunc, _ = env.step(action)
        rb.insert(obs, action, float(reward), mask_for(term, trunc), next_obs)
        ep_ret += float(reward)
        obs = next_obs
        if term or trunc:
            train_returns.append(ep_ret)
            ep_ret, obs = 0.0, env.reset()[0]
        if i >= args.start_training:
            info = agent.update(rb.sample(args.batch_size, rng), utd_ratio=args.utd_ratio)
        if i % 1000 == 0:
            print(f"[FRL] step {i}/{agent_steps} env_step {i * repeat} | train last10 "
                  f"{np.mean(train_returns[-10:]) if train_returns else float('nan'):.1f} | "
                  + " ".join(f"{k} {v:.4g}" for k, v in sorted(info.items()))
                  + f" | {i / (time.time() - t0):.1f} agent steps/s", flush=True)
        if args.eval_every and i % args.eval_every == 0 and i < agent_steps:
            rets = evaluate(agent, args.env, pixels, args.eval_episodes, args.eval_seed)
            curve.append([i * repeat, float(np.mean(rets))])
            print(f"[FRLEval] env_step {i * repeat} eval mean {np.mean(rets):.1f}", flush=True)

    rets = evaluate(agent, args.env, pixels, args.eval_episodes, args.eval_seed)
    curve.append([agent_steps * repeat, float(np.mean(rets))])
    result = {"arm": "C_frl", "config": args.config, "learner": kind, "env_id": args.env,
              "env_steps": agent_steps * repeat, "agent_steps": agent_steps, "action_repeat": repeat,
              "seed": args.seed, "utd_ratio": args.utd_ratio, "batch_size": args.batch_size,
              "target_entropy_final": agent.target_entropy,
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
