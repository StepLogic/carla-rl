"""Measure whether V predicts cost-to-go, against two independent truths.

    realised   successful episodes: V(s_0, g) vs steps actually taken
    geodesic   GlobalRoutePlanner along-road distance / 0.283 m per step

The second covers pairs the policy fails, where realised steps do not exist.

A NEGATIVE CONTROL is run alongside: the identical evaluation against an
UNTRAINED critic. If a random V scores rho = 0.6 through goal-block geometry
alone, then 0.8 from the trained one is a far weaker result than it looks.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from vae_nav.value_distance import (METRES_PER_STEP, edge_cost_steps,  # noqa: E402
                                    value_to_steps)
from vae_nav.value_validation import gate_report  # noqa: E402


def pairs_from_episodes(episodes, step_cost=0.1):
    """Successful episodes -> (pred_steps, true_steps, dist_m, v0).

    ONLY ARRIVALS. A timed-out episode has no ground-truth distance; treating
    its length as the truth would teach the gate that failure is just a long
    journey.
    """
    pred, true, dist, v0 = [], [], [], []
    for ep in episodes:
        if not ep.get("arrived"):
            continue
        pred.append(float(value_to_steps(np.array([ep["v0"]]), step_cost)[0]))
        true.append(float(ep["steps"]))
        dist.append(float(ep["goal_dist_m"]))
        v0.append(float(ep["v0"]))
    return (np.asarray(pred), np.asarray(true),
            np.asarray(dist), np.asarray(v0))


def run_episodes(agent, env, n_episodes, seed0, step_cost, max_steps):
    """Roll out, recording V at the START state against the realised length."""
    out = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed0 + i)
        start = np.asarray(info["hero_pos"], np.float64)[:2]
        goal = np.asarray(info["goal_pos"], np.float64)[:2]
        batched = {k: np.asarray(v)[None] for k, v in obs.items()}
        cost0, spread0 = edge_cost_steps(agent, batched, step_cost)
        v0 = -float(cost0[0]) * step_cost

        arrived, steps, done = False, 0, False
        while not done and steps < max_steps:
            obs, _r, term, trunc, info = env.step(agent.eval_actions(obs))
            steps += 1
            arrived = arrived or bool(info.get("arrived") or info.get("is_success"))
            done = bool(term or trunc)
        out.append({"arrived": arrived, "steps": steps,
                    "goal_dist_m": float(np.linalg.norm(goal - start)),
                    "v0": v0, "spread0": float(spread0[0])})
        print(f"  ep {i:3d}  arrived={arrived}  steps={steps:4d}  "
              f"V0={v0:7.2f}  pred={cost0[0]:6.1f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--towns", default="Town02,Town03,Town10HD")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--port", type=int, default=int(os.environ.get("CARLA_PORT", 2000)))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--step-cost", type=float,
                    default=float(os.environ.get("STEP_COST", 0.1)))
    ap.add_argument("--max-steps", type=int,
                    default=int(os.environ.get("TTG_MAX_STEPS", 300)))
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    import train_carla_tqc as T
    from vae_nav.latent_source import LatentSource
    from vae_nav.policy_io import load_agent_state, param_fingerprint

    with open(args.ckpt, "rb") as f:
        ck = pickle.load(f)
    if "agent" not in ck:
        raise SystemExit(f"{args.ckpt} holds no weights (keys: {sorted(ck)})")
    print(f"[Validate] checkpoint step={ck.get('step')}")
    print(f"[Validate] {param_fingerprint(ck['agent'])}")

    latent_source = LatentSource(os.environ.get("VAE_CKPT", ""))
    towns = [t for t in args.towns.split(",") if t]

    probe = T.make_env(towns[0], args.seed, latent_source, eval_mode=True,
                       port=args.port)
    trained = T.build_agent(probe.observation_space, probe.action_space, args.seed)
    # NEGATIVE CONTROL: same architecture, same seed, NO loaded weights.
    control = T.build_agent(probe.observation_space, probe.action_space,
                            args.seed + 1)
    probe.close()
    load_agent_state(trained, ck["agent"])

    os.makedirs(args.out_dir, exist_ok=True)
    for town in towns:
        env = T.make_env(town, args.seed, latent_source, eval_mode=True,
                         port=args.port)
        try:
            print(f"\n=== {town} (trained) ===", flush=True)
            eps = run_episodes(trained, env, args.episodes, args.seed,
                               args.step_cost, args.max_steps)
            print(f"\n=== {town} (UNTRAINED negative control) ===", flush=True)
            ctl = run_episodes(control, env, args.episodes, args.seed,
                               args.step_cost, args.max_steps)
        finally:
            env.close()

        rep = gate_report(*pairs_from_episodes(eps, args.step_cost))
        ctl_rep = gate_report(*pairs_from_episodes(ctl, args.step_cost))
        rep["negative_control_spearman"] = ctl_rep["gates"]["spearman"]["value"]
        rep["town"] = town
        rep["arrival_rate"] = float(np.mean([e["arrived"] for e in eps]))
        rep["heldout"] = town in ("Town03", "Town10HD")

        path = os.path.join(args.out_dir, f"value_validation_{town}.json")
        with open(path, "w") as f:
            json.dump(rep, f, indent=2)

        print(f"\n[Gate] {town}  {'PASS' if rep['passed'] else 'FAIL'}  "
              f"(n={rep['n_pairs']}, arrival={rep['arrival_rate']:.2f}"
              f"{', HELD OUT' if rep['heldout'] else ''})")
        for name, g in rep["gates"].items():
            mark = "ok  " if g["passed"] else "FAIL"
            print(f"   {mark} {name:18s} {g['value']} vs {g['threshold']}")
        print(f"   negative control rho = {rep['negative_control_spearman']:.3f} "
              "(trained rho must beat this by a clear margin)")
        for b, s in rep["bins"].items():
            print(f"     bin {b:12s} n={s['n']:4d}  rho={s['rho']:.3f}")
        print(f"   -> {path}", flush=True)


if __name__ == "__main__":
    main()
