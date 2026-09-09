#!/usr/bin/env python
"""Evaluate ONE arm of the gated ablation, on a fixed shared episode set.

ONE ARM PER PROCESS, DELIBERATELY. The four arms have different observation
widths (144 goal-only, 132 goal-free, 145 gated and gated+hindsight) and are
selected by environment variables that several modules read AT IMPORT. Switching
arms inside one process would mean reloading half the package and hoping nothing
cached a flag. A fresh process per arm cannot get that wrong. The driver
(run_four_way_eval.sh) starts CARLA once and runs this four times against it.

THE EPISODE SET IS IDENTICAL ACROSS ARMS, and that is the whole point. It comes
from train_carla_tqc.build_eval_episodes with a fixed seed, which draws from an
RNG never touched by training, so every arm drives the same towns from the same
start states. A comparison over different draws would measure the draw.

WHAT IS REPORTED, AND WHAT IS REFUSED.

    success        fraction of episodes reaching the goal. GOAL-CONDITIONED
                   ARMS ONLY.
    steps          median and mean steps to arrival over SUCCESSFUL episodes.
                   Under the SORB reward this IS the return, negated, so it is
                   the metric the goal critic is trained to predict.
    timeouts       fraction hitting the step cap.

SUCCESS IS REFUSED, NOT REPORTED AS ZERO, FOR THE GOAL-FREE ARM. It measures
arrival at a goal that policy cannot see and is not paid for. The existing
goal-free run reports 0.000 over 60 episodes for exactly that reason, and a
zero in a comparison table reads as "worse" rather than "not applicable". This
script prints "n/a (not a goal-conditioned arm)" instead.

THE GATED ARMS ARE EVALUATED IN GOAL MODE. make_env forces it when
eval_mode=True, so the bit is not drawn. Letting it be drawn would average a
success rate over two different tasks, one of which has no goal.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    help="label for the results file, e.g. gated_her")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--towns", default=os.environ.get("EVAL_TOWNS", "Town02"))
    ap.add_argument("--seed", type=int, default=0,
                    help="episode-set seed; MUST match across arms")
    ap.add_argument("--max-steps", type=int,
                    default=int(os.environ.get("EVAL_MAX_STEPS", 1000)))
    ap.add_argument("--goal-conditioned", type=int, default=1,
                    help="0 for the goal-free arm, whose success is meaningless")
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("CARLA_PORT", 2000)))
    args = ap.parse_args()

    import train_carla_tqc as T
    from vae_nav.latent_source import LatentSource
    from vae_nav.policy_io import load_agent_state

    towns = [t for t in args.towns.split(",") if t]
    episodes = T.build_eval_episodes(towns, args.episodes, seed=args.seed)

    latent = LatentSource.from_env()
    # eval_mode=True: forces GOAL mode on the gated arms and takes the eval
    # branch of every wrapper, so this is the same stack training evaluated.
    env = T.make_env(towns[0], seed=args.seed, latent_source=latent,
                     eval_mode=True, port=args.port)

    agent = T.build_agent(env.observation_space, env.action_space, args.seed)
    with open(args.ckpt, "rb") as fh:
        import pickle
        state = pickle.load(fh)
    load_agent_state(agent, state.get("agent", state))

    successes, steps_to_goal, timeouts, lengths = [], [], 0, []
    for ep in episodes:
        obs, _ = env.reset()
        done, n, ok = False, 0, False
        while not done and n < args.max_steps:
            action = agent.eval_actions(obs)
            obs, _r, term, trunc, info = env.step(action)
            n += 1
            ok = ok or bool(info.get("is_success", False))
            done = bool(term or trunc)
        successes.append(float(ok))
        lengths.append(n)
        if ok:
            steps_to_goal.append(n)
        elif n >= args.max_steps:
            timeouts += 1
    env.close()

    goal_conditioned = bool(args.goal_conditioned)
    out = {
        "arm": args.arm,
        "checkpoint": args.ckpt,
        "episodes": len(episodes),
        "towns": towns,
        "episode_seed": args.seed,
        "goal_conditioned": goal_conditioned,
        # REFUSED, not zeroed, for the goal-free arm.
        "success": (float(np.mean(successes)) if goal_conditioned else None),
        "steps_median": (float(np.median(steps_to_goal))
                         if steps_to_goal else None),
        "steps_mean": (float(np.mean(steps_to_goal)) if steps_to_goal else None),
        "n_arrived": len(steps_to_goal),
        "episode_len_mean": float(np.mean(lengths)),
        "timeout_frac": timeouts / max(1, len(episodes)),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    succ = (f"{out['success']:.3f}" if goal_conditioned
            else "n/a (not a goal-conditioned arm)")
    med = ("n/a" if out["steps_median"] is None else f"{out['steps_median']:.0f}")
    print(f"[Arm {args.arm}] success={succ} "
          f"steps_to_goal_median={med} (n={out['n_arrived']}) "
          f"episode_len_mean={out['episode_len_mean']:.1f} "
          f"timeouts={out['timeout_frac']:.3f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
