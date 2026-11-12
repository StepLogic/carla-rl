#!/usr/bin/env python
"""Per-episode FAULT-STEP counter for the v9 offline eval, requested by the
user 2026-09-21 ("lane crossings are happening alot" -> "measure it").

The eval_arm.py JSON records only HOW episodes ended (the truncation cause of
the final step), so a policy that crosses the lane line for 19 steps and then
arrives shows ZERO crossing evidence. The counting regime pays 10.0 per
crossing step and truncates at 20 cumulative, so the rate itself -- steps, not
events -- is the quantity, and only a per-step counter sees it.

This re-drives the SAME episode set as eval_arm.py (build_eval_episodes, same
seed, same env flags: the caller must replay results/run_env/ plus the gate
and action-box exports exactly as eval_v9_town.sh does) and counts, per
episode, the steps where info says lane_any / off_road / collision -- the same
keys the gated wrapper's counters read (gated_reward.py:576-580).

ONE TOWN PER INVOCATION, fresh server per invocation, as eval_v9_town.sh.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
# train_carla_tqc lives at the WORKTREE ROOT, not under src/ -- eval_arm.py
# gets it for free because it sits in the root; a scripts/ probe does not.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--towns", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--goal-min", type=float, default=None)
    ap.add_argument("--goal-max", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("CARLA_PORT", 2000)))
    args = ap.parse_args()

    os.environ["EVAL_FORCE_MODE"] = "1"
    if args.goal_min is not None:
        os.environ["GOAL_DIST_MIN"] = str(args.goal_min)
    if args.goal_max is not None:
        os.environ["GOAL_DIST_MAX"] = str(args.goal_max)
        os.environ["GOAL_DIST_START"] = str(args.goal_max)
        os.environ["GOAL_DIST_TARGET"] = str(args.goal_max)

    import train_carla_tqc as T
    from vae_nav.latent_source import from_env as latent_from_env
    from vae_nav.policy_io import load_agent_state

    towns = [t for t in args.towns.split(",") if t]
    episodes = T.build_eval_episodes(towns, args.episodes, seed=args.seed)
    latent = latent_from_env()
    env = T.make_env(episodes[0]["town"], seed=episodes[0]["seed"],
                     latent_source=latent, eval_mode=True, port=args.port)
    current_town = episodes[0]["town"]

    agent = T.build_agent(env.observation_space, env.action_space, args.seed)
    with open(args.ckpt, "rb") as fh:
        import pickle
        state = pickle.load(fh)
    if isinstance(state, dict) and "identity" in state:
        from vae_nav.latent_source import LatentSource
        LatentSource.assert_matches(state["identity"], latent)
    load_agent_state(agent, state.get("agent", state))

    per_ep = []
    for ep in episodes:
        if ep["town"] != current_town:
            env.close()
            env = T.make_env(ep["town"], seed=ep["seed"], latent_source=latent,
                             eval_mode=True, port=args.port)
            current_town = ep["town"]
        obs, _ = env.reset(seed=ep["seed"])
        done, n = False, 0
        cross = offroad = coll = 0
        arrived = False
        while not done and n < args.max_steps:
            action = agent.eval_actions(obs)
            obs, _r, term, trunc, info = env.step(action)
            n += 1
            cross += int(bool(info.get("lane_any",
                                       info.get("lane_invasion_any", False))))
            offroad += int(bool(info.get("off_road", False)))
            coll += int(bool(info.get("collision", False)))
            arrived = arrived or bool(info.get("is_success", False))
            done = bool(term or trunc)
        per_ep.append(dict(ep=n, arrived=bool(arrived), crossing=cross,
                           offroad=offroad, collision=coll))
    env.close()

    def col(key, only=None):
        vals = [e[key] for e in per_ep if only is None or e["arrived"] == only]
        v = np.asarray(vals, float)
        return dict(mean=float(v.mean()) if v.size else None,
                    median=float(np.median(v)) if v.size else None,
                    max=float(v.max()) if v.size else None,
                    frac_ep_with_gt0=float((v > 0).mean()) if v.size else None)

    total_steps = sum(e["ep"] for e in per_ep)
    out = dict(towns=towns, band=(f"{args.goal_min:g}-{args.goal_max:g}m"
               if args.goal_min is not None and args.goal_max is not None
               else "default"),
               checkpoint=args.ckpt, episodes=len(per_ep), seed=args.seed,
               all= dict(crossing=col("crossing"), offroad=col("offroad"),
                         collision=col("collision")),
               arrived_only=dict(crossing=col("crossing", only=True)),
               not_arrived=dict(crossing=col("crossing", only=False)),
               crossing_step_share_of_steps=float(
                   sum(e["crossing"] for e in per_ep) / max(1, total_steps)),
               total_steps=total_steps,
               per_episode=per_ep)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    c = out["all"]["crossing"]
    print(f"[Faults {args.towns}] crossing steps/ep mean={c['mean']:.1f} "
          f"median={c['median']:.0f} max={c['max']:.0f} "
          f"frac_eps_with>=1={c['frac_ep_with_gt0']:.2f} | "
          f"offroad={out['all']['offroad']['mean']:.1f} "
          f"collision={out['all']['collision']['mean']:.1f} | "
          f"crossing share of all steps={out['crossing_step_share_of_steps']:.3f}"
          f" -> {args.out}", flush=True)


if __name__ == "__main__":
    main()