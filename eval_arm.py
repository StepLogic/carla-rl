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
    ap.add_argument("--goal-min", type=float, default=None)
    ap.add_argument("--goal-max", type=float, default=None,
                    help="goal distance band in metres. Bands beyond 20 m are "
                         "EXTRAPOLATION: training is pinned to 5-20 m.")
    ap.add_argument("--mode", choices=("goal", "explore"), default="goal",
                    help="which half of a GATED arm to evaluate. A gated arm "
                         "serves both, so measuring only the goal half can "
                         "report it as better while its explore behaviour has "
                         "degraded, invisibly.")
    args = ap.parse_args()

    # Must precede the config import, like every other flag in this trainer.
    os.environ["EVAL_FORCE_MODE"] = "1" if args.mode == "goal" else "0"
    if args.mode == "explore":
        # SUCCESS IS NOT DEFINED IN EXPLORE MODE, whatever the caller asked for.
        # The goal block is zeroed there, so arrival would be measured against a
        # goal the policy cannot see -- the same meaningless zero the goal-free
        # arm reports. Overridden here rather than trusted to the caller.
        args.goal_conditioned = 0

    # THE BAND MUST BE SET BEFORE THE CONFIG IS IMPORTED, because the goal
    # sampler reads these at import like everything else in this trainer.
    #
    # WHY BANDS EXIST AT ALL. In-distribution success saturated at 1.000 for
    # both gated arms within the first evaluation, so a table built on 5-20 m
    # success shows four numbers that are all approximately one and separate
    # nothing. Longer bands are extrapolation and are EXPECTED to be weak --
    # that is what makes them able to discriminate.
    #
    # IT ALSO TESTS HINDSIGHT'S ACTUAL CLAIM. Relabelling supplies goal data at
    # distances the pinned 5-20 m curriculum cannot produce. Evaluating only
    # inside that curriculum could not show the benefit even if it were real.
    if args.goal_min is not None:
        os.environ["GOAL_DIST_MIN"] = str(args.goal_min)
    if args.goal_max is not None:
        os.environ["GOAL_DIST_START"] = str(args.goal_max)
        os.environ["GOAL_DIST_TARGET"] = str(args.goal_max)

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

    def q_at(observation):
        """The goal critic's value of this state under the policy's action.

        WHY THIS IS MEASURED AT ALL. Under the SORB minimum-time reward the
        value IS the negated time to goal, and that reading -- not the success
        rate -- is why the reward was chosen over a shaped one: a topological
        map builder consumes the critic as a DISTANCE. A policy can be excellent
        while its critic is useless as a metric, and nothing else in this
        harness would notice.

        Returns None for arms whose learner has no goal critic to read.
        """
        critic = getattr(agent, "_critic", None)
        if critic is None:
            return None
        try:
            import jax.numpy as jnp
            batched = jax.tree_util.tree_map(lambda x: np.asarray(x)[None],
                                             observation)
            action = np.asarray(agent.eval_actions(observation))[None]
            qs = critic.apply_fn({"params": critic.params}, batched, action)
            return float(jnp.mean(qs))
        except Exception:  # noqa: BLE001 - telemetry must never fail a run
            return None

    import jax  # noqa: E402  (only needed for the tree_map above)

    successes, steps_to_goal, timeouts, lengths = [], [], 0, []
    q0, q0_steps = [], []
    for ep in episodes:
        obs, _ = env.reset()
        q_start = q_at(obs)
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
            # CALIBRATION IS ONLY DEFINED ON EPISODES THAT ARRIVED. For a
            # timeout the true cost-to-go is unknown -- the goal was never
            # reached -- so including it would compare the critic against a
            # censored label and flatter or damn it arbitrarily.
            if q_start is not None:
                q0.append(q_start)
                q0_steps.append(float(n))
        elif n >= args.max_steps:
            timeouts += 1
    env.close()

    goal_conditioned = bool(args.goal_conditioned)
    band = (f"{args.goal_min:g}-{args.goal_max:g}m"
            if args.goal_min is not None and args.goal_max is not None
            else "default")
    out = {
        "arm": args.arm,
        "mode": args.mode,
        # Explore mode has no goal, so distance bands do not apply to it.
        "band": band if args.mode == "goal" else "n/a",
        # Training is pinned to 5-20 m, so anything beyond it is extrapolation
        # and is expected to be weak. Recorded per row so a reader cannot mistake
        # an extrapolation number for an in-distribution one.
        "in_distribution": (args.goal_max is None or args.goal_max <= 20.0),
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
        # CRITIC-AS-DISTANCE. Under minimum time Q(s0) should equal -(steps
        # this episode then took). q_bias is the mean signed error in STEPS, so
        # -570 means the critic thinks the goal is 570 steps further away than
        # it is. q_corr is the correlation between predicted and actual: a
        # critic can be badly biased yet still ORDER states correctly, which is
        # all a map builder needs if it compares distances rather than reading
        # them absolutely. Reporting both separates those two failures.
        "q_bias_steps": (float(np.mean(np.array(q0) + np.array(q0_steps)))
                         if len(q0) >= 2 else None),
        "q_corr": (float(np.corrcoef(np.array(q0), -np.array(q0_steps))[0, 1])
                   if len(q0) >= 2 and np.std(q0) > 0 else None),
        "q_n": len(q0),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    succ = (f"{out['success']:.3f}" if goal_conditioned
            else "n/a (not a goal-conditioned arm)")
    med = ("n/a" if out["steps_median"] is None else f"{out['steps_median']:.0f}")
    bias = ("n/a" if out["q_bias_steps"] is None
            else f"{out['q_bias_steps']:+.0f} steps")
    corr = "n/a" if out["q_corr"] is None else f"{out['q_corr']:.2f}"
    print(f"[Arm {args.arm}] success={succ} "
          f"steps_to_goal_median={med} (n={out['n_arrived']}) "
          f"episode_len_mean={out['episode_len_mean']:.1f} "
          f"timeouts={out['timeout_frac']:.3f} -> {args.out}", flush=True)
    print(f"[Arm {args.arm}] critic-as-distance: bias={bias} corr={corr} "
          f"(n={out['q_n']}) -- bias is how far wrong the value is IN STEPS; "
          f"corr is whether it still ORDERS states correctly", flush=True)


if __name__ == "__main__":
    main()
