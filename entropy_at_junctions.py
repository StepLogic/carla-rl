"""Policy entropy along a trajectory, labelled by whether the car is in a junction.

    python entropy_at_junctions.py --ckpt checkpoints/carla_lanefollow/checkpoint_50000.pkl \
        --town Town02 --episodes 8 --goal-min 40 --goal-max 113

WHAT IS MEASURED, AND WHY IT IS THE UNDERLYING GAUSSIAN.

The actor is NormalTanhPolicy: an MLP head emits `means` and `log_stds`, a
MultivariateNormalDiag is built from them, and a Tanh (plus an affine rescale to
the action bounds) is applied on top. Two different entropies exist here and
they answer different questions:

  * the TRANSFORMED entropy, of the squashed distribution actually sampled
    from. It has no closed form and is bounded by the action box, so it
    saturates: once the Gaussian is wide relative to the tanh's linear region,
    pushing it wider barely moves the squashed entropy at all.
  * the BASE entropy, of the Gaussian before the squash. Analytic:

        H = sum_i log(sigma_i) + (D/2) * log(2 * pi * e)      D = 2

    and unbounded, so it keeps responding after the squashed one has flattened.

jaxrl2 already commits to the base one -- normal_tanh_policy.py:50 overrides
`entropy()` to return `self.distribution.entropy()`, deliberately skipping the
bijector. This script uses that same quantity.

PER-DIMENSION SIGMA IS RECORDED TOO, and is the more informative number.
Entropy is a sum over dimensions, so a rise in H does not say WHICH control got
less certain. At a junction the interesting hypothesis is specifically that
STEER widens (several exits are viable) while throttle does not, and only the
per-dim sigmas can separate that from a uniform inflation of both.

THE ENTROPY IS READ BEFORE THE STEP, from the observation the policy is about
to act on, so the label `in_junction` and the entropy describe the same instant.
Reading it after stepping would pair each entropy with the NEXT state's label,
which at a junction boundary is exactly the wrong one.

ACTIONS ARE DETERMINISTIC (the distribution's mode). The trajectory is then a
clean, reproducible path; the entropy recorded at each visited state is still
the full stochastic policy's spread at that state. Sampling actions instead
would make the entropy trace a function of a random walk and two runs would not
be comparable.

CAVEAT ON THE NULL. Steps within an episode are heavily autocorrelated -- a
junction traversal is tens of consecutive steps -- so treating steps as
independent samples would inflate n by roughly the traversal length and
manufacture significance. The comparison here is therefore clustered BY
EPISODE: one in-junction mean and one on-road mean per episode, compared
pairwise, with the bootstrap resampling episodes rather than steps.
"""
import argparse
import json
import os
import pickle
import sys
from functools import partial

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


def make_entropy_fn():
    """(apply_fn, params, obs) -> (base-Gaussian entropy, per-dim sigma).

    Both come off the SAME forward pass. Calling the actor twice would double
    the cost and, if anything stochastic ever entered the network, would not be
    guaranteed to agree.
    """
    import jax

    @partial(jax.jit, static_argnames="apply_fn")
    def _fn(apply_fn, params, obs):
        dist = apply_fn({"params": params}, obs)
        # .distribution is the pre-tanh MultivariateNormalDiag. Reaching for it
        # explicitly rather than trusting the wrapper's entropy() override, so
        # this stays correct if that override is ever changed.
        base = dist.distribution
        return base.entropy(), base.stddev()

    return _fn


def rollout(agent, env, entropy_fn, n_episodes, max_steps, seed0):
    """One record per step, plus a per-episode summary."""
    apply_fn, params = agent._actor.apply_fn, agent._actor.params
    episodes = []

    for ep in range(n_episodes):
        seed = seed0 + ep * 97
        obs, info = env.reset(seed=seed)
        steps = []
        done, t, success = False, 0, False

        while not done and t < max_steps:
            # BEFORE the step: entropy and label describe the same instant.
            # `info` is REBOUND by env.step below, so the pre-step one is held
            # under its own name -- reading pose off the post-step `info` would
            # silently pair each entropy with the NEXT state's position.
            h, sigma = entropy_fn(apply_fn, params, obs)
            pre_info = info
            in_junction = bool(pre_info.get("in_junction", False))

            action = agent.eval_actions(obs)
            obs, reward, term, trunc, info = env.step(action)

            # POSE IS RECORDED FROM THE PRE-STEP INFO, like the entropy and the
            # junction label, so all three describe the same instant. Taking
            # the position from the post-step info would place every sample one
            # control tick ahead of the entropy that was computed for it --
            # about 0.28 m at the measured step size, which is small on a road
            # and is exactly the size of error that matters when the question
            # being asked is WHERE to put a node.
            pos = pre_info.get("hero_pos")
            steps.append({
                "t": t,
                "entropy": float(np.asarray(h).reshape(-1)[0]),
                "sigma": [float(x) for x in np.asarray(sigma).reshape(-1)],
                "in_junction": in_junction,
                "x": float(np.asarray(pos).reshape(-1)[0]) if pos is not None else None,
                "y": float(np.asarray(pos).reshape(-1)[1]) if pos is not None else None,
                "yaw": float(pre_info.get("hero_yaw", 0.0) or 0.0),
                # speed_factor, not speed: the env publishes no raw speed, and
                # this one is clip(speed/target_speed, 0, 1) -- so it SATURATES
                # at the 20 km/h target and cannot distinguish 20 from 40.
                # Recorded under its real name so it is not misread as m/s.
                # Also pre-step, for the same reason as the pose.
                "speed_factor": float(pre_info.get("speed_factor", 0.0) or 0.0),
                "heading_alignment": float(pre_info.get("heading_alignment", 0.0) or 0.0),
                # POST-step, and correctly so: this is the reward EARNED BY the
                # action taken from the state above, not a property of it.
                "reward": float(reward),
                "distance_completed": float(info.get("distance_travelled", 0.0) or 0.0),
            })
            success = success or bool(info.get("is_success", False))
            done = bool(term or trunc)
            t += 1

        reasons = [k for k, v in (info.get("done_reasons") or {}).items() if v]
        n_j = sum(s["in_junction"] for s in steps)
        print(f"  ep {ep:2d} seed={seed:5d} steps={len(steps):4d} "
              f"junction_steps={n_j:4d} ({n_j / max(len(steps), 1):.1%})  "
              f"success={success}  reasons={reasons}")
        episodes.append({
            "episode": ep, "seed": seed, "success": success,
            "reasons": reasons, "steps": steps,
        })

    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--port", type=int, default=int(os.environ.get("CARLA_PORT", 2000)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--goal-min", type=float, default=40.0)
    ap.add_argument("--goal-max", type=float, default=113.0)
    ap.add_argument("--max-steps", type=int,
                    default=int(os.environ.get("MAX_EPISODE_STEPS", 2000)))
    ap.add_argument("--out", default="results/entropy_junctions.json")
    args = ap.parse_args()

    # The band is set BEFORE the env is built; carla_core.reset_hero reads
    # these at every reset. Same mechanism eval_policy.py uses per cell.
    os.environ["GOAL_DIST_MIN"] = str(args.goal_min)
    os.environ["GOAL_DIST_MAX"] = str(args.goal_max)

    import train_carla_tqc as T
    from vae_nav.latent_source import LatentSource
    from vae_nav.policy_io import load_agent_state, param_fingerprint

    with open(args.ckpt, "rb") as f:
        ck = pickle.load(f)
    if "agent" not in ck:
        raise SystemExit(f"{args.ckpt} holds no weights (keys: {sorted(ck)})")
    print(f"[Entropy] checkpoint step={ck.get('step')}")
    print(f"[Entropy] {param_fingerprint(ck['agent'])}")
    print(f"[Entropy] goal band {args.goal_min}-{args.goal_max} m, "
          f"{args.episodes} episodes on {args.town}")

    latent_source = LatentSource(os.environ.get("VAE_CKPT", ""))
    env = T.make_env(args.town, args.seed, latent_source, eval_mode=True,
                     port=args.port)
    agent = T.build_agent(env.observation_space, env.action_space, args.seed)
    load_agent_state(agent, ck["agent"])
    print("[Entropy] weights loaded\n")

    try:
        episodes = rollout(agent, env, make_entropy_fn(), args.episodes,
                           args.max_steps, args.seed)
    finally:
        env.close()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "ckpt": args.ckpt, "step": ck.get("step"), "town": args.town,
            "goal_band": [args.goal_min, args.goal_max],
            "episodes": episodes,
        }, f)

    n_j = sum(s["in_junction"] for e in episodes for s in e["steps"])
    n_all = sum(len(e["steps"]) for e in episodes)
    print(f"\n[Entropy] {n_all} steps, {n_j} in junction ({n_j / max(n_all, 1):.1%})"
          f" -> {args.out}")
    if n_j == 0:
        print("[Entropy] WARNING: no junction steps recorded. The comparison "
              "cannot be made -- widen the goal band or add episodes.")


if __name__ == "__main__":
    main()
