"""Test the peer's registered prediction: does epistemic uncertainty INVERT?

THEIR CLAIM. Their uncertainty-based node placer is inverted -- they measure
corr(leg length, epistemic) at -0.59, -0.72 and -0.94 across configurations,
with adjacent legs refused 100% of the time at every percentile. Their proposed
mechanism is VALUE SCALE: epistemic is the standard deviation of critic MEANS,
Q is large near a +600 terminal and barely discounted, so disagreement tracks
VALUE rather than incompetence. Far legs then read as "confident" because every
critic agrees on a near-zero it knows nothing about.

WHY THIS IS WORTH RUNNING HERE. It is a falsifiable prediction about a
MECHANISM, not a number. This CARLA setup shares every ingredient their
explanation depends on -- TQC, a sparse reward, k_goal=600, a terminal on
arrival -- but shares none of the simulator, sensors, task or map construction.
If the mechanism is right the sign must replicate. If it does not, their
explanation is wrong and the negative result may be an artefact of their setup.

WHAT IS MEASURED. For sampled trace pairs (i, j), the observation the policy
would receive at i when asked to reach j -- built through the wrapper's OWN
goal_block, so it is the same transform training used -- then the critic
ensemble is queried and epistemic is taken as the std across critics of each
critic's MEAN quantile. Correlated against the euclidean leg length ||i - j||.

The measurement is deliberately RAW: no min-spacing mask, no thresholding. A
masked criterion reproduces the spacing grid by construction and would tell us
nothing about the criterion itself.
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="traces/town02_trace.pkl")
    ap.add_argument("--ckpt", default="checkpoints/carla_tqc_town02_200m/checkpoint_1000000.pkl")
    ap.add_argument("--pairs", type=int, default=4000)
    ap.add_argument("--max-leg", type=float, default=30.0)
    ap.add_argument("--min-leg", type=float, default=2.0,
                    help="must be >= the arrival radius; below it a 'leg' is "
                         "already inside the goal and the pair is degenerate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/epistemic_inversion.json")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from vae_nav.latent_source import LatentSource
    from vae_nav.vae_goal_wrapper import N_ACTION_HISTORY, VAEGoalWrapper

    with open(args.trace, "rb") as f:
        tr = pickle.load(f)
    pos, yaws, seg = tr["positions"], tr["yaws"], tr["segment_ids"]
    rgb = tr["rgb"]
    print(f"[Epi] trace {len(pos)} samples, {len(np.unique(seg))} segments")

    # Rebuild the agent exactly as training did, then load the weights.
    import train_carla_tqc as T
    from vae_nav.policy_io import load_agent_state

    latent_source = LatentSource(os.environ.get("VAE_CKPT", ""))

    # SPACES BUILT DIRECTLY, so this needs no simulator. make_env would spawn a
    # CARLA connection purely to read two Box shapes, and the port is in use by
    # trace collection. The shapes are pinned by the wrapper and the action
    # space is the env's -- Box([-1, 0], [1, 1]) because throttle has no
    # reverse. load_agent_state checks the loaded pytree against the built one,
    # so a mismatch here fails loudly rather than producing silent nonsense.
    from gymnasium import spaces
    from vae_nav.vae_goal_wrapper import GOAL_DIM
    obs_space = spaces.Dict({
        "goal": spaces.Box(-np.inf, np.inf, (GOAL_DIM,), np.float32),
        "latent": spaces.Box(-np.inf, np.inf, (latent_source.latent_dim,), np.float32)})
    act_space = spaces.Box(np.array([-1.0, 0.0], np.float32),
                           np.array([1.0, 1.0], np.float32), dtype=np.float32)
    agent = T.build_agent(obs_space, act_space, 0)
    with open(args.ckpt, "rb") as f:
        ck = pickle.load(f)
    load_agent_state(agent, ck["agent"])
    print(f"[Epi] policy from step {ck.get('step')}")

    # Sample pairs WITHIN a segment: a pair spanning a trace break is not a leg
    # any follower would ever be asked to execute.
    rng = np.random.default_rng(args.seed)
    idx_by_seg = {s: np.flatnonzero(seg == s) for s in np.unique(seg)}
    usable = [v for v in idx_by_seg.values() if len(v) >= 2]
    pairs = []
    while len(pairs) < args.pairs and usable:
        v = usable[int(rng.integers(0, len(usable)))]
        i, j = (int(rng.integers(0, len(v))) for _ in range(2))
        if i == j:
            continue
        a, b = int(v[min(i, j)]), int(v[max(i, j)])
        d = float(np.linalg.norm(pos[b] - pos[a]))
        # LEGS SHORTER THAN THE ARRIVAL RADIUS ARE NOT LEGS. With a 2.0 m
        # radius, a 1 m pair is already arrived: the episode terminates at once,
        # so Q reflects a single immediate payout rather than the value of
        # travelling. Including them put a low-Q degenerate cluster at the short
        # end and flattened the correlation from -0.63 to -0.18 -- an artifact
        # of the measurement, not a property of the critic.
        if args.min_leg <= d <= args.max_leg:
            pairs.append((a, b, d))
    print(f"[Epi] {len(pairs)} pairs, leg length {args.min_leg}-{args.max_leg} m "
          f"(shorter than arrival radius excluded as degenerate)")

    # Build the observation for each pair through the WRAPPER's transform.
    wrapper = VAEGoalWrapper.__new__(VAEGoalWrapper)      # transform only
    hist = np.zeros((N_ACTION_HISTORY, 2), np.float32)
    goals = np.stack([wrapper.goal_block({
        "hero_pos": pos[a], "hero_yaw": yaws[a],
        "goal_pos": pos[b], "action_history": hist}) for a, b, _ in pairs])
    lats = np.stack([latent_source.latent(rgb[a]) for a, _, _ in pairs]).astype(np.float32)
    legs = np.array([d for _, _, d in pairs])

    # The action the policy would actually take there -- epistemic is a property
    # of a state-ACTION pair, and querying at an arbitrary action would measure
    # something the agent never does.
    obs = {"goal": goals.astype(np.float32), "latent": lats}
    acts = np.asarray(agent.eval_actions(obs), np.float32)

    qs = agent._critic.apply_fn({"params": agent._critic.params}, obs, acts)
    qs = np.asarray(jax.device_get(qs), np.float64)   # (num_qs, batch, n_quant)
    if qs.ndim != 3:
        raise SystemExit(f"unexpected critic output shape {qs.shape}")
    per_critic_mean = qs.mean(axis=-1)                # (num_qs, batch)
    epistemic = per_critic_mean.std(axis=0)           # (batch,)
    qbar = per_critic_mean.mean(axis=0)

    r_epi = float(np.corrcoef(legs, epistemic)[0, 1])
    r_q = float(np.corrcoef(legs, qbar)[0, 1])
    print("\n=== RESULT ===")
    print(f"  corr(leg length, epistemic) = {r_epi:+.3f}"
          f"   [their prediction: NEGATIVE, they measured -0.59/-0.72/-0.94]")
    print(f"  corr(leg length, Qbar)      = {r_q:+.3f}"
          f"   [their mechanism needs Q to fall with leg length]")
    # COEFFICIENT OF VARIATION separates the two limbs of their explanation.
    #
    # Their stated mechanism has been reported as one thing but is two:
    #   (a) disagreement scales with |Q|, so epistemic falls because Q falls
    #   (b) critics agree TRIVIALLY about a far field none of them has seen
    # Raw epistemic cannot tell them apart. epistemic/Qbar can:
    #   CV flat with distance    -> (a), a pure scale effect, their text as-is
    #   CV falling with distance -> (b), a real drop in RELATIVE disagreement,
    #                               and (a) is not what is doing the work
    # Reported per band as well as pooled, because a pooled correlation can
    # hide a non-monotonic shape.
    cv = epistemic / np.maximum(np.abs(qbar), 1e-9)
    r_cv = float(np.corrcoef(legs, cv)[0, 1])

    # PARTIAL CORRELATION, the instrument that works on BOTH systems.
    #
    # CV needs a denominator that does not cross zero. It is valid here (Qbar
    # stays ~1000-1400) but not in the peer's system, where Qbar runs +570 to
    # -96 and epistemic/|Qbar| has a singularity at the crossing -- their CV
    # returned a plausible-looking -0.279 that was an artifact of that pole.
    # partial corr(leg, epistemic | |Qbar|) asks the same question with no
    # denominator, so the two systems can be compared on one instrument instead
    # of two different ones.
    aq = np.abs(qbar)
    r_le = float(np.corrcoef(legs, epistemic)[0, 1])
    r_lq = float(np.corrcoef(legs, aq)[0, 1])
    r_eq = float(np.corrcoef(epistemic, aq)[0, 1])
    r_partial = ((r_le - r_lq * r_eq)
                 / np.sqrt(max((1 - r_lq ** 2) * (1 - r_eq ** 2), 1e-12)))
    print(f"  partial corr(leg, epistemic | |Qbar|) = {r_partial:+.3f}"
          f"   [theirs: -0.607 full / -0.759 on positive Qbar, raw -0.681]")
    print(f"  corr(leg length, epistemic/Qbar) = {r_cv:+.3f}"
          f"   [flat => pure scale (a); falling => relative (b)]")
    print(f"\n{'leg band':>12s} {'n':>6s} {'epistemic':>11s} {'Qbar':>10s} {'CV':>9s}")
    for lo, hi in [(2, 4), (4, 6), (6, 10), (10, 15), (15, 30)]:
        m = (legs >= lo) & (legs < hi)
        if m.sum() < 5:
            continue
        print(f"{lo:5.1f}-{hi:<6.1f} {m.sum():6d} {epistemic[m].mean():11.3f} "
              f"{qbar[m].mean():10.1f} {cv[m].mean():9.4f}")
    verdict = ("REPLICATES -- their mechanism survives a second, independent system"
               if r_epi < -0.1 else
               "DOES NOT REPLICATE -- sign is not negative in CARLA"
               if r_epi > 0.1 else "AMBIGUOUS -- correlation near zero")
    print(f"\n  VERDICT: {verdict}")

    import json
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"corr_leg_epistemic": r_epi, "corr_leg_qbar": r_q,
                   "corr_leg_cv": r_cv, "partial_corr_leg_epi_given_absQ": float(r_partial),
                   "n_pairs": len(pairs), "town": tr["town"],
                   "policy_step": ck.get("step"), "verdict": verdict}, f, indent=2)
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
