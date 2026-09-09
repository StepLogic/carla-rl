"""Refuse gated runs that cannot learn, before the simulator starts.

WHY A SEPARATE CHECK FROM THE WRAPPER'S. vae_goal_wrapper.check_gated_coherence
sees only the observation flags, because that is all it owns. This one sees the
whole run: which reward is enabled, which perception backend, the batch split,
the discounts. Those are the settings that make a gated run silently
meaningless rather than loudly broken, and the failures they cause look like
"the method does not work" rather than "the run was misconfigured".

The house already has this pattern: tests/test_lane_explorer_config.py exists
purely to pin environment-variable coherence for the explorer arm, after a
stale goal reward leaked into a goal-free run.
"""
import os


def _flag(env, name, default="0"):
    return str(env.get(name, default)) == "1"


def _num(env, name, default):
    try:
        return float(env.get(name, default))
    except (TypeError, ValueError):
        raise ValueError(f"{name}={env.get(name)!r} is not a number")


def check_gated_run(env=None):
    """Raise ValueError on an incoherent gated configuration. Inert otherwise."""
    env = os.environ if env is None else env
    if not _flag(env, "GATED"):
        return

    if not _flag(env, "OBS_WITH_MODE"):
        raise ValueError(
            "GATED=1 requires OBS_WITH_MODE=1. Without the mode key the actor "
            "cannot tell which mode it is in, both critics are selected by a "
            "bit that is not in the observation, and the run degenerates to a "
            "goal-conditioned policy trained on two incompatible reward "
            "scales.")

    if not _flag(env, "OBS_WITH_GOAL", "1"):
        raise ValueError(
            "GATED=1 requires the goal block (OBS_WITH_GOAL=1). Gating zeroes "
            "the goal in explore mode; it does not remove it, because one "
            "actor cannot accept two input shapes. If you want the goal "
            "genuinely absent that is the separate goal-free arm, which is the "
            "third leg of the ablation rather than a mode of this one.")

    if not _flag(env, "OBS_WITH_VECTOR"):
        raise ValueError(
            "GATED=1 requires the vector block (OBS_WITH_VECTOR=1). BOTH gated "
            "rewards are speed-gated, so a policy that cannot observe its own "
            "speed sees the gate as an unobservable coin flip.")

    perception = str(env.get("PERCEPTION", "vae")).lower()
    if perception not in ("vae", ""):
        raise ValueError(
            f"GATED=1 requires PERCEPTION=vae, not {perception!r}. The gated "
            "arm unifies the two 128-d autoencoder lane arms. The explorer's "
            "384-d MobileNet backend would change the observation width, and "
            "policy_io.load_agent_state would then refuse every checkpoint on "
            "the other side -- loudly, and correctly.")

    if _flag(env, "LANE_EXPLORE"):
        raise ValueError(
            "GATED=1 is incompatible with LANE_EXPLORE=1. LaneExploreReward is "
            "the SEPARATE explorer arm, with its own observation wrapper, its "
            "own perception backend and three extra mechanisms (visit ring, "
            "pose, image novelty). Gating against it would vary four things at "
            "once and the ablation would measure none of them. The gated arm's "
            "explore half is the goal-free lane reward, which GatedReward "
            "computes itself.")

    explore_prob = _num(env, "EXPLORE_PROB", 0.5)
    if not 0.0 < explore_prob < 1.0:
        raise ValueError(
            f"EXPLORE_PROB={explore_prob} must lie strictly between 0 and 1. At "
            "0 or 1 only one mode is ever drawn, one critic never trains, and "
            "the run is an ordinary single-mode policy carrying a dead second "
            "critic and a constant input.")

    goal_frac = _num(env, "GOAL_BATCH_FRAC", 0.5)
    if not 0.0 < goal_frac < 1.0:
        raise ValueError(
            f"GOAL_BATCH_FRAC={goal_frac} must lie strictly between 0 and 1. A "
            "degenerate split starves one critic entirely, which is the exact "
            "failure the mode-split buffer exists to prevent: the peer measured "
            "goal-mode evaluation collapsing from 40-65% to 15% when the goal "
            "critic received 10-37% of each batch.")

    # NOT AN ERROR, BUT WORTH SAYING OUT LOUD. The two arms cap episodes very
    # differently (2000 against 500), so one mode contributes far more rows than
    # the other. That is exactly what the fixed batch composition neutralises.
    goal_cap = _num(env, "MAX_EPISODE_STEPS", 2000)
    explore_cap = _num(env, "GATED_EXPLORE_MAX_STEPS", 500)
    if goal_cap != explore_cap:
        print(f"[GatedConfig] episode caps differ on purpose: goal={goal_cap:g} "
              f"explore={explore_cap:g}. Row counts will be skewed toward "
              f"{'goal' if goal_cap > explore_cap else 'explore'}; "
              f"GOAL_BATCH_FRAC={goal_frac:g} fixes batch composition "
              "regardless.")

    print(f"[GatedConfig] OK  explore_prob={explore_prob:g} "
          f"goal_frac={goal_frac:g} perception={perception or 'vae'} "
          f"discount={_num(env, 'DISCOUNT', 0.99):g} "
          f"discount_e={_num(env, 'EXPLORE_DISCOUNT', 0.99):g}")
