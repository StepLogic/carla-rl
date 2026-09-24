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


def _positive_return_bound(env, discount):
    """The largest return the sorb goal reward can pay, with the bonuses on.

    GOAL_ARRIVAL_BONUS + max(0, GOAL_SPEED_BONUS - STEP_COST) / (1 - discount).
    The arrival term is exact; the velocity term is a per-step MAXIMUM
    (every remaining step earns at most speed_bonus - step_cost and the
    episode is capped), so the bound is an upper envelope, not a typical
    value. It is what the goal critic's target CEILING must be at least as
    large as -- the trainer's derived ceiling of 0 clips real targets away
    when the bonuses are on (the TQC_VALUE_CEILING warning below).

    At discount >= 1.0 the velocity term has no geometric tail; it collapses
    to the arrival bonus when the velocity bonus does not beat the step
    cost, and is unbounded when it does -- which the warning reports as inf.
    """
    arrival = _num(env, "GOAL_ARRIVAL_BONUS", 0.0)
    speed_bonus = _num(env, "GOAL_SPEED_BONUS", 0.0)
    step_cost = _num(env, "STEP_COST", 1.0)
    per_step = max(0.0, speed_bonus - step_cost)
    if discount >= 1.0:
        return arrival if per_step == 0.0 else float("inf")
    return arrival + per_step / (1.0 - discount)


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

    # THE 2-MODE IMAGE-GOAL PAIRING (IMAGE_GOAL_TASK=1) keeps its velocity in
    # the mode-independent `state` block (image_goal_coherence demands
    # OBS_WITH_STATE=1 there) instead of the 4-d `vector` block, so the
    # requirement below does not apply to it.
    #
    # THE v8 ESCAPE (ledger ruling, Tasks 1+2 review) skips the same
    # requirement: OBS_WITH_VECTOR=0 is permitted with the 9-d state block
    # (OBS_WITH_STATE=1 AND OBS_STATE_COLLISION=0), because the state block's
    # slot 0 IS the observable speed the progress reward assumes -- the same
    # clipped speed/TARGET_SPEED ratio the vector block carried. The collision
    # flag must be dropped: only the 9-d block has the speed ratio at slot 0.
    # With the flag unset every frozen arm keeps today's behaviour unchanged.
    _v8_state_block = (_flag(env, "OBS_WITH_STATE")
                       and not _flag(env, "OBS_STATE_COLLISION", "1"))
    if (not _flag(env, "OBS_WITH_VECTOR") and not _flag(env, "IMAGE_GOAL_TASK")
            and not _v8_state_block):
        raise ValueError(
            "GATED=1 requires the vector block (OBS_WITH_VECTOR=1). BOTH gated "
            "rewards are speed-gated, so a policy that cannot observe its own "
            "speed sees the gate as an unobservable coin flip. "
            "OBS_WITH_VECTOR=0 is permitted only when the velocity still "
            "reaches the observation: the 2-mode image-goal pairing "
            "(IMAGE_GOAL_TASK=1), or the v8 9-d state block (OBS_WITH_STATE=1 "
            "with OBS_STATE_COLLISION=0), whose slot 0 is the same clipped "
            "speed ratio.")

    perception = str(env.get("PERCEPTION", "vae")).lower()
    # PERCEPTION=vc1|vc1_vitb is the 2-mode image-goal arm's backend (added
    # additively; the vae/mobilenet acceptance is unchanged). The checkpoint
    # risk the mobilenet WARNING describes applies to vc1 too -- 768-d
    # observation, no checkpoint shared with any vae or mobilenet run in
    # either direction -- and is refused at load time the same way, by
    # LatentSource.assert_matches / policy_io naming both sides.
    if perception not in ("vae", "", "mobilenet", "vc1", "vc1_vitb"):
        raise ValueError(
            f"GATED=1 accepts PERCEPTION=vae, mobilenet, vc1 or vc1_vitb, "
            f"not {perception!r}.")
    if perception == "mobilenet":
        # RELAXED 2026-09-11 on the user's instruction. This used to be a hard
        # refusal, on the grounds that MobileNet's 384-d latent changes the
        # observation width and checkpoints would not load across the two.
        #
        # THAT RISK IS REAL BUT ALREADY COVERED, AND MORE PRECISELY: every
        # checkpoint stores its encoder identity and
        # LatentSource.assert_matches / policy_io.load_agent_state refuse a
        # mismatch at load time, naming both sides. This guard was a second,
        # blunter copy of that check, and it also blocked a legitimate
        # from-scratch run. It now warns instead of refusing.
        #
        # WHAT IS STILL TRUE AND IS WHY THE WARNING IS LOUD: a mobilenet run
        # shares NO checkpoint with a vae run in either direction, and it must
        # start from scratch -- INIT_FROM a vae checkpoint will fail.
        print("=" * 72)
        print("[GatedConfig] WARNING: PERCEPTION=mobilenet on a GATED run.")
        print("  observation is 384-d (mobilenet_v3_small_f4_pool4), NOT the")
        print("  145-d VAE observation. Checkpoints are NOT interchangeable")
        print("  with any vae run, in either direction. This run must be from")
        print("  scratch: INIT_FROM a vae checkpoint will be refused at load.")
        print("=" * 72, flush=True)

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

    goal_reward = str(env.get("GATED_GOAL_REWARD", "sorb")).lower()
    if goal_reward not in ("sorb", "shaped"):
        raise ValueError(
            f"GATED_GOAL_REWARD={goal_reward!r} is not sorb|shaped")

    discount = _num(env, "DISCOUNT", 0.99)
    discount_e = _num(env, "EXPLORE_DISCOUNT", discount)

    if goal_reward == "sorb":
        # THE SINGLE MOST EXPENSIVE MISCONFIGURATION AVAILABLE HERE.
        #
        # ESCAPE HATCH, 2026-09-11, ON THE USER'S EXPLICIT INSTRUCTION.
        # GATED_ALLOW_DISCOUNTED_SORB=1 downgrades this to a loud warning. It
        # is opt-in per run and deliberately ugly to type, so it cannot happen
        # by accident. The measured cost is unchanged and is printed.
        if discount < 1.0 and _flag(env, "GATED_ALLOW_DISCOUNTED_SORB"):
            _h = 1.0 / (1.0 - discount)
            print("=" * 72)
            print(f"[GatedConfig] WARNING: sorb goal reward with DISCOUNT="
                  f"{discount} (horizon ~{_h:.0f} steps).")
            print("  The goal critic's Q NO LONGER READS AS -(steps to goal).")
            print("  Measured here: goal episodes run ~195 steps, where V is")
            print("  -86.6 at 200 steps and -95.1 at 300 -- 8.5 apart; 400 vs")
            print("  750 steps are 1.7 apart. Cutoffs calibrated in STEPS stop")
            print("  meaning steps, and the SGM graph built on -V is an")
            print("  ORDERING, not a distance. Recalibrate before trusting it.")
            print("=" * 72, flush=True)
        elif discount < 1.0:
            raise ValueError(
                f"GATED_GOAL_REWARD=sorb requires DISCOUNT=1.0, not {discount}. "
                "Under -1 per step the critic IS the cost-to-go, and gamma=0.99 "
                "gives a 1/(1-gamma)=100-step horizon: Q saturates near -100 and "
                "cannot separate a 400-step goal from a 1000-step one (measured "
                "0.18 apart in this repo). The run would train, converge, and "
                "produce a value function that is useless as a distance -- which "
                "is the one thing the minimum-time reward exists to provide.")
        if discount_e >= 1.0:
            raise ValueError(
                f"EXPLORE_DISCOUNT={discount_e} must be < 1.0. The explore arm "
                "has no terminal state to bound its return -- it is paid a "
                "per-step progress bonus and truncates on the clock -- so an "
                "undiscounted explore return has no bounded fixed point and the "
                "critic diverges.")
        step_cost = _num(env, "STEP_COST", 1.0)
        if step_cost <= 0.0:
            raise ValueError(
                f"STEP_COST={step_cost} must be positive. At zero the goal arm "
                "is paid nothing and every policy is optimal; negative pays the "
                "agent to drive forever.")

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

    if goal_reward == "sorb":
        step_cost = _num(env, "STEP_COST", 1.0)
        cap = _num(env, "GATED_GOAL_MAX_STEPS", 1000)
        # THE COUNTING REGIME'S POSITIVE RETURNS (2026-09-24). With an arrival
        # bonus and/or a velocity bonus above the step cost, the goal return
        # is no longer confined to [-step_cost * cap, 0]: arrival pays
        # +GOAL_ARRIVAL_BONUS and clean steps at speed can pay up to
        # GOAL_SPEED_BONUS - STEP_COST each. The banner below says so, and the
        # warning after it points at the ceiling that must grow with them.
        arrival = _num(env, "GOAL_ARRIVAL_BONUS", 0.0)
        speed_bonus = _num(env, "GOAL_SPEED_BONUS", 0.0)
        if arrival > 0.0 or speed_bonus > 0.0 or discount < 1.0:
            bound = _positive_return_bound(env, discount)
            print(f"[GatedConfig] goal reward = SORB minimum time, "
                  f"-{step_cost:g}/step, return in [{-step_cost * cap:g}, "
                  f"{bound:g}]; POSITIVE RETURNS exist (arrival bonus "
                  f"{arrival:g}, velocity bonus {speed_bonus:g}), so the "
                  f"derived target ceiling of 0 CLIPS real targets -- the "
                  f"goal critic's Q no longer reads as -(steps to goal) "
                  f"alone; set TQC_VALUE_CEILING >= {bound:g}")
        else:
            print(f"[GatedConfig] goal reward = SORB minimum time, "
                  f"-{step_cost:g}/step, return in [{-step_cost * cap:g}, 0]; "
                  f"goal critic Q reads as -(steps to goal)")
        # THE START-UP WARNING: the trainer derives the goal-critic target
        # ceiling as 0 under SORB (train_carla_tqc.agent_kwargs), which was
        # exact when every return was negative and now CLIPS the bonuses'
        # positive targets to 0 -- the critic cannot fit them, silently. The
        # fix is the existing export TQC_VALUE_CEILING.
        bound = _positive_return_bound(env, discount)
        ceiling = _num(env, "TQC_VALUE_CEILING", 0.0)
        if bound > ceiling + 1e-9:
            print("=" * 72)
            print(f"[GatedConfig] WARNING: goal-critic target ceiling "
                  f"{ceiling:g} is BELOW the realisable return bound "
                  f"{bound:g}.")
            print("  The arrival bonus and velocity bonus put POSITIVE "
                  "returns in reach; a ceiling of 0 clips every such target")
            print("  to 0, so the critic cannot fit them. Fix: export "
                  f"TQC_VALUE_CEILING={bound:g}  (bound = GOAL_ARRIVAL_BONUS "
                  "+ max(0, GOAL_SPEED_BONUS - STEP_COST) / (1 - DISCOUNT)).")
            print("=" * 72, flush=True)
    else:
        print("[GatedConfig] goal reward = SHAPED (delta + projected), "
              "returns ~order 100")
    print(f"[GatedConfig] OK  explore_prob={explore_prob:g} "
          f"goal_frac={goal_frac:g} perception={perception or 'vae'} "
          f"discount={discount:g} discount_e={discount_e:g}")
