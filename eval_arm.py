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
    ap.add_argument("--minimum-time", type=int, default=0,
                    help="1 if this arm was trained on the SORB minimum-time "
                         "reward. Only then is the critic supposed to read as "
                         "-(steps to goal), so only then is the calibration "
                         "measurement defined. The shaped arm's critic "
                         "predicts a shaped return; scoring it against steps "
                         "measures nothing, and it reported a meaningless "
                         "+398 before this flag existed.")
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
    # GOAL_DIST_MIN AND *MAX* ARE THE ONES THE SAMPLER READS.
    #
    # GOAL_DIST_START and GOAL_DIST_TARGET are consumed ONLY by the curriculum,
    # once, at trainer startup (train_carla_tqc.py:720-722); the curriculum then
    # WRITES GOAL_DIST_MIN/MAX as it advances (:743-744), and those are what the
    # goal sampler actually uses.
    #
    # Setting START/TARGET from an evaluator therefore does NOTHING, because no
    # curriculum runs here -- and it fails silently, leaving GOAL_DIST_MAX at
    # its default. Measured: a "5-20 m" band produced an arrival after 427
    # steps, roughly 200 m of driving, because the band was never applied.
    if args.goal_min is not None:
        os.environ["GOAL_DIST_MIN"] = str(args.goal_min)
    if args.goal_max is not None:
        os.environ["GOAL_DIST_MAX"] = str(args.goal_max)
        # Kept consistent so anything that re-derives the band from these, such
        # as the [Eval] banner, agrees with what is actually sampled.
        os.environ["GOAL_DIST_START"] = str(args.goal_max)
        os.environ["GOAL_DIST_TARGET"] = str(args.goal_max)

    import train_carla_tqc as T
    # MODULE-LEVEL FUNCTION, not a classmethod. from_env is deliberately the
    # ONLY place that chooses the perception backend, so that a difference
    # between runs cannot come from two call sites picking differently. The
    # trainer imports it the same way at train_carla_tqc.py:52.
    from vae_nav.latent_source import from_env as latent_from_env
    from vae_nav.policy_io import load_agent_state
    # Per-episode rule accounting (2026-09-27): off-road, collision and every
    # crossing classified legal/illegal, plus path driven. See eval_faults.
    from vae_nav.eval_faults import EpisodeFaults, aggregate

    towns = [t for t in args.towns.split(",") if t]
    episodes = T.build_eval_episodes(towns, args.episodes, seed=args.seed)

    latent = latent_from_env()
    # eval_mode=True: forces the commanded mode on the gated arms and takes the
    # eval branch of every wrapper, so this is the same stack training evaluated.
    env = T.make_env(episodes[0]["town"], seed=episodes[0]["seed"],
                     latent_source=latent, eval_mode=True, port=args.port)
    current_town = episodes[0]["town"]

    agent = T.build_agent(env.observation_space, env.action_space, args.seed)
    with open(args.ckpt, "rb") as fh:
        import pickle
        state = pickle.load(fh)

    # REFUSE A PERCEPTION MISMATCH BEFORE DRIVING ANYWHERE. The identity record
    # is written into every checkpoint so evaluation cannot silently pair a
    # policy with an encoder it never saw -- a 384-d MobileNet checkpoint read
    # under a 128-d autoencoder would not crash, it would produce confident
    # nonsense. The trainer makes the same check in run_evaluation.
    if isinstance(state, dict) and "identity" in state:
        from vae_nav.latent_source import LatentSource
        LatentSource.assert_matches(state["identity"], latent)

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
            from vae_nav.lane_priv import strip_priv
            batched = jax.tree_util.tree_map(lambda x: np.asarray(x)[None],
                                             strip_priv(observation))
            action = np.asarray(agent.eval_actions(observation))[None]
            qs = critic.apply_fn({"params": critic.params}, batched, action)
            return float(jnp.mean(qs))
        except Exception:  # noqa: BLE001 - telemetry must never fail a run
            return None

    import jax  # noqa: E402  (only needed for the tree_map above)

    # PER-STEP Q TRACE (EVAL_Q_TRACE=1, user 2026-09-27: "does q function predict
    # fatal states correctly?"). q_at reads only the goal critic at the start
    # state. This reads, at EVERY step, the critic the actor actually follows in
    # this mode -- _critic in goal mode, _critic_e in explore mode -- at the
    # action the policy then takes, as the actor loss reads it (mean over the
    # ensemble and all quantiles, gated_tqc_learner._gated_actor_update). Rows
    # carry the reward and the fault flags of the step that action produced, so
    # an offline pass can ask whether Q falls BEFORE a fault and whether it
    # matches the realised return-to-go. Written to <out>.qtrace.jsonl.
    q_trace = os.environ.get("EVAL_Q_TRACE", "0") == "1"
    trace_rows = []
    from vae_nav.eval_faults import classify_crossing

    # PROBE CAPTURE (EVAL_Q_TRACE_OBS=1, user 2026-09-27 "can the critic learn
    # from the image latent?" -> "do it"). Alongside each trace row, saves the
    # observation the critic was fed (latent as float16, goal, state) and the
    # TRUE lane geometry of that same state from the CARLA map: signed lateral
    # offset from the lane centre, heading error to the lane, lane width and
    # the junction flag. A probe then asks whether each predicts a fault a few
    # steps ahead. Written to <out>.probe.npz.
    probe_obs = q_trace and os.environ.get("EVAL_Q_TRACE_OBS", "0") == "1"
    probe = {"latent": [], "goal": [], "state": [], "lane": []}

    def world_map_of(e):
        node = e
        for _ in range(16):
            if node is None:
                return None
            if getattr(node, "core", None) is not None:
                return getattr(node.core, "map", None)
            node = getattr(node, "env", None)
        return None

    def lane_geometry(info):
        """[lateral_m, heading_err_rad, lane_width_m, in_junction] at the hero,
        NaN when there is no pose or no map."""
        wmap = world_map_of(env)
        pos = info.get("hero_pos") if info else None
        if wmap is None or pos is None:
            return [np.nan] * 4
        import carla
        import math
        loc = carla.Location(x=float(pos[0]), y=float(pos[1]),
                             z=float(info.get("hero_z", 0.0)))
        wp = wmap.get_waypoint(loc, project_to_road=True,
                               lane_type=carla.LaneType.Driving)
        if wp is None:
            return [np.nan] * 4
        tf = wp.transform
        fwd = tf.get_forward_vector()
        dx, dy = loc.x - tf.location.x, loc.y - tf.location.y
        lateral = fwd.x * dy - fwd.y * dx
        herr = float(info.get("hero_yaw", 0.0)) - math.radians(tf.rotation.yaw)
        herr = (herr + math.pi) % (2 * math.pi) - math.pi
        return [lateral, herr, float(wp.lane_width), float(wp.is_junction)]

    def q_mode(observation, action, attr=None):
        # attr="_critic_p": the PRIVILEGED goal critic (OBS_PRIV runs), which
        # sees the full observation; the image-only critics never see 'priv'.
        attr = attr or ("_critic" if args.mode == "goal" else "_critic_e")
        critic = getattr(agent, attr, None)
        if critic is None:
            return None
        if attr != "_critic_p":
            from vae_nav.lane_priv import strip_priv
            observation = strip_priv(observation)
        import jax.numpy as jnp
        batched = jax.tree_util.tree_map(lambda x: np.asarray(x)[None],
                                         observation)
        qs = critic.apply_fn({"params": critic.params}, batched,
                             np.asarray(action)[None])
        return float(jnp.mean(qs))

    successes, steps_to_goal, timeouts, lengths = [], [], 0, []
    q0, q0_steps = [], []
    end_reasons = {}
    ep_detail = []
    explore_ends = {}
    for ep in episodes:
        # REBUILD ON A TOWN CHANGE, as run_evaluation does. A single-town sweep
        # never triggers this; it exists so a multi-town one is not silently
        # driving the wrong map.
        if ep["town"] != current_town:
            env.close()
            env = T.make_env(ep["town"], seed=ep["seed"], latent_source=latent,
                             eval_mode=True, port=args.port)
            current_town = ep["town"]
        # RESET WITH THE EPISODE'S OWN SEED, as run_evaluation does. Without it
        # the fixed episode set is decorative: every arm drives whatever the
        # env happens to sample, so "identical episodes across arms" -- the
        # property the whole comparison rests on -- silently does not hold.
        # It also produced a 0.000 success rate for an arm whose own training
        # evaluation reads 1.000, because the goal was never resampled.
        obs, info0 = env.reset(seed=ep["seed"])
        q_start = q_at(obs)
        faults = EpisodeFaults(start_pos=(info0 or {}).get("hero_pos"))
        _prev_info = info0 or {}
        done, n, ok = False, 0, False
        while not done and n < args.max_steps:
            action = agent.eval_actions(obs)
            _q = q_mode(obs, action) if q_trace else None
            _qp = (q_mode(obs, action, "_critic_p")
                   if q_trace and args.mode == "goal" else None)
            if probe_obs:
                probe["latent"].append(np.asarray(obs["latent"], np.float16))
                probe["goal"].append(np.asarray(obs["goal"], np.float32))
                probe["state"].append(np.asarray(obs["state"], np.float32))
                probe["lane"].append(lane_geometry(_prev_info))
            obs, _r, term, trunc, info = env.step(action)
            _prev_info = info
            n += 1
            if q_trace:
                _rem = info.get("route_remaining_m", None)
                trace_rows.append(dict(
                    ep=len(ep_detail), t=n - 1, q=_q, q_p=_qp, r=float(_r),
                    term=bool(term), trunc=bool(trunc),
                    off_road=bool(info.get("off_road", False)),
                    collision=bool(info.get("collision", False)),
                    crossing=classify_crossing(info),
                    arrived=bool(info.get("is_success", False)),
                    remaining_m=None if _rem is None else float(_rem),
                    speed_kmh=float(info.get("speed_kmh", 0.0)),
                    target_speed_kmh=float(info.get("target_speed_kmh", 0.0) or 0.0),
                    marks=[str(m).split(".")[-1]
                           for m in (info.get("lane_marking_types") or [])],
                    in_junction=bool(info.get("in_junction", False)),
                    lane_any=bool(info.get("lane_any",
                                           info.get("lane_invasion_any", False))),
                    lane_event_lag=info.get("lane_event_lag"),
                    collision_event_lag=info.get("collision_event_lag")))
            ok = ok or bool(info.get("is_success", False))
            done = bool(term or trunc)
            faults.step(info)
        # WHY THE EPISODE ENDED. Without this a zero success rate is just a
        # zero: it cannot distinguish "the policy drives badly" from "the
        # harness is configured so episodes end before it can arrive", which is
        # exactly the ambiguity that cost several runs here.
        # ARRIVAL IS ONLY A TERMINATOR IN GOAL MODE. In explore mode the goal
        # block is zeroed and arrival does not end the episode, yet the success
        # flag still passes through from the wrapper below -- so keying the
        # label off it reported "arrived=4" for episodes that in fact ran the
        # full 500-step cap. The reason must come from what actually ended the
        # episode.
        if n >= args.max_steps:
            reason = "step_cap"
        elif ok and args.goal_conditioned:
            reason = "arrived"
        else:
            reason = "truncated_or_env_done"
            for key in ("collision", "off_road", "lane_invasion", "wedged",
                        "lane_any"):
                if info.get(key):
                    reason = key
                    break
        end_reasons[reason] = end_reasons.get(reason, 0) + 1
        # EXPLORE MODE'S OWN VERDICT (reached / crossing / off_road / collision /
        # cap), from the wrapper that ended it. 'reached' means it drove
        # EXPLORE_GOAL_DIST_M of path without a terminal fault.
        if args.mode == "explore":
            _xe = str(info.get("explore_end", "step_cap" if n >= args.max_steps else "?"))
            explore_ends[_xe] = explore_ends.get(_xe, 0) + 1
        _arr = (bool(ok) if args.goal_conditioned
                else str(info.get("explore_end", "")) == "reached")
        ep_detail.append(dict(faults.summary(_arr), town=ep["town"],
                              seed=int(ep["seed"]), end=reason))
        # Counted, but NEVER as a termination reason: for an arm that cannot
        # see the goal this is driving past one, not reaching it.
        if ok and not args.goal_conditioned:
            end_reasons["incidental_pass"] = (
                end_reasons.get("incidental_pass", 0) + 1)
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
    # CALIBRATION IS DEFINED ONLY WHERE THE CRITIC IS SUPPOSED TO BE A DISTANCE:
    # a goal-conditioned arm trained on the minimum-time reward. Anything else
    # produces a number with no referent.
    _calib = goal_conditioned and bool(args.minimum_time)
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
        # STEPS-TO-GOAL IS REFUSED FOR NON-GOAL-CONDITIONED ARMS TOO, for the
        # same reason success is. A goal-free policy cannot see the goal and is
        # not paid for it, so any arrival is INCIDENTAL -- it drove past. A
        # median printed beside the gated arms' figures would invite comparing
        # deliberate arrival with stumbling into one, which is the category
        # error the n/a on success exists to prevent. Caught in a smoke test,
        # where the goal-free arm reported 331 steps from a single accidental
        # arrival.
        "steps_median": (float(np.median(steps_to_goal))
                         if steps_to_goal and goal_conditioned else None),
        "steps_mean": (float(np.mean(steps_to_goal))
                       if steps_to_goal and goal_conditioned else None),
        "n_arrived": (len(steps_to_goal) if goal_conditioned else 0),
        "n_incidental_arrivals": (0 if goal_conditioned else len(steps_to_goal)),
        "episode_len_mean": float(np.mean(lengths)),
        "timeout_frac": timeouts / max(1, len(episodes)),
        "end_reasons": end_reasons,
        # CRITIC-AS-DISTANCE. Under minimum time Q(s0) should equal -(steps
        # this episode then took). q_bias is the mean signed error in STEPS, so
        # -570 means the critic thinks the goal is 570 steps further away than
        # it is. q_corr is the correlation between predicted and actual: a
        # critic can be badly biased yet still ORDER states correctly, which is
        # all a map builder needs if it compares distances rather than reading
        # them absolutely. Reporting both separates those two failures.
        # ALSO REFUSED FOR NON-GOAL-CONDITIONED ARMS. Their critic predicts
        # their own lane-following return, not a distance to a goal they cannot
        # see, so scoring it against steps-to-goal would measure nothing.
        "minimum_time": bool(args.minimum_time),
        "q_bias_steps": (float(np.mean(np.array(q0) + np.array(q0_steps)))
                         if _calib and len(q0) >= 2 else None),
        "q_corr": (float(np.corrcoef(np.array(q0), -np.array(q0_steps))[0, 1])
                   if _calib and len(q0) >= 2 and np.std(q0) > 0 else None),
        "q_n": (len(q0) if _calib else 0),
        # Driving-rule accounting (vae_nav.eval_faults): how CLEANLY it drove,
        # not only whether it arrived.
        "faults": aggregate(ep_detail),
        "explore_ends": explore_ends if args.mode == "explore" else None,
        "explore_reached_rate": (round(explore_ends.get("reached", 0)
                                      / max(1, len(episodes)), 3)
                                if args.mode == "explore" else None),
        "episodes_detail": ep_detail,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    if q_trace:
        with open(args.out + ".qtrace.jsonl", "w") as fh:
            for row in trace_rows:
                fh.write(json.dumps(row) + "\n")
        print(f"[QTrace] {len(trace_rows)} steps -> {args.out}.qtrace.jsonl")
    if probe_obs:
        np.savez_compressed(args.out + ".probe.npz",
                            **{k: np.asarray(v) for k, v in probe.items()})
        print(f"[Probe] {len(probe['lane'])} rows -> {args.out}.probe.npz")

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
    why = "  ".join(f"{k}={v}" for k, v in
                    sorted(end_reasons.items(), key=lambda kv: -kv[1]))
    print(f"[Arm {args.arm}] why episodes ended: {why or '(none)'}", flush=True)
    _f = out["faults"]
    print(f"[Arm {args.arm}] rules: clean arrivals {_f['clean_arrival_rate']:.3f} "
          f"(strict {_f['strict_clean_arrival_rate']:.3f}) | any off-road "
          f"{_f['frac_any_off_road']:.3f} | any illegal crossing "
          f"{_f['frac_any_illegal_crossing']:.3f} | any collision "
          f"{_f['frac_any_collision']:.3f} | any lane change "
          f"{_f['frac_any_lane_change']:.3f} | path {_f['mean_path_m']:.1f} m", flush=True)
    if args.mode == "explore":
        print(f"[Arm {args.arm}] explore: reached {out['explore_reached_rate']:.3f} "
              f"ends {explore_ends}", flush=True)
    print(f"[Arm {args.arm}] critic-as-distance: bias={bias} corr={corr} "
          f"(n={out['q_n']}) -- bias is how far wrong the value is IN STEPS; "
          f"corr is whether it still ORDERS states correctly", flush=True)


if __name__ == "__main__":
    main()
