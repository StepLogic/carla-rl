#!/usr/bin/env python
"""SR and SPL per NODE-PLACEMENT METHOD, for one gated policy.

WHAT IS BEING COMPARED. Four ways of choosing graph nodes -- uniform, random,
policy (value coverage) and SGM -- with everything else held identical: one
policy, ONE TRACE, one edge rule, one navigation harness, one episode set. Node
placement is then the only variable, so a difference in SR/SPL is attributable
to it. That is the whole design; anything collected per method would measure
the collection instead.

ONE PROCESS DOES ALL FOUR METHODS, deliberately, and that is what makes the
trace shared. Running a method per process would give each its own rollouts and
quietly reintroduce the variable the comparison is supposed to remove.

THE THREE STAGES.

  1. COLLECT. Drive the gated policy in EXPLORE mode and record a trace:
     position, yaw, the observation, and the fields the goal block is built
     from. Explore mode is what produces coverage; the goal half would drive
     straight lines to sampled goals and map only those.

  2. BUILD. Choose nodes by each method from the SAME candidates, then connect
     them with edges weighted by the critic read as a distance, rejecting any
     edge beyond the horizon the critic was trained on.

  3. NAVIGATE. For each evaluation episode: localise start and goal to nodes,
     plan over the graph, and hand the policy each node in turn as its goal by
     REWRITING THE GOAL BLOCK. Success is the ENV's own arrival flag at the
     ENV's own goal, so the metric is unchanged from every other evaluation in
     this repo.

TWO TRAPS THIS FILE EXISTS TO AVOID, both of which produce confident nonsense
rather than an error:

  * THE MODE BIT IS A SEPARATE OBSERVATION KEY. A trace collected in explore
    mode carries mode=0, and the goal block is ZEROED at mode=0. Querying the
    goal critic with those observations reads the wrong half of a gated policy.
    Every observation is forced to goal mode before the critic ever sees it.

  * STEP_COST MUST BE THE ARM'S OWN, 1.0 for the gated arms against
    value_distance's 0.1 default. Passing the default scales every edge weight
    by ten, which does not fail -- it produces a graph whose distances are all
    out by an order of magnitude and whose horizon test therefore admits
    everything.

SPL'S DENOMINATOR IS CARLA'S OWN ROUTE, not euclidean distance. A straight line
through a building is not a shortest path, and using one would credit every
method with detours it did not avoid. GlobalRoutePlanner is the same planner
the expert oracle drives.
"""
import argparse
import json
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

GOAL_MODE, EXPLORE_MODE = 1, 0


def _assert_town(env, town, stage):
    """The simulator's ACTUAL map must be the town every result is labelled
    with. A quiet mismatch produced weeks of Town01 numbers labelled Town02;
    this fails the stage instead. Printed, so the log proves where it ran."""
    loaded = env.unwrapped.core.world.get_map().name
    if town.split("_")[0].lower() not in loaded.lower():
        raise RuntimeError(f"[{stage}] asked for {town} but the simulator has {loaded}")
    print(f"[Preflight] {stage}: simulator map is {loaded}", flush=True)


def goal_mode(obs):
    """A copy of `obs` commanded into GOAL mode.

    THE GOAL BLOCK IS NOT REPAIRED HERE, only the bit. Callers always follow
    with goal_relabelled_obs, which rebuilds the block through the wrapper's
    own transform; repairing it here as well would create a second code path
    for the same quantity, which is the drift this repo keeps out of the
    relabelling contract.
    """
    out = dict(obs)
    if "mode" in out:
        out["mode"] = np.array([float(GOAL_MODE)], np.float32)
    return out


def find_wrapper(env, attr):
    """Walk the wrapper chain for the one owning `attr`. None if absent."""
    node = env
    seen = 0
    while node is not None and seen < 32:
        if hasattr(node, attr) and not hasattr(getattr(node, attr), "__wrapped__"):
            return node
        node = getattr(node, "env", None)
        seen += 1
    return None


# ------------------------------------------------------------- stage 1 ----

# COLLECT_CONTINUE (user, 2026-09-12): after an explore episode ends -- crash,
# lane departure, off road or the step cap -- the NEXT episode starts at the
# nearest drivable waypoint to where it stopped, instead of teleporting to a
# fresh random spawn.
#
# WHY IT MATTERS FOR THE MAP. The trace IS the candidate set the graph is built
# from. Random respawns sample the town as disconnected islands: the policy
# drives 60-100 m, dies, and reappears somewhere unrelated, so candidates that
# are adjacent in the trace can be hundreds of metres apart in the world.
# Continuing from where it stopped walks the town instead, and adjacency in the
# trace starts to mean adjacency on the road.
# Set COLLECT_CONTINUE=0 for the old behaviour.
COLLECT_CONTINUE = os.environ.get("COLLECT_CONTINUE", "1") == "1"


def find_attr_owner(env, attr, max_depth=12):
    """Walk the wrapper chain by `.env` and return the first object that has
    `attr`, or None.

    NOT `.unwrapped`: the gated wrappers return themselves from it, so
    `env.unwrapped.experiment` raises AttributeError on the collection env --
    measured 2026-09-12, and it made the continuation rule fall back to the
    random sampler silently on every episode.
    """
    seen, cur = 0, env
    while cur is not None and seen < max_depth:
        if hasattr(cur, attr):
            return cur
        cur = getattr(cur, "env", None)
        seen += 1
    return None


def set_continue_origin(exp, wp, ahead_m=30.0):
    """Point the NEXT reset at `wp`.

    BOTH origin AND destination, because CarlaEnv.reset (rlib_integration
    carla_env.py:74) honours the experiment's waypoints only when both are set
    and truthy; with origin alone it silently calls reset_hero, the random
    sampler. MEASURED 2026-09-12: that is exactly what happened to every v6n
    placement trace (median restart gap 96-433 m). The destination is a
    waypoint `ahead_m` along the lane -- explore mode ignores the goal -- or
    the origin itself at a dead end, so a lane end cannot drop back to random.
    """
    ahead = wp.next(ahead_m) if hasattr(wp, "next") else []
    exp.origin = wp
    exp.destination = ahead[0] if ahead else wp


def clear_continue_origin(exp):
    """Remove both, so every later reset uses the default sampler again."""
    for a in ("origin", "destination"):
        if hasattr(exp, a):
            delattr(exp, a)


CONTINUE_SPAWN_TOL_M = 10.0


def continued_spawn_ok(spawn_xy, wp, tol_m=CONTINUE_SPAWN_TOL_M):
    """Did the reset really put the car at the continuation waypoint?

    The check that was missing: setting the attribute proved nothing, and the
    fallback to a random spawn raises no error.
    """
    loc = wp.transform.location
    return bool(np.linalg.norm(np.asarray(spawn_xy, np.float64)[:2]
                               - np.array([loc.x, loc.y])) <= tol_m)


COLLECT_END_CAUSES = ("collision", "lane", "off_road")


def collect_should_end(info, term, trunc, end_on):
    """Budget-mode trace collection: does this step END the episode?

    end_on is a set drawn from COLLECT_END_CAUSES. USER DECISIONS 2026-09-12/13:
    first "never ends unless collision"; then, for the Town02 recollection,
    "reset in nearest waypoint if lane invasion happens / goes offroad".
      collision -- info["collision"]
      lane      -- the explore arm's own crossing terminal (junction-exempt):
                   term with info["explore_end"] == "crossing"
      off_road  -- info["off_road"] on ANY step, because the v6n explore
                   reward does not terminate on it (EXPLORE_OFF_ROAD_FATAL=0)
    Everything else (the 100 m arrival, truncation) is driven through.
    Returns the cause string, or None to keep driving.
    """
    if "collision" in end_on and bool(info.get("collision", False)):
        return "collision"
    if "lane" in end_on and term and info.get("explore_end") == "crossing":
        return "lane"
    if "off_road" in end_on and bool(info.get("off_road", False)):
        return "off_road"
    return None


def collect_action(agent, obs, mode="eval"):
    """The collection action. 'eval' = agent.eval_actions (distribution mode,
    deterministic; every trace before 2026-09-13 ~11:25). 'sample' =
    agent.sample_actions (dist.sample with the agent's own PRNG key, seeded at
    build_agent, so a sampled trace is reproducible for a fixed seed)."""
    if mode == "eval":
        return agent.eval_actions(obs)
    if mode == "sample":
        return agent.sample_actions(obs)
    raise ValueError(f"collect action mode must be 'eval' or 'sample', got {mode!r}")


def parse_end_on(text):
    """'collision,lane,off_road' -> frozenset; rejects unknown causes."""
    causes = frozenset(p.strip() for p in str(text or "collision").split(",") if p.strip())
    bad = causes - set(COLLECT_END_CAUSES)
    if bad or not causes:
        raise ValueError(f"--collect-end-on: unknown or empty causes {sorted(bad)}; "
                         f"choose from {COLLECT_END_CAUSES}")
    return causes


def parse_start_xy(text):
    """'X,Y' -> np.array([x, y]) for --collect-start-xy; '' or None -> None."""
    if text is None or str(text).strip() == "":
        return None
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError(f"--collect-start-xy expects 'X,Y', got {text!r}")
    return np.array([float(parts[0]), float(parts[1])], np.float64)


def continue_origin(last_pos):
    """Where the next collection episode should start, or None for the sampler.

    None when the feature is off, or when there is no previous episode (the
    first one has nowhere to continue from). The caller snaps this to the
    nearest drivable waypoint -- a crash site is often off the road, and
    get_waypoint with project_to_road returns the nearest lane centre.
    """
    if not COLLECT_CONTINUE or last_pos is None:
        return None
    import numpy as _np
    return _np.asarray(last_pos, _np.float64)[:2]


def collect_trace(T, latent, agent, args, resume=None):
    """Drive the explore half and record candidate states.

    Returns a dict of parallel arrays. Candidates are THINNED BY DISTANCE as
    they are recorded: consecutive simulator steps are ~0.28 m apart, so an
    unthinned trace is mostly duplicates and the pairwise critic matrix is
    quadratic in exactly that redundancy.

    SAVED AFTER EVERY EPISODE, AND RESUMABLE, because the task supervisor on
    this machine kills background jobs at random with a spurious low-memory
    message (see run_placement_eval.sh). A 60-episode collection is the longest
    single step in the pipeline, and writing only at the end meant a kill at
    episode 59 cost the whole thing. `episodes_done` records how far it got so
    a later run continues rather than restarts.

    RESUMING CHANGES THE DRAW, and that is acknowledged rather than hidden: the
    RNG is reseeded from the episode index, so a resumed trace is the same
    sequence of seeds it would have drawn anyway. What it cannot preserve is
    the novelty buffer's history within the killed episode, which is not
    carried across episodes in eval_mode regardless.
    """
    explorer_kind = getattr(args, "explorer", "policy_explore")
    os.environ["EVAL_FORCE_MODE"] = str(GOAL_MODE if explorer_kind == "policy_goal"
                                        else EXPLORE_MODE)
    env = T.make_env(args.town, seed=args.collect_seed, latent_source=latent,
                     eval_mode=True, port=args.port)
    _assert_town(env, args.town, "collect")
    wrapper = find_wrapper(env, "goal_block")
    ring, reactive, lo, hi = None, None, None, None
    if explorer_kind == "reactive":
        from vae_nav.reactive_explorer import DepthRing, ReactiveExplorer
        ring = DepthRing(env)
        reactive = ReactiveExplorer(seed=args.collect_seed)
        lo, hi = env.action_space.low, env.action_space.high
    print(f"[Collect] explorer={explorer_kind} "
          f"lane_rule={getattr(args, 'collect_lane_rule', 'strict')}", flush=True)
    if wrapper is None:
        raise SystemExit(
            "no VAEGoalWrapper in the chain: the goal block cannot be rebuilt, "
            "so neither the critic queries nor the subgoal driving can run.")

    if resume is not None:
        obs_list = list(resume["observations"])
        fields_list = list(resume["fields"])
        pos = [np.asarray(p, np.float64) for p in resume["positions"]]
        yaw = list(resume["yaws"])
        seg = list(resume["segments"])
        first_ep = int(resume.get("episodes_done", 0))
    else:
        obs_list, fields_list, pos, yaw, seg = [], [], [], [], []
        first_ep = 0

    def _snapshot(done):
        return {"observations": obs_list, "fields": fields_list,
                "positions": np.asarray(pos, np.float64),
                "yaws": np.asarray(yaw, np.float64),
                "segments": np.asarray(seg, dtype=int),
                "episodes_done": int(done),
                "steps_done": int(steps_done),
                "collect_total_steps": int(getattr(args, "collect_total_steps", 0) or 0),
                "no_timelimit": os.environ.get("NO_TIMELIMIT", "0") == "1",
                "meta": {"explorer": explorer_kind,
                         "collect_lane_rule": getattr(args, "collect_lane_rule", "strict"),
                         "checkpoint": args.ckpt, "town": args.town,
                         "collect_seed": args.collect_seed,
                         "collect_start_xy": getattr(args, "collect_start_xy", "") or None,
                         "collect_end_on": sorted(_end_on),
                         "target_speed_kmh": os.environ.get("TARGET_SPEED"),
                         # USER 2026-09-13 "disable dist200 for collection mode":
                         # budget collection runs with MAX_DISTANCE_M=1e9, so the
                         # env's done_dist flag never sets.
                         "max_distance_m": os.environ.get("MAX_DISTANCE_M"),
                         "collect_action_mode": getattr(args, "collect_action_mode", "eval")}}
    # which events END a budget-mode episode (collision by default)
    _end_on = parse_end_on(getattr(args, "collect_end_on", "collision"))

    _last_end_pos = None          # first episode has nowhere to continue from
    # --collect-start-xy: a FRESH trace's first episode starts at the nearest
    # waypoint to this point, via the same continuation path + spawn check.
    # A resumed trace ignores it (its first episode is not the trace's first).
    _start = parse_start_xy(getattr(args, "collect_start_xy", ""))
    if _start is not None and resume is None:
        _last_end_pos = _start
    _cont_misses = 0              # consecutive continuations the reset ignored
    # STEP BUDGET MODE (--collect-total-steps > 0): run episodes until the
    # budget is spent; each episode may use whatever is left of it.
    _budget = int(getattr(args, "collect_total_steps", 0) or 0)
    steps_done = int(resume.get("steps_done", 0)) if resume is not None else 0
    ep = first_ep - 1
    while True:
        ep += 1
        if _budget > 0:
            if steps_done >= _budget:
                break
        elif ep >= args.collect_episodes:
            break
        # SEEDED FROM THE EPISODE INDEX, not from a running generator, so a
        # resumed run draws the same episode a fresh run would have.
        rng = np.random.default_rng((args.collect_seed, ep))
        _s = int(rng.integers(0, 2**31 - 1))
        import random as _random
        _random.seed(_s); np.random.seed(_s % (2 ** 32))   # spawn draw reproducible
        # CONTINUE FROM WHERE THE LAST EPISODE STOPPED. Same mechanism the
        # episodes-from-trace path uses: CarlaEnv.reset honours the
        # experiment's `origin` when it is set, and the attribute is removed
        # afterwards so the default sampler is untouched for everyone else.
        _cont = continue_origin(_last_end_pos)
        if _cont is not None:
            try:
                import carla as _carla
                _core = find_attr_owner(env, "core")
                _amap = _core.core.map
                _wp = _amap.get_waypoint(
                    _carla.Location(x=float(_cont[0]), y=float(_cont[1]), z=0.0),
                    project_to_road=True)
                if _wp is not None:
                    # SAME ACCESSOR navigate() uses and has proven in the
                    # episodes-from-trace path. My first attempt walked the
                    # wrapper chain looking for an `experiment` attribute and
                    # found GatedReward instead, so every episode fell back to
                    # the sampler -- silently, because the fallback is graceful.
                    _exp = find_attr_owner(env, "experiment").experiment
                    set_continue_origin(_exp, _wp)
                    print(f"[Collect] ep{ep}: continuing from "
                          f"({_cont[0]:.1f},{_cont[1]:.1f}) -> waypoint "
                          f"({_wp.transform.location.x:.1f},"
                          f"{_wp.transform.location.y:.1f})", flush=True)
            except Exception as _e:                      # noqa: BLE001
                print(f"[Collect] ep{ep}: could not continue "
                      f"({type(_e).__name__}: {_e}); using the sampler",
                      flush=True)
                _cont = None
        o, info = env.reset(seed=_s)
        if _cont is not None:
            try:
                _o = find_attr_owner(env, "experiment")
                if _o is not None:
                    clear_continue_origin(_o.experiment)
            except Exception:                            # noqa: BLE001
                pass
            # PROVE THE CONTINUATION HAPPENED. Setting the attribute proved
            # nothing on 2026-09-12: the reset ignored it and respawned at
            # random for every v6n trace, with no error anywhere. A miss is
            # counted, and three misses in a row stop the collection rather
            # than record another random-respawn trace under the rule's name.
            _spawn = np.asarray(info.get("hero_pos", (np.inf, np.inf)), np.float64)[:2]
            if continued_spawn_ok(_spawn, _wp):
                _cont_misses = 0
            else:
                _cont_misses += 1
                print(f"[Collect] ep{ep}: CONTINUATION MISSED -- spawned at "
                      f"({_spawn[0]:.1f},{_spawn[1]:.1f}), asked for "
                      f"({_wp.transform.location.x:.1f},{_wp.transform.location.y:.1f})",
                      flush=True)
                if _cont_misses >= 3:
                    raise SystemExit("[Collect] 3 continuation misses in a row: the reset is "
                                     "not honouring the continue origin; refusing to record a "
                                     "random-respawn trace as a continued one")
        last_kept = None
        p0 = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
        driven, t, why = 0.0, 0, "cap"
        _ignored_ends = {}        # budget mode: terminations driven through, by cause
        prev_p = p0.copy()
        for t in range((_budget - steps_done) if _budget > 0 else args.collect_steps):
            p = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
            driven += float(np.linalg.norm(p - prev_p)); prev_p = p
            if last_kept is None or np.linalg.norm(p - last_kept) >= args.candidate_spacing:
                obs_list.append(goal_mode(o))
                fields_list.append(wrapper._fields(info))
                pos.append(p)
                yaw.append(float(info.get("hero_yaw", 0.0)))
                seg.append(ep)
                last_kept = p
            if reactive is not None:
                a_ = reactive.act(ring.latest(), p)
                reactive.note(p)
                action = np.clip(a_, lo, hi).astype(np.float32)
            else:
                action = collect_action(agent, o, getattr(args, "collect_action_mode", "eval"))
            o, _r, term, trunc, info = env.step(action)
            # BUDGET MODE ENDS AN EPISODE ON A COLLISION ONLY. USER 2026-09-12:
            # "the collection episode never ends unless collision happens or max
            # budget 10,000 limit is reached". Lane crossings, off-road, the
            # 100 m explore arrival and any truncation are counted and driven
            # through, not obeyed.
            if _budget > 0:
                # --collect-end-on decides what ends the episode (collision by
                # default; the Town02 recollection adds lane and off_road).
                _cause = collect_should_end(info, term, trunc, _end_on)
                if _cause is None:
                    if term or trunc:
                        _k = info.get("explore_end") or ("trunc" if trunc else "term")
                        _ignored_ends[_k] = _ignored_ends.get(_k, 0) + 1
                    continue
                why = _cause
                _last_end_pos = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)[:2]
                break
            if term or trunc:
                why = ("trunc" if trunc and not term else
                       ",".join(k for k, v in (info.get("done_reasons") or {}).items() if v)
                       or ("collision" if info.get("collision") else
                           "lane" if info.get("lane_invasion") else "term"))
                # WHERE IT STOPPED, so the next episode can continue from here
                # rather than teleporting to a fresh spawn. Read from info
                # rather than the loop's `p`, which lags by one step.
                _last_end_pos = np.asarray(
                    info.get("hero_pos", (0.0, 0.0)), np.float64)[:2]
                break
        else:
            # ran to the step cap without terminating -- continue from here too
            _last_end_pos = np.asarray(
                info.get("hero_pos", (0.0, 0.0)), np.float64)[:2]

        # PER-EPISODE DIAGNOSTICS, because "2 candidates over 2 episodes" said
        # nothing about whether the car stood still or the episode ended at
        # the second metre -- and those call for opposite remedies.
        steps_done += t + 1
        _of = (f"{steps_done}/{_budget} steps" if _budget > 0
               else f"{ep + 1}/{args.collect_episodes}")
        print(f"[Collect] episode {ep + 1} ({_of}): "
              f"{t + 1} steps, {driven:.1f} m driven, ended={why}, "
              f"{len(pos)} candidates so far"
              + (f", driven through {_ignored_ends}" if _ignored_ends else ""), flush=True)
        # AFTER EVERY EPISODE. A kill then costs one episode, not sixty.
        if args.trace_cache:
            os.makedirs(os.path.dirname(args.trace_cache) or ".", exist_ok=True)
            with open(args.trace_cache, "wb") as fh:
                pickle.dump(_snapshot(ep + 1), fh)
    env.close()

    if ring is not None:
        ring.destroy()
    n_eps = ep if _budget > 0 else args.collect_episodes
    trace = _snapshot(n_eps)
    print(f"[Collect] {len(pos)} candidates over {n_eps} explore episodes "
          f"({steps_done} steps), thinned to {args.candidate_spacing} m")
    return trace


def path_length_m(positions):
    """Arc length of a polyline, in metres."""
    p = np.asarray(positions, np.float64)
    if len(p) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


# ------------------------------------------------------------- stage 2 ----

def candidate_distances(agent, trace, wrapper_goal_block, args):
    """The full candidate x candidate matrix of predicted steps.

    COMPUTED ONCE AND SHARED by every method that needs it, which is both a
    large saving and a correctness property: policy and sgm must see the same
    distances or their difference is not only the consistency test.
    """
    from vae_nav.sgm_placement import pairwise_steps
    n = len(trace["positions"])
    print(f"[Build] querying the critic for a {n}x{n} distance matrix", flush=True)
    return pairwise_steps(agent, trace["observations"], trace["fields"],
                          trace["positions"], wrapper_goal_block,
                          step_cost=args.step_cost,
                          percentile=args.percentile)


def embedding_cutoff_from_data(embeddings, percentile):
    """A cutoff in THIS latent space, not the reference's 0.05.

    SGM's published embedding cutoff belongs to the reference implementation's
    embedding, and a 128-d VAE latent is not on that scale. Left at 0.05 the
    prefilter finds no neighbours at all, no candidate is ever tested for
    consistency, and SGM degenerates into "keep everything" -- which looks like
    a result (many nodes, poor sparsity) rather than a misconfiguration. Taking
    a low percentile of the observed pairwise distances keeps the prefilter
    doing its job in whatever space it is handed.
    """
    emb = np.asarray(embeddings, np.float64)
    n = len(emb)
    if n < 2:
        return 0.05
    idx = np.random.default_rng(0).choice(n, size=min(n, 200), replace=False)
    sample = emb[idx]
    d = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)
    off = d[~np.eye(len(sample), dtype=bool)]
    return float(np.percentile(off, percentile))


def build_graph(dist, nodes, max_steps):
    """Directed graph over `nodes`, weighted in steps, horizon-rejected.

    DIRECTED BECAUSE THE CRITIC IS. dist[i, j] is i -> j and need not equal
    dist[j, i]; symmetrising would assert a return leg the policy may not have.
    """
    import networkx as nx
    from vae_nav.value_distance import edge_reject_mask

    G = nx.DiGraph()
    for n in nodes:
        G.add_node(int(n))
    sub = dist[np.ix_(nodes, nodes)]
    # tau is left unset: value_distance records that raw ensemble spread is a
    # proxy for leg length with the wrong sign, so the horizon test carries the
    # rejection alone.
    reject = edge_reject_mask(sub, spread=None, max_steps=max_steps, tau=None)
    for a in range(len(nodes)):
        for b in range(len(nodes)):
            if a != b and not reject[a, b]:
                G.add_edge(int(nodes[a]), int(nodes[b]),
                           steps=float(sub[a, b]))
    return G


# ------------------------------------------------------------- stage 3 ----

def route_length_m(planner, carla, a, b):
    """CARLA's own shortest driving route, in metres. None if unroutable."""
    try:
        route = planner.trace_route(carla.Location(x=float(a[0]), y=float(a[1]), z=0.0),
                                    carla.Location(x=float(b[0]), y=float(b[1]), z=0.0))
    except Exception:  # noqa: BLE001 - an unroutable pair is data, not a crash
        return None
    if not route:
        return None
    pts = np.array([[wp.transform.location.x, wp.transform.location.y]
                    for wp, _ in route], np.float64)
    return path_length_m(pts)


def plan(G, nodes, positions, start_pos, goal_pos, localize_radius):
    """Node path from the node nearest the start to the one nearest the goal.

    Returns (path, reason). A None path carries WHY, because "0.000 success"
    cannot distinguish a policy that drove badly from a graph that never
    offered it a route -- the distinction this whole comparison is about.
    """
    import networkx as nx

    if len(nodes) == 0:
        return None, "empty_graph"
    npos = positions[nodes]
    ds = np.linalg.norm(npos - np.asarray(start_pos, np.float64), axis=1)
    dg = np.linalg.norm(npos - np.asarray(goal_pos, np.float64), axis=1)
    if ds.min() > localize_radius:
        return None, "start_unlocalised"
    if dg.min() > localize_radius:
        return None, "goal_unlocalised"
    s = int(nodes[int(np.argmin(ds))])
    t = int(nodes[int(np.argmin(dg))])
    if s == t:
        return [s], "ok"
    try:
        return nx.shortest_path(G, s, t, weight="steps"), "ok"
    except nx.NetworkXNoPath:
        return None, "no_path"
    except nx.NodeNotFound:
        return None, "node_missing"


def navigate(T, latent, agent, trace, G, nodes, episodes, args, method):
    """Drive every episode under one graph. Returns the metric dict."""
    os.environ["EVAL_FORCE_MODE"] = str(GOAL_MODE)
    env = T.make_env(episodes[0]["town"], seed=episodes[0]["seed"],
                     latent_source=latent, eval_mode=True, port=args.port)
    _assert_town(env, episodes[0]["town"], f"navigate/{method}")
    wrapper = find_wrapper(env, "goal_block")
    from vae_nav.value_distance import goal_relabelled_obs

    planner = carla = None
    try:
        import carla as carla_mod
        # BOTH PATHS, because this module is imported one way from the repo
        # root (expert_oracle.py) and another with src/ on sys.path (here).
        try:
            from src.global_route_planner import GlobalRoutePlanner
        except ImportError:
            from global_route_planner import GlobalRoutePlanner
        core = getattr(getattr(env, "unwrapped", env), "core", None)
        hero = getattr(core, "hero", None)
        if hero is not None:
            planner = GlobalRoutePlanner(hero.get_world().get_map(), 2.0)
            carla = carla_mod
    except Exception as e:  # noqa: BLE001
        print(f"[SPL] WARNING: no route planner ({e}); SPL will be n/a.")

    positions = trace["positions"]
    # SR AND SPL ARE COMPUTED OVER ROUTED EPISODES ONLY.
    #
    # An episode is ROUTED when the graph actually supplied a path: start and
    # goal both localised, and a route between them exists. It is BYPASSED when
    # the graph offered nothing and the policy was handed the true goal as its
    # only waypoint -- in which case the map contributed nothing and the episode
    # measures the unaided policy.
    #
    # Averaging the two together makes the headline numbers a BLEND of "how good
    # is the map" and "how good is the policy without one", in a mixing ratio
    # that differs per method. Measured at 20-40 m: sgm and policy each bypassed
    # 17 of 20 episodes while uniform bypassed 11, so comparing their means
    # compared two different mixtures rather than two maps.
    #
    # THE COST OF THIS CHOICE, AND WHY routed_frac IS REPORTED NEXT TO IT: a
    # graph that routes 3 episodes of 20 is now scored on those 3, so a sparse
    # or disconnected map can post a strong SR while being useless in practice.
    # The routed fraction is the guard against reading one without the other.
    successes, spls, reasons, leg_timeouts = [], [], {}, 0
    n_routed, n_bypassed, bypassed_ok = 0, 0, []
    # WHERE ALONG THE ROUTE EPISODES END, one record per routed episode as
    # [leg reached, legs in plan, success].
    #
    # WHY THIS INTEGER MATTERS. A peer session found their mapped arm scoring
    # 19% on a PERFECT route, and the summary alone read as a weak policy. The
    # per-episode field said otherwise: all 34 failures ended on leg 1 of 11
    # while distance-to-goal still fell, which is not a competence gradient but
    # a leg that never advanced. A policy that cannot navigate fails at varying
    # points; one that can does not fail at leg 1 thirty-four times.
    #
    # AND THE OPPOSITE FAILURE, which is the one this harness is exposed to.
    # habitat_map.py warns that when node spacing is smaller than the arrival
    # radius, "a waypoint is already reached when it becomes current, the
    # follower skips down the chain, and the mapped arm degenerates toward the
    # unmapped one." Our advance radius is 4 m against an environment goal
    # radius of 2 m, so a skipped advance is counted: an advance whose NEXT
    # waypoint was ALREADY inside the radius at the moment of advancing.
    # Counted, never acted on -- changing the rule would change the experiment.
    leg_records, n_advances, n_skip_advances = [], 0, 0

    exp = getattr(env, "unwrapped", env).experiment
    amap = getattr(getattr(env, "unwrapped", env), "core").map
    for ep in episodes:
        # ONE EPISODE SET FOR EVERY ARM. carla_core.reset_hero draws the spawn
        # and goal from the GLOBAL random module and CarlaEnv.reset ignores its
        # seed kwarg, so without this each arm was scored on a different draw
        # -- the one variable the comparison exists to hold fixed.
        import random as _random
        _random.seed(int(ep["seed"])); np.random.seed(int(ep["seed"]) % (2 ** 32))
        if ep.get("start_xy") is not None:
            # EXPLICIT START AND GOAL (episodes-from-trace). CarlaEnv.reset
            # takes the experiment's `origin`/`destination` waypoints when
            # both are set, through reset_hero_for_experiments; the
            # experiment class never defines them, so they are attached here
            # per episode and removed after, and the default sampler is
            # untouched for everyone else.
            import carla as _carla
            exp.origin = amap.get_waypoint(_carla.Location(
                x=float(ep["start_xy"][0]), y=float(ep["start_xy"][1]), z=0.0))
            exp.destination = amap.get_waypoint(_carla.Location(
                x=float(ep["goal_xy"][0]), y=float(ep["goal_xy"][1]), z=0.0))
        obs, info = env.reset(seed=ep["seed"])
        if ep.get("start_xy") is not None:
            for _a in ("origin", "destination"):
                if hasattr(exp, _a):
                    delattr(exp, _a)
        start = np.asarray(info["hero_pos"], np.float64)
        goal = np.asarray(info["goal_pos"], np.float64)

        node_path, why = plan(G, nodes, positions, start, goal,
                              args.localize_radius)
        # THE FINAL GOAL IS ALWAYS THE LAST WAYPOINT. Without it the policy is
        # handed the nearest node to the goal and stops there, and the episode
        # fails within the arrival radius of a node it was never asked to leave.
        waypoints = ([positions[i] for i in node_path] if node_path else [])
        waypoints.append(goal)
        if node_path is None:
            reasons[why] = reasons.get(why, 0) + 1
            # STILL DRIVEN, with the true goal only -- but NOT scored in the
            # headline SR/SPL (see the BYPASSED branch below): its outcome goes
            # to success_bypassed as an unaided control, and routed_frac is the
            # guard against a sparse map posting a strong SR on few episodes.
            waypoints = [goal]

        geodesic = (route_length_m(planner, carla, start, goal)
                    if planner is not None else None)

        wp_i, leg_t, driven, ok = 0, 0, [start.copy()], False
        for _t in range(args.max_steps):
            target = waypoints[min(wp_i, len(waypoints) - 1)]
            drive_obs = goal_relabelled_obs(goal_mode(obs),
                                            wrapper._fields(info), target,
                                            wrapper.goal_block)
            obs, _r, term, trunc, info = env.step(agent.eval_actions(drive_obs))
            here = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
            driven.append(here.copy())
            ok = ok or bool(info.get("is_success", False))
            if term or trunc:
                break
            leg_t += 1
            if np.linalg.norm(here - target) <= args.arrival_radius:
                wp_i, leg_t = wp_i + 1, 0
                n_advances += 1
                # A SKIP: the waypoint we just adopted is already satisfied, so
                # the follower will retire it next step without the policy ever
                # driving toward it. Measured, not corrected.
                nxt = waypoints[min(wp_i, len(waypoints) - 1)]
                if (wp_i < len(waypoints)
                        and np.linalg.norm(here - nxt) <= args.arrival_radius):
                    n_skip_advances += 1
            elif leg_t >= args.leg_steps and wp_i < len(waypoints) - 1:
                # A LEG THE POLICY CANNOT DRIVE IS AN EDGE THE GRAPH SHOULD NOT
                # HAVE OFFERED, which is a placement result. Skipping on rather
                # than aborting keeps the episode's outcome about the whole
                # route; the count is reported so the two are not confused.
                wp_i, leg_t = wp_i + 1, 0
                leg_timeouts += 1

        if node_path is None:
            # BYPASSED: the graph gave nothing, so this episode says nothing
            # about placement. Counted and kept as a control, never mixed into
            # the headline SR/SPL.
            n_bypassed += 1
            bypassed_ok.append(float(ok))
            if ok:
                reasons["arrived"] = reasons.get("arrived", 0) + 1
            continue

        n_routed += 1
        leg_records.append([int(wp_i), int(len(waypoints)), int(ok)])
        successes.append(float(ok))
        # SPL IS ONLY DEFINED WHERE THE GEODESIC IS KNOWN, and an episode whose
        # geodesic is unknown contributes NOTHING -- not a zero.
        #
        # The earlier version appended 0.0 for every failure regardless, so with
        # no route planner an arm that failed once reported SPL 0.000 while an
        # arm that never failed reported n/a. Two arms, two different meanings
        # of "no measurement", and the 0.000 reads as "took an infinite detour"
        # rather than "was never measured". Caught by the smoke test.
        if geodesic:
            p = max(path_length_m(driven), 1e-9)
            spls.append((geodesic / max(p, geodesic)) if ok else 0.0)
        if ok:
            reasons["arrived"] = reasons.get("arrived", 0) + 1
    env.close()

    finite = [s for s in spls if not math.isnan(s)]
    if not finite:
        print("[SPL] n/a for this arm: no episode had a known geodesic. SPL is "
              "reported as n/a rather than 0.000 for every arm alike.")
    return {
        "method": method,
        "n_nodes": int(len(nodes)),
        "n_candidates": int(len(positions)),
        "nodes_per_100m": float(len(nodes)) / max(path_length_m(positions), 1e-9) * 100.0,
        "n_edges": int(G.number_of_edges()),
        "episodes": len(episodes),
        # ROUTED-ONLY. n/a rather than 0.0 when nothing routed, because a graph
        # that never supplied a path has not been measured -- it has not scored
        # zero. Reporting 0.000 there would rank an unusable map below a bad one
        # on the same axis, which is the n/a-versus-zero trap this harness keeps
        # out of every other column.
        "success": (float(np.mean(successes)) if successes else None),
        "n_routed": int(n_routed),
        "n_bypassed": int(n_bypassed),
        "routed_frac": float(n_routed) / max(len(episodes), 1),
        # THE CONTROL, not a headline metric: the unaided policy on the episodes
        # this graph could not route. It should be broadly similar across
        # methods, since it does not depend on the map; a large spread means the
        # methods are being scored on genuinely different episode subsets.
        "success_bypassed": (float(np.mean(bypassed_ok))
                             if bypassed_ok else None),
        # SPL IS n/a, NOT ZERO, WITHOUT A ROUTE PLANNER. A zero would read as
        # "took an infinite detour" rather than "was never measured".
        "spl": (float(np.mean(finite)) if finite else None),
        "spl_n": len(finite),
        "leg_timeouts": int(leg_timeouts),
        # DIAGNOSTICS, not metrics. See the comment at leg_records above.
        "legs": leg_records,
        "n_advances": int(n_advances),
        "n_skip_advances": int(n_skip_advances),
        # arrival_radius only: min_sep_m is written by main(), and two writers
        # for one key is what silently corrupted nodes_per_100m earlier.
        "arrival_radius": float(args.arrival_radius),
        "end_reasons": reasons,
    }


# ----------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", default="gated_her")
    ap.add_argument("--methods", default="sgm,madent,uniform,random")
    ap.add_argument("--town", default=os.environ.get("EVAL_TOWNS", "Town02"))
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("CARLA_PORT", 2000)))
    ap.add_argument("--goal-min", type=float, default=None)
    ap.add_argument("--goal-max", type=float, default=None)
    # collection
    ap.add_argument("--collect-episodes", type=int, default=20)
    ap.add_argument("--collect-steps", type=int, default=500)
    # USER DECISION 2026-09-12: "COLLECT TRACE JUST ONE SEED - RUN 10,000 step
    # budget remove timelimit wrapper for trace collection". When > 0 the
    # trace is ONE step budget spent across as many episodes as it takes.
    # USER, same day: "the collection episode never ends unless collision
    # happens or max budget 10,000 limit is reached" -- so an episode ends ONLY
    # on a collision; crossings, off-road, the 100 m arrival and truncation are
    # counted and driven through. The next episode continues from the crash.
    # --collect-episodes and --collect-steps are then ignored.
    ap.add_argument("--collect-total-steps", type=int, default=0)
    # USER DECISION 2026-09-13: Town04 traces from a random spawn drove the
    # outer highway once and never revisited (3/1200, 4/1230 within the 2 m SGM
    # gate), so SGM could not thin them. "X,Y" starts the FIRST episode at the
    # nearest drivable waypoint to that point (e.g. inside the NE city grid),
    # through the same continuation path and spawn check as every later one.
    ap.add_argument("--collect-start-xy", default="")
    # Budget mode only: which events END an episode (then continue from the
    # nearest waypoint). Default collision-only; see collect_should_end.
    ap.add_argument("--collect-end-on", default="collision")
    # USER 2026-09-13: "one trace in town 2 with sampled actions". eval = the
    # distribution mode (deterministic, the default and every earlier trace);
    # sample = agent.sample_actions (stochastic policy). Collection only;
    # navigation always uses eval_actions.
    ap.add_argument("--collect-action-mode", choices=("eval", "sample"), default="eval")
    ap.add_argument("--collect-seed", type=int, default=7)
    ap.add_argument("--explorer", default="policy_explore",
                    choices=("policy_explore", "policy_goal", "reactive"),
                    help="WHAT DRIVES THE TRACE. policy_explore: the gated "
                         "policy's explore half (docs sF; on v4 it crosses a "
                         "line within ~2 m of spawn). policy_goal: its goal "
                         "half driving to sampled goals. reactive: the policy-"
                         "free depth-ring explorer, the Habitat analogue.")
    ap.add_argument("--collect-lane-rule", default="strict",
                    choices=("strict", "noop"),
                    help="lane-crossing rule DURING COLLECTION ONLY. 'noop' "
                         "relaxes the fatal crossing so a non-lane-aware "
                         "explorer can cover ground; recorded in the trace "
                         "metadata so it is never silent. Navigation always "
                         "runs under the arm's own rule.")
    ap.add_argument("--candidate-spacing", type=float, default=2.0,
                    help="metres between recorded candidates. Consecutive "
                         "simulator steps are ~0.28 m apart, so an unthinned "
                         "trace is mostly duplicates in a quadratic matrix.")
    ap.add_argument("--trace-cache", default=None,
                    help="reuse a saved trace. The bands are separate "
                         "processes because the goal sampler reads its band at "
                         "import; the TRACE does not depend on the band, so "
                         "recollecting per band would be three different maps "
                         "for one comparison.")
    # placement
    ap.add_argument("--min-sep-m", type=float, default=2.0,
                    help="minimum arc-length separation between entropy peaks, "
                         "in metres. SET TO THE ENVIRONMENT'S GOAL RADIUS "
                         "(distance_to_goal < 2), so two nodes are never closer "
                         "than the distance at which the policy considers a "
                         "goal reached -- below that they are not two waypoints "
                         "in any operational sense. "
                         "node_placement_entropy.py used 8 m, which CAPPED this "
                         "arm: at 8 m a 7.5 km trace admits at most ~933 nodes, "
                         "so a sweep requesting 1194 would have returned ~933 "
                         "for policy and 1194 for uniform and random, silently "
                         "comparing arms at unequal node counts.")
    ap.add_argument("--smooth-w", type=int, default=9,
                    help="entropy smoothing window, odd")
    ap.add_argument("--step-cost", type=float,
                    default=float(os.environ.get("STEP_COST", 1.0)),
                    help="THE ARM'S OWN. 1.0 for the gated arms; "
                         "value_distance defaults to 0.1 and would scale every "
                         "edge weight by ten.")
    ap.add_argument("--percentile", type=float, default=None)
    ap.add_argument("--nodes-per-100m", type=float, default=4.0,
                    help="THE EQUAL BUDGET, as node density along the trace. "
                         "Every arm places this many. node_placement_entropy.py"
                         "'s default; at 4/100 m the nominal spacing is 25 m, "
                         "so --min-sep-m 8 is satisfiable by every arm.")
    ap.add_argument("--embedding-percentile", type=float, default=75.0,
                    help="SGM's prefilter width, as a percentile of observed "
                         "pairwise LATENT distances. sgm_placement calls this a "
                         "cheap prefilter and not the test -- too tight costs "
                         "sparsity, never correctness -- so it is opened wide "
                         "and the consistency test is left to decide. At the "
                         "reference's literal 0.05 it proposes no neighbours "
                         "at all in this latent space and nothing ever merges.")
    ap.add_argument("--consistency-cutoff", type=float, default=None,
                    help="SGM's two-way agreement threshold, IN STEPS. "
                         "Overrides --consistency-frac. sgm_placement's own "
                         "default of 5.0 is in the reference implementation's "
                         "units and is far too strict here, where the critic's "
                         "whole horizon is 106 steps.")
    ap.add_argument("--consistency-frac", type=float, default=0.1,
                    help="consistency cutoff as a fraction of the critic's "
                         "trained horizon H_STEPS, used when "
                         "--consistency-cutoff is not given")
    ap.add_argument("--max-edge-steps", type=float, default=None,
                    help="edge cutoff in STEPS for every arm. Overridden by "
                         "--calibration. Falls back to value_distance.H_STEPS "
                         "only with a loud warning: that constant belongs to "
                         "another critic.")
    ap.add_argument("--calibration", default="",
                    help="calibration JSON from calibrate_critic.py: sets the "
                         "edge cutoff (all arms) and SGM consistency for THIS "
                         "critic (protocol s0a/s3: re-measure, never inherit)")
    ap.add_argument("--sgm-edge-cutoff", type=float, default=None)
    ap.add_argument("--sgm-consistency", type=float, default=None)
    ap.add_argument("--sgm-gate-m", type=float, default=2.0,
                    help="metric gate of the SGM frame; the env's goal radius")
    ap.add_argument("--mad-z", type=float, default=3.0)
    ap.add_argument("--episodes-from-trace", action="store_true",
                    help="sample every mapped-eval episode's START and GOAL "
                         "from the trace's own candidate states (both inside "
                         "the explored region; goal within the band by "
                         "straight-line distance). User decision 2026-09-11 "
                         "for large towns: 'sample goals only in mapped "
                         "areas'. The set is derived from --seed and the "
                         "trace, so every arm gets the same episodes.")
    ap.add_argument("--stop-after", choices=("", "collect", "matrix"), default="",
                    help="run only the simulator-free-able prefix: 'collect' "
                         "records the trace, 'matrix' also caches the critic "
                         "matrix. Calibration (calibrate_critic.py) runs "
                         "between 'matrix' and the build, over ALL traces.")
    ap.add_argument("--match-to", default="sgm,madent",
                    help="which self-sizing arms uniform/random are count-"
                         "matched to (protocol s3/s4). Each produces a "
                         "uniform@n and random@n row.")
    # navigation
    ap.add_argument("--arrival-radius", type=float, default=4.0)
    ap.add_argument("--localize-radius", type=float, default=15.0)
    ap.add_argument("--leg-steps", type=int, default=200)
    ap.add_argument("--out-dir", default="results/placement")
    ap.add_argument("--tag", default="",
                    help="suffix for the result filename, e.g. '__d8'. Carries "
                         "the sweep point, so two densities cannot overwrite "
                         "each other and every file still looks complete.")
    args = ap.parse_args()
    # THE COLLECTION LANE RULE IS APPLIED BEFORE ANY vae_nav IMPORT: the reward
    # wrappers read LANE_STRICT / LANE_INVASION_NOOP / GATED_GOAL_LANE_TERMINAL
    # at import time, so setting them later would silently do nothing.
    # Navigation (stage 3) runs in a SEPARATE process under the arm's own rule.
    if args.collect_lane_rule == "noop" and args.stop_after:
        os.environ["LANE_STRICT"] = "0"
        os.environ["LANE_INVASION_NOOP"] = "1"
        os.environ["GATED_GOAL_LANE_TERMINAL"] = "0"
        print("[Collect] lane rule relaxed FOR COLLECTION ONLY "
              "(LANE_STRICT=0 LANE_INVASION_NOOP=1 GATED_GOAL_LANE_TERMINAL=0)",
              flush=True)
    elif args.collect_lane_rule == "noop":
        raise SystemExit("--collect-lane-rule noop is only valid with "
                         "--stop-after collect|matrix: navigation must run "
                         "under the arm's own rule, in its own process")

    os.environ["EVAL_FORCE_MODE"] = str(GOAL_MODE)
    if args.goal_min is not None:
        os.environ["GOAL_DIST_MIN"] = str(args.goal_min)
    if args.goal_max is not None:
        os.environ["GOAL_DIST_MAX"] = str(args.goal_max)
        os.environ["GOAL_DIST_START"] = str(args.goal_max)
        os.environ["GOAL_DIST_TARGET"] = str(args.goal_max)

    # PREFLIGHT: everything that names or shapes a result, in one block, from
    # the values actually in effect -- so a launch can be checked in the log
    # before hours are spent, instead of after (four restarts on 2026-09-11
    # were configuration errors visible here).
    _pf = {k: os.environ.get(k) for k in (
        "EXPLORE_LANE_MODE", "LANE_INVASION_NOOP", "LANE_STRICT",
        "GATED_GOAL_LANE_TERMINAL", "LANE_PROGRESS_MODE", "LANE_PROGRESS_SCALE",
        "COLLISION_TERMINAL", "STEP_COST", "POTENTIAL_K", "EXPERT_PROB",
        "CARLA_START_SERVER", "CARLA_QUALITY", "GOAL_DIST_MIN", "GOAL_DIST_MAX")}
    print("[Preflight] ckpt=%s arm=%s town=%s explorer=%s collect_lane_rule=%s "
          "episodes=%d seed=%d collect_seed=%d methods=%s match_to=%s "
          "calibration=%s stop_after=%s out=%s" % (
              args.ckpt, args.arm, args.town, args.explorer, args.collect_lane_rule,
              args.episodes, args.seed, args.collect_seed, args.methods,
              args.match_to, args.calibration or "-", args.stop_after or "-",
              args.out_dir), flush=True)
    print("[Preflight] env: " + " ".join(f"{k}={v}" for k, v in _pf.items()), flush=True)
    import train_carla_tqc as T
    from vae_nav.latent_source import from_env as latent_from_env
    from vae_nav.node_placement import (actor_entropy, arclength, place,
                                        sgm_cutoff_for_budget)
    from vae_nav.policy_io import load_agent_state
    from vae_nav.value_distance import H_STEPS

    calib = None
    if args.calibration and not args.stop_after:
        with open(args.calibration) as fh:
            calib = json.load(fh)
        if not calib.get("chosen"):
            raise SystemExit(f"{args.calibration} has no chosen setting -- the "
                             "sweep found no fully-reachable cutoff on every "
                             "trace; widen it before building maps.")
    sgm_edge = (args.sgm_edge_cutoff if args.sgm_edge_cutoff is not None
                else (calib["chosen"]["edge_cutoff"] if calib else None))
    sgm_cons = (args.sgm_consistency if args.sgm_consistency is not None
                else (calib["chosen"]["consistency"] if calib else None))
    if args.max_edge_steps is not None:
        max_edge_steps = args.max_edge_steps
    elif sgm_edge is not None:
        max_edge_steps = float(sgm_edge)      # ONE edge rule for every arm
    else:
        max_edge_steps = H_STEPS
        print("[Build] WARNING: no --calibration and no --max-edge-steps; "
              "using value_distance.H_STEPS, which was derived for a DIFFERENT "
              "critic. Treat any result as uncalibrated.", flush=True)
    latent = latent_from_env()
    episodes = T.build_eval_episodes([args.town], args.episodes, seed=args.seed)

    env0 = T.make_env(args.town, seed=episodes[0]["seed"], latent_source=latent,
                      eval_mode=True, port=args.port)
    agent = T.build_agent(env0.observation_space, env0.action_space, args.seed)
    with open(args.ckpt, "rb") as fh:
        state = pickle.load(fh)
    if isinstance(state, dict) and "identity" in state:
        from vae_nav.latent_source import LatentSource
        LatentSource.assert_matches(state["identity"], latent)
    load_agent_state(agent, state.get("agent", state))
    goal_block_fn = find_wrapper(env0, "goal_block").goal_block
    env0.close()

    trace = None
    if args.trace_cache and os.path.exists(args.trace_cache):
        with open(args.trace_cache, "rb") as fh:
            cached = pickle.load(fh)
        done = int(cached.get("episodes_done", args.collect_episodes))
        _bud = int(getattr(args, "collect_total_steps", 0) or 0)
        complete = (int(cached.get("steps_done", 0)) >= _bud if _bud > 0
                    else done >= args.collect_episodes)
        if complete:
            trace = cached
            print(f"[Collect] reusing {len(trace['positions'])} candidates "
                  f"from {args.trace_cache} ({done} episodes)")
        else:
            # A PARTIAL TRACE IS RESUMED, NOT DISCARDED AND NOT USED AS IS.
            # Using it as is would silently compare four placement methods on a
            # map built from however far the last kill happened to get.
            print(f"[Collect] {args.trace_cache} holds {done} of "
                  f"{args.collect_episodes} episodes; resuming collection")
            trace = collect_trace(T, latent, agent, args, resume=cached)
    if trace is None:
        trace = collect_trace(T, latent, agent, args)
    if args.stop_after == "collect":
        print("[Stage] stop-after collect: trace cached, exiting", flush=True)
        return

    # THE DISTANCE MATRIX IS CACHED BESIDE THE TRACE. It costs an n^2 critic
    # pass on the GPU and depends on nothing but the trace and the checkpoint,
    # while the merge cutoffs it feeds are exactly what needs tuning. Without
    # this, every cutoff experiment reruns CARLA and the critic; with it,
    # tuning is an offline second.
    dist_cache = (args.trace_cache + ".dist.npy") if args.trace_cache else None
    dist = None
    if dist_cache and os.path.exists(dist_cache):
        cached = np.load(dist_cache)
        if cached.shape == (len(trace["positions"]),) * 2:
            dist = cached
            print(f"[Build] reusing the {len(dist)}x{len(dist)} distance "
                  f"matrix from {dist_cache}")
        else:
            print(f"[Build] ignoring {dist_cache}: shape {cached.shape} does "
                  f"not match {len(trace['positions'])} candidates")
    if dist is None:
        dist = candidate_distances(agent, trace, goal_block_fn, args)
        if dist_cache:
            np.save(dist_cache, dist)
    if args.stop_after == "matrix":
        print("[Stage] stop-after matrix: trace + distance matrix cached, "
              "exiting (run calibrate_critic.py next)", flush=True)
        return
    embeddings = np.stack([np.asarray(o["latent"], np.float64)
                           for o in trace["observations"]])
    emb_cutoff = embedding_cutoff_from_data(embeddings, args.embedding_percentile)
    print(f"[Build] embedding cutoff {emb_cutoff:.4f} "
          f"(p{args.embedding_percentile} of observed latent distances)")

    order = list(range(len(trace["positions"])))
    s = arclength(trace["positions"], trace["segments"])
    path_m = float(s[-1]) if len(s) else 0.0
    P = np.asarray(trace["positions"], np.float64)[:, :2]
    from vae_nav.sgm_frame import (sgm_faithful, learned_edges,
                                   directed_reachability, undirected_components,
                                   edge_separation_stats)
    methods = [m for m in args.methods.split(",") if m]
    horizon_m = float((calib or {}).get("horizon_m",
                                        float(os.environ.get("GOAL_DIST_MAX", 50))))

    # THE SELF-SIZING ARMS FIRST: their counts are what uniform/random match.
    sgm_nodes, sgm_edges, cutoff = None, None, None
    if "sgm" in methods or "sgm" in args.match_to:
        if sgm_edge is None or sgm_cons is None:
            raise SystemExit("sgm needs --calibration or both --sgm-edge-cutoff "
                             "and --sgm-consistency (steps, for THIS critic)")
        cutoff = float(sgm_cons)
        sgm_nodes, sgm_edges = sgm_faithful(dist, P, consistency_cutoff=cutoff,
                                            edge_cutoff=float(sgm_edge),
                                            gate_m=args.sgm_gate_m)
        n_merged = len(order) - len(sgm_nodes)
        print(f"[Build] sgm[{cutoff:g}/{sgm_edge:g}] kept {len(sgm_nodes)} of "
              f"{len(order)} candidates ({n_merged} merged), "
              f"{len(sgm_edges)} learned directed edges, gate {args.sgm_gate_m} m",
              flush=True)
        if n_merged == 0:
            off = dist[~np.eye(len(dist), dtype=bool)]
            raise SystemExit(
                "SGM MERGED NOTHING at this calibration; the row would be the "
                "unthinned trace. pairwise distance percentiles (steps): "
                + ", ".join(f"p{p}={np.percentile(off, p):.1f}" for p in (1, 5, 25, 50)))
    entropy = actor_entropy(agent, trace["observations"], seed=args.seed,
                            analytic=True)
    print(f"[Build] analytic actor entropy over candidates: mean "
          f"{entropy.mean():.3f} nats, range [{entropy.min():.3f}, "
          f"{entropy.max():.3f}]")
    entropy_mc = (actor_entropy(agent, trace["observations"], seed=args.seed,
                                analytic=False) if "policy" in methods else None)
    madent_nodes = None
    if "madent" in methods or "madent" in args.match_to:
        madent_nodes, _ = place("madent", s=s, entropy=entropy,
                                min_sep_m=args.min_sep_m, smooth_w=args.smooth_w,
                                z_thresh=args.mad_z, nms_w=0)
        print(f"[Build] madent (MAD z>={args.mad_z:g}, endpoints off): "
              f"{len(madent_nodes)} nodes", flush=True)
    # THE ROWS. uniform/random appear once per arm they are matched to, so
    # "random did worse" can never be "random had fewer nodes".
    rows = []
    for m in methods:
        if m in ("uniform", "random"):
            for ref in [r for r in args.match_to.split(",") if r]:
                ref_nodes = sgm_nodes if ref == "sgm" else madent_nodes
                if ref_nodes is None or len(ref_nodes) == 0:
                    continue
                rows.append((f"{m}@{ref}", m, int(len(ref_nodes))))
        else:
            rows.append((m, m, None))

    if args.episodes_from_trace:
        # START and GOAL both drawn from the trace's candidate states, goal
        # within the band by straight-line distance and at least one node
        # away, from a generator seeded by --seed: the same list for every
        # arm, and reproducible. Positions only -- no map query, no planner.
        _P = np.asarray(trace["positions"], np.float64)[:, :2]
        _rng = np.random.default_rng(int(args.seed))
        lo_m = float(args.goal_min if args.goal_min is not None else 20.0)
        hi_m = float(args.goal_max if args.goal_max is not None else 50.0)
        eps_ft, tries = [], 0
        while len(eps_ft) < args.episodes and tries < 200 * args.episodes:
            tries += 1
            i = int(_rng.integers(0, len(_P)))
            d = np.linalg.norm(_P - _P[i], axis=1)
            ok = np.flatnonzero((d >= lo_m) & (d <= hi_m))
            if len(ok) == 0:
                continue
            j = int(_rng.choice(ok))
            eps_ft.append({"town": args.town, "seed": int(_rng.integers(0, 2 ** 31 - 1)),
                           "index": len(eps_ft), "start_xy": _P[i].tolist(),
                           "goal_xy": _P[j].tolist(), "start_node": i, "goal_node": j})
        if len(eps_ft) < args.episodes:
            raise SystemExit(f"episodes-from-trace: only {len(eps_ft)} of "
                             f"{args.episodes} pairs found in the band on this trace")
        episodes = eps_ft
        print(f"[Episodes] {len(episodes)} start/goal pairs sampled FROM THE "
              f"TRACE (band {lo_m:g}-{hi_m:g} m straight-line, seed {args.seed})",
              flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    band = (f"{args.goal_min:g}-{args.goal_max:g}m"
            if args.goal_min is not None and args.goal_max is not None
            else "default")

    import networkx as nx
    for label, method, budget in rows:
        row_edges = None
        if method == "sgm":
            nodes, row_edges = sgm_nodes, sgm_edges
        elif method == "madent":
            nodes = madent_nodes
        elif method == "sgm_paper":
            embeddings = np.stack([np.asarray(o["latent"], np.float64)
                                   for o in trace["observations"]])
            emb_cutoff = embedding_cutoff_from_data(embeddings,
                                                    args.embedding_percentile)
            nodes, _ = place("sgm_paper", dist=np.minimum(dist, max_edge_steps),
                             embeddings=embeddings, order=order,
                             embedding_cutoff=emb_cutoff,
                             consistency_cutoff=float(cutoff or H_STEPS * 0.1))
        elif method == "policy":
            nodes, _ = place("policy", s=s, entropy=entropy_mc, n=budget or
                             len(sgm_nodes or madent_nodes), seed=args.seed,
                             min_sep_m=args.min_sep_m, smooth_w=args.smooth_w)
        else:
            nodes, _ = place(method, s=s, n=budget, seed=args.seed,
                             min_sep_m=args.min_sep_m, smooth_w=args.smooth_w)
        nodes = np.asarray(nodes, dtype=int)
        # ONE EDGE RULE FOR EVERY ARM: directed, weight = critic steps, admitted
        # under the SAME cutoff. The SGM row's edges are the ones its frame
        # learned during aggregation (identical rule, built incrementally).
        if row_edges is None:
            row_edges = learned_edges(dist, nodes, float(max_edge_steps))
        G = nx.DiGraph()
        for n_ in nodes:
            G.add_node(int(n_))
        for a_, b_, w in row_edges:
            # floor at one step: a sub-unit weight breaks Dijkstra's ordering
            G.add_edge(int(nodes[a_]), int(nodes[b_]), steps=max(float(w), 1.0))
        reach = directed_reachability(len(nodes), row_edges)
        ncomp = undirected_components(len(nodes), row_edges)
        sep = edge_separation_stats(row_edges, P, nodes, horizon_m)
        print(f"[Build] {label}: {len(nodes)} nodes, {len(row_edges)} edges, "
              f"{ncomp} undirected comps, directed reach {reach:.3f}, "
              f"edge sep median {sep.get('sep_median_m', float('nan')):.1f} m, "
              f"{100 * sep.get('frac_beyond_horizon', 0):.0f}% beyond "
              f"{horizon_m:g} m", flush=True)
        out = navigate(T, latent, agent, trace, G, nodes, episodes, args, label)
        out.update({"placement": method, "matched_to": (label.split("@")[1]
                                                        if "@" in label else None),
                    "directed_reachability": reach,
                    "undirected_components": ncomp,
                    "edge_separation": sep, "horizon_m": horizon_m,
                    "sgm_edge_cutoff": sgm_edge, "sgm_consistency": sgm_cons,
                    "sgm_gate_m": args.sgm_gate_m, "mad_z": args.mad_z,
                    "calibration": args.calibration or None,
                    "episodes_from_trace": bool(args.episodes_from_trace)})
        out.update({"arm": args.arm, "band": band, "checkpoint": args.ckpt,
                    "episode_seed": args.seed, "town": args.town,
                    "step_cost": args.step_cost,
                    "max_edge_steps": float(max_edge_steps),
                    # THE BUDGET IS REPORTED PER ROW so a reader can confirm the
                    # arms were compared at equal node count rather than taking
                    # it on trust; n_nodes below it is what the arm actually
                    # placed, which can be lower when snapping collides.
                    "budget": (budget if budget is not None else int(len(nodes))),
                    # THE SWEEP COORDINATES, so a Pareto point can be placed
                    # without parsing its filename. For the three budgeted arms
                    # nodes_per_100m IS the independent variable; for SGM the
                    # independent variable is consistency_cutoff and n_nodes is
                    # whatever falls out of it.
                    "nodes_per_100m": args.nodes_per_100m,
                    "consistency_cutoff": (float(cutoff) if cutoff is not None else None),
                    "path_m": path_m,
                    "min_sep_m": args.min_sep_m,
                    "smooth_w": args.smooth_w,
                    "entropy_mean": float(entropy.mean())})
        # WRITTEN ATOMICALLY: a full file appears or the old one is untouched.
        #
        # The supervisor kills these jobs at arbitrary moments, and a kill
        # landing between open() and the end of json.dump leaves a TRUNCATED
        # file under the final name -- a banked result silently replaced by
        # half of one, which json.load may even parse if the truncation falls
        # on a boundary. os.replace is atomic within a filesystem, so a reader
        # sees either the previous complete result or the new complete one.
        # THE TAG CARRIES THE SWEEP POINT INTO THE FILENAME. Without it a second
        # density silently overwrites the first and the Pareto curve collapses
        # to whichever point ran last -- with every file looking complete.
        path = os.path.join(
            args.out_dir,
            f"{args.arm}__{label.replace('@', '_at_')}__{band}{args.tag}.json")
        tmp = path + ".part"
        with open(tmp, "w") as fh:
            json.dump(out, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        spl = "n/a" if out["spl"] is None else f"{out['spl']:.3f}"
        sr = "n/a" if out["success"] is None else f"{out['success']:.3f}"
        print(f"[Result] {label}: SR(routed) {sr}  SPL {spl}  "
              f"routed {out['n_routed']}/{out['episodes']}  "
              f"nodes {out['n_nodes']}  -> {path}", flush=True)


if __name__ == "__main__":
    main()
