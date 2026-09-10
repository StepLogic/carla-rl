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
    os.environ["EVAL_FORCE_MODE"] = str(EXPLORE_MODE)
    env = T.make_env(args.town, seed=args.collect_seed, latent_source=latent,
                     eval_mode=True, port=args.port)
    wrapper = find_wrapper(env, "goal_block")
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
                "episodes_done": int(done)}

    for ep in range(first_ep, args.collect_episodes):
        # SEEDED FROM THE EPISODE INDEX, not from a running generator, so a
        # resumed run draws the same episode a fresh run would have.
        rng = np.random.default_rng((args.collect_seed, ep))
        o, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        last_kept = None
        for t in range(args.collect_steps):
            p = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
            if last_kept is None or np.linalg.norm(p - last_kept) >= args.candidate_spacing:
                obs_list.append(goal_mode(o))
                fields_list.append(wrapper._fields(info))
                pos.append(p)
                yaw.append(float(info.get("hero_yaw", 0.0)))
                seg.append(ep)
                last_kept = p
            o, _r, term, trunc, info = env.step(agent.eval_actions(o))
            if term or trunc:
                break
        print(f"[Collect] episode {ep + 1}/{args.collect_episodes}: "
              f"{len(pos)} candidates so far", flush=True)
        # AFTER EVERY EPISODE. A kill then costs one episode, not sixty.
        if args.trace_cache:
            os.makedirs(os.path.dirname(args.trace_cache) or ".", exist_ok=True)
            with open(args.trace_cache, "wb") as fh:
                pickle.dump(_snapshot(ep + 1), fh)
    env.close()

    trace = _snapshot(args.collect_episodes)
    print(f"[Collect] {len(pos)} candidates over {args.collect_episodes} "
          f"explore episodes, thinned to {args.candidate_spacing} m")
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

    for ep in episodes:
        obs, info = env.reset(seed=ep["seed"])
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
            # STILL DRIVEN, with the true goal only. An unroutable episode is
            # not excluded: excluding it would score a graph on the subset it
            # happened to cover and flatter a sparse, badly placed map.
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
    ap.add_argument("--methods", default="uniform,random,policy,sgm")
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
    ap.add_argument("--collect-seed", type=int, default=7)
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
    ap.add_argument("--max-edge-steps", type=float, default=None)
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

    os.environ["EVAL_FORCE_MODE"] = str(GOAL_MODE)
    if args.goal_min is not None:
        os.environ["GOAL_DIST_MIN"] = str(args.goal_min)
    if args.goal_max is not None:
        os.environ["GOAL_DIST_MAX"] = str(args.goal_max)
        os.environ["GOAL_DIST_START"] = str(args.goal_max)
        os.environ["GOAL_DIST_TARGET"] = str(args.goal_max)

    import train_carla_tqc as T
    from vae_nav.latent_source import from_env as latent_from_env
    from vae_nav.node_placement import (actor_entropy, arclength, place,
                                        sgm_cutoff_for_budget)
    from vae_nav.policy_io import load_agent_state
    from vae_nav.value_distance import H_STEPS

    max_edge_steps = (args.max_edge_steps if args.max_edge_steps is not None
                      else H_STEPS)
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
        if done >= args.collect_episodes:
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
    embeddings = np.stack([np.asarray(o["latent"], np.float64)
                           for o in trace["observations"]])
    emb_cutoff = embedding_cutoff_from_data(embeddings, args.embedding_percentile)
    print(f"[Build] embedding cutoff {emb_cutoff:.4f} "
          f"(p{args.embedding_percentile} of observed latent distances)")

    # SGM FIRST, BECAUSE IT SETS THE BUDGET. It is the one method that derives
    # its own node count -- sparsity falls out of the merge test rather than
    # being imposed -- so that count becomes the equal budget the other three
    # are given. Without the match, "random did worse" cannot be separated from
    # "random had fewer nodes" and the ablation is about sparsity, not
    # placement. This is node_placement_entropy.py's convention.
    order = list(range(len(trace["positions"])))
    # THE CONSISTENCY CUTOFF IS IN STEPS HERE, NOT THE REFERENCE'S UNITS.
    # sgm_placement's default of 5.0 comes from the reference implementation,
    # whose distances are on a different scale; ours are steps under
    # step_cost=1.0, where the critic's whole trained horizon is H_STEPS=106.
    # Demanding that two states agree to within 5 steps about every other node
    # is far stricter than the reference intends, and the failure is SILENT:
    # nothing merges, SGM "keeps" every candidate, and the equal budget
    # silently becomes the entire candidate set.
    # THE BUDGET COMES FROM A DENSITY TARGET, NOT FROM SGM'S OWN COUNT.
    #
    # Deriving it from SGM looked principled and was not: SGM's natural count
    # here is denser than an 8 m-separated peak set can match, so the peaks arm
    # undershot it badly -- measured 25 nodes against a budget of 101, which is
    # a different map, not a different placement. node_placement_entropy.py sets
    # the count from nodes-per-100m for exactly this reason, and at its default
    # of 4/100 m the nominal spacing is 25 m, so the 8 m separation is
    # comfortably satisfiable by every arm.
    s = arclength(trace["positions"], trace["segments"])
    path_m = float(s[-1]) if len(s) else 0.0
    budget = max(2, int(round(path_m / 100.0 * args.nodes_per_100m)))
    print(f"[Build] trace {path_m:.0f} m, {args.nodes_per_100m}/100 m "
          f"-> budget {budget} nodes for every arm")

    # CLIPPED AT THE HORIZON FOR THE MERGE TEST, and this is the difference
    # between SGM working and SGM merging nothing.
    #
    # The consistency test is a MAX over every current node k, so it is decided
    # by the worst-disagreeing pair. Raw predicted distances here run to a
    # median of 221 steps against a trained horizon of 106: beyond H the critic
    # is extrapolating, and comparing two states on those values compares noise,
    # which the max then selects for. edge_reject_mask refuses such edges for
    # exactly this reason; the same reasoning applies to the merge test.
    #
    # Clipping says: two states must agree about everything they can ACTUALLY
    # reach, and where both are beyond the horizon the critic has no opinion
    # worth comparing, so they agree trivially. The graph edges below still use
    # the RAW matrix, where the horizon test rejects rather than clips.
    dist_merge = np.minimum(dist, max_edge_steps)
    if args.consistency_cutoff is not None:
        from vae_nav.sgm_placement import sgm_select
        cutoff = args.consistency_cutoff
        sgm_nodes, merged_into = sgm_select(
            dist_merge, embeddings, embedding_cutoff=emb_cutoff,
            consistency_cutoff=cutoff, order=order)
    else:
        # SGM DERIVES ITS OWN SPARSITY, WHICH IS THE PROPERTY THAT MAKES IT
        # INTERESTING AND ALSO MAKES IT INCOMPARABLE to arms that are told how
        # many nodes to place. Sweeping its one free parameter onto the budget
        # is the standard way to put a self-selecting method on the same axis.
        # The cutoff that was actually used is reported and recorded, not the
        # one that was asked for.
        cutoff, sgm_nodes, merged_into = sgm_cutoff_for_budget(
            dist_merge, embeddings, budget, emb_cutoff, order=order)
    n_merged = int(np.sum(np.asarray(merged_into) >= 0))
    print(f"[Build] sgm kept {len(sgm_nodes)} of {len(order)} candidates "
          f"({n_merged} merged) at consistency_cutoff={cutoff:.1f} steps, "
          f"embedding_cutoff={emb_cutoff:.4f}")

    # REFUSE TO RUN A DEGENERATE ABLATION. With no merges the budget is the
    # whole candidate set, every arm keeps almost everything, and the four rows
    # differ by nothing -- a table that renders perfectly and means nothing.
    # Better to spend a minute here than hours of GPU producing it.
    if n_merged == 0:
        off = dist[~np.eye(len(dist), dtype=bool)]
        raise SystemExit(
            "SGM MERGED NOTHING, so there is no sparsification and no budget.\n"
            f"  consistency_cutoff = {cutoff:.1f} steps\n"
            f"  embedding_cutoff   = {emb_cutoff:.4f}\n"
            "  pairwise distance percentiles (steps): "
            + ", ".join(f"p{p}={np.percentile(off, p):.1f}"
                        for p in (1, 5, 25, 50))
            + "\nRaise --consistency-frac (or set --consistency-cutoff) so the "
              "two-way test can pass, or widen --embedding-percentile so the "
              "prefilter proposes neighbours at all.")

    entropy = actor_entropy(agent, trace["observations"], seed=args.seed)
    print(f"[Build] actor entropy over candidates: mean {entropy.mean():.3f} "
          f"nats, range [{entropy.min():.3f}, {entropy.max():.3f}]")

    os.makedirs(args.out_dir, exist_ok=True)
    band = (f"{args.goal_min:g}-{args.goal_max:g}m"
            if args.goal_min is not None and args.goal_max is not None
            else "default")

    for method in [m for m in args.methods.split(",") if m]:
        if method == "sgm":
            nodes = sgm_nodes
        else:
            nodes, _ = place(method, s=s, entropy=entropy, dist=dist,
                             embeddings=embeddings, n=budget, seed=args.seed,
                             min_sep_m=args.min_sep_m, smooth_w=args.smooth_w,
                             max_steps=max_edge_steps, order=order)
        G = build_graph(dist, np.asarray(nodes, dtype=int), max_edge_steps)
        print(f"[Build] {method}: {len(nodes)} nodes, {G.number_of_edges()} edges",
              flush=True)
        out = navigate(T, latent, agent, trace, G, np.asarray(nodes, dtype=int),
                       episodes, args, method)
        out.update({"arm": args.arm, "band": band, "checkpoint": args.ckpt,
                    "episode_seed": args.seed, "town": args.town,
                    "embedding_cutoff": emb_cutoff,
                    "step_cost": args.step_cost,
                    "max_edge_steps": float(max_edge_steps),
                    # THE BUDGET IS REPORTED PER ROW so a reader can confirm the
                    # arms were compared at equal node count rather than taking
                    # it on trust; n_nodes below it is what the arm actually
                    # placed, which can be lower when snapping collides.
                    "budget": budget,
                    # THE SWEEP COORDINATES, so a Pareto point can be placed
                    # without parsing its filename. For the three budgeted arms
                    # nodes_per_100m IS the independent variable; for SGM the
                    # independent variable is consistency_cutoff and n_nodes is
                    # whatever falls out of it.
                    "nodes_per_100m": args.nodes_per_100m,
                    "consistency_cutoff": float(cutoff),
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
            f"{args.arm}__{method}__{band}{args.tag}.json")
        tmp = path + ".part"
        with open(tmp, "w") as fh:
            json.dump(out, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        spl = "n/a" if out["spl"] is None else f"{out['spl']:.3f}"
        sr = "n/a" if out["success"] is None else f"{out['success']:.3f}"
        print(f"[Result] {method}: SR(routed) {sr}  SPL {spl}  "
              f"routed {out['n_routed']}/{out['episodes']}  "
              f"nodes {out['n_nodes']}  -> {path}", flush=True)


if __name__ == "__main__":
    main()
