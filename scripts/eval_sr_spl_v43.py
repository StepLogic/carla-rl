#!/usr/bin/env python3
"""SR and SPL on fixed long-horizon episodes (>=50 m), mapless vs each placer.

PROTOCOL, matched to the Habitat peer's so the two sides are comparable:
  success   reaching within `--success-radius` of the TRUE goal
  SPL       success * geodesic / max(geodesic, DRIVEN metres). The planner's
            path never enters SPL -- only what the car actually drove.
  geodesic  the road-network route length from start to goal (OpenDRIVE), which
            is this task's analogue of their navmesh shortest path.
  episodes  fixed set, drawn once from a seed, with the md5 of the file printed
            before any driving so a later run cannot quietly use a different set.
  paired    every arm drives the SAME episodes from the same spawn.

WHAT THIS MEASURES, AND WHAT IT DOES NOT. Localisation is by TRUE POSE: the
start and goal map to their nearest node by distance. That is deliberate -- it
isolates the graph and the follower, which is what a placement comparison is
about. The peer's own finding is that localisation was the bigger lever, so a
number here is not a claim about an image-localised system.

ARMS
  mapless    the goal policy driven straight at the final goal, no graph
  <placer>   plan over that map with Dijkstra on the GOAL CRITIC's edge weights
             (predicted steps), then drive leg by leg, advancing a leg when
             within `--arrival-radius` of the current node

Faults are terminal, per the standing rule: a painted crossing on open road,
off-road and collision all end the episode as a failure.

usage: eval_sr_spl_v43.py --town Town02 --maps MAPS.json --shortcuts SC.json
                          --ckpt CKPT.pkl --out OUT.json
"""
import argparse
import hashlib
import heapq
import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
XODR = ("/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/"
        "OpenDrive/{town}.xodr")


def route_m(grp, a, b):
    """Road-network distance. Valid at the scale we use it (>=50 m); it is NOT
    valid for metre-scale pairs, where it measures sampling resolution."""
    import carla
    try:
        r = grp.trace_route(carla.Location(x=float(a[0]), y=float(a[1]), z=0.0),
                            carla.Location(x=float(b[0]), y=float(b[1]), z=0.0))
    except Exception:                                      # noqa: BLE001
        return float("inf")
    if not r or len(r) < 2:
        return float("inf")
    p = np.array([[w.transform.location.x, w.transform.location.y] for w, _ in r])
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def make_episodes(town, grp, wmap, n, min_m, seed, out_path):
    """A fixed set of (start, goal) poses whose ROAD route is at least min_m."""
    if os.path.exists(out_path):
        with open(out_path) as fh:
            eps = json.load(fh)
        print(f"[Episodes] reusing {out_path} ({len(eps['episodes'])} episodes, "
              f"md5 {_md5(out_path)})")
        return eps
    rng = np.random.default_rng(seed)
    wps = wmap.generate_waypoints(5.0)
    xy = np.array([[w.transform.location.x, w.transform.location.y] for w in wps])
    eps, tries = [], 0
    while len(eps) < n and tries < n * 400:
        tries += 1
        i, j = rng.integers(0, len(xy), 2)
        if i == j:
            continue
        d = route_m(grp, xy[i], xy[j])
        if not np.isfinite(d) or d < min_m:
            continue
        eps.append({"index": len(eps), "start_xy": xy[i].tolist(),
                    "goal_xy": xy[j].tolist(), "geodesic_m": d,
                    "euclid_m": float(np.linalg.norm(xy[j] - xy[i])),
                    "env_seed": int(rng.integers(0, 10 ** 6))})
    doc = {"town": town, "seed": seed, "min_geodesic_m": min_m,
           "n": len(eps), "episodes": eps}
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=1)
    g = np.array([e["geodesic_m"] for e in eps])
    print(f"[Episodes] built {out_path}: {len(eps)} episodes, geodesic "
          f"median {np.median(g):.0f} m [{g.min():.0f}, {g.max():.0f}], "
          f"md5 {_md5(out_path)}")
    return doc


def _md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()[:12]


def dijkstra(n, weighted_edges, src, dst):
    """Shortest path by edge weight. Hops are not interchangeable here -- a
    trace edge can be 40 m and a shortcut 1 m -- so BFS would pick nonsense."""
    adj = {}
    for (u, v), w in weighted_edges.items():
        adj.setdefault(u, []).append((v, w))
    dist = {src: 0.0}
    prev, seen, pq = {}, set(), [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == dst:
            break
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, np.inf):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    if dst not in seen:
        return None
    path, u = [dst], dst
    while u != src:
        u = prev[u]
        path.append(u)
    return path[::-1]


def load_graph(maps_path, shortcuts_path, placer, pos, seg):
    """-> (node_xy, {(u,v): weight_m}). Trace edges cost DRIVEN metres along the
    trace (the peer's rule: never prune a wormhole, price it), shortcuts cost
    their straight line, which at <=3 m is what the car would drive."""
    from vae_nav.online_maps import build_edges, trace_edge_cost_m
    res = json.load(open(maps_path))
    idx = np.asarray(res["maps"][placer]["indices"], int)
    xy = np.asarray(res["maps"][placer]["xy"], np.float64)
    w = {}
    for (u, v), c in trace_edge_cost_m(idx, pos, seg):
        # FORWARD ONLY. The trace proves the car drove u -> v; it proves nothing
        # about v -> u, and adding the reverse for free made 35-45% of planned
        # hops run against the direction of travel -- into the oncoming lane on
        # a two-way road, which is the failure these maps kept producing. The
        # reverse direction has to be earned, through the same critic
        # certification as any other shortcut.
        w[(u, v)] = c
    n_trace = len(w)
    if shortcuts_path and os.path.exists(shortcuts_path):
        sc = json.load(open(shortcuts_path))["placers"][placer]["shortcuts"]
        for u, v in sc:                    # DIRECTED: certified i -> j only
            w[(int(u), int(v))] = float(np.linalg.norm(xy[int(v)] - xy[int(u)]))
    print(f"[Graph] {placer}: {len(xy)} nodes, {n_trace // 2} trace edges, "
          f"{len(w) - n_trace} shortcuts")
    return xy, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--shortcuts", default="")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--n-episodes", type=int, default=30)
    ap.add_argument("--min-geodesic", type=float, default=50.0)
    ap.add_argument("--episode-seed", type=int, default=7)
    ap.add_argument("--arms", default="mapless,mad,temporal,uniform")
    ap.add_argument("--success-radius", type=float, default=4.0,
                    help="the repo's radius. The peer uses 1 m; both are "
                         "reported so the two protocols can be compared.")
    ap.add_argument("--arrival-radius", type=float, default=4.0)
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--m-per-step", type=float, default=0.24)
    ap.add_argument("--cap-slack", type=int, default=300)
    a = ap.parse_args()

    os.environ["EVAL_FORCE_MODE"] = "1"            # the GOAL arm drives
    os.environ.setdefault("EVAL_FAULT_ENDS", "fault")
    sys.path.append("/home/robotlab/Apps/CARLA_0.9.15/PythonAPI/carla")
    import carla
    import train_carla_tqc as T
    from src.global_route_planner import GlobalRoutePlanner
    from vae_nav.latent_source import from_env as latent_from_env
    from vae_nav.policy_io import load_agent_state
    from vae_nav.value_distance import goal_relabelled_obs
    from eval_placement import find_attr_owner, find_wrapper, goal_mode

    wmap = carla.Map(a.town, open(XODR.format(town=a.town)).read())
    grp = GlobalRoutePlanner(wmap, 2.0)
    eps = make_episodes(a.town, grp, wmap, a.n_episodes, a.min_geodesic,
                        a.episode_seed, a.episodes)["episodes"]

    with open(a.trace, "rb") as fh:
        tr = pickle.load(fh)
    pos = np.asarray(tr["positions"], np.float64)[:, :2]
    seg = np.asarray(tr["segments"], int)

    latent = latent_from_env()
    env = T.make_env(a.town, seed=a.episode_seed, latent_source=latent,
                     eval_mode=True, port=a.port)
    wrapper = find_wrapper(env, "goal_block")
    agent = T.build_agent(env.observation_space, env.action_space, 0)
    with open(a.ckpt, "rb") as fh:
        state = pickle.load(fh)
    load_agent_state(agent, state.get("agent", state))

    graphs = {}
    for arm in a.arms.split(","):
        if arm != "mapless":
            graphs[arm] = load_graph(a.maps, a.shortcuts, arm, pos, seg)

    zero = np.zeros(env.action_space.shape, np.float32)
    rows, t0 = [], time.time()
    for arm in a.arms.split(","):
        for e in eps:
            start_xy = np.asarray(e["start_xy"], np.float64)
            goal_xy = np.asarray(e["goal_xy"], np.float64)
            exp = find_attr_owner(env, "experiment").experiment
            exp.origin = wmap.get_waypoint(carla.Location(
                x=float(start_xy[0]), y=float(start_xy[1]), z=0.0),
                project_to_road=True)
            exp.destination = wmap.get_waypoint(carla.Location(
                x=float(goal_xy[0]), y=float(goal_xy[1]), z=0.0),
                project_to_road=True)
            obs, info = env.reset(seed=int(e["env_seed"]))
            for _attr in ("origin", "destination"):
                if hasattr(exp, _attr):
                    delattr(exp, _attr)
            for _ in range(10):
                obs, _r, _t, _tr, info = env.step(zero)

            here = np.asarray(info["hero_pos"], np.float64)[:2]
            path, plan_m = None, 0.0
            if arm != "mapless":
                xy, w = graphs[arm]
                s_node = int(np.argmin(np.linalg.norm(xy - here, axis=1)))
                g_node = int(np.argmin(np.linalg.norm(xy - goal_xy, axis=1)))
                path = dijkstra(len(xy), w, s_node, g_node)
                if path:
                    plan_m = sum(w[(u, v)] for u, v in zip(path[:-1], path[1:]))

            cap = int(max(e["geodesic_m"], plan_m) / a.m_per_step) + a.cap_slack
            leg, driven, prev, end = 1, 0.0, here.copy(), "cap"
            for t in range(1, cap + 1):
                if arm == "mapless" or not path or leg >= len(path):
                    tgt = goal_xy
                else:
                    tgt = graphs[arm][0][path[leg]]
                drive = goal_relabelled_obs(goal_mode(obs), wrapper._fields(info),
                                            tgt, wrapper.goal_block)
                obs, _r, term, trunc, info = env.step(agent.eval_actions(drive))
                cur = np.asarray(info["hero_pos"], np.float64)[:2]
                driven += float(np.linalg.norm(cur - prev))
                prev = cur
                if np.linalg.norm(cur - goal_xy) <= a.success_radius:
                    end = "success"
                    break
                if arm != "mapless" and path and leg < len(path):
                    # CARROT, not "next node". The arrival radius (4 m) is wider
                    # than the gap between MAD's nodes (~1.2 m), so advancing one
                    # leg per arrival let the whole plan be "reached" in a few
                    # steps and the car drove at the final goal with the graph
                    # doing nothing. Advance past every node already inside the
                    # radius, so the target is always genuinely ahead.
                    while (leg < len(path)
                           and np.linalg.norm(cur - graphs[arm][0][path[leg]])
                           <= a.arrival_radius):
                        leg += 1
                if term or trunc:
                    end = str(info.get("goal_end", info.get("explore_end",
                                                            "terminated")))
                    break
            geo = float(e["geodesic_m"])
            ok = end == "success"
            rows.append({
                "arm": arm, "episode": int(e["index"]), "end": end,
                "success": bool(ok), "geodesic_m": geo, "driven_m": driven,
                "final_dist_m": float(np.linalg.norm(prev - goal_xy)),
                "spl": (geo / max(geo, driven)) if ok else 0.0,
                "planned": path is not None, "plan_m": plan_m,
                "plan_hops": (len(path) - 1) if path else 0, "steps": t,
            })
            print(f"[{arm:8s}] ep{e['index']:02d} {end:12s} geo {geo:6.0f} m "
                  f"driven {driven:6.0f} m spl {rows[-1]['spl']:.2f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            with open(a.out, "w") as fh:
                json.dump({"town": a.town, "protocol": vars(a), "rows": rows},
                          fh, indent=1)

    env.close()
    print(f"\n=== {a.town}: SR and SPL over {len(eps)} episodes "
          f"(geodesic >= {a.min_geodesic:g} m) ===")
    for arm in a.arms.split(","):
        R = [r for r in rows if r["arm"] == arm]
        if not R:
            continue
        sr = float(np.mean([r["success"] for r in R]))
        spl = float(np.mean([r["spl"] for r in R]))
        unplanned = sum(1 for r in R if arm != "mapless" and not r["planned"])
        print(f"  {arm:9s} SR {sr:.2f}  SPL {spl:.2f}  "
              f"(n={len(R)}, no route found for {unplanned})")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
