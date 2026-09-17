"""Drive the planned paths with option C (user decisions 2026-09-13) and score success.

Maps: the two-way sequence maps (results/maps/sequence/*_sequence.json), all 7.
Episodes: the 40 cached episodes (20 short, 20 long incl. 5 off-graph goals).

Per episode:
  1. Spawn at the start (origin + destination waypoints, as the placement harness),
     hold 10 zero-action ticks, image-localise the live latent among ALL nodes.
  2. Goal node: image-localise the captured goal view (episode_views pickle).
  3. Plan: BFS hop path, edges usable both ways.
  4. Drive in goal mode: the next path node's position is the policy's goal.
     On arrival within 4 m -- or after --leg-steps without arriving -- re-localise
     the live latent, accepting only the expected node or the next 3 along the
     plan; a different match replans from it (plan_c.on_node_event). On a
     timeout whose match is still the expected node, skip to the next node.
  5. Plan finished: target the TRUE goal in goal mode for one leg; if not reached,
     switch to EXPLORE mode (goal block zeroed) until the cap.
  End: success = within 4 m of the true goal; collision; or the step cap
  2 x (planned path metres / 0.24) + 300. Lane crossings and off-road are logged,
  never terminal (v6n env: LANE_WEDGE unset, so a crossing does not freeze the car).

Edges driven against the trace's driving direction are driven as planned and logged.
Writes one JSON line per episode; re-running resumes (skips episodes already written).
"""
import argparse, json, os, pickle, sys, time
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))

from plan_c import bfs_path, candidate_window, nearest_latent, on_node_event, step_cap  # noqa: E402

TRACE = "results/placement_v6n_town02_budget10k_laneoff_speed10/trace_s7.pkl"
MAP_NAMES = ["junction", "uniform6_trace", "uniform2_trace", "entropy_smooth9_high", "entropy_smooth9_low",
             "entropy_nosmooth_high", "entropy_nosmooth_low"]


def load_sequence_map(name, seq_dir="results/maps/sequence"):
    m = json.load(open(f"{seq_dir}/{name}_sequence.json"))
    xy = np.asarray([[d["x_m"], d["y_m"]] for d in m["nodes"]], np.float64)
    tidx = np.asarray([int(d["trace_index"]) for d in m["nodes"]], int)
    edges = [(int(e["from"]), int(e["to"])) for e in m["edges"]]
    legs = {}
    for e in m["edges"]:
        u, v = int(e["from"]), int(e["to"])
        pts = np.vstack([xy[u], np.asarray([[w["x_m"], w["y_m"]] for w in e["waypoints"]]), xy[v]])
        legs[(u, v)] = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    return xy, tidx, edges, legs


def path_metres(path, xy, legs):
    tot = 0.0
    for u, v in zip(path[:-1], path[1:]):
        tot += legs.get((u, v), legs.get((v, u), float(np.linalg.norm(xy[u] - xy[v]))))
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", default=MAP_NAMES)
    ap.add_argument("--episodes", default="results/episodes/town02_speed10_s7_episodes_short20_long20_seed0.pkl")
    ap.add_argument("--views", default="results/episodes/town02_speed10_s7_episode_views_seed0.pkl")
    ap.add_argument("--ckpt", default="checkpoints/carla_gated_her_v6n/checkpoint_250000.pkl")
    ap.add_argument("--town", default="Town02")
    # user 2026-09-15 Town07 repeat: that town's trace and sequence maps; defaults are the Town02 run
    ap.add_argument("--trace", default=TRACE)
    ap.add_argument("--seq-dir", default="results/maps/sequence")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--out", default="results/drive_plan_c/sequence_undirected.jsonl")
    ap.add_argument("--arrival-radius", type=float, default=4.0)
    ap.add_argument("--success-radius", type=float, default=4.0)
    ap.add_argument("--window-hops", type=int, default=3)
    ap.add_argument("--leg-steps", type=int, default=200, help="per-leg timeout (placement harness default)")
    ap.add_argument("--settle-ticks", type=int, default=10)
    ap.add_argument("--m-per-step", type=float, default=0.24)
    ap.add_argument("--cap-slack", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="drive only the first N episodes per map (smoke test)")
    # VIDEOS (user 2026-09-13 "record some videos for entropy"): passive -- frames are drawn from info["rgb"]
    # and the poses already computed; nothing the car does depends on them. Off unless --video-dir is set.
    ap.add_argument("--video-dir", default="")
    ap.add_argument("--video-maps", nargs="*", default=["entropy_smooth9_high", "entropy_smooth9_low",
                                                        "entropy_nosmooth_high", "entropy_nosmooth_low"])
    ap.add_argument("--video-episodes", nargs="*", default=["short2", "short10", "long0", "long1"])
    a = ap.parse_args()

    import carla
    import train_carla_tqc as T
    from vae_nav.latent_source import from_env as latent_from_env
    from vae_nav.policy_io import load_agent_state
    from vae_nav.value_distance import goal_relabelled_obs
    from vae_nav.vae_goal_wrapper import apply_mode
    from eval_placement import find_attr_owner, goal_mode, find_wrapper, _assert_town, GOAL_MODE, EXPLORE_MODE

    os.environ["EVAL_FORCE_MODE"] = str(GOAL_MODE)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            if line.strip():
                r = json.loads(line); done.add((r["map"], r["set"], r["index"]))
    bundle = pickle.load(open(a.episodes, "rb"))
    V = pickle.load(open(a.views, "rb"))
    views = {(v["set"], v["index"], v["role"]): v for v in V["views"]}
    tr = pickle.load(open(a.trace, "rb"))
    Z = np.stack([np.asarray(o["latent"], np.float64) for o in tr["observations"]])
    episodes = [(s, e) for s in ("short", "long") for e in bundle[s]["episodes"]]
    if a.limit:
        episodes = episodes[:a.limit]
    todo = [(mn, s, e) for mn in a.maps for s, e in episodes if (mn, s, int(e["index"])) not in done]
    print(f"[drive] {len(todo)} episodes to drive ({len(done)} already in {a.out}); maps {a.maps}", flush=True)
    if not todo:
        return

    latent = latent_from_env()
    env = T.make_env(a.town, seed=int(todo[0][2]["env_seed"]), latent_source=latent, eval_mode=True, port=a.port)
    _assert_town(env, a.town, "drive_plan_c")
    agent = T.build_agent(env.observation_space, env.action_space, 0)
    state = pickle.load(open(a.ckpt, "rb"))
    if isinstance(state, dict) and "identity" in state:
        from vae_nav.latent_source import LatentSource
        LatentSource.assert_matches(state["identity"], latent)
    load_agent_state(agent, state.get("agent", state))
    wrapper = find_wrapper(env, "goal_block")
    amap = find_attr_owner(env, "core").core.map
    zero = np.zeros(env.action_space.shape, np.float32)

    maps = {mn: load_sequence_map(mn, a.seq_dir) for mn in a.maps}
    topdown = {}
    if a.video_dir:
        from drive_video import TopDown, open_writer, render_frame
        from render_placement_maps import lane_polylines
        os.makedirs(a.video_dir, exist_ok=True)
        _lanes = lane_polylines(a.town)
        for mn in a.maps:
            if mn in a.video_maps:
                topdown[mn] = TopDown(_lanes, maps[mn][0], maps[mn][2])
    t_all = time.time()
    for k, (mn, set_name, e) in enumerate(todo):
        xy, tidx, edges, legs = maps[mn]
        n, Zn = len(xy), Z[tidx]
        driven_dirs = set(edges)
        idx = int(e["index"])
        start_xy = np.asarray(e["start_xy"], np.float64); goal_xy = np.asarray(e["goal_xy"], np.float64)

        exp = find_attr_owner(env, "experiment").experiment
        exp.origin = amap.get_waypoint(carla.Location(x=float(start_xy[0]), y=float(start_xy[1]), z=0.0), project_to_road=True)
        exp.destination = amap.get_waypoint(carla.Location(x=float(goal_xy[0]), y=float(goal_xy[1]), z=0.0), project_to_road=True)
        obs, info = env.reset(seed=int(e["env_seed"]))
        for _a in ("origin", "destination"):
            if hasattr(exp, _a):
                delattr(exp, _a)
        for _t in range(a.settle_ticks):
            obs, _r, _term, _trunc, info = env.step(zero)
        here = np.asarray(info["hero_pos"], np.float64)[:2]
        spawn_err = float(np.linalg.norm(here - start_xy))

        start_node = nearest_latent(obs["latent"], Zn, range(n))
        goal_node = nearest_latent(views[(set_name, idx, "goal")]["obs"]["latent"], Zn, range(n))
        path = bfs_path(n, edges, start_node, goal_node, undirected=True)
        plan0 = list(path) if path is not None else None
        plan_m = path_metres(path, xy, legs) if path is not None else 0.0
        cap = step_cap(plan_m, a.m_per_step, a.cap_slack)
        i = 1 if path is not None else 0                       # path[0] is where we already are
        if path is None:
            path = []

        log = {"relocalisations": 0, "reloc_changed": 0, "replans": 0, "leg_timeouts": 0, "skips": 0,
               "reversed_legs_driven": 0, "lane_steps": 0, "off_road_steps": 0, "explore_steps": 0,
               "final_goal_leg_started": None, "explore_started": None, "reloc_events": []}
        mode, leg_t, driven_m, prev, end, t = GOAL_MODE, 0, 0.0, here.copy(), "cap", 0
        final_leg = False

        def target_now():
            return xy[path[i]] if i < len(path) else goal_xy

        writer, video_path, trail = None, None, [here.copy()]
        if mn in topdown and f"{set_name}{idx}" in a.video_episodes:
            video_path = os.path.join(a.video_dir, f"{mn}_{set_name}{idx:02d}.mp4")
            writer = open_writer(video_path)

        for t in range(1, cap + 1):
            tgt = target_now()
            if mode == GOAL_MODE:
                drive_obs = goal_relabelled_obs(goal_mode(obs), wrapper._fields(info), tgt, wrapper.goal_block)
            else:
                drive_obs = dict(obs)
                drive_obs["goal"] = apply_mode(np.array(drive_obs["goal"], np.float32, copy=True), EXPLORE_MODE)
                drive_obs["mode"] = np.array([float(EXPLORE_MODE)], np.float32)
                log["explore_steps"] += 1
            obs, _r, _term, _trunc, info = env.step(agent.eval_actions(drive_obs))
            here = np.asarray(info.get("hero_pos", prev), np.float64)[:2]
            driven_m += float(np.linalg.norm(here - prev)); prev = here.copy()
            if writer is not None:
                trail.append(here.copy())
                writer.write(render_frame(
                    topdown[mn], info.get("rgb"), trail, path, i, tgt, goal_xy, start_node, goal_node,
                    [f"{mn} {set_name}#{idx}", f"Step {t}/{cap}  {'GOAL' if mode == GOAL_MODE else 'EXPLORE'} mode",
                     f"Target {'node %d' % path[i] if i < len(path) else 'true goal'}  hops left {max(len(path) - i, 0)}",
                     f"Relocs {log['relocalisations']}  replans {log['replans']}",
                     f"To goal {np.linalg.norm(here - goal_xy):.0f} m"]))
            log["lane_steps"] += int(bool(info.get("lane_any", info.get("lane_invasion_any", False))))
            log["off_road_steps"] += int(bool(info.get("off_road", False)))
            if np.linalg.norm(here - goal_xy) <= a.success_radius:
                end = "success"; break
            if bool(info.get("collision", False)):
                end = "collision"; break
            if mode == EXPLORE_MODE:
                continue
            leg_t += 1
            if i >= len(path):                                   # final leg to the true goal
                if log["final_goal_leg_started"] is None:
                    log["final_goal_leg_started"] = t
                if leg_t >= a.leg_steps:
                    mode = EXPLORE_MODE; log["explore_started"] = t
                continue
            arrived = np.linalg.norm(here - tgt) <= a.arrival_radius
            if not arrived and leg_t < a.leg_steps:
                continue
            # RE-LOCALISE at the node (or on a leg timeout), only within the plan window
            window = candidate_window(path, i, a.window_hops)
            matched = nearest_latent(obs["latent"], Zn, window)
            expected = path[i]
            log["relocalisations"] += 1
            prev_node = path[i - 1] if i >= 1 else None
            if matched == expected and not arrived:
                i += 1; log["leg_timeouts"] += 1; log["skips"] += 1; replanned = False
            else:
                if not arrived:
                    log["leg_timeouts"] += 1
                path, i, replanned = on_node_event(path, i, matched, goal_node, n, edges, undirected=True)
            if matched != expected:
                log["reloc_changed"] += 1
            log["replans"] += int(replanned)
            if prev_node is not None and arrived and (prev_node, expected) not in driven_dirs and (expected, prev_node) in driven_dirs:
                log["reversed_legs_driven"] += 1
            if len(log["reloc_events"]) < 200:
                log["reloc_events"].append([t, int(expected), int(matched), bool(arrived), bool(replanned)])
            leg_t = 0

        if writer is not None:
            writer.release()
        final_d = float(np.linalg.norm(here - goal_xy))
        route = e.get("route_m")
        rec = {"map": mn, "set": set_name, "index": idx, "goal_kind": e.get("goal_kind", "on_graph"),
               "route_m": route, "spawn_error_m": spawn_err, "start_node": int(start_node), "goal_node": int(goal_node),
               "start_node_err_m": float(np.linalg.norm(xy[start_node] - start_xy)),
               "goal_node_err_m": float(np.linalg.norm(xy[goal_node] - goal_xy)),
               "plan_found": plan0 is not None, "plan_hops": (len(plan0) - 1) if plan0 else None, "plan_m": plan_m,
               "plan_nodes": plan0, "step_cap": cap, "steps": t, "end": end, "success": end == "success",
               "final_dist_m": final_d, "driven_m": driven_m,
               "spl": (float(route) / max(driven_m, float(route))) if (end == "success" and route) else 0.0,
               "reached_plan_end": log["final_goal_leg_started"] is not None, "video": video_path, **log}
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        el = time.time() - t_all
        print(f"[drive] {k+1}/{len(todo)} {mn} {set_name}#{idx} ({rec['goal_kind']}): {end} in {t}/{cap} steps, "
              f"final {final_d:.1f} m, plan {rec['plan_hops']} hops {plan_m:.0f} m, relocs {log['relocalisations']} "
              f"(changed {log['reloc_changed']}, replans {log['replans']}), explore {log['explore_steps']} [{el:.0f}s]",
              flush=True)
    env.close()
    print(f"[drive] done: {a.out}", flush=True)


if __name__ == "__main__":
    main()
