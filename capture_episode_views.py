"""Spawn the agent at every episode START and GOAL and cache what it sees.

For each episode in an episodes pickle (short + long sets), the car is spawned at
the start pose and at the goal pose through the SAME continuation mechanism trace
collection uses (set_continue_origin: origin + destination waypoints, snapped to
the lane; then the spawn check). After each reset the live observation (latent,
vector, mode, goal block), the per-step fields (hero_pos, hero_yaw, goal_pos,
action_history) and the ground-truth pose are stored. Off-graph goals are captured
the same way, so every goal gets a live image.

Plan-only pipeline, stage A. The ground-truth pose is stored for SCORING only;
localisation in stage B uses the latent (and, for the critic step, the fields --
the v6n goal block is pose-based).
"""
import argparse, os, pickle, sys, time
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="results/episodes/town02_speed10_s7_episodes_short20_long20_seed0.pkl")
    ap.add_argument("--ckpt", default="checkpoints/carla_gated_her_v6n/checkpoint_250000.pkl")
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--settle-ticks", type=int, default=10,
                    help="zero-action ticks after each spawn before reading the latent (user: yes)")
    ap.add_argument("--out", default="results/episodes/town02_speed10_s7_episode_views_seed0.pkl")
    a = ap.parse_args()

    import carla
    import train_carla_tqc as T
    from vae_nav.latent_source import from_env as latent_from_env
    from vae_nav.policy_io import load_agent_state
    from eval_placement import (find_attr_owner, set_continue_origin, clear_continue_origin,
                                continued_spawn_ok, goal_mode, find_wrapper, _assert_town, GOAL_MODE)

    os.environ["EVAL_FORCE_MODE"] = str(GOAL_MODE)
    bundle = pickle.load(open(a.episodes, "rb"))
    jobs = []
    for set_name in ("short", "long"):
        for e in bundle[set_name]["episodes"]:
            for role in ("start", "goal"):
                jobs.append((set_name, int(e["index"]), role, np.asarray(e[f"{role}_xy"], np.float64),
                             float(e[f"{role}_yaw_rad"]), int(e["env_seed"])))
    print(f"[views] {len(jobs)} poses to capture ({len(jobs)//2} episodes x start/goal)", flush=True)

    latent = latent_from_env()
    env = T.make_env(a.town, seed=jobs[0][5], latent_source=latent, eval_mode=True, port=a.port)
    _assert_town(env, a.town, "views")
    agent = T.build_agent(env.observation_space, env.action_space, 0)
    state = pickle.load(open(a.ckpt, "rb"))
    if isinstance(state, dict) and "identity" in state:
        from vae_nav.latent_source import LatentSource
        LatentSource.assert_matches(state["identity"], latent)
    load_agent_state(agent, state.get("agent", state))
    wrapper = find_wrapper(env, "goal_block")
    core = find_attr_owner(env, "core").core
    amap = core.map

    views, misses, t0 = [], 0, time.time()
    for k, (set_name, idx, role, xy, yaw, env_seed) in enumerate(jobs):
        wp = amap.get_waypoint(carla.Location(x=float(xy[0]), y=float(xy[1]), z=0.0), project_to_road=True)
        exp = find_attr_owner(env, "experiment").experiment
        set_continue_origin(exp, wp)
        obs, info = env.reset(seed=env_seed)
        clear_continue_origin(find_attr_owner(env, "experiment").experiment)
        spawn_xy = np.asarray(info.get("hero_pos", (np.inf, np.inf)), np.float64)[:2]
        ok = continued_spawn_ok(spawn_xy, wp)
        # SETTLE (user: yes): the first frame after a teleport can be stale, so hold a
        # zero action for --settle-ticks before reading the latent and the pose.
        for _t in range(a.settle_ticks):
            obs, _r, term, trunc, info = env.step(np.zeros(env.action_space.shape, np.float32))
            if term or trunc:
                break
        hero = np.asarray(info.get("hero_pos", (np.inf, np.inf)), np.float64)[:2]
        misses += int(not ok)
        fields = wrapper._fields(info)
        views.append({"set": set_name, "index": idx, "role": role,
                      "requested_xy": xy.tolist(), "requested_yaw_rad": yaw,
                      "waypoint_xy": [wp.transform.location.x, wp.transform.location.y],
                      "gt_xy": hero.tolist(), "gt_yaw_rad": float(info.get("hero_yaw", np.nan)),
                      "spawn_ok": bool(ok), "spawn_error_m": float(np.linalg.norm(hero - xy)),
                      "obs": {kk: np.asarray(vv).copy() for kk, vv in goal_mode(obs).items()},
                      "fields": {kk: (np.asarray(vv).copy() if not isinstance(vv, float) else vv)
                                 for kk, vv in fields.items()}})
        if (k + 1) % 10 == 0 or not ok:
            print(f"[views] {k+1}/{len(jobs)} {set_name}#{idx} {role}: spawned ({hero[0]:.1f},{hero[1]:.1f}) "
                  f"asked ({xy[0]:.1f},{xy[1]:.1f}) err {views[-1]['spawn_error_m']:.1f} m ok={ok} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    env.close()
    out = {"meta": {"episodes": a.episodes, "checkpoint": a.ckpt, "town": a.town, "n_views": len(views),
                    "spawn_misses": misses, "mode": "goal (EVAL_FORCE_MODE)",
                    "note": "gt_* are for scoring only; obs['latent'] is the live camera latent"},
           "views": views}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pickle.dump(out, open(a.out, "wb"), protocol=4)
    errs = [v["spawn_error_m"] for v in views]
    print(f"[views] wrote {a.out}: {len(views)} views, spawn misses {misses}, spawn error median "
          f"{np.median(errs):.2f} m max {np.max(errs):.2f} m", flush=True)


if __name__ == "__main__":
    main()
