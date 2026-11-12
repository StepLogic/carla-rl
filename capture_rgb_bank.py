"""RGB bank for image localisation (user 2026-09-19: "replace localization with dinov2").

The trace pickle keeps only the 384-d VAE latent per sample -- no pixels -- so a DINOv2 (or any other
encoder's) localiser has nothing to match against. This captures, ONCE, the frame the policy's camera
sees at every trace pose and at every episode start/goal pose, through the SAME path as the live
drive: experiment.obs_img (180x180 sensor -> 120x120 uint8 RGB), read off info["rgb"].

Teleport, not reset: hero.set_transform + settle ticks, as drive_nomad.py captures its node images --
a reset per pose would cost an hour. The live VAE latent is stored beside each frame, so the two
localisers can later be compared on IDENTICAL inputs rather than trace-latent vs teleported-frame.
"""
import argparse, os, pickle, sys, time
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))

TRACE = "results/placement_v6n_town02_budget10k_laneoff_speed10/trace_s7.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=TRACE)
    ap.add_argument("--episodes", default="results/episodes/town02_speed10_s7_episodes_short20_long20_seed0.pkl")
    ap.add_argument("--ckpt", default="checkpoints/carla_gated_her_v6n/checkpoint_250000.pkl")
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--settle-ticks", type=int, default=10)
    ap.add_argument("--out", default="results/episodes/town02_speed10_s7_rgb_bank.pkl")
    ap.add_argument("--limit", type=int, default=0, help="first N trace poses only (smoke test)")
    a = ap.parse_args()

    import carla
    import train_carla_tqc as T
    from vae_nav.latent_source import from_env as latent_from_env
    from vae_nav.policy_io import load_agent_state
    from eval_placement import find_attr_owner, goal_mode, find_wrapper, _assert_town, GOAL_MODE

    os.environ["EVAL_FORCE_MODE"] = str(GOAL_MODE)
    tr = pickle.load(open(a.trace, "rb"))
    P = np.asarray(tr["positions"], np.float64)[:, :2]; YAW = np.asarray(tr["yaws"], np.float64)
    bundle = pickle.load(open(a.episodes, "rb"))

    jobs = [("trace", int(i), None, P[i], float(YAW[i])) for i in range(len(P))]
    if a.limit:
        jobs = jobs[:a.limit]
    for s in ("short", "long"):
        for e in bundle[s]["episodes"]:
            for role in ("start", "goal"):
                jobs.append((s, int(e["index"]), role, np.asarray(e[f"{role}_xy"], np.float64),
                             float(e[f"{role}_yaw_rad"])))
    seed0 = int(bundle["short"]["episodes"][0]["env_seed"])
    print(f"[bank] {len(jobs)} poses ({sum(1 for j in jobs if j[0]=='trace')} trace + "
          f"{sum(1 for j in jobs if j[0]!='trace')} episode views)", flush=True)

    latent = latent_from_env()
    env = T.make_env(a.town, seed=seed0, latent_source=latent, eval_mode=True, port=a.port)
    _assert_town(env, a.town, "bank")
    agent = T.build_agent(env.observation_space, env.action_space, 0)
    state = pickle.load(open(a.ckpt, "rb"))
    if isinstance(state, dict) and "identity" in state:
        from vae_nav.latent_source import LatentSource
        LatentSource.assert_matches(state["identity"], latent)
    load_agent_state(agent, state.get("agent", state))
    core = find_attr_owner(env, "core").core
    amap = core.map
    obs, info = env.reset(seed=seed0)
    zero = np.zeros(env.action_space.shape, np.float32)

    rows, t0 = [], time.time()
    for k, (kind, idx, role, xy, yaw) in enumerate(jobs):
        wp = amap.get_waypoint(carla.Location(x=float(xy[0]), y=float(xy[1]), z=0.0), project_to_road=True)
        core.hero.set_transform(carla.Transform(
            carla.Location(x=float(xy[0]), y=float(xy[1]), z=wp.transform.location.z + 0.3),
            carla.Rotation(yaw=float(np.degrees(yaw)))))
        core.hero.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        for _t in range(a.settle_ticks):
            obs, _r, _term, _trunc, info = env.step(zero)
        rgb = info.get("rgb")
        if rgb is None:
            raise SystemExit("info['rgb'] is missing -- the env did not expose the camera frame")
        hero = np.asarray(info.get("hero_pos", (np.inf, np.inf)), np.float64)[:2]
        rows.append({"kind": kind, "index": idx, "role": role,
                     "requested_xy": xy.tolist(), "requested_yaw_rad": yaw,
                     "gt_xy": hero.tolist(), "gt_yaw_rad": float(info.get("hero_yaw", np.nan)),
                     "teleport_error_m": float(np.linalg.norm(hero - xy)),
                     "rgb": np.ascontiguousarray(np.asarray(rgb, np.uint8)),
                     "latent": np.asarray(goal_mode(obs)["latent"], np.float32).copy()})
        if (k + 1) % 100 == 0:
            print(f"[bank] {k+1}/{len(jobs)} teleport err median "
                  f"{np.median([r['teleport_error_m'] for r in rows]):.2f} m [{time.time()-t0:.0f}s]", flush=True)
    env.close()
    errs = np.array([r["teleport_error_m"] for r in rows])
    out = {"meta": {"trace": a.trace, "episodes": a.episodes, "town": a.town, "checkpoint": a.ckpt,
                    "rgb_shape": list(rows[0]["rgb"].shape), "settle_ticks": a.settle_ticks,
                    "n": len(rows), "teleport_error_median_m": float(np.median(errs)),
                    "note": "rgb is experiment.obs_img via info['rgb'] -- the same frame the live drive sees"},
           "rows": rows}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pickle.dump(out, open(a.out, "wb"), protocol=4)
    print(f"[bank] wrote {a.out}: {len(rows)} frames {rows[0]['rgb'].shape}, teleport error median "
          f"{np.median(errs):.2f} m p90 {np.percentile(errs,90):.2f} m max {errs.max():.2f} m", flush=True)


if __name__ == "__main__":
    main()
