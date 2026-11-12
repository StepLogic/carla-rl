"""NoMaD baseline: the exploration trace as NoMaD's topological map (user decisions 2026-09-13).

Runs in the `nomad_carla` conda env (clone of oatomobile + gymnasium, opencv, tensorboard; torch CUDA).
  map       uniform[2 m] trace nodes (547), edges = the two-way trace sequence graph (same as the v6n maps)
  images    raw 300x300 RGB from the baseline camera, captured by teleporting to each node pose (trace yaw)
            and to each episode's goal pose; NoMaD's own transform resizes them to 96x96
  localise  start = argmin NoMaD predicted temporal distance from the live context (last 4 frames) to all
            547 node images; goal = the same from the goal image (context = goal image x4)
  plan      BFS hop path on the two-way trace graph; topomap = path node images + the goal image
  follow    navigation_policies NoMaD.eval_action rules (radius-3 window, subgoal = closest node if its
            distance < 3 else the next, 10 diffusion steps x 8 samples, waypoints = sample 1) -> SetPointAgent
            PID (baseline setup; its (steer, throttle) output goes through the baseline env compute_action); when NoMaD's stop rule fires (within 2 nodes of the goal image) it switches
            to exploration mode (goal masked) until the cap
  end       success = within 4 m of the true goal; collision; or cap 2 x (plan m / 0.24) + 300 steps
Writes one JSON line per episode in drive_plan_c.py's schema (map = nomad_uniform2_trace); resumes.
"""
import argparse, atexit, json, os, signal, socket, subprocess, sys, time
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/robotlab/scratch/carla-rl"
for p in (os.path.join(REPO, "src"), REPO, os.path.join(REPO, "dependencies/rlib_integration"),
          os.path.join(REPO, "dependencies/navigation_policies"), W):
    if p not in sys.path:
        sys.path.insert(0, p)

from plan_c import bfs_path, step_cap                                      # noqa: E402
from nomad_follow import localise_all, plan_finished, select_subgoal, window  # noqa: E402

MAP_NAME_DEFAULT = "nomad_uniform2_trace"
SEQ_MAP = "results/maps/sequence/uniform2_trace_sequence.json"
CKPT = os.path.join(REPO, "dependencies/navigation_policies/navigation_policies/pretrained_models/nomad.pth")
_SERVER = {"proc": None}


def ensure_carla(port, root, quality="Epic", timeout=180.0):
    """Start CARLA from Python on a known port (same launch as train_carla_tqc.ensure_carla)."""
    import carla

    def answering():
        with socket.socket() as sk:
            sk.settimeout(1.0)
            if sk.connect_ex(("localhost", int(port))) != 0:
                return False
        try:
            c = carla.Client("localhost", int(port)); c.set_timeout(4.0); c.get_world().get_map()
            return True
        except Exception:  # noqa: BLE001
            return False

    if answering():
        print(f"[CARLA] port {port} already serving -- attaching to a server this process did NOT start", flush=True)
        return
    cmd = [f"{root}/CarlaUE4.sh", f"-carla-port={port}", f"-quality-level={quality}", "-windowed", "-ResX=640",
           "-ResY=360", "-prefernvidia"]
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"carla_py_{port}_{int(time.time())}.log")
    shell = " ".join(cmd) + " 2>&1 | sed -u '/Invalid session: no stream available/d' > " + log_path
    print(f"[CARLA] starting: {' '.join(cmd)}  (log: {log_path})", flush=True)
    proc = subprocess.Popen(shell, shell=True, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"CARLA exited during startup (rc={proc.returncode})")
        if answering():
            print(f"[CARLA] ready on port {port} after {time.time() - t0:.0f} s", flush=True)
            break
        time.sleep(1.0)
    else:
        os.killpg(proc.pid, signal.SIGKILL)
        raise RuntimeError(f"CARLA did not answer on port {port} in {timeout}s")
    _SERVER["proc"] = proc

    def _stop():
        pr = _SERVER["proc"]
        if pr is not None and pr.poll() is None:
            try:
                os.killpg(pr.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    atexit.register(_stop)


def load_sequence_map(path):
    m = json.load(open(path))
    xy = np.asarray([[d["x_m"], d["y_m"]] for d in m["nodes"]], np.float64)
    edges = [(int(e["from"]), int(e["to"])) for e in m["edges"]]
    legs = {}
    for e in m["edges"]:
        u, v = int(e["from"]), int(e["to"])
        pts = np.vstack([xy[u], np.asarray([[w["x_m"], w["y_m"]] for w in e["waypoints"]]), xy[v]])
        legs[(u, v)] = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    return xy, edges, legs


def path_metres(path, xy, legs):
    return float(sum(legs.get((u, v), legs.get((v, u), float(np.linalg.norm(xy[u] - xy[v]))))
                     for u, v in zip(path[:-1], path[1:])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/nomad/nomad_uniform2_trace.jsonl")
    ap.add_argument("--image-dir", default="results/nomad/images")
    ap.add_argument("--node-poses", default="results/nomad/uniform2_node_poses.json")
    ap.add_argument("--episodes", default="results/nomad/episodes_short20_long20_seed0.json")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--settle-ticks", type=int, default=10)
    ap.add_argument("--success-radius", type=float, default=4.0)
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--close-threshold", type=float, default=3.0)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--m-per-step", type=float, default=0.24)
    ap.add_argument("--cap-slack", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="first N episodes only (smoke test)")
    ap.add_argument("--capture-only", action="store_true")
    ap.add_argument("--debug-steps", type=int, default=0, help="print control/speed/waypoints for the first N steps")
    # user 2026-09-14: "use same step cap for nomad and same target speed as in our experiments"
    ap.add_argument("--map-name", default=MAP_NAME_DEFAULT)
    # user 2026-09-14 "evaluate nomad on raw trace": the planning graph must match --node-poses
    ap.add_argument("--seq-map", default=SEQ_MAP, help="sequence map JSON whose nodes/edges NoMaD plans over")
    ap.add_argument("--town", default="Town02", help="user 2026-09-15 Town07 repeat")
    ap.add_argument("--seed", type=int, default=None, help="torch/numpy seed for the diffusion sampler (repeat runs)")
    ap.add_argument("--target-speed-kmh", type=float, default=6.0,
                    help="PID target speed; SetPointAgent hardcodes 6 km/h, overridden here (v6n TARGET_SPEED = 20)")
    ap.add_argument("--cap-from", default="", help="drive_plan_c JSONL: use its step_cap per episode (same map key)")
    ap.add_argument("--cap-map", default="uniform2_trace", help="map name to read caps from in --cap-from")
    a = ap.parse_args()

    import carla
    import torch
    from PIL import Image
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from navigation_policies.baseline_policies.misc import get_action, load_model, to_numpy, transform_images
    from navigation_policies.baseline_policies.baselines_config import nomad_config
    from carla_eval import CarlaEvalEnv
    from agent_wrapper import SetPointAgent

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # user 2026-09-15 "repeat driving experiments for 3 seeds" (same 40 episodes, 3 runs): seeds NoMaD's diffusion
    # sampling so each repeat is reproducible. Default None keeps the original unseeded behaviour (run 1).
    if a.seed is not None:
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(a.seed)
        print(f"[nomad] seed {a.seed}", flush=True)
    MAP_NAME = a.map_name
    caps_from = {}
    if a.cap_from:
        for line in open(a.cap_from):
            if line.strip():
                r = json.loads(line)
                if r["map"] == a.cap_map:
                    caps_from[(r["set"], int(r["index"]))] = int(r["step_cap"])
        print(f"[nomad] step caps from {a.cap_from} ({a.cap_map}): {len(caps_from)} episodes", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    os.makedirs(os.path.join(a.image_dir, "nodes"), exist_ok=True); os.makedirs(os.path.join(a.image_dir, "goals"), exist_ok=True)
    nodes = json.load(open(a.node_poses))["nodes"]
    episodes = json.load(open(a.episodes))["episodes"]
    if a.limit:
        episodes = episodes[:a.limit]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            if line.strip():
                r = json.loads(line); done.add((r["set"], r["index"]))
    todo = [e for e in episodes if (e["set"], e["index"]) not in done]
    node_img = lambda k: os.path.join(a.image_dir, "nodes", f"{k:04d}.png")
    goal_img = lambda e: os.path.join(a.image_dir, "goals", f"{e['set']}{e['index']:02d}.png")
    missing_nodes = [k for k in range(len(nodes)) if not os.path.exists(node_img(k))]
    missing_goals = [e for e in episodes if not os.path.exists(goal_img(e))]
    print(f"[nomad] {len(todo)} episodes to drive ({len(done)} done); images to capture: {len(missing_nodes)} nodes, "
          f"{len(missing_goals)} goals; device {dev}", flush=True)
    if not todo and not missing_nodes and not missing_goals:
        print(f"[nomad] done: {a.out}", flush=True)
        return

    ensure_carla(a.port, os.environ.get("CARLA_ROOT", "/home/robotlab/Apps/CARLA_0.9.15"),
                 os.environ.get("CARLA_QUALITY", "Epic"))
    # the baseline config asks CarlaCore for a pygame display; keep it headless for THIS process only
    # (setting it before the server start made CarlaUE4 inherit it and segfault)
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    env = CarlaEvalEnv(start_server=False, town=a.town)
    amap = env.core.map
    loaded = env.core.world.get_map().name
    if a.town.lower() not in loaded.lower():
        raise RuntimeError(f"asked for {a.town} but the simulator has {loaded}")
    print(f"[Preflight] drive_nomad: simulator map is {loaded}", flush=True)
    brake = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

    def frame_of(sd):
        arr = np.asarray(sd["rgb"][1])
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return Image.fromarray(np.ascontiguousarray(arr[..., ::-1]).astype(np.uint8))   # CARLA BGR -> RGB

    def spawn(xy_start, xy_goal):
        wp_s = amap.get_waypoint(carla.Location(x=float(xy_start[0]), y=float(xy_start[1]), z=0.0), project_to_road=True)
        # carla_core.reset_hero_for_experiments reads `origin.location` when it must SPAWN the hero (a Transform) and
        # `origin.transform` when it only moves an existing hero (a Waypoint) -- hand it the type each branch expects.
        if getattr(env.core, "hero", None) is None:
            env.experiment.origin = carla.Transform(
                carla.Location(x=wp_s.transform.location.x, y=wp_s.transform.location.y, z=wp_s.transform.location.z),
                wp_s.transform.rotation)
        else:
            env.experiment.origin = wp_s
        env.set_destination_transform(carla.Location(x=float(xy_goal[0]), y=float(xy_goal[1]), z=0.0))
        env.reset()
        sd = None
        for _ in range(a.settle_ticks):
            sd = env.core.tick(brake)
        return sd

    def capture_at(x, y, yaw_rad):
        wp = amap.get_waypoint(carla.Location(x=float(x), y=float(y), z=0.0), project_to_road=True)
        hero = env.core.hero
        hero.set_transform(carla.Transform(carla.Location(x=float(x), y=float(y), z=wp.transform.location.z + 0.3),
                                           carla.Rotation(yaw=float(np.degrees(yaw_rad)))))
        hero.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        sd = None
        for _ in range(a.settle_ticks):
            sd = env.core.tick(brake)
        return frame_of(sd)

    if missing_nodes or missing_goals:
        e0 = episodes[0]
        spawn(e0["start_xy"], e0["goal_xy"])
        t0 = time.time()
        for n_done, k in enumerate(missing_nodes):
            nd = nodes[k]
            capture_at(nd["x_m"], nd["y_m"], nd["yaw_rad"]).save(node_img(k))
            if (n_done + 1) % 100 == 0:
                print(f"[capture] {n_done + 1}/{len(missing_nodes)} node images [{time.time() - t0:.0f}s]", flush=True)
        for e in missing_goals:
            capture_at(e["goal_xy"][0], e["goal_xy"][1], e["goal_yaw_rad"]).save(goal_img(e))
        print(f"[capture] {len(missing_nodes)} node + {len(missing_goals)} goal images in {time.time() - t0:.0f}s", flush=True)
    if a.capture_only:
        print(f"[nomad] done: capture only", flush=True)
        return

    model = load_model(CKPT, nomad_config, dev).to(dev).eval()
    isz = nomad_config["image_size"]; ctx_n = nomad_config["context_size"] + 1
    sched = DDPMScheduler(num_train_timesteps=nomad_config["num_diffusion_iters"], beta_schedule="squaredcos_cap_v2",
                          clip_sample=True, prediction_type="epsilon")
    node_t = torch.cat([transform_images(Image.open(node_img(k)).convert("RGB"), isz, center_crop=False)
                        for k in range(len(nodes))], 0).to(dev)
    xy, edges, legs = load_sequence_map(a.seq_map)
    assert len(xy) == len(nodes), f"--seq-map has {len(xy)} nodes but --node-poses has {len(nodes)}"
    driven_dirs = set(edges)

    @torch.no_grad()
    def distances(ctx_imgs, goals_t, chunk=128):
        obs = transform_images(list(ctx_imgs), isz, center_crop=False).to(dev)
        out = []
        for i in range(0, len(goals_t), chunk):
            g = goals_t[i:i + chunk]
            oc = model("vision_encoder", obs_img=obs.repeat(len(g), 1, 1, 1), goal_img=g,
                       input_goal_mask=torch.zeros(len(g), dtype=torch.long, device=dev))
            out.append(to_numpy(model("dist_pred_net", obsgoal_cond=oc).flatten()))
        return np.concatenate(out), obs

    @torch.no_grad()
    def sample_waypoints(cond):
        cond = cond.repeat(a.num_samples, 1)
        x = torch.randn((a.num_samples, nomad_config["len_traj_pred"], 2), device=dev)
        sched.set_timesteps(nomad_config["num_diffusion_iters"])
        for k in sched.timesteps:
            x = sched.step(model_output=model("noise_pred_net", sample=x, timestep=k, global_cond=cond),
                           timestep=k, sample=x).prev_sample
        return to_numpy(get_action(x))[1]          # the wrapper's choice: sample 1, full predicted trajectory

    for k_ep, e in enumerate(todo):
        sx, gx = np.asarray(e["start_xy"]), np.asarray(e["goal_xy"])
        sd = spawn(e["start_xy"], e["goal_xy"])
        hero = env.core.hero
        here = np.array([hero.get_location().x, hero.get_location().y])
        spawn_err = float(np.linalg.norm(here - sx))
        ctx = [frame_of(sd)] * ctx_n
        d_start, _ = distances(ctx, node_t)
        g_im = Image.open(goal_img(e)).convert("RGB")
        d_goal, _ = distances([g_im] * ctx_n, node_t)
        start_node, goal_node_map = localise_all(d_start), localise_all(d_goal)
        path = bfs_path(len(nodes), edges, start_node, goal_node_map, undirected=True)
        plan_m = path_metres(path, xy, legs) if path else 0.0
        cap = caps_from[(e["set"], int(e["index"]))] if a.cap_from else step_cap(plan_m, a.m_per_step, a.cap_slack)
        reversed_legs = sum(1 for u, v in zip(path[:-1], path[1:]) if (u, v) not in driven_dirs) if path else 0
        topo_nodes = list(path) if path else [start_node]
        topo_t = torch.cat([node_t[topo_nodes], transform_images(g_im, isz, center_crop=False).to(dev)], 0)
        goal_idx = len(topo_t) - 1
        sp = SetPointAgent(hero)
        _pid_step = sp._vehicle_controller.run_step          # SetPointAgent passes a hardcoded target_speed=6
        sp._vehicle_controller.run_step = lambda target_speed, waypoint, _f=_pid_step: _f(a.target_speed_kmh, waypoint)
        closest, explore, prev_closest = 0, False, 0
        log = dict(relocalisations=0, lane_steps=0, off_road_steps=0, explore_steps=0,
                   final_goal_leg_started=None, explore_started=None)
        driven_m, prev, end, t = 0.0, here.copy(), "cap", 0
        t_ep = time.time()
        for t in range(1, cap + 1):
            obs_t = transform_images(ctx, isz, center_crop=False).to(dev)
            with torch.no_grad():
                if not explore and plan_finished(closest, goal_idx):
                    explore = True; log["explore_started"] = t
                    if log["final_goal_leg_started"] is None:
                        log["final_goal_leg_started"] = t
                if explore:
                    oc = model("vision_encoder", obs_img=obs_t,
                               goal_img=torch.randn((1, 3, *isz), device=dev), input_goal_mask=torch.ones(1, dtype=torch.long, device=dev))
                    cond = oc
                    log["explore_steps"] += 1
                else:
                    s, en = window(closest, a.radius, goal_idx)
                    g = topo_t[s:en + 1]
                    oc = model("vision_encoder", obs_img=obs_t.repeat(len(g), 1, 1, 1), goal_img=g,
                               input_goal_mask=torch.zeros(len(g), dtype=torch.long, device=dev))
                    dd = to_numpy(model("dist_pred_net", obsgoal_cond=oc).flatten())
                    closest, sg = select_subgoal(dd, s, a.close_threshold)
                    if closest != prev_closest:
                        log["relocalisations"] += 1; prev_closest = closest
                    if closest >= goal_idx - 1 and log["final_goal_leg_started"] is None:
                        log["final_goal_leg_started"] = t
                    cond = oc[sg:sg + 1]
            wps = sample_waypoints(cond)
            ctrl = sp.run_step(wps)
            raw_ctrl = ctrl
            if not isinstance(ctrl, carla.VehicleControl):
                # SetPointAgent returns the PID output as a (steer, throttle) ACTION (or (0, 0) with no setpoint);
                # the baseline scripts hand it to env.step, which converts it with compute_action -- do the same.
                ctrl = env.experiment.compute_action((float(ctrl[0]), float(ctrl[1])))
            if t <= a.debug_steps:
                v = hero.get_velocity()
                print(f"[debug] t={t} closest={closest} explore={explore} ctrl={type(raw_ctrl).__name__} "
                      f"thr={ctrl.throttle:.2f} brk={ctrl.brake:.2f} str={ctrl.steer:.2f} hb={ctrl.hand_brake} "
                      f"speed={np.hypot(v.x, v.y):.2f} m/s physics={hero.attributes.get('role_name')} "
                      f"wps0={np.round(wps[0], 3).tolist()} wps3={np.round(wps[min(3, len(wps) - 1)], 3).tolist()}", flush=True)
                buf = sp._setpoints_buffer
                tf = hero.get_transform()
                sp_loc = carla.Location(*[float(v) for v in buf[sp._setpoint_index]]) if buf is not None else None
                wp_proj = amap.get_waypoint(sp_loc, project_to_road=True) if sp_loc is not None else None
                print(f"[debug]   hero loc=({tf.location.x:.2f},{tf.location.y:.2f},{tf.location.z:.2f}) rot=(r{tf.rotation.roll:.1f},p{tf.rotation.pitch:.1f},y{tf.rotation.yaw:.1f}) "
                      f"buffer shape={None if buf is None else np.shape(buf)} setpoint={None if buf is None else np.round(buf[sp._setpoint_index], 2).tolist()} "
                      f"wp_proj={None if wp_proj is None else (round(wp_proj.transform.location.x, 2), round(wp_proj.transform.location.y, 2))}", flush=True)
            sd = env.core.tick(ctrl)
            ctx = ctx[1:] + [frame_of(sd)]
            loc = hero.get_location(); here = np.array([loc.x, loc.y])
            driven_m += float(np.linalg.norm(here - prev)); prev = here.copy()
            log["lane_steps"] += int("lane_invasion" in sd)
            log["off_road_steps"] += int(amap.get_waypoint(loc, project_to_road=False) is None)
            if np.linalg.norm(here - gx) <= a.success_radius:
                end = "success"; break
            if "collision" in sd:
                end = "collision"; break
        final_d = float(np.linalg.norm(here - gx))
        # user 2026-09-15 Town07: a goal unreachable along lane directions has route_m = None (one short episode).
        # drive_plan_c.py keeps it None and scores SPL 0.0; mirror that exactly rather than inventing a distance.
        route = None if e["route_m"] is None else float(e["route_m"])
        rec = {"map": MAP_NAME, "set": e["set"], "index": e["index"], "goal_kind": e["goal_kind"], "route_m": route,
               "spawn_error_m": spawn_err, "start_node": int(start_node), "goal_node": int(goal_node_map),
               "start_node_err_m": float(np.linalg.norm(xy[start_node] - sx)),
               "goal_node_err_m": float(np.linalg.norm(xy[goal_node_map] - gx)),
               "plan_found": path is not None, "plan_hops": (len(path) - 1) if path else None, "plan_m": plan_m,
               "plan_nodes": path, "step_cap": cap, "steps": t, "end": end, "success": end == "success",
               "final_dist_m": final_d, "driven_m": driven_m,
               "spl": (float(route) / max(driven_m, float(route))) if (end == "success" and route) else 0.0,
               "reached_plan_end": log["final_goal_leg_started"] is not None,
               "reloc_changed": 0, "replans": 0, "leg_timeouts": 0, "skips": 0,
               "reversed_legs_driven": reversed_legs, "reloc_events": [], "video": None,
               "localiser": "nomad dist_pred_net", "controller": f"SetPointAgent PID ({a.target_speed_kmh:g} km/h)",
               "target_speed_kmh": a.target_speed_kmh, "cap_source": (f"{a.cap_from}:{a.cap_map}" if a.cap_from else "formula"), **log}
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[nomad] {k_ep + 1}/{len(todo)} {e['set']}#{e['index']} ({e['goal_kind']}): {end} in {t}/{cap} steps, "
              f"final {final_d:.1f} m, plan {rec['plan_hops']} hops {plan_m:.0f} m, start node err "
              f"{rec['start_node_err_m']:.1f} m, goal node err {rec['goal_node_err_m']:.1f} m, relocs {log['relocalisations']}, "
              f"explore {log['explore_steps']} [{time.time() - t_ep:.0f}s]", flush=True)
    print(f"[nomad] done: {a.out}", flush=True)


if __name__ == "__main__":
    main()
