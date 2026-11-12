"""Drive a town to produce an exploration TRACE, for self-built mapping.

This replaces the previous mapping capture, which TELEPORTED the vehicle to
every waypoint of CARLA's road graph. That was wrong twice over: it consumed
privileged map data the experiment is supposed to be testing whether an agent
can do without, and it produced no continuous trace at all -- the "sequential"
edges linked node i to node i+1 in capture order, so at road-segment boundaries
they jumped up to 234 m and every planned route was a teleport.

CARLA ANALOGUES of the peer Habitat setup, each chosen and stated:

  EXPLORATION POLICY -- TWO MODES, and the default is NOT the trained agent.

  --explorer reactive (DEFAULT) is the analogue of theirs: a reactive
  wall-hugger reading only the depth ring, with a visited-position buffer. It
  does not use the trained policy, the road graph, or any goal. THE MAP IS
  THEREFORE BUILT WITHOUT THE AGENT, which is how their pipeline is structured
  and is the point: if the explorer IS the policy, map quality is coupled to
  policy quality and the mapped-vs-unmapped comparison is confounded by a
  shared cause.

  MEASURED consequence of getting this wrong: the first CARLA traces were
  driven by the trained policy and fragmented into 497 segments over 15,000
  steps -- one break every ~30 steps -- because the policy terminates on lane
  crossings. The map inherited the policy's failure mode wholesale.

  --explorer policy is kept because those traces are still the right data for
  measuring properties OF the policy (the epistemic-inversion test uses them),
  and because "map built by the agent" is a legitimate contrast arm. It is
  never the default.

  CARLA's autopilot is not used in either mode: it drives on the road graph,
  which is exactly the privileged information being excluded.

  SEGMENT BREAKS. Their trace is severed by wedge recovery -- stuck detection
  followed by a teleport back to a banked safe pose -- and they were explicit
  that the fragmentation everyone measures is manufactured by that mechanism.
  The CARLA analogue is an EPISODE ENDING: a lane crossing or collision ends
  the run and the hero respawns elsewhere, which is the same discontinuity for
  the same reason. Each episode is one segment.

  THE RANGE RING. Theirs is 16 beams from the depth frame the agent held. Here
  a depth camera with a matching horizontal field of view is sampled into the
  same number of beams. A depth camera rather than the semantic camera because
  certification needs RANGE, and rather than lidar because the ring must come
  from what the forward-facing agent actually saw.

  BUDGET. Theirs is a fixed 15,000 steps with no coverage criterion; that is
  copied rather than improved on, so node counts stay comparable.
"""
import argparse
import math
import os
import pickle
import queue
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

N_BEAMS = int(os.environ.get("RING_BEAMS", 16))

# Reactive explorer constants, mirroring theirs where CARLA allows it.
# Below this the way ahead is genuinely blocked and the fixed-sign turn fires.
# 8.0 made "blocked" the normal state on a street and the car spun in place.
BLOCKED_M = float(os.environ.get("EXPLORE_BLOCKED_M", 3.5))
STEER_GAIN = float(os.environ.get("EXPLORE_STEER_GAIN", 0.7))
TURN_STEER = float(os.environ.get("EXPLORE_TURN_STEER", 0.6))
# 0.45 gave 1.27 m per 1.5 s segment -- about 3 km/h from a standing start,
# against the trained policy's 21.6 km/h over the same window. The explorer
# needs to actually cover ground for the trace to contain anything.
CRUISE_THROTTLE = float(os.environ.get("EXPLORE_THROTTLE", 0.75))
RING_HFOV = float(os.environ.get("RING_HFOV", 120.0))
DEPTH_W, DEPTH_H = 256, 64          # wide and short: the ring is horizontal


def depth_to_ring(depth_m, n_beams=N_BEAMS):
    """(H, W) metric depth -> (n_beams,) ranges across the horizontal field.

    The MINIMUM over each angular column, not the mean: certification asks
    whether anything blocks the line, and averaging a wall with the sky behind
    it reports open space that is not there.

    The middle band of rows is used rather than the whole frame, so road surface
    at the bottom and sky at the top do not dominate. Sky comes back as the
    camera's far plane and is left as a large range, which is correct -- it
    blocks nothing.
    """
    d = np.asarray(depth_m, np.float32)
    h, w = d.shape[:2]
    band = d[int(h * 0.35):int(h * 0.75), :]
    cols = np.array_split(np.arange(w), n_beams)
    ring = np.empty(n_beams, np.float32)
    for i, c in enumerate(cols):
        seg = band[:, c[0]:c[-1] + 1]
        seg = seg[np.isfinite(seg) & (seg > 0.05)]
        ring[i] = float(seg.min()) if seg.size else 0.0   # 0 = DEAD, not clear
    return ring


class ReactiveExplorer:
    """Wall-hugging on the depth ring alone. No map, no goal, no policy.

    Their controller proposes targets ahead, rejects any within 1.2 m of a
    visited position, and fans candidates in alternating +/-30 degree steps. The
    CARLA analogue works in steering rather than target positions because the
    action space is (steer, throttle) and a car cannot rotate in place.

    THE FIXED TURN DIRECTION IS LOAD-BEARING. They measured that rotating
    ALWAYS THE SAME WAY on losing clearance follows a wall around a corner,
    where reversing bounces back down the corridor it came from -- worth 3.6x
    coverage. So `self.turn_sign` is set once and never flipped.
    """

    def __init__(self, seed=0, turn_sign=+1.0):
        self.rng = np.random.default_rng(seed)
        self.turn_sign = float(turn_sign)
        self.visited = []                 # bounded buffer, world frame

    def act(self, ring, pos):
        """Steer toward the FURTHEST visible direction, throttle on clearance.

        REWRITTEN after the first version drove in circles. That one treated the
        ring as a binary "clear ahead / blocked" test at 8 m and did a fixed-sign
        turn whenever it was blocked. On a CARLA street something is nearly
        always within 8 m -- a kerb, a parked car, a building at a corner -- so
        it sat permanently in the turn branch. MEASURED: 2000 samples covering
        440 m of path inside an 8.4 m BOUNDING BOX. Long trace, no exploration.

        Steering toward the argmax of the ring instead follows roads for free:
        down a street the furthest return IS the street, so the controller
        tracks it without any notion of a road. The fixed-sign turn is kept but
        demoted to a genuine dead end, where their wall-following argument
        applies and nothing else works.
        """
        n = len(ring)
        r = np.where(ring > 1e-3, ring, np.nan)      # dead beams are UNKNOWN
        half = math.radians(RING_HFOV) / 2.0
        angles = np.linspace(-half, half, n)

        if np.all(np.isnan(r)):
            return np.array([self.turn_sign * TURN_STEER, 0.30], np.float32)

        # Forward clearance from the central fifth only: the beams that decide
        # whether we can keep going straight.
        c0, c1 = int(n * 0.4), max(int(n * 0.6), int(n * 0.4) + 1)
        fwd_band = r[c0:c1]
        fwd = np.nanmin(fwd_band) if np.any(~np.isnan(fwd_band)) else np.nan

        if not np.isnan(fwd) and fwd < BLOCKED_M:
            # Genuine dead end. ALWAYS the same direction -- they measured that
            # this follows a wall around a corner where reversing bounces back
            # down the corridor, worth 3.6x coverage.
            return np.array([self.turn_sign * TURN_STEER, 0.30], np.float32)

        # Head for the furthest thing we can see, damped so the car does not
        # saw at the wheel, and nudged away from ground already covered.
        rr = np.where(np.isnan(r), -np.inf, r)
        target = float(angles[int(np.argmax(rr))])
        bias = 0.0
        if self.visited:
            v = np.asarray(self.visited)
            if float(np.linalg.norm(v - np.asarray(pos)[:2], axis=1).min()) < 1.2:
                bias = self.turn_sign * 0.15
        steer = np.clip(STEER_GAIN * (target / half) + bias, -1.0, 1.0)

        # Throttle on clearance: slow into tight places, push on open road.
        open_m = fwd if not np.isnan(fwd) else 50.0
        throttle = float(np.clip(0.35 + 0.05 * open_m, 0.35, CRUISE_THROTTLE))
        return np.array([steer, throttle], np.float32)

    def note(self, pos):
        self.visited.append(np.asarray(pos, np.float64)[:2])
        if len(self.visited) > 4000:
            self.visited.pop(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--steps", type=int, default=int(os.environ.get("TRACE_STEPS", 15000)))
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", default=os.environ.get(
        "CKPT", "checkpoints/carla_tqc_town02_200m/checkpoint_1000000.pkl"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("CARLA_PORT", 2000)))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--explorer", choices=["reactive", "policy"], default="reactive",
                    help="reactive = no agent involved (default, matches Habitat); "
                         "policy = drive the trace with the trained policy")
    args = ap.parse_args()
    out = args.out or f"traces/{args.town.lower()}_trace.pkl"

    import carla  # noqa: F401
    import train_carla_tqc as T
    from vae_nav.latent_source import LatentSource
    from vae_nav.policy_io import load_agent_state

    # Short goals keep the policy DRIVING rather than terminating early; the
    # trace wants coverage, not long-horizon success.
    os.environ["GOAL_DIST_MIN"] = os.environ.get("GOAL_DIST_MIN", "5.0")
    os.environ["GOAL_DIST_MAX"] = os.environ.get("GOAL_DIST_MAX", "20.0")

    latent_source = LatentSource(os.environ.get("VAE_CKPT", ""))
    env = T.make_env(args.town, args.seed, latent_source, eval_mode=True,
                     port=args.port)

    agent, ck, explorer = None, {}, None
    if args.explorer == "policy":
        agent = T.build_agent(env.observation_space, env.action_space, args.seed)
        with open(args.ckpt, "rb") as f:
            ck = pickle.load(f)
        if "agent" not in ck:
            raise SystemExit(f"{args.ckpt} carries no weights")
        load_agent_state(agent, ck["agent"])
        print(f"[Trace] EXPLORER=policy (step {ck.get('step')}) -- the map will "
              "be coupled to the agent; this is the contrast arm, not the "
              "default", flush=True)
    else:
        explorer = ReactiveExplorer(seed=args.seed)
        print("[Trace] EXPLORER=reactive -- depth ring only, no policy, no road "
              "graph, no goal. The map is built WITHOUT the agent.", flush=True)
    print(f"[Trace] exploring {args.town} for {args.steps} steps", flush=True)

    # Reach through the wrappers for the live CARLA handles.
    core = env
    while not hasattr(core, "core"):
        core = getattr(core, "env", None)
        if core is None:
            raise SystemExit("could not reach CarlaEnv.core through the wrappers")
    world = core.core.world
    hero = core.core.hero

    bp = world.get_blueprint_library().find("sensor.camera.depth")
    bp.set_attribute("image_size_x", str(DEPTH_W))
    bp.set_attribute("image_size_y", str(DEPTH_H))
    bp.set_attribute("fov", str(RING_HFOV))
    cam = world.spawn_actor(
        bp, carla.Transform(carla.Location(x=1.5, z=1.6)), attach_to=hero)
    dq = queue.Queue()
    cam.listen(dq.put)

    def latest_ring():
        img = None
        while True:
            try:
                img = dq.get_nowait()
            except queue.Empty:
                break
        if img is None:
            return np.zeros(N_BEAMS, np.float32)
        a = np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)
        # CARLA packs depth as 24-bit across BGR; decode to metres.
        b, g, r = (a[:, :, 0].astype(np.float32), a[:, :, 1].astype(np.float32),
                   a[:, :, 2].astype(np.float32))
        norm = (r + g * 256.0 + b * 65536.0) / (256.0 ** 3 - 1.0)
        return depth_to_ring(norm * 1000.0)

    pos, yaw, rings, seg, rgbs = [], [], [], [], []
    segment, steps = 0, 0
    try:
        obs, info = env.reset(seed=args.seed)
        ring_now = latest_ring()
        while steps < args.steps:
            if explorer is not None:
                # The reactive controller sees ONLY the ring and its own visited
                # buffer -- no observation, no goal, no latent.
                action = explorer.act(ring_now, np.asarray(info["hero_pos"]))
                explorer.note(np.asarray(info["hero_pos"]))
            else:
                action = agent.eval_actions(obs)
            obs, _r, term, trunc, info = env.step(action)
            steps += 1
            ring_now = latest_ring()
            pos.append(np.asarray(info["hero_pos"], np.float64)[:2])
            yaw.append(float(info["hero_yaw"]))
            rings.append(ring_now)
            seg.append(segment)
            rgbs.append(np.asarray(info["rgb"], np.uint8))
            if term or trunc:
                # A break in the trace, exactly like their wedge recovery.
                segment += 1
                obs, info = env.reset(seed=args.seed + segment)
            if steps % 1000 == 0:
                print(f"  [{args.town}] {steps}/{args.steps} steps, "
                      f"{segment + 1} segments", flush=True)
    finally:
        try:
            cam.stop(); cam.destroy()
        except Exception:  # noqa: BLE001
            pass
        env.close()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"town": args.town,
                     "positions": np.asarray(pos, np.float64),
                     "yaws": np.asarray(yaw, np.float64),
                     "rings": np.asarray(rings, np.float32),
                     "segment_ids": np.asarray(seg, np.int32),
                     "rgb": np.asarray(rgbs, np.uint8),
                     "ring_beams": N_BEAMS, "ring_hfov": RING_HFOV,
                     "explorer": args.explorer,
                     "policy_step": ck.get("step")}, f)
    print(f"[Trace] {steps} samples, {segment + 1} segments -> {out}")


if __name__ == "__main__":
    main()
