"""Drive a trained policy in a CARLA window you can watch.

    DISPLAY=:1 CARLA_PORT=2000 CKPT=checkpoints/<run>/checkpoint_50000.pkl \
        python watch_policy.py --episodes 3

WHAT YOU SEE. The CARLA server window follows the hero in third person
(carla_core moves the spectator every tick when rendering is enabled), and a
second OpenCV window shows the policy's OWN inputs: the camera frame it is fed,
the VAE's lane/obstacle decode, and -- the point of this script -- a live
readout of the goal block.

WHY THE GOAL READOUT IS THE POINT. The policy's failure mode was invisible in
a plain camera recording: it drove smoothly, kept its lane, covered a hundred
metres and simply never arrived. Nothing on screen said why, because the fault
was in a number the video did not show. The lateral term of the goal bearing
was sign-inverted (CARLA is left-handed; the transform used a right-handed
rotation), so every goal off the centreline was reported on the wrong side and
the policy steered away from it, correctly, on corrupted input.

So this window prints forward/left in metres and draws the bearing arrow. If
the arrow ever points away from where the car should go, that is visible in one
frame instead of six failed evaluations.

RUNS WITHOUT WEIGHTS TOO. With no CKPT, or a checkpoint that predates weight
saving, the agent is freshly initialised and the run is only an observation
harness -- it says so loudly rather than implying you are watching a trained
policy.
"""
import argparse
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

PANEL = 256


def hud(rgb, masks, goal, info, step, ep, note):
    """One composite frame: camera | VAE decode | goal geometry."""
    import cv2

    def up(img):
        a = np.asarray(img)
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        if a.dtype != np.uint8:
            a = np.clip(a * 255 if a.max() <= 1.5 else a, 0, 255).astype(np.uint8)
        return cv2.resize(a, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)

    left = up(rgb) if rgb is not None else np.zeros((PANEL, PANEL, 3), np.uint8)

    mid = np.zeros((PANEL, PANEL, 3), np.uint8)
    if masks is not None:
        m = np.asarray(masks, np.float32)
        col = np.zeros((*m.shape[:2], 3), np.float32)
        col[..., 1] = m[..., 0]                       # lane   -> green
        if m.shape[-1] > 1:
            col[..., 0] = m[..., 1]                   # obstacle -> red
        mid = up(col)

    # Bird's-eye goal panel: the car at centre facing UP, the goal plotted from
    # the SAME two numbers the network receives. Nothing is recomputed from the
    # simulator here -- if the block is wrong, this panel is wrong with it, and
    # that is exactly what makes it a check.
    right = np.full((PANEL, PANEL, 3), 22, np.uint8)
    cx = cy = PANEL // 2
    cv2.line(right, (cx, 0), (cx, PANEL), (45, 45, 45), 1)
    cv2.line(right, (0, cy), (PANEL, cy), (45, 45, 45), 1)
    fwd, lft = float(goal[0]), float(goal[1])
    scale = PANEL / 2.0 / max(12.0, abs(fwd), abs(lft))
    # Screen: +forward is UP, +left is LEFT.
    gx, gy = int(cx - lft * scale), int(cy - fwd * scale)
    cv2.arrowedLine(right, (cx, cy), (np.clip(gx, 4, PANEL - 4),
                                      np.clip(gy, 4, PANEL - 4)),
                    (80, 220, 255), 2, tipLength=0.18)
    cv2.circle(right, (cx, cy), 5, (255, 255, 255), -1)

    side = "LEFT" if lft > 0.5 else ("RIGHT" if lft < -0.5 else "AHEAD")
    dist = float(np.hypot(fwd, lft))
    for i, t in enumerate([f"fwd  {fwd:+7.1f} m",
                           f"left {lft:+7.1f} m",
                           f"dist  {dist:6.1f} m   -> {side}"]):
        cv2.putText(right, t, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (235, 235, 235), 1, cv2.LINE_AA)

    frame = np.concatenate([left, mid, right], axis=1)
    for x, t in ((6, "camera"), (PANEL + 6, "VAE lane/obst"),
                 (2 * PANEL + 6, "goal (as the net sees it)")):
        cv2.putText(frame, t, (x, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
    bar = np.full((26, frame.shape[1], 3), 15, np.uint8)
    flags = " HIT" if info.get("collision") else ""
    flags += " LANE" if info.get("lane_invasion") else ""
    cv2.putText(bar, f"ep {ep}  step {step}{flags}   {note}", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 255), 1, cv2.LINE_AA)
    return np.concatenate([frame, bar], axis=0)


def draw_goal(world, carla, goal_xy, hero_loc, radius=2.0, life=0.12):
    """Draw the goal in the CARLA world, visible in the spectator view.

    Drawn from info["goal_pos"] -- the ACTUAL goal the policy chases and the
    reward is scored against -- not carla_core.draw_goal, which snaps the point
    to the nearest road waypoint via get_waypoint() and can therefore show a
    position several metres from the one the agent was asked for.

    life_time is slightly longer than one 0.05 s tick so the marker persists
    between redraws without accumulating; CARLA expires each primitive itself.
    """
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    # Height from the hero rather than a constant: Town02 is not flat and a
    # fixed z buries the marker on slopes.
    z = hero_loc.z
    g = carla.Location(x=gx, y=gy, z=z)
    world.debug.draw_line(g, carla.Location(x=gx, y=gy, z=z + 6.0),
                          thickness=0.15, color=carla.Color(0, 255, 60),
                          life_time=life)
    # The ARRIVAL RADIUS, not a bare point: success is "within 2 m", so the
    # circle is what the agent actually has to enter.
    n = 16
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        world.debug.draw_line(
            carla.Location(x=gx + radius * math.cos(a0),
                           y=gy + radius * math.sin(a0), z=z + 0.2),
            carla.Location(x=gx + radius * math.cos(a1),
                           y=gy + radius * math.sin(a1), z=z + 0.2),
            thickness=0.08, color=carla.Color(0, 255, 60), life_time=life)
    world.debug.draw_line(carla.Location(x=hero_loc.x, y=hero_loc.y, z=z + 1.0),
                          carla.Location(x=gx, y=gy, z=z + 1.0),
                          thickness=0.05, color=carla.Color(255, 200, 0),
                          life_time=life)
    d = math.hypot(gx - hero_loc.x, gy - hero_loc.y)
    world.debug.draw_string(carla.Location(x=gx, y=gy, z=z + 6.4),
                            f"GOAL  {d:.1f} m", draw_shadow=False,
                            color=carla.Color(0, 255, 60), life_time=life)


def decode(src, rgb):
    if src is None or getattr(src, "_model", None) is None:
        return None
    try:
        import jax
        import jax.nn as jnn
        x = src._prepare(rgb)
        logits, _, _ = src._model.apply({"params": src._params}, x,
                                        deterministic=True)
        return np.asarray(jax.device_get(jnn.sigmoid(logits))[0], np.float32)
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("CKPT", ""))
    ap.add_argument("--town", default=os.environ.get("WATCH_TOWN", "Town02"))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("CARLA_PORT", 2000)))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--record", default="", help="also write an mp4 here")
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()

    import cv2
    import carla  # noqa: F401
    os.environ.setdefault("CARLA_PORT", str(args.port))
    import train_carla_tqc as T
    from vae_nav.latent_source import LatentSource
    from vae_nav.policy_io import load_agent_state, param_fingerprint

    src = LatentSource(os.environ.get("VAE_CKPT", ""))
    env = T.make_env(args.town, args.seed, src, eval_mode=True, port=args.port)
    agent = T.build_agent(env.observation_space, env.action_space, args.seed)

    # Live CARLA handles for debug drawing, reached through the wrapper stack.
    _c = env
    while _c is not None and not hasattr(_c, "core"):
        _c = getattr(_c, "env", None)
    world = _c.core.world if _c is not None else None
    if world is None:
        print("[Watch] could not reach the CARLA world; goal will not be drawn")

    trained = False
    if args.ckpt:
        with open(args.ckpt, "rb") as f:
            ck = pickle.load(f)
        if "agent" in ck:
            load_agent_state(agent, ck["agent"])
            trained = True
            print(f"[Watch] loaded step {ck.get('step')} "
                  f"{param_fingerprint(ck['agent'])}")
        else:
            print(f"[Watch] {args.ckpt} has NO weights (keys: "
                  f"{sorted(ck)}). It was written before the trainer saved "
                  "them, so there is nothing to replay.")
    note = "TRAINED POLICY" if trained else "UNTRAINED (random init)"
    print(f"[Watch] {note} | town={args.town} port={args.port}")
    if not trained:
        print("[Watch] Not a demo of learned behaviour. The goal panel is "
              "still meaningful: it shows the exact bearing the net receives.")

    writer = None
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            done, step, success = False, 0, False
            while not done and step < T.MAX_EPISODE_STEPS:
                action = agent.eval_actions(obs)
                obs, _r, term, trunc, info = env.step(action)
                done, step = bool(term or trunc), step + 1
                success = success or bool(info.get("is_success", False))
                if world is not None and "goal_pos" in info:
                    try:
                        draw_goal(world, carla, info["goal_pos"],
                                  _c.core.hero.get_transform().location)
                    except Exception:      # noqa: BLE001
                        pass               # drawing must never kill a run

                rgb = info.get("rgb")
                frame = hud(rgb, decode(src, rgb) if rgb is not None else None,
                            obs["goal"], info, step, ep, note)
                if args.record:
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(
                            args.record, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
                    writer.write(frame[:, :, ::-1])
                if not args.no_window:
                    cv2.imshow("policy inputs", frame[:, :, ::-1])
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        done = True
            g = obs["goal"]
            print(f"[Watch] ep {ep}: {step} steps  success={success}  "
                  f"final goal fwd={g[0]:+.1f} left={g[1]:+.1f} "
                  f"dist={np.hypot(g[0], g[1]):.1f} m  "
                  f"reasons={info.get('done_reasons')}")
    finally:
        env.close()
        if writer is not None:
            writer.release()
            print(f"[Watch] wrote {args.record}")
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
