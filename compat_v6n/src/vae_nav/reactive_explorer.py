"""A reactive, policy-free explorer for trace collection -- ported from
explore_trace.py (main checkout) so the mapping harness can build the map
WITHOUT the agent, which is how the peer Habitat pipeline is structured.

WHY IT EXISTS HERE. The harness's documented trace source is the gated policy
in explore mode. Measured on carla_gated_her_v4 (2026-09-11): every explore
episode ends on a lane crossing within ~2 m of spawn (31-41 steps, 1.8-3.2 m
driven), because that arm trained with crossings fatal in both modes and
never learned to hold a lane. ~1.5 candidates per episode, all beside spawn
points, is not a trace. This explorer reads ONLY a depth ring from a camera
on the hero plus its own visited buffer: no map, no goal, no policy, no road
graph (CARLA's autopilot is not used precisely because it drives on the road
graph, the privileged information the experiment excludes).

The controller is the main checkout's ReactiveExplorer verbatim: steer toward
the FURTHEST return on the ring (down a street that IS the street), a fixed-
sign turn only at a genuine dead end (their measured 3.6x coverage argument),
throttle on forward clearance, a nudge away from ground already covered.
"""
import math
import os
import queue

import numpy as np

N_BEAMS = int(os.environ.get("RING_BEAMS", 16))
BLOCKED_M = float(os.environ.get("EXPLORE_BLOCKED_M", 3.5))
STEER_GAIN = float(os.environ.get("EXPLORE_STEER_GAIN", 0.7))
TURN_STEER = float(os.environ.get("EXPLORE_TURN_STEER", 0.6))
CRUISE_THROTTLE = float(os.environ.get("EXPLORE_THROTTLE", 0.75))
RING_HFOV = float(os.environ.get("RING_HFOV", 120.0))
DEPTH_W, DEPTH_H = 256, 64


def depth_to_ring(depth_m, n_beams=N_BEAMS):
    """(H, W) metric depth -> (n_beams,) ranges; MIN per angular column of the
    middle rows. 0 means a DEAD beam, not a clear one."""
    d = np.asarray(depth_m, np.float32)
    h, w = d.shape[:2]
    band = d[int(h * 0.35):int(h * 0.75), :]
    cols = np.array_split(np.arange(w), n_beams)
    ring = np.empty(n_beams, np.float32)
    for i, c in enumerate(cols):
        seg = band[:, c[0]:c[-1] + 1]
        seg = seg[np.isfinite(seg) & (seg > 0.05)]
        ring[i] = float(seg.min()) if seg.size else 0.0
    return ring


class ReactiveExplorer:
    def __init__(self, seed=0, turn_sign=+1.0):
        self.rng = np.random.default_rng(seed)
        self.turn_sign = float(turn_sign)
        self.visited = []

    def act(self, ring, pos):
        n = len(ring)
        r = np.where(ring > 1e-3, ring, np.nan)
        half = math.radians(RING_HFOV) / 2.0
        angles = np.linspace(-half, half, n)
        if np.all(np.isnan(r)):
            return np.array([self.turn_sign * TURN_STEER, 0.30], np.float32)
        c0, c1 = int(n * 0.4), max(int(n * 0.6), int(n * 0.4) + 1)
        fwd_band = r[c0:c1]
        fwd = np.nanmin(fwd_band) if np.any(~np.isnan(fwd_band)) else np.nan
        if not np.isnan(fwd) and fwd < BLOCKED_M:
            return np.array([self.turn_sign * TURN_STEER, 0.30], np.float32)
        rr = np.where(np.isnan(r), -np.inf, r)
        target = float(angles[int(np.argmax(rr))])
        bias = 0.0
        if self.visited:
            v = np.asarray(self.visited)
            if float(np.linalg.norm(v - np.asarray(pos)[:2], axis=1).min()) < 1.2:
                bias = self.turn_sign * 0.15
        steer = np.clip(STEER_GAIN * (target / half) + bias, -1.0, 1.0)
        open_m = fwd if not np.isnan(fwd) else 50.0
        throttle = float(np.clip(0.35 + 0.05 * open_m, 0.35, CRUISE_THROTTLE))
        return np.array([steer, throttle], np.float32)

    def note(self, pos):
        self.visited.append(np.asarray(pos, np.float64)[:2])
        if len(self.visited) > 4000:
            self.visited.pop(0)


class DepthRing:
    """A depth camera on the hero, read as a ring. destroy() when done."""

    def __init__(self, env):
        import carla
        core = env
        while not hasattr(core, "core"):
            core = getattr(core, "env", None)
            if core is None:
                raise RuntimeError("could not reach CarlaEnv.core through the wrappers")
        world, hero = core.core.world, core.core.hero
        bp = world.get_blueprint_library().find("sensor.camera.depth")
        bp.set_attribute("image_size_x", str(DEPTH_W))
        bp.set_attribute("image_size_y", str(DEPTH_H))
        bp.set_attribute("fov", str(RING_HFOV))
        self.cam = world.spawn_actor(
            bp, carla.Transform(carla.Location(x=1.5, z=1.6)), attach_to=hero)
        self.q = queue.Queue()
        self.cam.listen(self.q.put)

    def latest(self):
        img = None
        while True:
            try:
                img = self.q.get_nowait()
            except queue.Empty:
                break
        if img is None:
            return np.zeros(N_BEAMS, np.float32)
        a = np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)
        b, g, r = (a[:, :, 0].astype(np.float32), a[:, :, 1].astype(np.float32),
                   a[:, :, 2].astype(np.float32))
        norm = (r + g * 256.0 + b * 65536.0) / (256.0 ** 3 - 1.0)
        return depth_to_ring(norm * 1000.0)

    def destroy(self):
        try:
            self.cam.stop(); self.cam.destroy()
        except Exception:  # noqa: BLE001
            pass
