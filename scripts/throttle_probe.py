#!/usr/bin/env python3
"""Does the commanded throttle actually move the car? (2026-11-03)

v36's deterministic Town02 eval at 600k ended 30/40 episodes on the clock with
0.0 m driven, while the burnt-in action read steer ~+0.85, throttle ~0.08. The
control path passes the action straight through (no brake, no hand brake), so
either that throttle is below the vehicle's standstill threshold or something
else holds the car.

This sweeps a FIXED (steer, throttle) from a normal episode start and reports
the metres driven and top speed, which turns the question into a measurement.

    THROTTLE_PROBE_STEPS=200 python3 scripts/throttle_probe.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

STEPS = int(os.environ.get("THROTTLE_PROBE_STEPS", 200))
TOWN = os.environ.get("ENV_TOWNS", "Town02").split(",")[0]
PORT = int(os.environ.get("CARLA_PORT", 3400))
SEED = int(os.environ.get("EVAL_SEED", 10000))
THROTTLES = [float(x) for x in os.environ.get(
    "THROTTLE_PROBE_VALUES", "0.05,0.08,0.10,0.15,0.20,0.30,0.50,1.00").split(",")]
STEERS = [float(x) for x in os.environ.get("THROTTLE_PROBE_STEERS", "0.0,0.85").split(",")]


def main():
    from torch_rl.carla_env_build import build_env
    from torch_rl.carla_server import CarlaServer
    srv = CarlaServer(PORT, quality=os.environ.get("CARLA_QUALITY", "Epic"),
                      log_dir=os.path.join(ROOT, "logs", "throttle_probe"))
    srv.start()
    print(f"carla ready in {srv.wait_ready():.0f}s", flush=True)
    # encoder=None: the probe never asks the policy anything, so DEBiT is not
    # loaded and the whole thing runs without the checkpoint.
    env = build_env(TOWN, seed=SEED, encoder=None, port=PORT, state_keys=(),
                    eval_mode=True)
    box = env.action_space
    print(f"action box low={box.low.tolist()} high={box.high.tolist()}", flush=True)
    print(f"{'steer':>6} {'throttle':>9} {'metres':>8} {'top km/h':>9} "
          f"{'end':>10}  (after {STEPS} steps)", flush=True)
    for steer in STEERS:
        for thr in THROTTLES:
            import random as _random
            import torch
            _random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            _obs, info = env.reset(seed=SEED)
            p0 = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
            prev, driven, top, end = p0.copy(), 0.0, 0.0, "cap"
            a = np.clip(np.array([steer, thr], np.float32), box.low, box.high)
            for t in range(STEPS):
                _obs, _r, term, trunc, info = env.step(a)
                pos = np.asarray(info.get("hero_pos", prev), np.float64)
                driven += float(np.linalg.norm(pos - prev))
                prev = pos
                top = max(top, float(info.get("speed_kmh", 0.0) or 0.0))
                if term or trunc:
                    end = ("collision" if info.get("collision")
                           else "off_road" if info.get("off_road")
                           else "crossing" if info.get("lane_any") else "term")
                    break
            print(f"{steer:>6.2f} {thr:>9.2f} {driven:>8.2f} {top:>9.1f} {end:>10}",
                  flush=True)
    env.close()
    srv.stop()


if __name__ == "__main__":
    main()
