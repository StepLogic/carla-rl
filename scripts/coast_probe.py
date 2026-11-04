#!/usr/bin/env python3
"""How far does the car roll before it is slow enough to arrive? (2026-11-04)

Arrival needs <= GOAL_ARRIVAL_MAX_SPEED (10 km/h) inside a 2 m radius, the
progress reward is paid only below TARGET_SPEED (20 km/h), and the action space
has NO BRAKE -- throttle is [0, 1] -- so the only way to slow down is to lift
off and coast. This measures that coast: hold the lane at full throttle until
the car reaches COAST_FROM_KMH, then throttle 0 and record the metres and steps
until it drops to COAST_TO_KMH. A simple heading controller on the lane's own
direction (info["wp_forward"]) keeps the car on the road so a straight is not
needed.

    COAST_FROM_KMH=20 COAST_TO_KMH=10 python3 scripts/coast_probe.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

TOWN = os.environ.get("ENV_TOWNS", "Town02").split(",")[0]
PORT = int(os.environ.get("CARLA_PORT", 3450))
SEEDS = [int(s) for s in os.environ.get("COAST_SEEDS", "10000,10001,10002,10003,10004,10005,10006,10007").split(",")]
FROM_KMH = float(os.environ.get("COAST_FROM_KMH", 20.0))
TO_KMH = float(os.environ.get("COAST_TO_KMH", 10.0))
MAX_STEPS = int(os.environ.get("COAST_MAX_STEPS", 600))


def steer_to_lane(info):
    """Proportional steer onto the lane direction; 0 if it is not published."""
    fwd = info.get("wp_forward")
    if fwd is None:
        return 0.0
    yaw = float(info.get("hero_yaw", 0.0) or 0.0)
    want = float(np.arctan2(float(fwd[1]), float(fwd[0])))
    err = (want - yaw + np.pi) % (2 * np.pi) - np.pi
    return float(np.clip(1.5 * err, -1.0, 1.0))


def main():
    import random as _random
    import torch
    from torch_rl.carla_env_build import build_env
    from torch_rl.carla_server import CarlaServer
    srv = CarlaServer(PORT, quality=os.environ.get("CARLA_QUALITY", "Epic"),
                      log_dir=os.path.join(ROOT, "logs", "coast_probe"))
    srv.start()
    print(f"carla ready in {srv.wait_ready():.0f}s", flush=True)
    env = build_env(TOWN, seed=SEEDS[0], encoder=None, port=PORT, state_keys=(),
                    eval_mode=True)
    box = env.action_space
    print(f"action box low={box.low.tolist()} high={box.high.tolist()} | coast "
          f"{FROM_KMH:g} -> {TO_KMH:g} km/h, throttle 0, no brake exists", flush=True)
    print(f"{'seed':>6} {'reached':>8} {'accel m':>8} {'coast m':>8} {'coast steps':>11} {'end':>10}",
          flush=True)
    rows = []
    for seed in SEEDS:
        _random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        _o, info = env.reset(seed=seed)
        prev = np.asarray(info.get("hero_pos", (0.0, 0.0)), np.float64)
        phase, accel_m, coast_m, coast_t, end = "accel", 0.0, 0.0, 0, "cap"
        top = 0.0
        for _t in range(MAX_STEPS):
            thr = 1.0 if phase == "accel" else 0.0
            a = np.clip(np.array([steer_to_lane(info), thr], np.float32), box.low, box.high)
            _o, _r, term, trunc, info = env.step(a)
            pos = np.asarray(info.get("hero_pos", prev), np.float64)
            d = float(np.linalg.norm(pos - prev))
            prev = pos
            v = float(info.get("speed_kmh", 0.0) or 0.0)
            top = max(top, v)
            if phase == "accel":
                accel_m += d
                if v >= FROM_KMH:
                    phase = "coast"
            else:
                coast_m += d
                coast_t += 1
                if v <= TO_KMH:
                    end = "slowed"
                    break
            if term or trunc:
                end = ("collision" if info.get("collision") else "off_road" if info.get("off_road")
                       else "crossing" if info.get("lane_any") else "term")
                break
        reached = phase == "coast"
        print(f"{seed:>6} {str(reached):>8} {accel_m:>8.1f} {coast_m:>8.1f} {coast_t:>11d} {end:>10}"
              f"  (top {top:.1f} km/h)", flush=True)
        if reached and end == "slowed":
            rows.append((coast_m, coast_t))
    if rows:
        m = np.array(rows)
        print(f"COAST {FROM_KMH:g}->{TO_KMH:g} km/h over {len(rows)} clean runs: "
              f"median {np.median(m[:, 0]):.1f} m, {np.median(m[:, 1]):.0f} steps "
              f"({np.median(m[:, 1]) * 0.05:.1f} s); range {m[:, 0].min():.1f}-{m[:, 0].max():.1f} m",
              flush=True)
    env.close()
    srv.stop()


if __name__ == "__main__":
    main()
