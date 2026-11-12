"""Can the car reach < 0.2 m/s without a brake? The image-goal stop-gate check.

The image-goal success condition (user spec 2026-09-18) requires a sustained
low-speed stop: speed < IG_STOP_SPEED_MPS (0.2 m/s) for IG_STOP_WINDOW_STEPS
(10) consecutive steps. The action space has no STOP: under
INCREMENTAL_ACTIONS the throttle is a clamped delta with brake hardwired to 0,
so the ONLY way down is coasting. This script measures whether coasting
actually reaches the gate in the live env, or floors out above it -- in which
case the task needs absolute-mode brake and the constant needs revisiting
BEFORE any training run, per the plan's sign-off gate.

WHAT IT DOES. One episode in one town: accelerate under full throttle, bleed
the throttle with negative deltas until the applied control reads zero, then
coast and log speed every step until either the gate is held for the full
window (REACHABLE) or a step cap expires (reports the floor it coasted to).

RUNNING IT. CARLA is started by Python (CARLA_START_SERVER=1 default, same
as the trainer's ensure_carla). Preflight rules apply as always: no other
CARLA of ours left running, one env, prove-alive check before trusting the
output. Kill the server with the process, as CarlaCore would.

    IMAGE_GOAL_TASK=1 python scripts/image_goal_coastdown_check.py \
        --town Town02 --seed 0
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

from vae_nav.image_goal_reward import (ImageGoalReward,  # noqa: E402
                                       STOP_SPEED_MPS, STOP_WINDOW_STEPS)

TOWN02_SPEED_KMH_TARGET = 18.0        # drive to ~5 m/s before bleeding
BLEED_STEPS = 60                      # cap on the negative-delta bleed phase
COAST_CAP_STEPS = 600                 # ~60 s of coasting before calling it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--drive-speed-kmh", type=float,
                    default=TOWN02_SPEED_KMH_TARGET)
    args = ap.parse_args()

    # The env's two discarded MobileNets burn 17 cores for image features the
    # coast-down never reads: skip them (measured, 2026-09). And the coast-down
    # never reads latents either, so PERCEPTION=mobilenet keeps a VAE
    # checkpoint out of the build: latent_source must come from from_env(),
    # because VAEGoalWrapper's own default is the VAE path and would demand
    # VAE_CKPT just to drive in a straight line.
    os.environ.setdefault("ENV_SKIP_IMAGE_FEATURES", "1")
    os.environ.setdefault("PERCEPTION", "mobilenet")

    import train_carla_tqc as T   # reuse make_env: same attach/quality rules
    from vae_nav.latent_source import from_env as latent_from_env

    env = T.make_env(args.town, args.seed,
                     latent_source=latent_from_env(),
                     eval_mode=True, port=args.port)
    env = ImageGoalReward(env)

    try:
        obs, info = env.reset()
        act = np.array([0.0, 1.0], np.float32)     # full throttle, steer 0

        # PHASE 1: accelerate.
        for _ in range(400):
            _, _, term, trunc, info = env.step(act)
            if term or trunc:
                break
            if float(info.get("speed_kmh", 0.0)) >= args.drive_speed_kmh:
                break
        start_speed = float(info.get("speed_kmh", 0.0)) / 3.6
        print(f"[CoastDown] reached {start_speed:.2f} m/s, bleeding throttle")

        # PHASE 2: bleed the throttle to zero (INCREMENTAL_ACTIONS is a
        # clamped delta; the applied control is what info carries).
        for _ in range(BLEED_STEPS):
            _, _, term, trunc, info = env.step(np.array([0.0, -1.0], np.float32))
            if term or trunc:
                break
            if float(info.get("applied_throttle", 0.0)) <= 0.0:
                break
        bleed_speed = float(info.get("speed_kmh", 0.0)) / 3.6
        print(f"[CoastDown] throttle zeroed at {bleed_speed:.2f} m/s, coasting")

        # PHASE 3: coast. Log speed; the gate is STOP_SPEED_MPS held for the
        # full window.
        trace, held = [], 0
        verdict, floor = "CAP", None
        for i in range(COAST_CAP_STEPS):
            _, _, term, trunc, info = env.step(np.array([0.0, 0.0], np.float32))
            speed = float(info.get("speed_kmh", 0.0)) / 3.6
            trace.append(round(speed, 3))
            if speed < STOP_SPEED_MPS:
                held += 1
                if held >= STOP_WINDOW_STEPS:
                    verdict = "REACHABLE"
                    break
            else:
                held = 0
                floor = speed
        if verdict == "CAP":
            floor = trace[-1] if trace else float("nan")

        print(f"[CoastDown] speed trace (m/s): {trace}")
        print(f"[CoastDown] VERDICT: {verdict} -- gate {STOP_SPEED_MPS} m/s "
              f"{'held for the full window' if verdict == 'REACHABLE' else f'NOT reached; coast floor {floor} m/s above it'}")
        if verdict == "CAP":
            print("[CoastDown] The low-speed success gate is NOT reachable by "
                  "coasting in this action mode: the task needs absolute-mode "
                  "brake (INCREMENTAL_ACTIONS=0) or a raised IG_STOP_SPEED_MPS. "
                  "Do not train before this is signed off.")
    finally:
        env.close()


if __name__ == "__main__":
    main()