"""Demo drive of the image-goal v2 pre-divergence checkpoint (user ask 2026-09-20).

WHY A SEPARATE SCRIPT, NOT image_goal_video_eval.py. That driver pins the v1
band (5-100 m) and deterministic actions (which parked: 7/12 episodes never
moved). The demo shows the V2 policy: band 5-25 m, SAMPLED actions (the honest
mode -- training success 0.88 is sampled; deterministic is tanh(mean)), and
DRAW_GOAL=1 so run_episodes marks the goal with carla.world.debug.draw_line --
a red 6 m vertical pole at info["goal_pos"], redrawn every tick, visible in the
hero camera whenever it faces the goal.

PORT 2002. The v2 training run owns port 2000 and its lock; a demo must not
touch either. ensure_carla() spawns a second, independent server on 2002
(Low + offscreen, ~2 GB) and tears it down at process exit. The trainer and
the demo share the GPU only.

EXPLORE MODE IS NOT RUN, ON PURPOSE. The image-goal arm has no explore mode:
IMAGE_GOAL_TASK=1 returns from make_env before the GATED branch, so there is
no mode bit in the observation and no explore reward; the v4 config's
explore_mode coin is written twice and read nowhere in this config. The
checkpoint is goal-mode only -- demoing "explore" would mean driving with a
forged goal block, which is off-distribution input, not a mode.
"""
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "src"))

import train_carla_tqc as T
from eval_policy import run_episodes
from vae_nav.eval_video import EvalVideoRecorder
from vae_nav.latent_source import from_env as latent_from_env
from vae_nav.policy_io import load_agent_state, param_fingerprint

CK = os.environ.get("IG_DEMO_CKPT",
                    "checkpoints/carla_image_goal_v2/checkpoint_350000.pkl")
TOWN = os.environ.get("IG_DEMO_TOWN", "Town01")
N_EPISODES = int(os.environ.get("IG_DEMO_EPISODES", "8"))
SEED0 = int(os.environ.get("IG_DEMO_SEED", "4242"))
PORT = int(os.environ.get("IG_DEMO_PORT", "2002"))
OUT = os.environ.get("IG_DEMO_DIR", "results/image_goal_demo_v2_350k")
ACTION_MODE = os.environ.get("IG_DEMO_ACTION", "sample")
MAX_STEPS = int(os.environ.get("MAX_EPISODE_STEPS", "1000"))
BAND = (float(os.environ.get("IG_DEMO_BAND_LO", "5.0")),
        float(os.environ.get("IG_DEMO_BAND_HI", "25.0")))

# The values the v2 run trained under, pinned the same way
# image_goal_video_eval.py pins them (recorded in
# results/run_env/carla_image_goal_v2.env).
os.environ.setdefault("IMAGE_GOAL_TASK", "1")
os.environ.setdefault("PERCEPTION", "vc1")
os.environ.setdefault("OBS_WITH_COLLISION", "1")
os.environ.setdefault("ENV_SKIP_IMAGE_FEATURES", "1")
os.environ.setdefault("DRAW_GOAL", "1")
os.environ.setdefault("CARLA_START_SERVER", "1")
os.environ.setdefault("CARLA_QUALITY", "Low")
os.environ.setdefault("CARLA_OFFSCREEN", "1")

latent_source = latent_from_env()
env = T.make_env(TOWN, SEED0, latent_source, eval_mode=True, port=PORT)
agent = T.build_agent(env.observation_space, env.action_space, SEED0)
with open(CK, "rb") as f:
    ck = pickle.load(f)
load_agent_state(agent, ck["agent"])
print(f"[IGDemo] checkpoint step={ck.get('step')} "
      f"{param_fingerprint(ck['agent'])}", flush=True)

rec = EvalVideoRecorder(os.path.join(OUT, TOWN), ck.get("step") or 0,
                        max_videos=N_EPISODES, latent_source=latent_source,
                        decode=False)

t0 = time.time()
records = run_episodes(agent, env, TOWN, BAND[0], BAND[1], N_EPISODES, SEED0,
                       max_steps=MAX_STEPS, recorder=rec,
                       action_mode=ACTION_MODE)
env.close()

out = os.path.join(OUT, f"demo_{TOWN}_{ACTION_MODE}.json")
with open(out, "w") as f:
    json.dump({"checkpoint": CK, "town": TOWN, "seed0": SEED0,
               "action_mode": ACTION_MODE, "band": BAND,
               "success_frac": round(float(
                   sum(r["success"] for r in records) / len(records)), 4),
               "records": records}, f, indent=2)
print(f"[IGDemo] done: {len(records)} episodes, success="
      f"{round(sum(r['success'] for r in records) / len(records), 3)} "
      f"in {time.time() - t0:.0f}s -> {out}; videos in "
      f"{os.path.join(OUT, TOWN, 'videos')}", flush=True)