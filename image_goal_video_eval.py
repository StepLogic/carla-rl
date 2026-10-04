"""Offline video eval of the image-goal arm's final checkpoint (user ask 2026-09-19).

WHY NOT eval_policy.py DIRECTLY. That driver hardcodes the VAE LatentSource
(`LatentSource(os.environ.get("VAE_CKPT", ""))`), while this arm's
observation pipeline needs the VcSource selected by PERCEPTION=vc1 -- a
different latent source AND a 768-d observation the VAE source would silently
mismatch. So this wrapper builds the source with latent_from_env(), the ONE
place a backend is chosen, and pins every flag to the values the run trained
under (the record of those values is results/run_env/carla_image_goal.env;
it is READ, not sourced -- the values are re-stated here so the videos
provably show the trained configuration).

Videos land in results/image_goal_videos/<town>/videos/<ep>.mp4. The VAE
lane/obstacle decode panel is OFF (decode=False): VcSource has no decoder,
and per-frame VAE decode on the GPU is the measured CARLA-killer
(memory: video-decode-panel-kills-carla).

Success for THIS arm is info["is_success"] (the wrapper's own flag), which
eval_policy.run_episodes already reads via info.get("arrived") or
info.get("is_success").
"""
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

CK = os.environ.get("IG_EVAL_CKPT",
                    "checkpoints/carla_image_goal/checkpoint_1000000.pkl")
TOWN = os.environ.get("IG_EVAL_TOWN", "Town01")
N_EPISODES = int(os.environ.get("IG_EVAL_EPISODES", "12"))
SEED0 = int(os.environ.get("IG_EVAL_SEED", "1234"))
VIDEO_DIR = os.environ.get("IG_EVAL_DIR", "results/image_goal_videos")
MAX_STEPS = int(os.environ.get("MAX_EPISODE_STEPS", "1000"))

# THE BAND THE RUN TRAINED UNDER, FIXED (user 2026-09-19: 5-100 m). carla_core
# reads MIN/MAX at every reset; run_episodes rewrites both per call, so this
# is set here AND passed as the cell's (lo, hi) below -- same numbers.
os.environ.setdefault("GOAL_DIST_MIN", "5.0")
os.environ.setdefault("GOAL_DIST_MAX", "100.0")

latent_source = latent_from_env()
env = T.make_env(TOWN, SEED0, latent_source, eval_mode=True,
                 port=int(os.environ.get("CARLA_PORT", "2000")))
agent = T.build_agent(env.observation_space, env.action_space, SEED0)
with open(CK, "rb") as f:
    ck = pickle.load(f)
load_agent_state(agent, ck["agent"])
print(f"[IGEval] checkpoint step={ck.get('step')} "
      f"{param_fingerprint(ck['agent'])}", flush=True)

rec = EvalVideoRecorder(os.path.join(VIDEO_DIR, TOWN),
                        ck.get("step") or 0, max_videos=N_EPISODES,
                        latent_source=latent_source, decode=False)

t0 = time.time()
records = run_episodes(agent, env, TOWN, 5.0, 100.0, N_EPISODES, SEED0,
                       max_steps=MAX_STEPS, recorder=rec,
                       action_mode="deterministic")
env.close()

import json
import numpy as np
out = os.path.join(VIDEO_DIR, f"eval_{TOWN}.json")
with open(out, "w") as f:
    json.dump({"checkpoint": CK, "town": TOWN, "seed0": SEED0,
               "action_mode": "deterministic",
               "success_frac": round(float(np.mean(
                   [r["success"] for r in records])), 4),
               "records": records}, f, indent=2)
print(f"[IGEval] done: {len(records)} episodes, "
      f"success={records and round(sum(r['success'] for r in records) / len(records), 3)} "
      f"in {time.time() - t0:.0f}s -> {out}; videos in "
      f"{os.path.join(VIDEO_DIR, TOWN, 'videos')}", flush=True)