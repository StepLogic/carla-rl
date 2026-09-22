"""Probe v10's actor and goal-critic offline, from checkpoints only.

Rebuilds GatedTQCLearner under the run's recorded snapshot env (plus the
two flags the snapshot classically misses: GATED / OBS_WITH_MODE), loads a
checkpoint, and runs functional probes:

  A. goal-critic value vs forward goal distance (ordering check)
  B. actor mean action vs speed ratio and distance (throttle/steer shape)
  C. TQC ensemble disagreement across the num_qs nets
  D. raw policy log-std (how deterministic the actor is)

No CARLA is touched: the learner is built on a synthetic obs spec with the
same key shapes the pyramid env publishes (goal 4, latent 3840, mode 1,
state 9), so the trainer process and its sim keep running undisturbed.

Run with the ig-venv interpreter. Latent is ZERO for every probe -- an
off-manifold input, but constant, so within-probe ordering is still valid;
the caveat applies to absolute values.
"""
import os
import sys
import pickle
import numpy as np

# -- recorded env ------------------------------------------------------------
snap = os.environ.get("SNAP", "results/run_env/carla_gated_her_v10_vc1.env")
with open(snap) as f:
    for line in f:
        line = line.strip()
        if "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k, v)
os.environ["GATED"] = "1"
os.environ["OBS_WITH_MODE"] = "1"

import gymnasium as gym
import jax
import jax.numpy as jnp

sys.path.insert(0, "src")
from vae_nav.gated_tqc_learner import GatedTQCLearner  # noqa: E402
from vae_nav.policy_io import load_agent_state  # noqa: E402

obs_space = gym.spaces.Dict({
    "goal": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
    "latent": gym.spaces.Box(-np.inf, np.inf, (3840,), np.float32),
    "mode": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
    "state": gym.spaces.Box(-np.inf, np.inf, (9,), np.float32),
})
act_space = gym.spaces.Box(np.array([-1.0, 0.0], np.float32),
                           np.array([1.0, 1.0], np.float32))

# MIRROR agent_kwargs() by importing the trainer's builder itself.
from train_carla_tqc import build_agent  # noqa: E402

agent = build_agent(obs_space, act_space, seed=0)

CK_DIR = "checkpoints/carla_gated_her_v10_vc1"
CK_MAIN = os.environ.get("CK", f"{CK_DIR}/checkpoint_130000.pkl")


def base_obs(fwd, left, ratio, mode=1):
    """Yaw=0, so the goal block is [forward, left, 0, 1]."""
    return {
        "goal": np.array([fwd, left, 0.0, 1.0], np.float32),
        "latent": np.zeros(3840, np.float32),
        "mode": np.array([float(mode)], np.float32),
        "state": np.concatenate([[ratio], np.zeros(8)], 0).astype(np.float32),
    }


def load(path):
    ck = pickle.load(open(path, "rb"))
    load_agent_state(agent, ck["agent"])
    print(f"# loaded {path} (step {ck['step']})")


def actor_mean(observations):
    """Deterministic (tanh-mean) action for a batch of obs dicts."""
    npobs = jax.tree_util.tree_map(lambda a: np.asarray(a)[None], observations)
    return np.asarray(agent.eval_actions(npobs), np.float32)


def critic_q(observations, action=None):
    """Goal-critic ensemble: mean value and cross-net std.

    StateActionQuantileEnsemble applies to (obs, act) and returns
    (n_nets, batch, n_quantiles); the value is the mean over the KEPT
    quantiles (TQC drops top_quantiles_to_drop_per_net of the union).
    NOTE: `observations` is the UNBATCHED dict; batching happens here,
    once -- a double batch (1,1,k) is ndim>=3 and the PixelMultiplexer
    would route the key through the conv encoder.
    """
    if action is None:
        action = actor_mean(observations)
        if action.ndim == 1:
            action = action[None]
    obs = jax.tree_util.tree_map(
        lambda a: jnp.asarray(a)[None], observations)
    q = agent._critic.apply_fn(
        {"params": agent._critic.params}, obs, jnp.asarray(action))
    q = np.asarray(q)
    if q.ndim == 3:               # (nets, batch, quantiles)
        per_net = q[:, 0, :].mean(-1)
        return per_net.mean(), per_net.std(), q[0, 0].shape[0]
    return float(np.asarray(q).ravel()[0]), 0.0, -1


load(CK_MAIN)   # EVERY probe below runs on the loaded weights

print("\n=== A. GOAL-CRITIC VALUE vs FORWARD GOAL DISTANCE "
      "(mode=1, speed ratio 0.5, latent 0) ===")
print(f"{'F_m':>6} {'Q_mean':>10} {'net_std':>9} {'n_quantiles':>11}")
for f in (5, 10, 20, 40, 80, 120, 200, 400):
    obs = base_obs(f, 0.0, 0.5)
    q, s, nq = critic_q(obs)
    print(f"{f:>6} {q:>10.2f} {s:>9.2f} {nq:>11}")

print("\n=== B. ACTOR MEAN ACTION vs SPEED RATIO (mode=1, goal ahead) ===")
print(f"{'F_m':>6} {'ratio':>6} {'steer':>7} {'throttle':>9}")
for f in (20.0, 40.0, 80.0, 200.0):
    for r in (0.0, 0.25, 0.5, 0.75, 1.0):
        obs = base_obs(f, 0.0, r)
        a = actor_mean(obs)[0]
        print(f"{f:>6} {r:>6.2f} {a[0]:>7.3f} {a[1]:>9.3f}")

print("\n=== B2. ACTOR vs LATERAL OFFSET (steering sanity, ratio 0.5) ===")
print(f"{'F_m':>6} {'left_m':>7} {'steer':>7} {'throttle':>9}")
for f in (40.0, 80.0):
    for l in (-10.0, 0.0, 10.0):
        obs = base_obs(f, l, 0.5)
        a = actor_mean(obs)[0]
        print(f"{f:>6} {l:>7} {a[0]:>7.3f} {a[1]:>9.3f}")

print("\n=== C. EXPLORE-CRITIC VALUE vs SPEED RATIO (mode=0, goal block zeroed) ===")
print(f"{'ratio':>6} {'Q_e':>10}")
for r in (0.0, 0.25, 0.5, 0.75, 1.0):
    obs = base_obs(0.0, 0.0, r, mode=0)
    a = actor_mean(obs)
    obsb = jax.tree_util.tree_map(lambda a: jnp.asarray(a)[None], obs)
    q = agent._critic_e.apply_fn(
        {"params": agent._critic_e.params}, obsb, jnp.asarray(a))
    print(f"{r:>6.2f} {float(np.asarray(q).mean()):>10.2f}")

print("\n=== D. POLICY LOG-STD (raw Dense_1 head, mode=1, F=40, ratio 0.5) ===")
obsb = jax.tree_util.tree_map(lambda a: jnp.asarray(a)[None],
                              base_obs(40.0, 0.0, 0.5))
d = agent._actor.apply_fn({"params": agent._actor.params}, obsb)
base = getattr(d, "_distribution", None) or getattr(d, "distribution", d)
print("  dist:", type(d).__name__, "| base:", type(base).__name__)
try:
    print("  raw gaussian loc:", np.asarray(base.loc)[0],
          " scale:", np.asarray(base.scale)[0])
except Exception as e:
    print("  (no loc/scale:", e, ")")
samp = np.stack([np.asarray(d.sample(seed=jax.random.PRNGKey(i)))[0]
                 for i in range(64)], 0)
print("  sample mean over 64 draws:", samp.mean(0),
      " sample std:", samp.std(0))

print("\n=== E. DRIFT: value-vs-distance on ck_45000 (post-freeze) vs the loaded ck ===")
ck45 = f"{CK_DIR}/checkpoint_45000.pkl"
if os.path.exists(ck45):
    load(ck45)
    for f in (5, 20, 80, 200, 400):
        q, s, _ = critic_q(base_obs(f, 0.0, 0.5))
        print(f"  F={f:>4}: Q={q:>9.2f} net_std={s:.2f}")
    load(CK_MAIN)