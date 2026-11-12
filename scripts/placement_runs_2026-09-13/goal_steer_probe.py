"""Does each checkpoint's MEAN action steer toward a goal it is given?
Relabel cached states with synthetic goals 30 m away at known bearings (via
the wrapper's own goal_block formula), mode=1, and measure the deterministic
steer against the bearing sign. Also the gap between tanh(mu) and E[samples].
Simulator-free, so it can run beside training."""
import os, pickle, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np
from gymnasium import spaces
import train_carla_tqc as T
from vae_nav.policy_io import load_agent_state
N_HIST = 4
tr = pickle.load(open("results/placement_v4_town02/trace_s7.pkl", "rb"))
obs, fields = tr["observations"][:300], tr["fields"][:300]

def goal_block(f, goal):
    hero = np.asarray(f["hero_pos"], np.float64); yaw = float(f["hero_yaw"])
    d = goal - hero; cy, sy = np.cos(yaw), np.sin(yaw)
    forward = d[0] * cy + d[1] * sy; left = -(-d[0] * sy + d[1] * cy)
    hist = np.asarray(f.get("action_history", np.zeros((N_HIST, 2))), np.float32).reshape(-1)[:2 * N_HIST]
    return np.concatenate([[forward, left], [np.sin(yaw), np.cos(yaw)], hist]).astype(np.float32)

bearings = np.deg2rad([-40, -15, 0, 15, 40])       # +ve = LEFT of heading
rows, lefts = [], []
for o, f in zip(obs, fields):
    hero = np.asarray(f["hero_pos"], np.float64); yaw = float(f["hero_yaw"])
    for b in bearings:
        ang = yaw + b                                # CARLA left-handed: +b is left? checked via block sign below
        goal = hero + 30.0 * np.array([np.cos(ang), np.sin(ang)])
        g = goal_block(f, goal)
        rows.append({"latent": np.asarray(o["latent"], np.float32), "goal": g,
                     "vector": np.asarray(o["vector"], np.float32), "mode": np.array([1.0], np.float32)})
        lefts.append(g[1])                           # the block's own LEFT component is the truth
stacked = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
lefts = np.asarray(lefts); print("states x bearings:", len(rows), "| left-component sign check:", np.sign(lefts[:5]).astype(int))
obs_space = spaces.Dict({k: spaces.Box(-np.inf, np.inf, v.shape[1:], np.float32) for k, v in stacked.items()})
act_space = spaces.Box(np.array([-0.5, 0.0], np.float32), np.array([0.5, 0.4], np.float32))
agent = T.build_agent(obs_space, act_space, 0)
cks = [("v4 350k", "checkpoints/carla_gated_her_v4/checkpoint_350000.pkl"),
       ("v5 125k", "checkpoints/carla_gated_her_v5/checkpoint_125000.pkl"),
       ("v5 150k", "checkpoints/carla_gated_her_v5/checkpoint_150000.pkl"), ("v5 175k", "checkpoints/carla_gated_her_v5/checkpoint_175000.pkl"), ("v5 200k", "checkpoints/carla_gated_her_v5/checkpoint_200000.pkl")]
print(f"{'ckpt':>9} {'corr(steer,-left)':>18} {'sign agree':>11} {'|det steer|':>11} {'|E[smp]-det| steer':>19} {'det throttle':>13}")
for name, path in cks:
    load_agent_state(agent, pickle.load(open(path, "rb"))["agent"])
    det = np.asarray(agent.eval_actions(stacked))
    smp = np.mean([np.asarray(agent.sample_actions(stacked)) for _ in range(8)], axis=0)
    st = det[:, 0]
    corr = np.corrcoef(st, -lefts)[0, 1]                      # steer +ve = right; goal left -> steer -ve
    # bootstrap over STATES (each state contributes 5 bearings), 95% CI
    nb = len(bearings); ns = len(st) // nb
    rng_ = np.random.default_rng(0); cs = []
    for _ in range(300):
        idx = rng_.integers(0, ns, ns)
        sel = (idx[:, None] * nb + np.arange(nb)[None, :]).ravel()
        cs.append(np.corrcoef(st[sel], -lefts[sel])[0, 1])
    lo, hi = np.percentile(cs, [2.5, 97.5])
    agree = np.mean(np.sign(st[np.abs(lefts) > 5]) == -np.sign(lefts[np.abs(lefts) > 5]))
    print(f"{name:>9} {corr:>18.3f} [{lo:.2f},{hi:.2f}] {agree:>11.2f} {np.abs(st).mean():>11.3f} {np.abs(smp[:,0]-st).mean():>19.3f} {det[:,1].mean():>13.3f}")
