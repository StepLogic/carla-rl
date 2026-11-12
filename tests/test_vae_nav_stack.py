"""Offline integration test for the vae_nav wrapper stack.

Runs the whole chain -- CoreStateWrapper -> SparseGoalReward -> VAEGoalWrapper
-> EpisodeRelabelBuffer -- against a mock CarlaEnv, so the wiring can be
checked without a simulator. Catches the failures that would otherwise only
appear minutes into a real run: wrong info keys, a goal block on the wrong
scale, a dense reward leaking through, HER minting free positives.

Run:  JAX_PLATFORMS=cpu python tests/test_vae_nav_stack.py
"""
import sys, numpy as np; sys.path.insert(0,'src')
from gymnasium import spaces, Env
from vae_nav.core_state import CoreStateWrapper
from vae_nav.sparse_reward import SparseGoalReward
from vae_nav.vae_goal_wrapper import VAEGoalWrapper, GOAL_DIM
from vae_nav.latent_source import LatentSource
from vae_nav.episode_her import EpisodeRelabelBuffer

class L:
    def __init__(s,x,y,z=0.): s.x,s.y,s.z=x,y,z
    def distance(s,o): return float(np.hypot(s.x-o.x, s.y-o.y))
class R:
    def __init__(s,yaw): s.yaw=yaw
class T:
    def __init__(s,x,y,yaw): s.location=L(x,y); s.rotation=R(yaw)
class Hero:
    def __init__(s): s.x,s.y,s.yaw=0.,0.,0.
    def get_transform(s): return T(s.x,s.y,s.yaw)
class Core:
    def __init__(s): s.hero=Hero(); s.destination=T(30.,0.,0.)
class Exp:
    # Mirrors the real experiment: passed_chckp is set on the arrival step,
    # and the goal is RESAMPLED away on that same step -- which is why the
    # wrapper must read the flag rather than recompute a distance.
    def __init__(s): s.collision=False; s.diff_lane=False; s.passed_chckp=False
    obs_img=np.zeros((84,84,3),np.uint8)

class MockCarlaEnv(Env):
    def __init__(s):
        s.core=Core(); s.experiment=Exp(); s.t=0
        s.action_space=spaces.Box(np.array([-1.,0.],np.float32), np.array([1.,1.],np.float32))
        s.observation_space=spaces.Dict({"state":spaces.Box(0,255,(84,84,3),np.uint8)})
    def reset(s,**kw):
        s.t=0; s.core.hero.x=0.; s.experiment.collision=False
        return {"state":s.experiment.obs_img}, {"ok":"ok"}
    def step(s,a):
        s.t+=1; s.core.hero.x += 1.0        # 1 m per step toward goal at x=30
        # Real-env behaviour: on arrival, flag it AND move the goal.
        gx = s.core.destination.location.x
        if abs(gx - s.core.hero.x) < 2.0:
            s.experiment.passed_chckp = True
            s.core.destination = T(s.core.hero.x + 60.0, 0., 0.)   # resampled
        else:
            s.experiment.passed_chckp = False
        s.experiment.obs_img=(np.random.rand(84,84,3)*255).astype(np.uint8)
        return {"state":s.experiment.obs_img}, 0.123, False, False, {}

src_ = LatentSource("")                      # zero-latent stub, no VAE needed
env = VAEGoalWrapper(SparseGoalReward(CoreStateWrapper(MockCarlaEnv()),
                                      max_episode_steps=100), source=src_)
obs, info = env.reset()
print("obs keys:", sorted(obs.keys()), "| goal", obs["goal"].shape, "| latent", obs["latent"].shape)
assert obs["goal"].shape==(GOAL_DIM,) and obs["latent"].shape==(128,)
print("initial fwd offset:", round(float(obs["goal"][0]),2), "(goal 30 m ahead)")

buf=[]
class B:
    def insert(s,r): buf.append(r)
her = EpisodeRelabelBuffer(B(), env.goal_block, goal_radius=2.0, k_goal=600.,
                           k_collision=5., n_relabel=1, seed=0)

rewards=[]
for i in range(32):
    a = env.action_space.sample()
    o2, r, term, trunc, inf = env.step(a)
    rewards.append(r)
    her.record(dict(observations=obs, actions=a, rewards=np.float32(r),
                    masks=np.float32(0. if term else 1.),
                    dones=np.float32(float(term or trunc)), next_observations=o2),
               fields={"hero_pos":inf["hero_pos"],"hero_yaw":inf["hero_yaw"],
                       "goal_pos":inf["goal_pos"],
                       "action_history":np.zeros((4,2))},
               pos=inf["hero_pos"], hit=bool(inf["collision"]))
    obs=o2
    if term:
        print(f"terminated at step {i+1}: dist={inf['distance_to_goal']:.2f} success={inf['is_success']}")
        break

print("sparse rewards seen:", sorted(set(rewards)))
assert 600.0 in rewards, "should reach goal at ~30 steps"
assert set(rewards) <= {0.0, 600.0, -5.0}, "sparse reward must be 0/+600/-5 only"
n_on=len(buf); ins=her.flush_episode()
print(f"buffer: {n_on} on-policy + {ins} hindsight = {len(buf)}")
hr=[float(r['rewards']) for r in buf[n_on:]]
print(f"hindsight: {sum(1 for x in hr if x==600.)} paying, {sum(1 for x in hr if x==0.)} zero, free_pos={her.stats()['her_free_positive_frac']}")
assert her.stats()['her_free_positive_frac']==0.0
print("STACK OK")
