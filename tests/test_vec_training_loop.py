import jax, jax.extend.core
if not hasattr(jax.core,'Var'): jax.core.Var = jax.extend.core.Var
import sys, numpy as np; sys.path.insert(0,'src')
from gymnasium import Env, spaces
from jaxrl2.agents import TQCLearner
from jaxrl2.data import ReplayBuffer
from vae_nav.vec_env import ThreadedVecEnv
from vae_nav.episode_her import EpisodeRelabelBuffer
from vae_nav.vae_goal_wrapper import VAEGoalWrapper

GOAL_RADIUS, K_GOAL, K_COLL = 2.0, 600.0, 5.0

class MockTownEnv(Env):
    """Drives 1 m/step toward a goal; different distance per town."""
    def __init__(s, town, goal_x):
        s.town, s.goal_x, s.x, s.t = town, goal_x, 0.0, 0
        s.observation_space = spaces.Dict({
            'goal': spaces.Box(-np.inf,np.inf,(12,),np.float32),
            'latent': spaces.Box(-np.inf,np.inf,(128,),np.float32)})
        s.action_space = spaces.Box(np.array([-1.,0.],np.float32), np.array([1.,1.],np.float32))
        s.goal_block = lambda f: VAEGoalWrapper.goal_block(s, f)
    def _obs(s):
        return {'goal': np.array([s.goal_x-s.x,0,0,1,0,0,0,0,0,0,0,0], np.float32),
                'latent': np.zeros(128, np.float32)}
    def _info(s):
        return {'hero_pos': np.array([s.x,0.0]), 'hero_yaw': 0.0,
                'goal_pos': np.array([s.goal_x,0.0]), 'collision': False,
                'action_history': np.zeros((4,2)), 'is_success': s.x >= s.goal_x-GOAL_RADIUS}
    def reset(s, **kw): s.x, s.t = 0.0, 0; return s._obs(), s._info()
    def step(s, a):
        s.x += 1.0; s.t += 1
        reached = (s.goal_x - s.x) < GOAL_RADIUS
        r = K_GOAL if reached else 0.0
        return s._obs(), r, bool(reached), s.t >= 40, s._info()

towns  = ['Town01','Town02','Town04','Town05']
goals  = [8.0, 12.0, 20.0, 999.0]   # last is unreachable -> truncation path
venv = ThreadedVecEnv([(lambda t=t,g=g: MockTownEnv(t,g)) for t,g in zip(towns,goals)], names=towns)
n = venv.n

agent = TQCLearner(42, venv.observation_space.sample(), venv.action_space.sample(),
    actor_lr=3e-4, critic_lr=3e-4, temp_lr=1e-5, hidden_dims=(256,256), discount=0.99,
    tau=0.005, init_temperature=0.2, target_entropy=-2.0, backup_entropy=True,
    num_qs=5, num_quantiles=25, top_quantiles_to_drop_per_net=2, kappa=60.0,
    action_space=venv.action_space)
buf = ReplayBuffer(venv.observation_space, venv.action_space, 50000)
hers = [EpisodeRelabelBuffer(buf, e.goal_block, goal_radius=GOAL_RADIUS,
                             k_goal=K_GOAL, k_collision=K_COLL, seed=i)
        for i, e in enumerate(venv.envs)]

obs, _ = venv.reset(seed=0)
ep_done = np.zeros(n, np.int64); step = 0; n_trunc = 0; n_term = 0
while step < 480:
    a = np.stack([venv.action_space.sample() for _ in range(n)])
    nobs, r, d, infos, finals = venv.step(a)
    for i in range(n):
        o_i = ThreadedVecEnv.unstack(obs, i); n_i = ThreadedVecEnv.unstack(nobs, i)
        inf = infos[i]
        mask = 0.0 if inf['terminated'] else 1.0
        n_term += int(inf['terminated']); n_trunc += int(inf['truncated'])
        hers[i].record(dict(observations=o_i, actions=a[i], rewards=np.float32(r[i]),
                            masks=np.float32(mask), dones=np.float32(float(d[i])),
                            next_observations=n_i),
                       fields={k: inf[k] for k in ('hero_pos','hero_yaw','goal_pos','action_history')},
                       pos=inf['hero_pos'], hit=bool(inf['collision']))
        if finals[i] is not None:
            ep_done[i] += 1; hers[i].flush_episode()
    obs = nobs; step += n

print(f'transitions stepped: {step}  buffer: {len(buf)} (on-policy + hindsight)')
print(f'episodes per env: {dict(zip(towns, ep_done.tolist()))}')
print(f'terminated={n_term}  truncated={n_trunc}  (both paths exercised)')
hs = [h.stats() for h in hers]
print('HER relabels per env:', [h["her_relabelled"] for h in hs])
print('HER free positives  :', [h["her_free_positive_frac"] for h in hs])
assert all(h["her_free_positive_frac"] == 0.0 for h in hs)
assert len(buf) > step, 'hindsight rows should add to the on-policy ones'
assert n_term > 0 and n_trunc > 0
b = buf.sample(256*4); agent.update(b, utd_ratio=4)
print('agent.update on mixed multi-town batch OK')
act = np.asarray(agent.sample_actions(obs))
print('batched actions', act.shape, '| throttle>=0:', bool((act[:,1]>=0).all()))
venv.close(); print('VEC TRAINING LOOP OK')
