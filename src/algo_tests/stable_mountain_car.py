import gymnasium as gym

from jaxrl2.wrappers.timelimit import TimeLimit
from sbx import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.box2d.car_racing import CarRacing

# Parallel environments
# vec_env = make_vec_env("CarRacing-v3", n_envs=4)
env = gym.make("Pendulum-v1", render_mode="human",)  # default goal_velocity=0
eval_env = gym.make("Pendulum-v1", render_mode="rgb_array")  # default goal_velocity=0

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=int(1e6))
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # vec_env.render("human")