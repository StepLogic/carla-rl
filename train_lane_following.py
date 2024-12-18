#!/usr/bin/env python

import argparse
from collections import deque
from typing import Any, Dict, Optional

import gym.wrappers
import torch
import gym
# from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DQN, SAC,PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from vision_rl.stb3.jax_lane_experiment import JAXLaneExperiment
from vision_rl.rllib_integration.carla_env_v2 import CarlaEnv

experiment_config = {
    "framework": "torch",
    "num_workers": 1,
    "num_gpus_per_worker": 1,
    "num_cpus_per_worker": 3,
    "rollout_fragment_length": 16,
    "timesteps_per_iteration": 2000,
    "train_batch_size": 16,
    "learning_starts": 5000,
    "buffer_size": 15000,
    "lr": 0.0003,
    "exploration_config": {
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.0,
        "final_epsilon": 0.1,
        "epsilon_timesteps": 50000
    },
    "env_config": {
        "carla": {
            "host": "localhost",
            "timeout": 20.0,
            "timestep": 0.1,
            "retries_on_error": 25,
            "resolution_x": 600,
            "resolution_y": 600,
            "quality_level": "Low",
            "enable_map_assets": True,
            "enable_rendering": True,
            "show_display": True,
            "town":"Town02"
        },
        "experiment": {
            "type":JAXLaneExperiment,
            "hero": {
                "blueprint": "vehicle.mercedes.coupe_2020",
                "sensors": {
                    "collision": {
                        "type": "sensor.other.collision"
                    },
                    "rgb": {
                        "type": "sensor.camera.rgb",
                        "image_size_x": 300,
                        "image_size_y": 300,
                        "transform": "1.9, 0.0, 1.7, 0.0, -15.0, 0.0"
                    },
                    # "goal": {
                    #     "type": "sensor.goal",
                    #     "image_size_x": 300,
                    #     "image_size_y": 300,
                    #     # "transform": "1.9, 0.0, 1.7, 0.0, -15.0, 0.0",
                    #     # "attach":False
                    # },
                    "lane_invasion": {
                        "type": "sensor.other.lane_invasion"
                    }
                },
                # "spawn_points": [
                #     "-115.60, -207.60, 11.02, -0.00, -0.01, -179.92",  # tl_l
                #     "-243.70, 71.50, 11.98, -0.00, -0.08, 90.81",      # bl_d
                #     "76.60, 202.10, 1.00, -0.00, 0.00, 0.29",          # br_r
                #     "210.20, -50.30, 1.00, -0.00, 0.00, -90.67",       # ur_u
                #     "-114.30, 191.00, 9.27, -0.00, 0.61, 179.97",      # bl_l
                #     "-233.00, -51.50, 11.00, -0.00, 0.00, -90.84",     # tl_u
                #     "83.90, -190.30, 1.00, -0.00, 0.00, -0.24",        # tr_r
                #     "189.30, 88.20, 1.00, -0.00, 0.00, 90.66"          # br_d
                # ]
            },
            "background_activity": {
                "n_vehicles": 0,
                "n_walkers": 0,
                "tm_hybrid_mode": True
            },
            # "town": "Town05",
            "weather": "CloudySunset",
            "others": {
                "framestack": 1,
                "max_time_idle": 600,
                "max_dist": 200,
                "target_speed": 5.0
            }
        }
    }
}


    # def batch_compute_reward_from_observation(self,
    #                                        observations: Dict[str, np.ndarray],
    #                                        actions: np.ndarray,
    #                                        next_observations: Dict[str, np.ndarray],
    #                                        infos: Dict[str, np.ndarray]) -> np.ndarray:
    #     """Compute rewards using metrics from info dictionary."""
    #     # Use metrics from info
    #     speed = infos['velocity']
    #     lane_deviation = infos['lane_deviation']
    #
    #     # Calculate goal distances using positions from info
    #     goal_distances = infos['distance_to_goal']
    #
    #     terminates = infos['collision']>0.0
    #     # Compute rewards using the same formula as in step function
    #     rewards = ((1 - np.exp(-speed)) * np.exp(-lane_deviation)) * 10
    #     # Add success reward for goals within threshold
    #     successes = goal_distances <= 2.0
    #     rewards = np.where(successes, rewards + 10, rewards)
    #     rewards = np.where(terminates, rewards - 10.0, rewards)
    #     return rewards
    # def step(self, action):
    #     observations, reward, done, terminate, info=self.sim.step(action)
    #     observations={k: observations[k] for k in self.observation_space.keys()}
    #     if done or terminate:
    #         self.sr_counter.append(int(done))
    #     return observations,reward,done,terminate,info
    #
    # def reset(self):
    #     return self.sim.reset()
    #
    # def render(self, mode='human'):
    #     return self.sim.render(mode=mode)
    #
    # def close(self):
    #     self.sim.__del__()

def select(samples, batch_size):
    """
    Select goals from a list of samples based on random indices.
    
    Args:
        samples: List of arrays/tensors with shape (batch_size, *feature_dims)
        batch_size: Integer specifying the batch size
    
    Returns:
        Selected goals with same shape as individual samples
    """
    # Generate random indices for selection
    indices = np.random.randint(0, len(samples), size=batch_size)
    
    # Initialize output array with same shape as samples
    selected = np.zeros_like(samples[0])
    
    # Use boolean indexing to select goals
    for i, sample in enumerate(samples):
        mask = (indices == i)
        selected[mask] = sample[mask]
    
    return selected

def relabel(batch: Dict[str, Any], env) -> Dict[str, Any]:
    """Relabel experiences with info metrics."""
    batch_size = batch['actions'].shape[0]
    observation = batch['observations']
    next_observation = batch['next_observations']
    future_observation = batch['future_observations']
    info = batch['infos']
    # next_info = batch['next_infos']
    
    # Extract image goals
    original_goal = observation['goal']
    original_goal_next = next_observation['goal']
    # breakpoint()
    future_goal = future_observation['obs']  # Use future observation as goal
    
    # Generate random goal (not needed for image-based goals, keeping original)
    random_goal = original_goal.copy()  # Could implement image perturbation here if desired
    
    # Select new goals
    # def select(samples):
    #     indices = np.random.randint(0, len(samples), size=(batch_size, 1))
    #     return sum((indices == i).squeeze() * sample for i, sample in enumerate(samples))
    
    goals = select([
        original_goal,   # Keep original goal
        original_goal,   # Keep original goal
        future_goal,     # Use future observation as goal
        random_goal      # Use random goal (same as original for now)
    ],batch_size=batch_size)
    
    goals_next = goals.copy()
    
    # Check which goals are achieved using distances from info
    goals_complete = info['distance_to_goal'] <= 2.0
    
    # Update next goals for completed ones
    goals_next[goals_complete] = original_goal_next[goals_complete]
    
    # Update observations with new goals
    observation['goal'] = goals
    next_observation['goal'] = goals_next
    
    # Update info metrics for new goals
    future_info = batch['future_infos']
    relabeled_info = {
        'velocity': info['velocity'],  # Speed remains same
        'lane_deviation': info['lane_deviation'],  # Lane deviation remains same
        'distance_to_goal': future_info['distance_to_goal'] , # Update distance to new goal
        'collision':info['collision']
    }
    
    # relabeled_next_info = {
    #     'velocity': next_info['velocity'],
    #     'lane_deviation': next_info['lane_deviation'],
    #     'distance_to_goal': future_info['distance_to_goal'],
    #     'collision':info['collision']
    # }
    
    # Compute new rewards with updated info
    reward = env.batch_compute_reward_from_observation(
        observation, batch['actions'], next_observation, relabeled_info)
    
    return {
        **batch,
        'observations': observation,
        'next_observations': next_observation,
        'rewards': reward,
        'infos': relabeled_info,
        # 'next_infos': relabeled_next_info
    }

# Example usage with memory efficient replay buffer
def create_info_example():
    """Create example info dict for buffer initialization."""
    return {
        'velocity': 0.0,
        'lane_deviation': 0.0,
        'distance_to_goal': 0.0,
        "collision":0.0
    }


# %%
import os
import numpy as np
import tqdm
from torch.utils.tensorboard import SummaryWriter
import glob
import ssl
# import gymnasium as gym
import gym
# from gymnasium.wrappers.time_limit import TimeLimit
import ml_collections
from ml_collections.config_dict import config_dict
from flax.training import checkpoints
from flax.core.frozen_dict import FrozenDict

# Import your modules
from jaxrl5.utils.misc import load_checkpoints, load_pretrained
# from jaxrl5.wrappers.gym_video import VideoRecorder
from jaxrl5.wrappers.time_limit import TimeLimit
from jaxrl5.wrappers.frame_stack_modified import FrameStack
from jaxrl5.agents.drq.drq_learner import DrQLearner
from jaxrl5.data.memory_efficient_replay_buffer import MemoryEfficientReplayBuffer
# SSL Certificate fix
ssl._create_default_https_context = ssl._create_stdlib_context
class OrnsteinUhlenbeckActionNoise():
    """
    An Ornstein Uhlenbeck action noise, this is designed to approximate Brownian motion with friction.

    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

    :param mean: Mean of the noise
    :param sigma: Scale of the noise
    :param theta: Rate of mean reversion
    :param dt: Timestep for the noise
    :param initial_noise: Initial value for the noise output, (if None: 0)
    :param dtype: Type of the output noise
    """

    def __init__(
        self,
        mean: np.ndarray,
        sigma: np.ndarray,
        theta: float = 0.15,
        dt: float = 1e-2,
        initial_noise: Optional[np.ndarray] = None,
        dtype = np.float32,
    ) -> None:
        self._theta = theta
        self._mu = mean
        self._sigma = sigma
        self._dt = dt
        self._dtype = dtype
        self.initial_noise = initial_noise
        self.noise_prev = np.zeros_like(self._mu)
        self.reset()

    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * (self._mu - self.noise_prev) * self._dt
            + self._sigma * np.sqrt(self._dt) * np.random.normal(size=self._mu.shape)
        )
        self.noise_prev = noise
        return noise.astype(self._dtype)

    def reset(self) -> None:
        """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
        self.noise_prev = self.initial_noise if self.initial_noise is not None else np.zeros_like(self._mu)

    def __repr__(self) -> str:
        return f"OrnsteinUhlenbeckActionNoise(mu={self._mu}, sigma={self._sigma})"





def get_config():
    config = ml_collections.ConfigDict()
    
    # Model configuration
    config.model = ml_collections.ConfigDict()
    config.model.model_cls = "DrQLearner"
    config.model.actor_lr = 3e-4
    config.model.critic_lr = 3e-4
    config.model.temp_lr = 1e-4
    config.model.hidden_dims = (256, 256)
    config.model.cnn_features = (32, 32, 32, 32)
    config.model.cnn_filters = (3, 3, 3, 3)
    config.model.cnn_strides = (2, 2, 2, 2)
    config.model.cnn_padding = "VALID"
    config.model.latent_dim = 50
    config.model.encoder = "d4pg"
    config.model.discount = 0.998
    config.model.num_qs = 2
    config.model.num_min_qs = 2
    config.model.critic_layer_norm = True
    config.model.tau = 0.005
    config.model.init_temperature = 1.0
    config.model.target_entropy = config_dict.placeholder(float)
    config.model.backup_entropy = False
    config.model.pixel_keys = ("obs",)

    # Training configuration
    config.training = ml_collections.ConfigDict()
    config.training.env_name = 'carla_lane'
    config.training.goal_config = 'small_inner_graph'
    config.training.image_size = 84
    config.training.num_stack = 6

    config.training.seed = 42
    config.training.batch_size = 256
    config.training.max_steps = int(2e6)
    config.training.start_training = int(1e3)
    config.training.replay_buffer_size = int(1e6)
    config.training.utd_ratio = 8
    
    # Evaluation configuration
    config.eval = ml_collections.ConfigDict()
    config.eval.eval_episodes = 100
    config.eval.eval_interval = 500000
    config.eval.log_interval = 100
    
    # Paths and saving
    config.paths = ml_collections.ConfigDict()
    config.paths.save_dir = './tmp/'
    config.paths.expert_replay_buffer = ''
    config.paths.pretrained_weights = ''
    config.paths.checkpoints = ''
    
    # Features
    config.features = ml_collections.ConfigDict()
    config.features.save_video = False
    config.features.save_buffer = False
    config.features.freeze_encoder = False
    config.features.tqdm = True
    
    config.comment = ''
    
    return config

def filter_obs(obs):
    return FrozenDict({k: obs[k] for k in obs.keys()})
def filter_obs_with_config(obs,config):
    return FrozenDict({k: obs[k] for k in config.model.pixel_keys.keys()})

def filter_batch(obs):
    return FrozenDict({
        'observations': filter_obs(obs['observations']),
        'actions': obs['actions'],
        'next_observations': filter_obs(obs['next_observations']),
        'rewards': obs['rewards'],
        'dones': obs['dones'],
        'masks': obs['masks'],
    })

def setup_training():
    config = get_config()
    
    # Set target entropy (required)
    # config.model.target_entropy = -3.0  # Typically negative dimension of action space
    
    # Optional: Modify any config parameters
    config.training.max_steps = int(1e9)  # Reduced for example
    config.training.start_training = 5000
    config.training.batch_size = 32
    config.eval.eval_interval = 50000
    return config
def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
):
    # if save_video:
    #     env = WANDBVideo(env, name="eval_video", max_videos=1)
    # env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    # env =VideoRecorder(env,"debug_viz",lambda x: x>0)
    sr=deque(maxlen=num_episodes)
    rewards=list()
    episode_length=list()
    for i in range(num_episodes):
        (observation,info),done,termintate = env.reset(), False,False
        step=0
        # sr=deque(maxlen=num_episodes)
        # breakpoint()
        while not (done or termintate):
            action = agent.eval_actions(filter_obs(observation))
            action = np.clip(action, env.action_space.low, env.action_space.high)
            # print("eval_action",action)
            observation, reward, done,termintate, info = env.step(action)
            rewards.append(reward)
            # env.render()
            step+=1
        if done or termintate:
            episode_length.append(step)
            # sr.append(int(info["is_success"]))
    return {"return": np.mean(rewards), "length": np.mean(episode_length) ,"success_rate":np.mean(sr)}

import jax
import jax.numpy as jnp

def create_ou_noise(mean, std_deviation, theta=0.15, dt=0.01):
    def apply_noise(action, state, key):
        noise = state + theta * (mean - state) * dt + \
                std_deviation * jnp.sqrt(dt) * jax.random.normal(key, state.shape)
        noisy_action = action + noise
        return noisy_action, noise
    
    def reset():
        return jnp.zeros_like(mean)
    return apply_noise, reset
def train_agent(config):
    # Initialize environment and agent
    writer = SummaryWriter(f"./runs/run_{len(glob.glob('./runs/*'))+1}")
    policy_folder = os.path.join("checkpoints", f"model-{len(glob.glob('./runs/*'))}")
    os.makedirs(policy_folder, exist_ok=True)
    # Create environment
    # print(experiment_config["experiment"])
    env = CarlaEnv(experiment_config["env_config"])
    env = FrameStack(env,num_stack=config.training.num_stack, stacking_key="obs")
    env = TimeLimit(env, max_episode_steps=10000)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    action_dim = 2
    mean = np.zeros(action_dim)
    sigma = 0.5 * np.ones(action_dim)
    noise = OrnsteinUhlenbeckActionNoise(mean=mean, sigma=sigma)
    noise.reset()

    # Create agent
    model_cls=config.model.model_cls
    del config.model.model_cls
    agent = globals()[model_cls].create(
        config.training.seed,
        env.observation_space,
        env.action_space,
        **config.model.to_dict()
    )

    # Load pretrained weights or checkpoints if specified
    if config.paths.pretrained_weights:
        agent = load_pretrained(config.paths.pretrained_weights, agent)
    if config.paths.checkpoints:
        agent = load_checkpoints(config.paths.checkpoints, agent)

    # Setup replay buffer
    replay_buffer = MemoryEfficientReplayBuffer(
        env.observation_space,
        env.action_space,
        config.training.replay_buffer_size,
        pixel_keys=config.model.pixel_keys,
        # info_example={
        #         'velocity': 0.0,
        #         'lane_deviation': 0.0,
        #         'distance_to_goal': 0.0,
        #         "collision":0
        # },
    )
    
    replay_buffer.seed(config.training.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(sample_args={
        'batch_size': config.training.batch_size,
        'sample_futures': True,
        'relabel': True,
    })

    # # # Helper function for relabeling
    # def do_relabel_batch(batch):
    #     return filter_batch(relabel(batch, env.unwrapped))
    
    # replay_buffer._relabel_fn = do_relabel_batch

    # Training loop
    (observation, info),done,terminate = env.reset(), False, False
    # noise_state = reset_noise()
    key = jax.random.PRNGKey(0)


    for i in tqdm.tqdm(range(1, config.training.max_steps + 1),
                       smoothing=0.1,
                       disable=not config.features.tqdm):
        # while not done or not terminate:
            # Update target entropy with decay
            agent = agent.replace(target_entropy=agent.target_entropy - 25e-7)
 
            # breakpoint()
            # Sample action
            if i < config.training.start_training:
                action = env.action_space.sample()
            else:
                action, agent = agent.sample_actions(filter_obs(observation))

            # action = np.clip(action + noise(), -1, 1)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_observation, reward, done, terminate, info = env.step(action)
            # env.render()
            # print(info)
            # breakpoint()
            # Calculate mask
            if not done or not terminate or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0
            # if  "truncated" in info:
            #     mask = 1.0
            # print(done,terminate)
            # Store transition
        
            replay_buffer.insert(
                dict(observations=observation,
                    actions=action,
                    rewards=reward,
                    masks=mask,
                    # infos=info,
                    dones=done,
                    next_observations=next_observation))

            if done or terminate:
                # breakpoint()
                for k, v in info['episode'].items():
                        decode = {'r': 'return', 'l': 'length', 't': 'time'}
                        writer.add_scalar(f'training/{decode[k]}', v, i)
                # writer.add_scalar(f'training/success_rate', np.mean(env.unwrapped.sr_counter), i)
                (observation, info), done = env.reset(), False
                noise.reset()

            observation = next_observation

            # Update agent
            if i >= config.training.start_training:

                batch = next(replay_buffer_iterator)
                agent, update_info = agent.update(batch, utd_ratio=config.training.utd_ratio)

                    # Log training metrics
                if i % config.eval.log_interval == 0:
                    writer.add_scalar('training/target_entropy', 
                                    agent.target_entropy.tolist(), i)
                    for k, v in update_info.items():
                        writer.add_scalar(f'training/{k}', v.tolist(), i)

            # Evaluate agent
            if i % config.eval.eval_interval == 0:
            #     # pass
                eval_info = evaluate(agent,
                        env,
                        num_episodes=config.eval.eval_episodes)
                for k,v in eval_info.items():
                    writer.add_scalar(f'eval/{k}', v, i)
                checkpoints.save_checkpoint(os.path.abspath(policy_folder),
                                        agent,
                                        step=i,
                                        overwrite=True,
                                        keep=1000)


    return agent, env, replay_buffer



# %%

# Setup
config = setup_training()
agent, env, replay_buffer = train_agent(config)

# Save final model
final_checkpoint_dir = "checkpoints/final_model"
os.makedirs(final_checkpoint_dir, exist_ok=True)
checkpoints.save_checkpoint(final_checkpoint_dir, agent,overwrite=True, step=config.training.max_steps)
eval_info = evaluate(agent,
            env,
            num_episodes=config.eval.eval_episodes)
print("EVAL",eval_info)
