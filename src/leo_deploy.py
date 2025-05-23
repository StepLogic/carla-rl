#! /usr/bin/env python
from collections import deque
import os
import pickle
import random
import sys

import gym
import gymnasium
from jaxrl2.utils.misc import Logger
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.timelimit import TimeLimit
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
import ml_collections
import tqdm
# import wandb
from absl import app, flags
from ml_collections import config_flags
from flax.training import checkpoints
import jaxrl2.extra_envs.dm_control_suite
from jaxrl2.agents import PixelIQLLearner,PixelResNetIQLLearner,PixelResNetBCLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.data.hindsight_replay_buffer import HindsightReplayBuffer
from jaxrl2.evaluation import evaluate
from jaxrl2.wrappers import wrap_pixels
from flax.core.frozen_dict import freeze
import glob
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from gym import spaces
import time
# from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
# from stable_baselines3 import SAC
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from stable_baselines3.common.callbacks import CheckpointCallback,EvalCallback
# from vision_rl.rllib_integration.carla_env import CarlaEnv
# from vision_rl.stb3.jax_experiments import JAXExperiments

# from rlib_integration.carla_goal_env import CarlaGoalEnv
# from configs.train_env_config import config as carla_config
import flax
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
import rospy
from leo_env import LeoEnv
flax.config.update('flax_use_orbax_checkpointing', True)
    # ML config
import jax
from ml_collections.config_dict import config_dict
jax.config.update("jax_debug_nans", True)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

config = ml_collections.ConfigDict()
config.actor_lr = 3e-4
config.hidden_dims = (256, 256)
config.cnn_features = (32, 64, 128, 256)
config.cnn_filters = (3, 3, 3, 3)
config.cnn_strides = (2, 2, 2, 2)
config.cnn_padding = "VALID"
config.latent_dim = 50
config.encoder = "d4pg"
config.dropout_rate=0.2
bc_config = config.to_dict()

config = ml_collections.ConfigDict()
config.actor_lr = 3e-4
config.critic_lr = 3e-4
config.value_lr = 3e-4
config.hidden_dims = (256, 256)
config.cnn_features = (32, 64, 128, 256)
config.cnn_filters = (3, 3, 3, 3)
config.cnn_strides = (2, 2, 2, 2)
config.cnn_padding = "VALID"
config.latent_dim = 50
config.discount = 0.99
config.expectile = 0.7  # The actual tau for expectiles.
config.A_scaling = 3.0
config.dropout_rate = config_dict.placeholder(float)
# config.cosine_decay = True
config.tau = 0.005
config.critic_reduction = "min"
config.share_encoder = False
sac_config = config.to_dict()
checkpoint_path="/workspaces/carla-rl/checkpoints/leo_bc/checkpoint_70"



def load_checkpoint(agent, checkpoint_path):
    """Load agent parameters from checkpoint."""
    state_dict = {
        'actor_params': agent._actor,
    }
    state_dict = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_path,
        target=state_dict
    )

    agent._actor = state_dict['actor_params']
    
    return agent


image_size=64
def initialize_spaces():
    """Initialize the replay buffer with proper spaces"""
    image_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(image_size,image_size,3,1),
        dtype=np.float32,
    )
    vec_space = gym.spaces.Box(
        low=-5.1,
        high=5.1,
        shape=(4,),
        dtype=np.float32,
    )
    action_space = gym.spaces.Box(
        low=np.array([-1.0, -1.0]),
        high=np.array([1.0, 1.0]),
        dtype=np.float32
    )
    observation_space = gym.spaces.Dict({"pixels": image_space, "vector": vec_space})
    return action_space,observation_space
def main(_):

    # Create environment
    # carla_config["env_config"]["carla"]["town"]="Town07"
    heading=np.pi

    action_space,observation_space=initialize_spaces()
    # Initialize agent and replay buffer
    agent = PixelResNetBCLearner(
        0, 
        observation_space.sample(), 
        action_space.sample(), 
        # num_qs=10,
        **bc_config
    )
    rospy.init_node("SAC_TRAINING", anonymous=False)
    env = LeoEnv(target_heading=np.deg2rad(90))
    rate = rospy.Rate(env.RATE)
    env = FrameStack(env=env, num_stack=1,stacking_key="pixels")
    env = TimeLimit(env,max_episode_steps=12000)
    env = RecordEpisodeStatistics(env)
    # logger = Logger(log_dir="./logs",prefix="SAC")
    env.unwrapped.target_speed=100
    np.random.seed(42)
    random.seed(42)
    agent = load_checkpoint(agent, checkpoint_path)

    # Track success metrics
    success_history = deque(maxlen=100)  # Track last 100 episodes
    eval_success_history = deque(maxlen=100)

    distance_to_goal_history = deque(maxlen=100)  # Track last 100 episodes
    eval_distance_to_goal_history = deque(maxlen=100)  # Track last 100 episodes
    # Main training loop
    eval_obs, info, done = *env.reset(), False
    training_start_time = time.time()

    # Run evaluation
    eval_successes = []
    eval_rewards = []
    eval_dists = []
    eval_slack = []
    headings=[]
    speeds=[]
    n_eval_episodes=10
    for _ in range(n_eval_episodes):
        eval_obs, eval_info = env.reset()
        eval_done = False
        episode_reward = 0
        
        while not eval_done:
            target=4.5
            vecs=eval_obs["vector"]
            current_velocity=env.unwrapped.speed
            current_heading=env.unwrapped.current_heading
            headings.append(current_heading)
            speeds.append(current_velocity)
            # breakpoint()
            # vecs[2] = np.clip(current_velocity/(target+1e-8), 0.0, 1.1)
            # print(vecs[2])
            # vecs[3]= np.cos(abs(current_heading-heading))
            # vecs[4]= np.clip(heading/np.pi,-5.1,5.1) 
            eval_obs["vector"]=vecs
            eval_action = agent.sample_actions(eval_obs)  # No exploration
            print(eval_action)
            eval_obs, eval_reward, eval_done, eval_truncated, eval_info = env.step(eval_action)
            episode_reward += eval_reward
            
            if eval_done or eval_truncated:
                if "is_success" in eval_info or "TimeLimit.truncated" in eval_info:
                    eval_successes.append(float(eval_info["is_success"]))
                if "distance_completed" in eval_info:
                    eval_dists.append(float(eval_info["distance_completed"]))
                if "slack" in eval_info:
                    eval_slack.append(float(eval_info["slack"]))
            print(f"\rCurrent heading {np.rad2deg(current_heading)} Target Heading {np.rad2deg(heading)} {current_velocity}", end="")
            sys.stdout.flush()

        eval_rewards.append(episode_reward)
        
    # Compute evaluation metrics
    eval_info = {
        "eval_reward_mean": np.mean(eval_rewards),
        "eval_reward_std": np.std(eval_rewards)
    }
    
    if len(eval_successes)>0:
        success_rate = np.mean(eval_successes)
        eval_success_history.append(success_rate)
        eval_info["success_rate"] = success_rate
        eval_info["avg_success_rate"] = np.mean(eval_success_history)
        eval_info["distance_completed"] = np.mean(eval_dists)
        eval_info["slack"] = np.mean(eval_slack)
    # save_checkpoint(agent,policy_folder,i)
    # logger.log_eval(eval_info, i)
    # logger.print_status(i, FLAGS.max_steps)
    print(eval_info)
    training_duration = time.time() - training_start_time
    print(f"\nTraining completed in {training_duration/3600:.2f} hours")
    with open(f"bc_test_results_{random.randint(0,int(1e5))}.pkl", "wb") as f:
        pickle.dump(dict(eval_info), f)
    # print(f"Logs saved to: {logger.log_dir}")

if __name__ == "__main__":
    app.run(main)