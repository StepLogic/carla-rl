#! /usr/bin/env python
from collections import deque
import os
import pickle
import random

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
from jaxrl2.agents import PixelIQLLearner,PixelResNetIQLLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.data.hindsight_replay_buffer import HindsightReplayBuffer
from jaxrl2.evaluation import evaluate
from jaxrl2.wrappers import wrap_pixels
from flax.core.frozen_dict import freeze,unfreeze
import glob
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from gym import spaces
# from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
# from stable_baselines3 import SAC
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from stable_baselines3.common.callbacks import CheckpointCallback,EvalCallback
# from vision_rl.rllib_integration.carla_env import CarlaEnv
# from vision_rl.stb3.jax_experiments import JAXExperiments
import itertools
# from rlib_integration.carla_goal_env import CarlaGoalEnv
# from src.configs.train_env_config import config as carla_config
import flax
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise

from leo.leo_env import LeoEnv
flax.config.update('flax_use_orbax_checkpointing', True)
    # ML config
import jax
from ml_collections.config_dict import config_dict
jax.config.update("jax_debug_nans", True)
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
config.cosine_decay = True
config.tau = 0.005
config.critic_reduction = "min"
config.share_encoder = False
config.freeze_encoders = False
sac_config = config.to_dict()



FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "cheetah-run-v0", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 5, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 100, "Logging interval.")
flags.DEFINE_integer("eval_interval", int(5e4), "Eval interval.")
flags.DEFINE_integer("batch_size", 32, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(5e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(2000), "Number of training steps to start training."
)
flags.DEFINE_integer("image_size", 64, "Image size.")
flags.DEFINE_integer("num_stack", 3, "Stack frames.")
flags.DEFINE_integer(
    "replay_buffer_size", int(1e5), "Number of training steps to start training."
)
flags.DEFINE_integer(
    "action_repeat", None, "Action repeat, if None, uses 2 or PlaNet default values."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("save_buffer", False, "Save the replay buffer.")



def save_checkpoint(agent, path, step):
    os.makedirs(path, exist_ok=True)
    state_dict = {
        'actor_params': agent._actor,
        'critic_params': agent._critic,
        "value_params":agent._value
        # 'target_critic_params': agent._target_critic_params,
        # 'temp': agent._temp,
        # 'rng': agent._rng,
        # Add any other numerical state you need to save
    }
    checkpoints.save_checkpoint(
        ckpt_dir=os.path.abspath(path),
        target=state_dict,
        step=step,
        overwrite=True,
        keep=3  # Keep last 3 checkpoints
    )

import os
import pickle
import time
from datetime import datetime
import gymnasium
import tqdm
from absl import app, flags
import rospy
from typing import Dict, Any

# expert_buffer="/home/kojogyaase/Projects/Research/carla-rl/datasets/goal_condition_Town05_data_0.pkl"
# expert_buffers=list(glob.glob("/workspaces/ROS1/carla-rl/real_robot_dataset/*.pkl"))
expert_buffers=None

checkpoint_path="/workspaces/ROS1/carla-rl/checkpoints/iql_checkpoint/checkpoint_1"
# rb_path="/workspaces/ROS1/carla-rl/savepoint/lane_following_buffer.pkl"
# rb_path="/workspaces/ROS1/carla-rl/savepoint/lane_following_buffer.pkl"
rb_path=None
def load_checkpoint(agent, checkpoint_path):
    """Load agent parameters from checkpoint."""
    state_dict = {
        'actor_params': agent._actor,
        'critic_params': agent._critic,
        "value_params":agent._value
    }
    state_dict = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_path,
        target=state_dict
    )

    # Update agent parameters
    # breakpoint()
    agent._actor = state_dict['actor_params']
    agent._critic = state_dict['critic_params'] 
    agent._value = state_dict['value_params'] 
    # agent._target_critic_params = state_dict['target_critic_params']
    # agent._temp = state_dict['temp']
    # agent._rng = state_dict['rng']
    
    return agent

def main(_):
    rospy.init_node("IQL", anonymous=False)
    # Create environment
    env=LeoEnv()
    env = FrameStack(env=env, num_stack=1,stacking_key="pixels")
    env = TimeLimit(env,max_episode_steps=12500)
    env = RecordEpisodeStatistics(env)
    # action_dim = 2
    # mean = np.zeros(action_dim)
    # sigma = 0.2 * np.ones(action_dim)
    # noise = OrnsteinUhlenbeckActionNoise(mean=mean, sigma=sigma)
  
    # Initialize logger
    logger = Logger(log_dir="./logs",prefix="SAC")

    # Initialize checkpoints dir
    policy_folder = os.path.join("checkpoints", f"model-sac-{len(glob.glob('./logs/*'))}")
    os.makedirs(policy_folder, exist_ok=True)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    # Initialize agent and replay buffer
    agent = PixelResNetIQLLearner(
        0, 
        env.observation_space.sample(), 
        env.action_space.sample(), 
        **sac_config,
    )
    agent = load_checkpoint(agent, checkpoint_path)

    replay_buffer_size = FLAGS.replay_buffer_size
    expert_replay_buffers=[]
    if not expert_buffers is None:
        for path in expert_buffers:
            with open(path, 'rb') as f:
                expert_replay_buffer = pickle.load(f)
            expert_replay_buffers.append(expert_replay_buffer)
    if rb_path:
        with open(rb_path, 'rb') as f:
             replay_buffer = pickle.load(f)
    else:
        replay_buffer = ReplayBuffer(
            env.observation_space, 
            env.action_space, 
            replay_buffer_size
        )


    replay_buffer.seed(FLAGS.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size}
    )
    expert_replay_buffer_iterators=[]
    
    if not expert_buffers is None:
        for expert_replay_buffer in expert_replay_buffers:
            if expert_replay_buffer:
                expert_replay_buffer_iterators.append(expert_replay_buffer.get_iterator(
                        sample_args={"batch_size": FLAGS.batch_size}))
        expert_replay_buffer_iterators=itertools.cycle(expert_replay_buffer_iterators)
    # Track success metrics
    success_history = deque(maxlen=100)  # Track last 100 episodes
    eval_success_history = deque(maxlen=100)

    distance_to_goal_history = deque(maxlen=100)  # Track last 100 episodes
    # eval_distance_to_goal_history = deque(maxlen=100)  # Track last 100 episodes
    # Main training loop
    observation, info, done = *env.reset(), False
    training_start_time = time.time()
    rollout=0
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        if not expert_buffers is None:
                expert_replay_buffer_iterator = next(expert_replay_buffer_iterators)
                for _ in range(2):
                    batch_expert = next(expert_replay_buffer_iterator)
                    batch_expert=unfreeze(batch_expert)
                    batch_expert["actions"]=np.clip(batch_expert["actions"], env.action_space.low, env.action_space.high)
                    batch_expert=freeze(batch_expert)
                    update_info_expert = agent.update(
                        batch_expert)
                if i % FLAGS.log_interval == 0:
                    logger.log_training(update_info, i)
                    logger.print_status(i, FLAGS.max_steps)

        # if i < FLAGS.start_training:
        #     action = env.action_space.sample()
        # else:
        action = agent.sample_actions(observation)
            # if i>int(5e5):
            # action = action + noise()
        action = np.clip(action, env.action_space.low, env.action_space.high)
        next_observation, reward, done, truncated, info = env.step(action)
        rollout+=1
        # Handle episode termination
        if not done or not truncated or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0
            
        # Store transition in replay buffer
        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )
        
        observation = next_observation
        
        # Handle episode completion
        if done or truncated or "TimeLimit.truncated" in info:
            # print(info)
            if "episode" in info:
                # Prepare episode metrics
                episode_info = {
                    "return": info["episode"]["r"],
                    "length": info["episode"]["l"],
                    "time": info["episode"]["t"]
                }
                
                # Track success if available
                if "is_success" in info:
                    success = float(info["is_success"])
                    success_history.append(success)
                    episode_info["is_success"] = success
                    episode_info["success_rate"] = np.mean(success_history)
                if "distance_completed" in info:
                    distance_completed = float(info["distance_completed"])
                    distance_to_goal_history.append(distance_completed)
                    episode_info["distance_completed"] = distance_completed
                    episode_info["distance_completed"] = np.mean(distance_to_goal_history)
                if "slack" in info:
                    episode_info["slack"] = float(info["slack"])
                episode_info.update({
                    "mean_reward":info.get("mean_reward",0),
                    "max_reward":info.get("max_reward",0),
                    "min_reward":info.get("min_reward",0)
                })
                logger.log_episode(episode_info, i)
            observation, info, done = *env.reset(), False
            # noise.reset()
        
        # Training updates
        if i >= FLAGS.start_training and rollout >100:
            rollout=0
            batch = next(replay_buffer_iterator)
            update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                logger.log_training(update_info, i,prefix="_expert")
                logger.print_status(i, FLAGS.max_steps)
            save_checkpoint(agent,f"checkpoints/iql_checkpoint",1)
            # if FLAGS.save_buffer:
            if i % FLAGS.log_interval == 0:
                dataset_folder ="savepoint"
                os.makedirs(dataset_folder, exist_ok=True)
                dataset_file = os.path.join(dataset_folder, f"lane_following_buffer.pkl")
                with open(dataset_file, "wb") as f:
                    pickle.dump(replay_buffer, f)
        # Periodic evaluation
        if i % FLAGS.eval_interval == 0:
            # Save replay buffer if requested
            if FLAGS.save_buffer:
                dataset_folder = os.path.join("datasets")
                os.makedirs(dataset_folder, exist_ok=True)
                dataset_file = os.path.join(dataset_folder, f"img_goal_ds")
                with open(dataset_file, "wb") as f:
                    pickle.dump(replay_buffer, f)
            
            # Run evaluation
            eval_successes = []
            eval_rewards = []
            eval_dists = []
            eval_slack = []
            
            for _ in range(FLAGS.eval_episodes):
                eval_obs, eval_info = env.reset()
                eval_done = False
                episode_reward = 0
                
                while not eval_done:
                    eval_action = agent.eval_actions(eval_obs)  # No exploration
                    eval_obs, eval_reward, eval_done, eval_truncated, eval_info = env.step(eval_action)
                    episode_reward += eval_reward
                    
                    if eval_done or eval_truncated:
                        if "is_success" in eval_info:
                            eval_successes.append(float(eval_info["is_success"]))
                        if "distance_completed" in eval_info:
                            eval_dists.append(float(eval_info["distance_completed"]))
                        if "slack" in eval_info:
                            eval_slack.append(float(eval_info["slack"]))
                
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
            save_checkpoint(agent,policy_folder,i)
            logger.log_eval(eval_info, i)
            logger.print_status(i, FLAGS.max_steps)
    
    # Print final training statistics

    training_duration = time.time() - training_start_time
    print(f"\nTraining completed in {training_duration/3600:.2f} hours")
    print(f"Logs saved to: {logger.log_dir}")

if __name__ == "__main__":
    app.run(main)