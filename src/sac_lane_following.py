#! /usr/bin/env python
from collections import defaultdict, deque
from functools import partial
import itertools
import math
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
from jaxrl2.agents import DrQLearner,PixelResNetDrQLearner
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
# from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
# from stable_baselines3 import SAC
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from stable_baselines3.common.callbacks import CheckpointCallback,EvalCallback
# # from vision_rl.rllib_integration.carla_env import CarlaEnv
# from vision_rl.stb3.jax_experiments import JAXExperiments

from rlib_integration.carla_goal_env import CarlaGoalEnv
from configs.train_env_config import config as carla_config
import flax
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
flax.config.update('flax_use_orbax_checkpointing', True)
    # ML config
import jax
jax.config.update("jax_debug_nans", True)
config = ml_collections.ConfigDict()
config.actor_lr = 3e-4
config.critic_lr = 3e-4
config.temp_lr = 3e-4
config.hidden_dims = (256, 256)
config.cnn_features = (8, 16, 32, 32)
config.cnn_filters = (3, 3, 3, 3)
config.cnn_strides = (2, 2, 2, 2)
config.cnn_padding = "VALID"
config.latent_dim = 50
config.encoder = "d4pg"
config.discount = 0.997
config.tau = 0.005
config.init_temperature = 1.0
# config.target_entropy = 0.1
config.backup_entropy = True
# config.num_qs=10
config.critic_reduction = "mean"
sac_config = config.to_dict()



FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "cheetah-run-v0", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 5, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", int(5e4), "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(2e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e3), "Number of training steps to start training."
)
flags.DEFINE_integer("image_size", 64, "Image size.")
flags.DEFINE_integer("num_stack", 3, "Stack frames.")
flags.DEFINE_integer(
    "replay_buffer_size", int(3e5), "Number of training steps to start training."
)
flags.DEFINE_integer(
    "action_repeat", None, "Action repeat, if None, uses 2 or PlaNet default values."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("save_buffer", False, "Save the replay buffer.")

flags.DEFINE_string("checkpoint_path", "/home/robotlab/scratch/carla-rl/best_models/model-sac-3/checkpoint_750000", "Save the replay buffer.")



def save_checkpoint(agent, path, step):
    os.makedirs(path, exist_ok=True)
    state_dict = {
        'actor_params': agent._actor,
        'critic_params': agent._critic,
        'target_critic_params': agent._target_critic_params,
        'temp': agent._temp,
        'rng': agent._rng,
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

from typing import Dict, Any

# expert_buffer="/home/kojogyaase/Projects/Research/carla-rl/datasets/goal_condition_Town05_data_0.pkl"
# expert_buffers=list(glob.glob("/home/robotlab/scratch/carla-rl/datasets/*.pkl"))
# expert_buffers=list(glob.glob("/home/robotlab/scratch/carla-rl/datasets/*.pkl"))
expert_buffer=None


def load_checkpoint(agent, checkpoint_path):
    """Load agent parameters from checkpoint."""
    state_dict = {
        'actor_params': agent._actor,
        'critic_params': agent._critic,
        'target_critic_params': agent._target_critic_params,
        'temp': agent._temp,
        'rng': agent._rng,
        # Add any other numerical state you need to save
    }
    state_dict = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_path,
        target=state_dict
    )

    # Update agent parameters
    # breakpoint()
    agent._actor = state_dict['actor_params']
    agent._critic = state_dict['critic_params'] 
    agent._target_critic_params = state_dict['target_critic_params']
    agent._temp = state_dict['temp']
    agent._rng = state_dict['rng']
    
    return agent
def main(_):

    # Create environment
    # carla_config["env_config"]["carla"]["town"]="Town01"
    # # carla_config["env_config"]["carla"]["start_server"]=False
    # env = CarlaGoalEnv(carla_config["env_config"])
    # env = FrameStack(env=env, num_stack=1,stacking_key="pixels")
    # env = TimeLimit(env,max_episode_steps=2500)
    # env = RecordEpisodeStatistics(env)
    env=None
    # _towns=['Town04',"Town03","Town01"]
    _towns=["Town07","Town06","Town15","Town01"]
    towns=itertools.cycle(_towns)
    carla_config["env_config"]["carla"]["town"]="Town01"
    env = CarlaGoalEnv(carla_config["env_config"])
    env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
    env = TimeLimit(env, max_episode_steps=1500)
    env = RecordEpisodeStatistics(env)
    def reset_env(eval_town=False):
        return  env
    #     nonlocal env
    #     if not env is None:
    #          env.close()
    #     if eval_town:
    #         carla_config["env_config"]["carla"]["town"]="Town02"
    #     else:    
    #         carla_config["env_config"]["carla"]["town"]=next(towns)
    #     # config["env_config"]["carla"]["start_server"]=False
    #     env = CarlaGoalEnv(carla_config["env_config"])
    #     env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
    #     env = TimeLimit(env, max_episode_steps=1500)
    #     env = RecordEpisodeStatistics(env)
    #     return env
    action_dim = 2
    mean = np.zeros(action_dim)
    sigma = .3* np.ones(action_dim)
    noise = OrnsteinUhlenbeckActionNoise(mean=mean, sigma=sigma)
    timelimit=10
    env=reset_env()
    # Initialize logger
    logger = Logger(log_dir="./logs",prefix="SAC")

    # Initialize checkpoints dir
    policy_folder = os.path.join("checkpoints", f"model-sac-{len(glob.glob('./logs/*'))}")
    os.makedirs(policy_folder, exist_ok=True)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    # Initialize agent and replay buffer
    agent = DrQLearner(
        0, 
        env.observation_space.sample(), 
        env.action_space.sample(), 
        # num_qs=10,
        **sac_config
    )
    # agent=load_checkpoint(agent,FLAGS.checkpoint_path)
    replay_buffer_size = FLAGS.replay_buffer_size
    # expert_replay_buffers=[]
    # if not expert_buffers is None:
    #     for path in expert_buffers:
    # expert_buffer="/home/robotlab/scratch/carla-rl/datasets/goal_condition_Town01_data_1.pkl"
    # with open(expert_buffer, 'rb') as f:
    #     expert_buffer = pickle.load(f)
    #     replay_buffer.freeze()
            # expert_replay_buffers.append(expert_replay_buffer)
    
    replay_buffer = ReplayBuffer(
        env.observation_space, 
        env.action_space, 
        replay_buffer_size
    )

    replay_buffer.seed(FLAGS.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": int(FLAGS.batch_size/4)}
    )
    # expert_replay_buffer_iterators=[]
    # expert_buffer_iterator=None
    # if not expert_buffer is None:
    # #     for expert_replay_buffer in expert_replay_buffers:
    # #         # expert_replay_buffer.optimize()
    # #         expert_replay_buffer_iterators.append(expert_replay_buffer.get_iterator(
    # #                 sample_args={"batch_size": FLAGS.batch_size}))
    #     # expert_replay_buffer_iterators=itertools.cycle(expert_replay_buffer_iterators)
    #     expert_buffer_iterator=expert_buffer.get_iterator(
    #                 sample_args={"batch_size": FLAGS.batch_size})
    # Track success metrics
    success_history = deque(maxlen=100)  # Track last 100 episodes
    eval_success_history = deque(maxlen=100)

    distance_to_goal_history = deque(maxlen=100)  # Track last 100 episodes
    eval_distance_to_goal_history = deque(maxlen=100)  # Track last 100 episodes
    aux_data = defaultdict(lambda:deque(maxlen=2500))  # Track last 100 episodes
    # Main training loop
    observation, info, done = *env.reset(), False
    training_start_time = time.time()
    episodes_per_environment=0
    # save_checkpoint(agent,policy_folder,1)
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
            # action = action + noise()
            action = np.clip(action, env.action_space.low, env.action_space.high)
        next_observation, reward, done, truncated, info = env.step(action)
        
        # Handle episode termination
        if not done or not truncated or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0
        for k,v in info.items():
            if isinstance(v,(int,float)): 
                aux_data[k].append(v)
            
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
            episode_info=dict()
            episode_info.update({
            k:np.mean(v)  for k,v in aux_data.items()
            })
            if "episode" in info:
                # Prepare episode metrics
                episode_info .update({
                    "return": info["episode"]["r"],
                    "length": info["episode"]["l"],
                    "time": info["episode"]["t"]
                })
                
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
            
                # print([list(v) for k,v in aux_data.items()])
                logger.log_episode(episode_info, i)
                if episodes_per_environment > 0 and episodes_per_environment % 100 == 0:
                    if i> int(3e5):
                        timelimit+=1000
                    else:
                        timelimit+=100
                    episodes_per_environment = 1
                    
                    max_attempts = 50
                    for attempt in range(max_attempts):
                        try:
                            env = reset_env()
                            
                            break  # Exit the loop if successful
                        except Exception as e:
                            if attempt == max_attempts - 1:
                                raise  # Re-raise the exception if all attempts failed
                            time.sleep(0.1)  # Small delay before retry
                else:
                    episodes_per_environment += 1
            observation, info, done = *env.reset(), False
            noise.reset()

        # expert_replay_buffer_iterators=itertools.cycle(expert_replay_buffer_iterators)
        # Training updates
        if i >= FLAGS.start_training:
            # for i in 
            batch = next(replay_buffer_iterator)
            update_info = agent.update(batch) #prevent entropy from dying too quickly

            # if not expert_buffer_iterator is None:
            # #     # for expert_replay_buffer_iterator in expert_replay_buffer_iterators:
            # #     # expert_replay_buffer_iterator=next(expert_replay_buffer_iterators)
            #     batch_expert = next(expert_buffer_iterator)
            #     update_info_expert = agent.update(
            #         batch_expert,
            #         enable_update_temperature=False,utd_ratio=8)
            #     if i % FLAGS.log_interval == 0:
            #         logger.log_training(update_info_expert, i,prefix="_expert")
            if i % FLAGS.log_interval == 0:
                logger.log_training(update_info, i)
                logger.print_status(i, FLAGS.max_steps)
            # logger.print_status(i, FLAGS.max_steps)
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
            env=reset_env(True)
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
            env=reset_env()
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
    save_checkpoint(agent,f"checkpoints/final_drq",1)
    # if FLAGS.save_buffer:
    dataset_folder ="datasets"
    os.makedirs(dataset_folder, exist_ok=True)
    dataset_file = os.path.join(dataset_folder, f"lane_following_buffer")
    with open(dataset_file, "wb") as f:
        pickle.dump(replay_buffer, f)
    training_duration = time.time() - training_start_time
    print(f"\nTraining completed in {training_duration/3600:.2f} hours")
    print(f"Logs saved to: {logger.log_dir}")

if __name__ == "__main__":
    app.run(main)