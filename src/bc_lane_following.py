#! /usr/bin/env python
from collections import defaultdict, deque
from copy import copy
import itertools
import os
import pickle
import random

import gym
import gymnasium
from jaxrl2.agents.pixel_bc.pixel_bc_learner import PixelBCLearner
from jaxrl2.agents.pixel_bc_resnet.pixel_bc_resnet_learner import PixelResNetBCLearner
from jaxrl2.utils.misc import Logger
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.timelimit import TimeLimit
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
import ml_collections
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
from flax.training import checkpoints
import jaxrl2.extra_envs.dm_control_suite
from jaxrl2.agents import DrQLearner
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
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback,EvalCallback
# from vision_rl.rllib_integration.carla_env import CarlaEnv
# from vision_rl.stb3.jax_experiments import JAXExperiments
from rlib_integration.carla_goal_env import CarlaGoalEnv
from src.configs.train_env_config import config as carla_config
import flax
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
flax.config.update('flax_use_orbax_checkpointing', True)

import jax
jax.config.update("jax_debug_nans", True)
    # ML config
config = ml_collections.ConfigDict()
config.actor_lr = 3e-4
config.hidden_dims = (256, 256)
config.cnn_features = (32, 64, 128, 256)
config.cnn_filters = (3, 3, 3, 3)
config.cnn_strides = (2, 2, 2, 2)
config.cnn_padding = "VALID"
config.latent_dim = 50
config.encoder = "pretrained-resnet"
bc_config = config.to_dict()



FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "cheetah-run-v0", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 5, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", int(10), "Eval interval.")
flags.DEFINE_integer("batch_size", 32, "Mini batch size.")
flags.DEFINE_integer("epochs", int(5e4), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e3), "Number of training steps to start training."
)
flags.DEFINE_integer("image_size", 64, "Image size.")
flags.DEFINE_integer("num_stack", 3, "Stack frames.")
# flags.DEFINE_integer(
#     "replay_buffer_size", int(1e6), "Number of training steps to start training."
# )
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
        # 'critic_params': agent._critic,
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

from typing import Dict, Any

# expert_buffer="/home/kojogyaase/Projects/Research/carla-rl/datasets/basic_agent_data_20241229_093438.pkl"
expert_buffers=list(glob.glob("/home/robotlab/scratch/carla-rl/datasets/*.pkl"))
# expert_buffers=list(glob.glob("/home/kojogyaase/Projects/Research/carla-rl/datasets/*.pkl"))
def sample_from_buffers():
    pass
def main(_):
    # Create environment
    carla_config["env_config"]["carla"]["start_server"]=True
    env = CarlaGoalEnv(carla_config["env_config"])
    env = FrameStack(env=env, num_stack=1,stacking_key="pixels")
    env = TimeLimit(env,max_episode_steps=2500)
    env = RecordEpisodeStatistics(env)
    action_dim = 2
    mean = np.zeros(action_dim)
    sigma = 0.2 * np.ones(action_dim)
    noise = OrnsteinUhlenbeckActionNoise(mean=mean, sigma=sigma)
  
    # Initialize logger
    logger = Logger(log_dir="./logs",prefix="BC")

    # Initialize checkpoints dir
    policy_folder = os.path.join("checkpoints", f"model-sac-{len(glob.glob('./logs/*'))}")
    os.makedirs(policy_folder, exist_ok=True)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    # breakpoint()
    # Initialize agent and replay buffer
    agent = PixelResNetBCLearner(
        0, 
        env.observation_space.sample(), 
        env.action_space.sample(), 
        # num_qs=10,
        **bc_config
    )
    expert_replay_buffers=[]
    if not expert_buffers is None:
        for path in expert_buffers:
            with open(path, 'rb') as f:
                expert_replay_buffer = pickle.load(f)
                # expert_replay_buffer.optimize()
            expert_replay_buffers.append(expert_replay_buffer)
    # breakpoint()
    expert_replay_buffer_iterators=[]
    # if not expert_buffers is None:
    #     for expert_replay_buffer in expert_replay_buffers:
    #         if expert_replay_buffer:
    #             # expert_replay_buffer.optimize()
    #             expert_replay_buffer_iterators.append(expert_replay_buffer.get_sequential_iterator(
    #                     sample_args={"batch_size": FLAGS.batch_size}))
            

    training_start_time = time.time()
    
    p_bar = tqdm.tqdm(range(1,FLAGS.epochs + 1))
    p_bar.update(5)
    p_bar.refresh()
    

    i=1
    run_eval=False
    # expert_replay_buffer_iterators=itertools.cycle(expert_replay_buffer_iterators)
    expert_replay_buffers=itertools.cycle(expert_replay_buffers)
    while i <  FLAGS.epochs + 1:
        if not expert_buffers is None:
            total_metrics = defaultdict(list)
            expert_replay_buffer = next(expert_replay_buffers)
            expert_replay_buffer_iterator=expert_replay_buffer.get_sequential_iterator(sample_args={"batch_size": FLAGS.batch_size})
            for batch_expert in expert_replay_buffer_iterator:
                update_info_expert = agent.update(
                    batch_expert)
                for key, value in update_info_expert.items():
                        total_metrics[key].append(float(value))
                        # Calculate averages for each metric
                # print(total_metrics)
            average_metrics = {
                key: np.mean(value)  
                for key, value in total_metrics.items()
            }
            # print(average_metrics,total_metrics)
            logger.log_training(average_metrics, i,prefix="_expert")
            # logger.log_training(update_info_expert, i,prefix="_expert")
            i+=1
            p_bar.n = i  
            if i % FLAGS.eval_interval == 0:
                    save_checkpoint(agent,policy_folder,i)
            logger.print_status(i, FLAGS.epochs)

        if i % FLAGS.eval_interval == 0:
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
                # try:
                    eval_action = agent.eval_actions(eval_obs)  # No exploration
                    # eval_action=jax
                    # print(eval_action)
                    # except:
                    #     pass
                    eval_obs, eval_reward, eval_done, eval_truncated, eval_info = env.step(eval_action)
                    episode_reward += eval_reward
                    
                    if eval_done or eval_truncated:
                        if "is_success" in eval_info:
                            eval_successes.append(float(eval_info["is_success"]))
                        if "distance_completed" in eval_info:
                            eval_dists.append(float(eval_info["distance_completed"]))
                        if "slack" in eval_info:
                            eval_slack.append(float(eval_info["slack"]))
                        if "episode" in eval_info:
                            eval_info.update(eval_info["episode"])
                            del eval_info["episode"]
                
                eval_rewards.append(episode_reward)
            # save_checkpoint(agent,policy_folder,i)
            # print(eval_info)
            logger.log_eval(eval_info, i)
            logger.print_status(i, FLAGS.epochs)
        
    
    # Print final training statistics
    save_checkpoint(agent,f"checkpoints/final_iql",1)
    training_duration = time.time() - training_start_time
    print(f"\nTraining completed in {training_duration/3600:.2f} hours")
    print(f"Logs saved to: {logger.log_dir}")

if __name__ == "__main__":
    app.run(main)