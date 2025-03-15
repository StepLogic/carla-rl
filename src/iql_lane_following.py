#! /usr/bin/env python
from collections import defaultdict, deque
import itertools
import math
import os
import pickle
import random

# import gym
import gymnasium as gym
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
config.A_scaling = 10.0
config.dropout_rate = 0.5
config.cosine_decay = True
config.tau = 0.005
config.critic_reduction = "min"
config.share_encoder = False
sac_config = config.to_dict()



FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "cheetah-run-v0", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 5, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", int(1), "Eval interval.")
flags.DEFINE_integer("batch_size", 16, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(70), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e3), "Number of training steps to start training."
)
flags.DEFINE_integer("image_size", 64, "Image size.")
flags.DEFINE_integer("num_stack", 3, "Stack frames.")
flags.DEFINE_integer(
    "replay_buffer_size", int(1e3), "Number of training steps to start training."
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
        "value_params":agent._value,
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

# expert_buffer="/home/kojogyaase/Projects/Research/carla-rl/datasets/goal_condition_Town05_data_0.pkl"
expert_buffers=list(glob.glob("/home/kojogyaase/Projects/Research/carla-rl/datasets/*.pkl"))
# print(expert_buffers)
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

    logger = Logger(log_dir="./logs",prefix="SAC")
    # Initialize checkpoints dir
    policy_folder = os.path.join("checkpoints", f"model-sac-{len(glob.glob('./logs/*'))}")
    os.makedirs(policy_folder, exist_ok=True)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    action_space,observation_space=initialize_spaces()
    # Initialize agent and replay buffer
    agent = PixelResNetIQLLearner(
        0, 
        observation_space.sample(), 
        action_space.sample(), 
        # num_qs=10,
        **sac_config
    )

    expert_replay_buffers=[]
    if not expert_buffers is None:
        for path in expert_buffers:
            with open(path, 'rb') as f:
                expert_replay_buffer = pickle.load(f)
            expert_replay_buffers.append(expert_replay_buffer)
    # breakpoint()
    expert_replay_buffer_iterators=[]
    if not expert_buffers is None:
        for expert_replay_buffer in expert_replay_buffers:
            if expert_replay_buffer:
                # expert_replay_buffer.optimize()
                # breakpoint()
                expert_replay_buffer_iterators.append(expert_replay_buffer.get_iterator(
                        sample_args={"batch_size": FLAGS.batch_size}))

    training_start_time = time.time()
    
    p_bar = tqdm.tqdm(range(1,FLAGS.max_steps + 1))
    p_bar.update(5)
    p_bar.refresh()
    

    i=1
    run_eval=False
    expert_replay_buffers=itertools.cycle(expert_replay_buffers)
    while i <  FLAGS.max_steps + 1:
        if not expert_buffers is None:
            total_metrics = defaultdict(list)
            expert_replay_buffer = next(expert_replay_buffers)
            expert_replay_buffer_iterator=expert_replay_buffer.get_sequential_iterator(sample_args={"batch_size": FLAGS.batch_size})
                
            epoch_bar = tqdm.tqdm(
                total=math.ceil(expert_replay_buffer._size / FLAGS.batch_size),  # Total number of batches
                desc="Epoch Progress",  # Description for the progress bar
                dynamic_ncols=True,  # Adjust progress bar width to the terminal
            )

            # Dictionary to store metrics
            total_metrics = defaultdict(list)

            try:
                # Iterate over the expert replay buffer
                # with cProfile.Profile() as pr:

                    for ix, batch_expert in enumerate(expert_replay_buffer_iterator):
                        # Update the agent with the current batch
                        update_info_expert = agent.update(batch_expert)
                        
                        # Log metrics
                        for key, value in update_info_expert.items():
                            total_metrics[key].append(float(value))
                        
                        # Update the progress bar by 1 step
                        epoch_bar.update(1)
                        
                        # Optionally, display metrics in the progress bar
                        epoch_bar.set_postfix({key: f"{np.mean(value):.4f}" for key, value in total_metrics.items()})
                        # ... do something ...

                        # pr.dump_stats("sample.prof")        
                        # exit(0)
                    
            finally:
                # Ensure the progress bar is closed even if an error occurs
                epoch_bar.close()
            
            average_metrics = {
                key: np.mean(value)  
                for key, value in total_metrics.items()
            }
            # print(average_metrics,total_metrics)
            logger.log_training(average_metrics, i,prefix="_expert")
            # logger.log_training(update_info_expert, i,prefix="_expert")
            i+=1
            p_bar.n = i  
            # breakpoint()   
            # if i%FLAGS.en
            if i % FLAGS.eval_interval == 0:
                save_checkpoint(agent,f"checkpoints/final_iql",1)
            logger.print_status(i, FLAGS.max_steps)

        
    
    # Print final training statistics
    save_checkpoint(agent,f"checkpoints/final_iql",1)
    training_duration = time.time() - training_start_time
    print(f"\nTraining completed in {training_duration/3600:.2f} hours")
    print(f"Logs saved to: {logger.log_dir}")

if __name__ == "__main__":
    app.run(main)