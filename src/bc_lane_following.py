#! /usr/bin/env python
from collections import defaultdict, deque
from copy import copy
import itertools
import math
import os
import pickle
import random

import gym
import gymnasium
# from jaxrl2.agents.pixel_bc.pixel_bc_learner import PixelBCLearner
from jaxrl2.agents.resnet_agents import PixelResNetBCLearner
from jaxrl2.utils.misc import Logger
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.timelimit import TimeLimit
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
import ml_collections
import optax
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

# from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
# from vision_rl.rllib_integration.carla_env import CarlaEnv
# from vision_rl.stb3.jax_experiments import JAXExperiments
from rlib_integration.carla_goal_env import CarlaGoalEnv
from src.configs.train_env_config import config as carla_config
import flax
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
import cProfile
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
# config.dropout_rate=0.5
bc_config = config.to_dict()



FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "cheetah-run-v0", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 5, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", int(1), "Eval interval.")
flags.DEFINE_integer("batch_size", 64, "Mini batch size.")
flags.DEFINE_integer("epochs", int(70), "Number of training steps.")
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
class EarlyStopping:
    """
    Early stopping handler to monitor training progress and stop when evaluation
    metric doesn't improve for a specified number of epochs.
    """
    def __init__(self, patience=4, mode='max', min_delta=0.0):
        """
        Initialize the EarlyStopping handler.
        
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            mode (str): 'min' or 'max' depending on whether we want to minimize or maximize the metric.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, current_score, epoch):
        """
        Call the early stopping handler.
        
        Args:
            current_score (float): Current value of the metric being monitored.
            epoch (int): Current epoch number.
            
        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            # For metrics we want to minimize (like loss)
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1
        else:
            # For metrics we want to maximize (like accuracy, reward)
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False


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


def update(expert_replay_buffers,agent,train_encoder,logger,i,update_func=None,prefix="_expert"):


    if not expert_buffers is None:
        total_metrics = defaultdict(list)
        expert_replay_buffer = next(expert_replay_buffers)
        expert_replay_buffer_iterator=expert_replay_buffer.get_sequential_iterator(sample_args={"batch_size": FLAGS.batch_size})
            
        epoch_bar = tqdm.tqdm(
            total=math.ceil(expert_replay_buffer._size / FLAGS.batch_size),  # Total number of batches
            desc=f"{prefix} Epoch Progress",  # Description for the progress bar
            dynamic_ncols=True,  # Adjust progress bar width to the terminal
        )

        # Dictionary to store metrics
        total_metrics = defaultdict(list)

        try:
            # Iterate over the expert replay buffer
            # with cProfile.Profile() as pr:

                for ix, batch_expert in enumerate(expert_replay_buffer_iterator):
                    # Update the agent with the current batch
                    update_info_expert = update_func(batch_expert,train_encoder=train_encoder)
                    
                    # Log metrics
                    extra={}
                    for key, value in update_info_expert.items():
                        # if isinstance(value,(int,float)):
                        total_metrics[key].append(float(value))
                        # else:
                            # extra[key]=value

                    print_dict={key: f"{np.mean(value):.4f}" for key, value in total_metrics.items()}
                    print_dict.update(extra)
                    # Update the progress bar by 1 step
                    epoch_bar.update(1)
                    
                    # Optionally, display metrics in the progress bar
                    epoch_bar.set_postfix(print_dict)
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
        logger.log_training(average_metrics, i,prefix=prefix)

def main(_):
    # Create environment
    # carla_config["env_config"]["carla"]["town"]="Town04"
    # carla_config["env_config"]["carla"]["start_server"]=False
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

    # Initialize agent
    agent = PixelResNetBCLearner(
        0, 
        env.observation_space.sample(), 
        env.action_space.sample(), 
        **bc_config
    )
    
    # Initialize expert replay buffers
    expert_replay_buffers=[]
    if not expert_buffers is None:
        for path in expert_buffers:
            with open(path, 'rb') as f:
                expert_replay_buffer = pickle.load(f)
            expert_replay_buffers.append(expert_replay_buffer)
    
    # Initialize early stopping with patience of 4 epochs
    early_stopper = EarlyStopping(patience=7, mode='max')  # Using 'max' since higher reward is better
    best_model_saved = False

    training_start_time = time.time()
    
    p_bar = tqdm.tqdm(range(1, FLAGS.epochs + 1))
    p_bar.update(5)
    p_bar.refresh()
    
    i = 1
    train_encoder = True
    expert_replay_buffers = itertools.cycle(expert_replay_buffers)
    
    # Training loop
    while i < FLAGS.epochs + 1:
        # Update on expert data
        update(expert_replay_buffers, agent, train_encoder, logger, i, update_func=agent.update, prefix="_expert")

        # Save checkpoint periodically
        if i % FLAGS.eval_interval == 0:
            save_checkpoint(agent, policy_folder, i)
        
        # Run evaluation every eval_interval epochs
        if i % FLAGS.eval_interval == 0:
            eval_successes = []
            eval_rewards = []
            eval_dists = []
            eval_slack = []
            
            for _ in range(FLAGS.eval_episodes):
                eval_obs, eval_info = env.reset()
                eval_done = False
                episode_reward = 0
                
                while not eval_done:
                    eval_action = agent.eval_actions(eval_obs)
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
            
            # Log evaluation results
            mean_eval_reward = np.mean(eval_rewards) if eval_rewards else 0
            logger.log_eval(eval_info, i)
            logger.print_status(i, FLAGS.epochs)
            
            # Add early stopping check with current evaluation reward
            if early_stopper(mean_eval_reward, i):
                print(f"\nEarly stopping triggered after {i} epochs. Best performance at epoch {early_stopper.best_epoch}")
                
                # Save the best model if not already saved
                if not best_model_saved:
                    save_checkpoint(agent, f"checkpoints/best_model_early_stopped", early_stopper.best_epoch)
                    best_model_saved = True
                break
            
            # Save best model so far
            if early_stopper.best_epoch == i:
                save_checkpoint(agent, f"checkpoints/best_model", i)
                best_model_saved = True
        
        i += 1
        p_bar.update(1)
        p_bar.refresh()
    
    # Save final model
    save_checkpoint(agent, f"checkpoints/final_bc", i)
    
    training_duration = time.time() - training_start_time
    print(f"\nTraining completed in {training_duration/3600:.2f} hours")
    print(f"Logs saved to: {logger.log_dir}")
    print(f"Best model saved at epoch {early_stopper.best_epoch}")

if __name__ == "__main__":
    app.run(main)