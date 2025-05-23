#! /usr/bin/env python
from collections import deque
import os
import pickle
import random
import glob
import time

import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import ml_collections
import jax
import flax
import rospy
import copy
from jaxrl2.utils.misc import Logger
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.timelimit import TimeLimit
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.agents import DrQLearner,PixelResNetDrQLearner
from jaxrl2.data import ReplayBuffer

from leo_env import LeoEnv

# Environment setup
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
flax.config.update('flax_use_orbax_checkpointing', True)
jax.config.update("jax_debug_nans", True)

# SAC configuration
config = ml_collections.ConfigDict()
config.actor_lr = 3e-4
config.critic_lr = 3e-4
config.temp_lr = 3e-4
config.hidden_dims = (256, 256)
config.cnn_features = (32, 64, 128, 256)
config.cnn_filters = (3, 3, 3, 3)
config.cnn_strides = (2, 2, 2, 2)
config.cnn_padding = "VALID"
config.latent_dim = 50
config.encoder = "d4pg"
config.discount = 0.98
config.tau = 0.005
config.init_temperature = 1.0
config.target_entropy = None
config.backup_entropy = True
config.num_qs=10
config.critic_reduction = "mean"
sac_config = config.to_dict()

# Command line flags
FLAGS = flags.FLAGS
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir")
flags.DEFINE_integer("seed", 42, "Random seed")
flags.DEFINE_integer("eval_episodes", 5, "Number of episodes used for evaluation")
flags.DEFINE_integer("log_interval", 1000, "Logging interval")
flags.DEFINE_integer("eval_interval", int(5e4), "Eval interval")
flags.DEFINE_integer("batch_size", 32, "Mini batch size")
flags.DEFINE_integer("max_steps", int(5e6), "Number of training steps")
flags.DEFINE_integer("start_training", int(1e3), "Steps before starting training")
flags.DEFINE_integer("replay_buffer_size", int(5e4), "Replay buffer size")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar")
flags.DEFINE_boolean("save_buffer", False, "Save the replay buffer")
flags.DEFINE_string("checkpoint_path", "/workspaces/carla-rl/checkpoints/model-sac-42/checkpoint_101200", "Path to load checkpoint from")

def save_checkpoint(agent, path, step):
    """Save agent checkpoint"""
    os.makedirs(path, exist_ok=True)
    state_dict = {
        'actor_params': agent._actor,
        'critic_params': agent._critic,
        'target_critic_params': agent._target_critic_params,
        'temp': agent._temp,
        'rng': agent._rng,
    }
    checkpoints.save_checkpoint(
        ckpt_dir=os.path.abspath(path),
        target=state_dict,
        step=step,
        overwrite=True,
        keep=3
    )

def load_checkpoint(agent, checkpoint_path):
    """Load agent parameters from checkpoint"""
    state_dict = {
        'actor_params': agent._actor,
        'critic_params': agent._critic,
        'target_critic_params': agent._target_critic_params,
        'temp': agent._temp,
        'rng': agent._rng,
    }
    state_dict = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_path,
        target=state_dict
    )

    agent._critic = state_dict['critic_params'] 
    agent._target_critic_params = state_dict['target_critic_params']
    agent._temp = state_dict['temp']
    agent._rng = state_dict['rng']
    return agent

def main(_):
    # Initialize ROS node
    rospy.init_node("SAC_TRAINING", anonymous=False)
    
    # Create environment
    env = LeoEnv()
    # rate = rospy.Rate(env.RATE)
    env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
    env = TimeLimit(env, max_episode_steps=2500)
    env = RecordEpisodeStatistics(env)
    
    # Set random seeds
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    
    # Initialize logger
    logger = Logger(log_dir="./logs", prefix="SAC")

    # Initialize checkpoints dir
    policy_folder = os.path.join("checkpoints", f"model-sac-{len(glob.glob('./logs/*'))}")
    os.makedirs(policy_folder, exist_ok=True)

    # Initialize agent
    agent = PixelResNetDrQLearner(
        0, 
        env.observation_space.sample(), 
        env.action_space.sample(), 
        **sac_config
    )
    
    # Load checkpoint if provided
    if FLAGS.checkpoint_path:
        agent = load_checkpoint(agent, FLAGS.checkpoint_path)
        print(f"Loaded checkpoint from {FLAGS.checkpoint_path}")
    
    # Setup replay buffer
    with open("/workspaces/carla-rl/datasets/bc_data_0.pkl", 'rb') as f:
        replay_buffer = pickle.load(f)
    
    # replay_buffer = ReplayBuffer(
    #     env.observation_space, 
    #     env.action_space, 
    #     FLAGS.replay_buffer_size
    # )
    replay_buffer.seed(FLAGS.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size}
    )
    
    # Load expert data if available
    expert_replay_buffers = []
    expert_replay_buffer_iterators = []
    expert_buffers = list(glob.glob("/workspaces/ROS1/carla-rl/real_robot_dataset/*.pkl"))
    
    if expert_buffers:
        pass
        # for path in expert_buffers:

        #     expert_replay_buffers.append(expert_replay_buffer)
        #     expert_replay_buffer_iterators.append(
        #         expert_replay_buffer.get_iterator(
        #             sample_args={"batch_size": FLAGS.batch_size}
        #         )
        #     )
    
    # Track metrics
    success_history = deque(maxlen=100)
    # eval_success_history = deque(maxlen=100)/
    distance_to_goal_history = deque(maxlen=100)
    
    # Start training
    observation, info = env.reset()
    done = False
    training_start_time = time.time()
    
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        # Sample action
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
            action = np.clip(action, env.action_space.low, env.action_space.high)
        current_velocity=env.unwrapped.speed
        current_heading=env.unwrapped.current_heading
        heading=env.unwrapped.heading
        # Take step in environment
        next_observation, reward, done, truncated, info = env.step(action)
        
        # Handle episode termination
        mask = 1.0 if not done or not truncated or "TimeLimit.truncated" in info else 0.0
            
        # Store transition
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
        print(f"\rCurrent heading {np.rad2deg(current_heading)} Target Heading {np.rad2deg(heading)} {current_velocity}", end="")
        # Handle episode completion
        if done or truncated or "TimeLimit.truncated" in info:
            if "episode" in info:
                # Record episode metrics
                episode_info = {
                    "return": info["episode"]["r"],
                    "length": info["episode"]["l"],
                    "time": info["episode"]["t"]
                }
                
                # Track additional metrics if available
                if "is_success" in info:
                    success_rate = float(info["is_success"])
                    success_history.append(success_rate)
                    episode_info["is_success"] = success_rate
                    episode_info["success_rate"] = np.mean(success_history)
                    
                if "distance_completed" in info:
                    distance = float(info["distance_completed"])
                    distance_to_goal_history.append(distance)
                    episode_info["distance_completed"] = distance
                    episode_info["avg_distance"] = np.mean(distance_to_goal_history)
                    
                if "slack" in info:
                    episode_info["slack"] = float(info["slack"])
                    
                # Add reward statistics
                episode_info.update({
                    "mean_reward": info.get("mean_reward", 0),
                    "max_reward": info.get("max_reward", 0),
                    "min_reward": info.get("min_reward", 0)
                })
                
                logger.log_episode(episode_info, i)
                
            # Reset environment
            observation, info = env.reset()
            done = False
        
        # Training updates
        if i >= FLAGS.start_training and i%4000==0:
            # Update from replay buffer
            for _ in  range(4):
                batch = next(replay_buffer_iterator)
                update_info = agent.update(batch)
                
            if i % FLAGS.log_interval == 0:
                logger.log_training(update_info, i)
                logger.print_status(i, FLAGS.max_steps)
            
            # Update from expert data if available
            if expert_replay_buffer_iterators:
                for expert_iterator in expert_replay_buffer_iterators:
                    batch_expert = next(expert_iterator)
                    update_info_expert = agent.update(
                        batch_expert,
                        enable_update_temperature=False
                    )
                if i % FLAGS.log_interval == 0:
                    logger.log_training(update_info_expert, i, prefix="_expert")
            save_checkpoint(agent, policy_folder, i)
       
    # Save final model and buffer
    save_checkpoint(agent, f"checkpoints/final_drq", 1)
    
    if FLAGS.save_buffer:
        dataset_folder = "datasets"
        os.makedirs(dataset_folder, exist_ok=True)
        dataset_file = os.path.join(dataset_folder, "lane_following_buffer")
        with open(dataset_file, "wb") as f:
            pickle.dump(replay_buffer, f)
    
    # Print final training statistics
    training_duration = time.time() - training_start_time
    print(f"\nTraining completed in {training_duration/3600:.2f} hours")
    print(f"Logs saved to: {logger.log_dir}")

if __name__ == "__main__":
    app.run(main)