#! /usr/bin/env python
from collections import defaultdict, deque
import itertools
import os
import pickle
import time
import argparse
import numpy as np
import random
import tqdm

import gymnasium
from jaxrl2.utils.misc import Logger
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.timelimit import TimeLimit
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
import ml_collections
from absl import app, flags
from flax.training import checkpoints
import jaxrl2.extra_envs.dm_control_suite
from jaxrl2.agents import DrQLearner
from gymnasium.wrappers.utils import RunningMeanStd

from rlib_integration.carla_goal_env import CarlaGoalEnv
from configs.train_env_config import config as carla_config
import flax
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
flax.config.update('flax_use_orbax_checkpointing', True)

# ML config
import jax
jax.config.update("jax_debug_nans", True)

# Define agent configuration
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
config.backup_entropy = True
config.num_qs = 10
config.critic_reduction = "mean"
sac_config = config.to_dict()

# Define command line arguments
FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_path", None, "Path to model checkpoint.")
flags.DEFINE_string("eval_towns", "Town02,Town05,Town10HD", "Comma-separated list of towns to evaluate on.")
flags.DEFINE_string("save_dir", "./eval_results/", "Directory to save evaluation results.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 20, "Number of episodes per town for evaluation.")
flags.DEFINE_integer("max_episode_steps", 3500, "Maximum steps per episode.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("record_trajectory", True, "Record full trajectory data.")
flags.DEFINE_boolean("visualize", False, "Visualize evaluation (set to True for rendering).")
flags.mark_flag_as_required("checkpoint_path")

def load_checkpoint(agent, checkpoint_path):
    """Load agent parameters from checkpoint."""
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

    # Update agent parameters
    agent._actor = state_dict['actor_params']
    agent._critic = state_dict['critic_params'] 
    agent._target_critic_params = state_dict['target_critic_params']
    agent._temp = state_dict['temp']
    agent._rng = state_dict['rng']
    
    return agent

def setup_environment(town, max_episode_steps, visualize=False):
    """Set up the CARLA environment with the given town."""
    carla_config["env_config"]["carla"]["town"] = town
    carla_config["env_config"]["carla"]["visualize"] = visualize
    
    env = CarlaGoalEnv(carla_config["env_config"])
    env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = RecordEpisodeStatistics(env)
    
    return env

def evaluate_on_town(agent, town, num_episodes, max_episode_steps, visualize, record_trajectory=False):
    """Evaluate the agent on a specific town."""
    env = setup_environment(town, max_episode_steps, visualize)
    
    episode_rewards = []
    episode_successes = []
    episode_distances = []
    episode_lengths = []
    episode_slacks = []
    trajectories = []
    
    for episode in tqdm.tqdm(range(num_episodes), desc=f"Evaluating on {town}", disable=not FLAGS.tqdm):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        # Initialize trajectory recording if needed
        if record_trajectory:
            trajectory = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'next_observations': [],
                'dones': [],
                'infos': []
            }
        
        while not done:
            action = agent.eval_actions(obs)
            
            if record_trajectory:
                trajectory['observations'].append(obs)
                trajectory['actions'].append(action)
            
            next_obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            if record_trajectory:
                trajectory['rewards'].append(reward)
                trajectory['next_observations'].append(next_obs)
                trajectory['dones'].append(done)
                trajectory['infos'].append({k: v for k, v in info.items() if isinstance(v, (int, float, bool, str))})
            
            obs = next_obs
            
            if done or truncated:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_steps)
                
                if "is_success" in info:
                    episode_successes.append(float(info["is_success"]))
                if "distance_completed" in info:
                    episode_distances.append(float(info["distance_completed"]))
                if "slack" in info:
                    episode_slacks.append(float(info["slack"]))
                
                if record_trajectory:
                    trajectories.append(trajectory)
                break
    
    # Close environment
    env.close()
    
    # Compile results
    results = {
        "town": town,
        "reward_mean": np.mean(episode_rewards),
        "reward_std": np.std(episode_rewards),
        "length_mean": np.mean(episode_lengths),
        "length_std": np.std(episode_lengths)
    }
    
    if episode_successes:
        results["success_rate"] = np.mean(episode_successes)
    if episode_distances:
        results["distance_completed_mean"] = np.mean(episode_distances)
        results["distance_completed_std"] = np.std(episode_distances)
    if episode_slacks:
        results["slack_mean"] = np.mean(episode_slacks)
    
    return results, trajectories if record_trajectory else None

def main(_):
    # Set random seeds
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    # Initialize logger
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(FLAGS.save_dir, f"eval_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir=log_dir, prefix="Evaluation")
    
    # Parse towns to evaluate on
    eval_towns = FLAGS.eval_towns.split(",")
    
    # Create environment to get action/observation spaces
    temp_env = setup_environment(eval_towns[0], FLAGS.max_episode_steps)
    
    # Initialize agent
    agent = DrQLearner(
        0, 
        temp_env.observation_space.sample(), 
        temp_env.action_space.sample(),
        **sac_config
    )
    
    # Close temporary environment
    temp_env.close()
    
    # Load checkpoint
    print(f"Loading checkpoint from {FLAGS.checkpoint_path}")
    agent = load_checkpoint(agent, FLAGS.checkpoint_path)
    
    # Evaluation results
    all_results = []
    all_trajectories = {}
    
    # Evaluate on each town
    print(f"Starting evaluation on {len(eval_towns)} towns: {eval_towns}")
    for town in eval_towns:
        print(f"\nEvaluating on {town}...")
        results, trajectories = evaluate_on_town(
            agent, 
            town, 
            FLAGS.eval_episodes, 
            FLAGS.max_episode_steps,
            FLAGS.visualize,
            FLAGS.record_trajectory
        )
        
        all_results.append(results)
        if trajectories:
            all_trajectories[town] = trajectories
        
        # Log town results
        print(f"Results for {town}:")
        for key, value in results.items():
            if key != "town":
                print(f"  {key}: {value}")
        
        # Log to logger
        logger.log_eval(results, 0, prefix=f"{town}_")
    
    # Aggregate results across towns
    aggregated_results = {
        "overall_reward_mean": np.mean([r["reward_mean"] for r in all_results]),
        "overall_reward_std": np.std([r["reward_mean"] for r in all_results]),
    }
    
    if all("success_rate" in r for r in all_results):
        aggregated_results["overall_success_rate"] = np.mean([r["success_rate"] for r in all_results])
    
    if all("distance_completed_mean" in r for r in all_results):
        aggregated_results["overall_distance_completed"] = np.mean([r["distance_completed_mean"] for r in all_results])
    
    # Log aggregated results
    print("\nAggregated results across all towns:")
    for key, value in aggregated_results.items():
        print(f"  {key}: {value}")
    
    logger.log_eval(aggregated_results, 0, prefix="overall_")
    
    # Save full results
    results_file = os.path.join(log_dir, "evaluation_results.pkl")
    with open(results_file, "wb") as f:
        pickle.dump({
            "per_town_results": all_results,
            "aggregated_results": aggregated_results,
            "eval_config": {
                "checkpoint_path": FLAGS.checkpoint_path,
                "eval_towns": eval_towns,
                "eval_episodes": FLAGS.eval_episodes,
                "max_episode_steps": FLAGS.max_episode_steps,
            }
        }, f)
    
    # Save trajectories if recorded
    if FLAGS.record_trajectory:
        trajectories_file = os.path.join(log_dir, "evaluation_trajectories.pkl")
        with open(trajectories_file, "wb") as f:
            pickle.dump(all_trajectories, f)
    
    print(f"\nEvaluation completed. Results saved to {log_dir}")

if __name__ == "__main__":
    app.run(main)