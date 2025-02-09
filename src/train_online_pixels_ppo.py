import os
import time
from datetime import datetime
from collections import deque

import gymnasium as gym
from jaxrl2.wrappers.single_obs_to_dict import SingleObsToDict
import ml_collections
import numpy as np
import tqdm
import jax
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any
from rlib_integration.carla_goal_env import CarlaGoalEnv
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit

from src.jax_experiments_goal import JAXGoalExperiments



env_config = {
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
            "town":"Town04"
        },
        "experiment": {
            "type":JAXGoalExperiments,
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
                    "goal": {
                        "type": "sensor.goal",
                        "image_size_x": 300,
                        "image_size_y": 300,
                        # "transform": "1.9, 0.0, 1.7, 0.0, -15.0, 0.0",
                        # "attach":False
                    },

                    "imu":{
                        "type":"sensor.other.imu"
                    },
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
            "town": "Town02",
            # "weather": "CloudySunset",
            "others": {
                "framestack": 1,
                "max_time_idle": 150,
                "max_dist": 200,
                "target_speed": 13.0,
                "use_rgb":False
            }
        }
    }
}

# Logger class
class Logger:
    def __init__(self, log_dir: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, timestamp)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        self.train_metrics = {}
        self.eval_metrics = {}
        self.episode_metrics = {}
        
        print(f"\nLogging to: {self.log_dir}\n")
    
    def log_training(self, metrics: Dict[str, Any], step: int):
        for k, v in metrics.items():
            self.writer.add_scalar(f"training/{k}", np.array(v), step)
            self.train_metrics[k] = np.array(v)
    
    def log_eval(self, metrics: Dict[str, Any], step: int):
        for k, v in metrics.items():
            self.writer.add_scalar(f"evaluation/{k}", np.array(v), step)
            self.eval_metrics[k] = np.array(v)
    
    def log_episode(self, metrics: Dict[str, Any], step: int):
        for k, v in metrics.items():
            self.writer.add_scalar(f"episode/{k}", np.array(v), step)
            self.episode_metrics[k] = np.array(v)
    
    def print_status(self, step: int, total_steps: int):
        print("\n" + "="*80)
        print(f"Step: {step}/{total_steps} ({step/total_steps*100:.1f}%)")
        
        if self.train_metrics:
            print("\nTraining Metrics:")
            for k, v in self.train_metrics.items():
                print(f"  {k:<20} {np.array(v):>10.4f}")
        
        if self.episode_metrics:
            print("\nLatest Episode:")
            for k, v in self.episode_metrics.items():
                print(f"  {k:<20} {np.array(v):>10.4f}")
        
        if self.eval_metrics:
            print("\nLatest Evaluation:")
            for k, v in self.eval_metrics.items():
                print(f"  {k:<20} {np.array(v):>10.4f}")
        
        print("="*80 + "\n")

# Main function
def main():
    # Training parameters
    MAX_STEPS = int(2e6)
    EVAL_INTERVAL = 5000
    LOG_INTERVAL = 1000
    EVAL_EPISODES = 5
    BATCH_SIZE = 64
    SEED = 42

    # Create environment
    env = CarlaGoalEnv(env_config["env_config"])
    env = FrameStack(env=env, num_stack=1,stacking_key="pixels")
    # env = FrameStack(env=env, num_stack=1,stacking_key="goal")
    env = TimeLimit(env,max_episode_steps=2500)
    env = RecordEpisodeStatistics(env)
    # Set seeds
    env.reset(seed=SEED)
    np.random.seed(SEED)

    # Initialize logger
    logger = Logger(log_dir="./logs_car_racing")

    # Initialize PPO agent
    from jaxrl2.agents import PPOLearner
    config = ml_collections.ConfigDict()
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.hidden_dims = (256, 256)
    config.cnn_features = (32, 64, 128, 256)
    config.cnn_filters = (3, 3, 3, 3)
    config.cnn_strides = (2, 2, 2, 2)
    config.cnn_padding = "VALID"
    config.latent_dim = 50
    config.encoder = "d4pg"
    config.discount = 0.98
    config.critic_reduction = "mean"
    config.clip_ratio = 0.2  # Add clip ratio
    config.gae_lambda = 0.95  # Add GAE lambda
    config = config.to_dict()

    agent = PPOLearner(
        seed=SEED,
        observations=env.observation_space.sample(),
        actions=env.action_space.sample(),
        **config
    )

    # Initialize replay buffer
    from jaxrl2.data import RolloutBuffer
    replay_buffer = RolloutBuffer(
        env.observation_space,
        env.action_space,
        capacity=BATCH_SIZE * 100  # Store 100 batches worth of data
    )
    replay_buffer.seed(SEED)
    replay_buffer_iterator = replay_buffer.get_iterator(sample_args={"batch_size": BATCH_SIZE})

    # Training loop
    observation, info = env.reset()
    episode_return = 0
    episode_length = 0
    training_start_time = time.time()

    for step in tqdm.tqdm(range(1, MAX_STEPS + 1), smoothing=0.1):
        # Sample action from policy
        action, logp, value = agent.sample_actions(observation)

        # Take step in environment
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        mask = 1.0 if not done else 0.0

        # Store transition in replay buffer
        replay_buffer.insert(
            dict(
                observations=observation,
                logps=logp,
                values=value,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
            )
        )

        observation = next_observation
        episode_return += reward
        episode_length += 1

        # Handle episode termination
        if done:
            # Log episode metrics
            episode_info = {
                "return": episode_return,
                "length": episode_length,
            }
            logger.log_episode(episode_info, step)
            if mask==1.0:
                _,_,last_value = agent.sample_actions(observation)
            else:
                last_value=0.0
            replay_buffer.compute_advantage_v2(last_value)
            # replay_buffer.compute_advantage_v2()
            # Reset episode stats and environment
            observation, info = env.reset()
            episode_return = 0
            episode_length = 0
            # if len(replay_buffer) >= BATCH_SIZE: 

            # Training update
            if len(replay_buffer) >= BATCH_SIZE and replay_buffer.can_sample():
                batch = next(replay_buffer_iterator)
                update_info = agent.update(batch, utd_ratio=3)  # Perform 3 updates per batch
                logger.log_training(update_info, step)
            if step % LOG_INTERVAL == 0:
                logger.print_status(step, MAX_STEPS)

            # Periodic evaluation
            if step % EVAL_INTERVAL == 0:
                eval_returns = []
                eval_lengths = []

                for _ in range(EVAL_EPISODES):
                    eval_obs, _ = eval_env.reset()
                    eval_done = False
                    eval_return = 0
                    eval_length = 0

                    while not eval_done:
                        eval_action = agent.eval_actions(eval_obs)
                        eval_obs, eval_reward, eval_terminated, eval_truncated, _ = eval_env.step(eval_action)
                        eval_done = eval_terminated or eval_truncated
                        eval_return += eval_reward
                        eval_length += 1

                    eval_returns.append(eval_return)
                    eval_lengths.append(eval_length)

                eval_info = {
                    "eval_return_mean": np.mean(eval_returns),
                    "eval_return_std": np.std(eval_returns),
                    "eval_length_mean": np.mean(eval_lengths),
                    "eval_length_std": np.std(eval_lengths),
                }

                logger.log_eval(eval_info, step)
                logger.print_status(step, MAX_STEPS)

    # Print final training statistics
    training_duration = time.time() - training_start_time
    print(f"\nTraining completed in {training_duration/3600:.2f} hours")
    print(f"Logs saved to: {logger.log_dir}")

    # Close environments
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()