#! /usr/bin/env python
import glob
import random
from collections import deque

import flax
import numpy as np
from absl import flags
from flax.training import checkpoints
from ml_collections import config_flags
from rlib_integration.carla_goal_env import CarlaGoalEnv
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from jaxrl2.agents import DrQLearner
# from drq_with_value_function import DrQLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit

# from jax_experiments_goal import JAXGoalExperiments
from src.jax_experiments_goal import JAXGoalExperiments

flax.config.update('flax_use_orbax_checkpointing', True)
# from flax
# config = {
#     "framework": "torch",
#     "num_workers": 1,
#     "num_gpus_per_worker": 1,
#     "num_cpus_per_worker": 3,
#     "rollout_fragment_length": 16,
#     "timesteps_per_iteration": 2000,
#     "train_batch_size": 16,
#     "learning_starts": 5000,
#     "buffer_size": 15000,
#     "lr": 0.0003,
#     "exploration_config": {
#         "type": "EpsilonGreedy",
#         "initial_epsilon": 1.0,
#         "final_epsilon": 0.1,
#         "epsilon_timesteps": 50000
#     },
#     "env_config": {
#         "carla": {
#             "host": "localhost",
#             "timeout": 20.0,
#             "timestep": 0.1,
#             "retries_on_error": 25,
#             "resolution_x": 600,
#             "resolution_y": 600,
#             "quality_level": "Low",
#             "enable_map_assets": True,
#             "enable_rendering": True,
#             "show_display": True,
#             "town":"Town02"
#         },
#         "experiment": {
#             "type":JAXGoalExperiments,
#             "hero": {
#                 "blueprint": "vehicle.mercedes.coupe_2020",
#                 "sensors": {
#                     "collision": {
#                         "type": "sensor.other.collision"
#                     },
#                     "rgb": {
#                         "type": "sensor.camera.rgb",
#                         "image_size_x": 180,
#                         "image_size_y": 180,
#                         "transform": "1.9, 0.0, 1.7, 0.0, -15.0, 0.0"
#                     },
#                     "imu":{
#                         "type":"sensor.other.imu"
#                     },
#                     "lane_invasion": {
#                         "type": "sensor.other.lane_invasion"
#                     }
#                 },
#                 # "spawn_points": [
#                 #     "-115.60, -207.60, 11.02, -0.00, -0.01, -179.92",  # tl_l
#                 #     "-243.70, 71.50, 11.98, -0.00, -0.08, 90.81",      # bl_d
#                 #     "76.60, 202.10, 1.00, -0.00, 0.00, 0.29",          # br_r
#                 #     "210.20, -50.30, 1.00, -0.00, 0.00, -90.67",       # ur_u
#                 #     "-114.30, 191.00, 9.27, -0.00, 0.61, 179.97",      # bl_l
#                 #     "-233.00, -51.50, 11.00, -0.00, 0.00, -90.84",     # tl_u
#                 #     "83.90, -190.30, 1.00, -0.00, 0.00, -0.24",        # tr_r
#                 #     "189.30, 88.20, 1.00, -0.00, 0.00, 90.66"          # br_d
#                 # ]
#             },
#             "background_activity": {
#                 "n_vehicles": 0,
#                 "n_walkers": 0,
#                 "tm_hybrid_mode": True
#             },
#             # "town": "Town05",
#             "weather": "CloudySunset",
#             "others": {
#                 "framestack": 1,
#                 "max_time_idle": 600,
#                 "max_dist": 200,
#                 "target_speed": 5.0
#             }
#         }
#     }
# }

config = {
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
                "target_speed": 5.0,
                "use_rgb":False
            }
        }
    }
}

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "cheetah-run-v0", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 5, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", int(5e4), "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(5e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e3), "Number of training steps to start training."
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
config_flags.DEFINE_config_file(
    "config",
    "./src/configs/drq_default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)



def save_checkpoint(agent, path, step):
    # Create checkpoint directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    # Save checkpoint
    # print(type(agent))
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
import tqdm
from absl import app
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any

class Logger:
    def __init__(self, log_dir: str):
        # Create timestamped log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, timestamp)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Store metrics for console printing
        self.train_metrics = {}
        self.eval_metrics = {}
        self.episode_metrics = {}
        
        print(f"\nLogging to: {self.log_dir}\n")
    
    def log_training(self, metrics: Dict[str, Any], step: int,prefix=""):
        """Log training metrics to both tensorboard and console."""
        for k, v in metrics.items():
            self.writer.add_scalar(f"training{prefix}/{k}", np.array(v), step)
            self.train_metrics[f"{k}{prefix}"] = np.array(v)
    
    def log_eval(self, metrics: Dict[str, Any], step: int):
        """Log evaluation metrics to both tensorboard and console."""
        for k, v in metrics.items():
            self.writer.add_scalar(f"evaluation/{k}",np.array(v), step)
            self.eval_metrics[k] = np.array(v)
    
    def log_episode(self, metrics: Dict[str, Any], step: int):
        """Log episode metrics to both tensorboard and console."""
        for k, v in metrics.items():
            self.writer.add_scalar(f"episode/{k}", np.array(v), step)
            self.episode_metrics[k] = np.array(v)
    
    def print_status(self, step: int, total_steps: int):
        """Print current status in a nicely formatted way."""
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
# expert_buffer="/home/kojogyaase/Projects/Research/carla-rl/datasets/basic_agent_data_20241229_093438.pkl"
expert_buffer=None

def relabel_obs_fn(original_dict,virtual_dict,is_near_goal,max_len):
    # Check current reward condition
    current_reward = virtual_dict["rewards"]
    
    # If reward < 0, return without relabeling
    if current_reward < 0:
        return original_dict
        
    # Only relabel if reward > 10
    if current_reward > 0:
        # Change goal to some achievable observation i.e next observation
        virtual_goal = virtual_dict["next_observations"]["pixels"]
        virtual_goal_heading = virtual_dict["next_observations"]["vector"][-2]
        
        # Update goal and heading in both current and next observations
        original_dict["next_observations"]["vector"][-1] = virtual_goal_heading
        original_dict["next_observations"]["goal"] = virtual_goal
        original_dict["observations"]["goal"] = virtual_goal
        original_dict["observations"]["vector"][-1] = virtual_goal_heading    
        
        # Set termination signals and reward
        original_dict["dones"] = True
        original_dict["masks"] = 0.0  if is_near_goal else original_dict["masks"]
        original_dict["rewards"] = 10.0 if is_near_goal else original_dict["rewards"]
    
    return original_dict

def main(_):
    # Create environment
    env = CarlaGoalEnv(config["env_config"])
    env = FrameStack(env=env, num_stack=1,stacking_key="pixels")
    # env = FrameStack(env=env, num_stack=1,stacking_key="goal")
    env = TimeLimit(env,max_episode_steps=2500)
    env = RecordEpisodeStatistics(env)
    action_dim = 2
    mean = np.zeros(action_dim)
    sigma = 0.2 * np.ones(action_dim)
    noise = OrnsteinUhlenbeckActionNoise(mean=mean, sigma=sigma)
  
    # Initialize logger
    logger = Logger(log_dir="./logs")

    # Initialize checkpoints dir
    policy_folder = os.path.join("checkpoints", f"model-{len(glob.glob('./logs/*'))}")
    os.makedirs(policy_folder, exist_ok=True)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    # Initialize agent and replay buffer
    kwargs = dict(FLAGS.config)
    # kwargs["target_entropy"]=-0.1*env.action_space.sample().shape[-1]
    agent = DrQLearner(
        FLAGS.seed, 
        env.observation_space.sample(), 
        env.action_space.sample(), 
        # num_qs=10,
        **kwargs
    )
    
    replay_buffer_size = FLAGS.replay_buffer_size
    if not expert_buffer is None:
        with open(expert_buffer, 'rb') as f:
            expert_replay_buffer = pickle.load(f)

    
    replay_buffer = ReplayBuffer(
        env.observation_space, 
        env.action_space, 
        replay_buffer_size
    )

    # =========================================
    # replay_buffer = HindsightReplayBuffer(
    #     env.observation_space, 
    #     env.action_space, 
    #     replay_buffer_size,
    #     relabel_obs_fn
    # )
    # =========================================


    replay_buffer.seed(FLAGS.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size}
    )

    if not expert_buffer is None:
        expert_replay_buffer_iterator = expert_replay_buffer.get_iterator(
                sample_args={"batch_size": FLAGS.batch_size})
    # Track success metrics
    success_history = deque(maxlen=100)  # Track last 100 episodes
    eval_success_history = deque(maxlen=100)

    distance_to_goal_history = deque(maxlen=100)  # Track last 100 episodes
    eval_distance_to_goal_history = deque(maxlen=100)  # Track last 100 episodes
    # Main training loop
    observation, info, done = *env.reset(), False
    training_start_time = time.time()
    
    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
            # if i>int(5e5):
            action = action + noise()
            action = np.clip(action, env.action_space.low, env.action_space.high)
        next_observation, reward, done, truncated, info = env.step(action)
        
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
            observation, info, done = *env.reset(), False
            noise.reset()
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
                
                logger.log_episode(episode_info, i)
        
        # Training updates
        if i >= FLAGS.start_training:
            batch = next(replay_buffer_iterator)
            update_info = agent.update(batch,utd_ratio=8)
            
            if i % FLAGS.log_interval == 0:
                logger.log_training(update_info, i)
                logger.print_status(i, FLAGS.max_steps)
            if not expert_buffer is None:
                batch_expert = next(expert_replay_buffer_iterator)
                update_info_expert = agent.update(
                    batch_expert,
                    enable_update_temperature=False)
                if i % FLAGS.log_interval == 0:
                    logger.log_training(update_info_expert, i,prefix="_expert")
                    logger.print_status(i, FLAGS.max_steps)
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
    save_checkpoint(agent,f"checkpoints/final_drq",1)
    training_duration = time.time() - training_start_time
    print(f"\nTraining completed in {training_duration/3600:.2f} hours")
    print(f"Logs saved to: {logger.log_dir}")

if __name__ == "__main__":
    app.run(main)