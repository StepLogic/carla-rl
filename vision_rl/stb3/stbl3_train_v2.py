#!/usr/bin/env python

import os
import argparse
import gym.wrappers
import numpy as np
import torch
import torch.nn as nn
from gym import spaces
import gym
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from vision_rl.rllib_integration.carla_goal_env import CarlaGoalEnv
from stbl3_experiments_v2 import STBL3GoalExperiment

class CarlaCNN(BaseFeaturesExtractor):
    """CNN feature extractor for CARLA images"""
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, 1, 84, 84)).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten*2 + 4, features_dim),  # +4 for the vector observations
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Split observations into image and vector parts
        image = observations['image']
        goal = observations['goal']
        vector = observations['vector']
        
        # Process image through CNN
        image_features = self.cnn(image.permute(0,3,1,2))
        goal_features = self.cnn(goal.permute(0,3,1,2))
        
        # Concatenate with vector observations
        combined = torch.cat([image_features,goal_features,vector], dim=1)
        
        # breakpoint()
        return self.linear(combined)


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
            "town":"Town05"
        },
        "experiment": {
            "type":STBL3GoalExperiment,
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
                "max_time_idle": 600,
                "max_dist": 200,
                "target_speed": 5.0
            }
        }
    }
}
os.environ["CARLA_ROOT"]='/home/robotlab/Apps/CARLA_0.9.15'
def main():
    parser = argparse.ArgumentParser(description="SAC training script for CARLA")
    parser.add_argument("--output_dir", default="./results")
    args = parser.parse_args()

    # Create environment
    env = CarlaGoalEnv(config["env_config"])
    env=gym.wrappers.TimeLimit(env,max_episode_steps=2500)
    n_actions = env.action_space.shape[0]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.5 * np.ones(n_actions),
        theta=0.15,
        dt=1e-2,
        initial_noise=None
    )
    # Create SAC model
    model = SAC(
        "MultiInputPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=CarlaCNN,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=dict(
                pi=[256, 256],  # Actor (policy) network
                qf=[256, 256]   # Critic (Q-function) network
            )
        ),
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=5000,
        batch_size=16,
        tau=0.005,              # Target network update rate
        gamma=0.99,
        # train_freq=1,
        # gradient_steps=1,
        # action_noise=action_noise,      # SAC handles exploration internally
        # optimize_memory_usage=True,
        ent_coef="auto",        # Automatic entropy tuning
        # target_entropy="auto",  # Automatically set target entropy
        tensorboard_log=os.path.join(args.output_dir, "tensorboard"),
        verbose=1
    )

    # Setup callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=10000,
            save_path=args.output_dir,
            name_prefix="sac_carla"
        ),
    ]

    # Train model
    model.learn(
        total_timesteps=1000000,
        callback=callbacks
    )

    # Save final model
    model.save(os.path.join(args.output_dir, "final_model"))

if __name__ == "__main__":
    main()