#!/usr/bin/env python

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from gym import spaces
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from vision_rl.rllib_integration.carla_env import CarlaEnv
from stbl3_continous_experiments import STBL3Experiment
from config import config
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
            nn.Linear(n_flatten + 5, features_dim),  # +4 for the vector observations
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Split observations into image and vector parts
        image = observations['image']
        vector = observations['vector']
        # breakpoint()
        # Process image through CNN
        image_features = self.cnn(image.permute(0,3,1,2))
        
        # Concatenate with vector observations
        combined = torch.cat([image_features, vector], dim=1)
        
        
        return self.linear(combined)




def main():
    parser = argparse.ArgumentParser(description="SAC training script for CARLA")
    parser.add_argument("--output_dir", default="./results")
    args = parser.parse_args()

    # Create environment
    env = CarlaEnv(config["env_config"])
    n_actions = env.action_space.shape[0]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.5 * np.ones(n_actions),
        theta=0.15,
        dt=1e-2,
        initial_noise=None
    )
    # Create SAC model
    model = PPO(
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
        # buffer_size=100000,
        # learning_starts=5000,
        batch_size=256,
        # tau=0.005,              # Target network update rate
        gamma=0.99,
        # train_freq=1,
        # gradient_steps=1,
        # action_noise=action_noise,      # SAC handles exploration internally
        # optimize_memory_usage=True,
        # ent_coef="auto",        # Automatic entropy tuning
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