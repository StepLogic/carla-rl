import os
import time
from datetime import datetime
from collections import deque

import gymnasium as gym
from jaxrl2.wrappers.single_obs_to_dict import SingleObsToDict
import ml_collections
import numpy as np
import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any

# Logger class from your code, slightly modified for Gymnasium
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



def main():
    # Training parameters
    MAX_STEPS = 100000
    EVAL_INTERVAL = 5000
    LOG_INTERVAL = 1000
    EVAL_EPISODES = 5
    BATCH_SIZE = 64
    SEED = 42
    

    # Create environment
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=True, continuous=True)
    eval_env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=True, continuous=True)
    env=SingleObsToDict(env)
    eval_env=SingleObsToDict(eval_env)
    # Set seeds
    env.reset(seed=SEED)
    eval_env.reset(seed=SEED + 1)
    np.random.seed(SEED)
    
    # Initialize logger
    logger = Logger(log_dir="./logs_car_racing")
    
    # Initialize PPO agent with suitable hyperparameters for CartPole
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
    config=config.to_dict()

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
        BATCH_SIZE * 8  # Store 8 batches worth of data
    )
    replay_buffer.seed(SEED)
    replay_buffer_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": BATCH_SIZE}
    )
    
    # Training loop
    observation, info = env.reset()
    episode_return = 0
    episode_length = 0
    training_start_time = time.time()
    
    for step in tqdm.tqdm(range(1, MAX_STEPS + 1), smoothing=0.1):
        # Sample action from policy
        action,logp,value = agent.sample_actions(observation)
        
        # Take step in environment
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
                # Handle episode termination
        if not done or not truncated or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0
            
        # Store transition in replay buffer
        replay_buffer.insert(
            dict(
                observations=observation,
                logps=logp,
                value=value,
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
            if mask==1.0:
                last_value=0.0
            else:
                _,_,last_value = agent.sample_actions(observation)
            replay_buffer.compute_advantage(last_value)
            episode_info = {
                "return": episode_return,
                "length": episode_length,
            }
            logger.log_episode(episode_info, step)
            
            # Reset episode stats and environment
            observation, info = env.reset()
            episode_return = 0
            episode_length = 0
        
        # Training update
        if len(replay_buffer) >= BATCH_SIZE:
            batch = next(replay_buffer_iterator)
            update_info = agent.update(batch)
            if step % LOG_INTERVAL == 0:
                logger.log_training(update_info, step)
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