import os
import time
from datetime import datetime
from collections import deque

import gymnasium as gym
from gymnasium.envs.box2d.car_racing import CarRacing

from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.single_obs_to_dict import SingleObsToDict
from jaxrl2.wrappers.timelimit import TimeLimit
import ml_collections
import numpy as np
import tqdm
import jax
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any

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
    EVAL_INTERVAL = 500000
    LOG_INTERVAL = 1000
    EVAL_EPISODES = 5
    BATCH_SIZE = 64
    SEED = 42
    ROLLOUT_CAPACITY=4096
    LOCAL_STEPS=2048

    # Create environment
    # env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, max_episode_steps=2048, continuous=True)
    env=CarRacing(render_mode="human")
    env = TimeLimit(env,max_episode_steps=2500)
    env = RecordEpisodeStatistics(env)
    eval_env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
    env = SingleObsToDict(env,num_stack=1)
    eval_env = SingleObsToDict(eval_env,num_stack=1)
    # env = FrameStack(env=env, num_stack=3,stacking_key="pixels")

    # Set seeds
    env.reset(seed=SEED)
    eval_env.reset(seed=SEED + 1)
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
        capacity=ROLLOUT_CAPACITY  # Store 100 batches worth of data
    )
    replay_buffer.seed(SEED)
    replay_buffer_iterator = replay_buffer.get_iterator(sample_args={"batch_size": BATCH_SIZE})

    # Training loop
    observation, info = env.reset()
    episode_return = 0
    episode_length = 0
    training_start_time = time.time()
    # n_steps=int(ROLLOUT_CAPACITY/2)  #rollout steps

    for step in tqdm.tqdm(range(1, MAX_STEPS + 1,LOCAL_STEPS), smoothing=0.1):
       
        for  _ in range(LOCAL_STEPS-1):
            action, logp, value = agent.sample_actions(observation)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            timeout = "TimeLimit.truncated" in info
            mask = 1.0 if not done or timeout else 0.0
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
            if done:
                break

        # Handle episode end
        epoch_ended = not done

        episode_info = {
            "return": episode_return,
            "length": episode_length,
        }
        logger.log_episode(episode_info, step)
        if mask==1.0 or epoch_ended or timeout:
            _,_,last_value = agent.sample_actions(observation)
        else:
            last_value=0.0
        replay_buffer.compute_advantage_v2(last_value)
        observation, info = env.reset()
        episode_return = 0
        episode_length = 0

        # print(replay_buffer.can_sample(),len(replay_buffer),replay_buffer._path_start_idx)
        if len(replay_buffer) >= BATCH_SIZE and replay_buffer.can_sample():
                for _ in range(6):
                    batch = next(replay_buffer_iterator)
                    update_info = agent.update(batch, utd_ratio=1)  # Perform 3 updates per batch
                    logger.log_training(update_info, step)
                # if step % LOG_INTERVAL==0:
                    logger.print_status(step+_, MAX_STEPS)

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