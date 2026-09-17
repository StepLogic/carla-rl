#! /usr/bin/env python
import glob
import random
from collections import deque

import flax
import numpy as np
from absl import flags
from flax.training import checkpoints
from ml_collections import config_flags
from rlib_integration.carla_env import CarlaEnv
from jaxrl2.agents import DrQLearner
from jaxrl2.data import ReplayBuffer
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit
from src.configs.train_env_config_v4_image_goal import config

flax.config.update('flax_use_orbax_checkpointing', True)

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "cheetah-run-v0", "Environment name.")
flags.DEFINE_string("save_dir", "./tmp/", "Tensorboard logging dir.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 5, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", int(5e4), "Eval interval.")
flags.DEFINE_integer("batch_size", 128, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(5.3e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(5e3), "Number of training steps to start training."
)
flags.DEFINE_integer("image_size", 64, "Image size.")
flags.DEFINE_integer("num_stack", 3, "Stack frames.")
flags.DEFINE_integer(
    "replay_buffer_size", int(8e5), "Number of training steps to start training."
)
flags.DEFINE_integer(
    "action_repeat", None, "Action repeat, if None, uses 2 or PlaNet default values."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("save_video", False, "Save videos during evaluation.")
flags.DEFINE_boolean("save_buffer", False, "Save the replay buffer.")
flags.DEFINE_float("goal_mask_prob", 0.1, "Probability of masking goal image per episode.")
config_flags.DEFINE_config_file(
    "config",
    "./src/configs/drq_default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)



def save_checkpoint(agent, path, step):
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, timestamp)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.train_metrics = {}
        self.eval_metrics = {}
        self.episode_metrics = {}

        print(f"\nLogging to: {self.log_dir}\n")

    def log_training(self, metrics: Dict[str, Any], step: int, prefix=""):
        for k, v in metrics.items():
            try:
                self.writer.add_scalar(f"training{prefix}/{k}", np.array(v), step)
                self.train_metrics[f"{k}{prefix}"] = np.array(v)
            except:
                pass

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

expert_buffer = None

def main(_):
    # Create environment
    env = CarlaEnv(config=config)
    env = TimeLimit(env, max_episode_steps=2500)
    env = RecordEpisodeStatistics(env)
    action_dim = 2
    mean = np.zeros(action_dim)
    noise = OrnsteinUhlenbeckActionNoise(mean=mean, sigma=np.array([
        0.05, 0.3
    ]))

    # Initialize logger
    logger = Logger(log_dir="./logs")

    # Initialize checkpoints dir
    policy_folder = os.path.join("checkpoints", f"model-{len(glob.glob('./logs/*'))}")
    os.makedirs(policy_folder, exist_ok=True)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    # Initialize agent and replay buffer
    kwargs = dict(FLAGS.config)
    agent = DrQLearner(
        FLAGS.seed,
        env.observation_space.sample(),
        env.action_space.sample(),
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

    replay_buffer.seed(FLAGS.seed)
    replay_buffer_iterator = replay_buffer.get_iterator(
        sample_args={"batch_size": FLAGS.batch_size}
    )

    if not expert_buffer is None:
        expert_replay_buffer_iterator = expert_replay_buffer.get_iterator(
                sample_args={"batch_size": FLAGS.batch_size})
    # Track success metrics
    success_history = deque(maxlen=100)
    eval_success_history = deque(maxlen=100)

    distance_to_goal_history = deque(maxlen=100)
    # Main training loop
    observation, info = env.reset()
    done = False
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
            action = action + noise()
            action = np.clip(action, env.action_space.low, env.action_space.high)

        # Per-step goal masking
        env.unwrapped.experiment.mask_goal = random.random() < FLAGS.goal_mask_prob

        next_observation, reward, done, truncated, info = env.step(action)

        # Handle episode termination
        if not done or "TimeLimit.truncated" in info:
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
            observation, info = env.reset()
            done = False
            noise.reset()
            if "episode" in info:
                episode_info = {
                    "return": info["episode"]["r"],
                    "length": info["episode"]["l"],
                    "time": info["episode"]["t"]
                }

                if "is_success" in info:
                    success = float(info["is_success"])
                    success_history.append(success)
                    episode_info["is_success"] = success
                    episode_info["success_rate"] = np.mean(success_history)
                if "distance_completed" in info:
                    distance_completed = float(info["distance_completed"])
                    distance_to_goal_history.append(distance_completed)
                    episode_info["distance_completed"] = distance_completed
                    episode_info["avg_distance_completed"] = np.mean(distance_to_goal_history)
                if "checkpoints_passed" in info:
                    episode_info["checkpoints_passed"] = info["checkpoints_passed"]
                if "slack" in info:
                    episode_info["slack"] = float(info["slack"])

                logger.log_episode(episode_info, i)

        # Training updates
        if i >= FLAGS.start_training:
            batch = next(replay_buffer_iterator)
            update_info = agent.update(batch, utd_ratio=2)

            if i % FLAGS.log_interval == 0:
                logger.log_training(update_info, i)
                logger.print_status(i, FLAGS.max_steps)
            if not expert_buffer is None:
                batch_expert = next(expert_replay_buffer_iterator)
                update_info_expert = agent.update(
                    batch_expert,
                    enable_update_temperature=False)
                if i % FLAGS.log_interval == 0:
                    logger.log_training(update_info_expert, i, prefix="_expert")
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

            # Run evaluation (no goal masking)
            env.unwrapped.experiment.mask_goal = False
            eval_successes = []
            eval_rewards = []
            eval_dists = []
            eval_slack = []
            eval_checkpoints = []

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
                        if "checkpoints_passed" in eval_info:
                            eval_checkpoints.append(eval_info["checkpoints_passed"])
                        if "slack" in eval_info:
                            eval_slack.append(float(eval_info["slack"]))

                eval_rewards.append(episode_reward)

            # Compute evaluation metrics
            eval_info = {
                "eval_reward_mean": np.mean(eval_rewards) if eval_rewards else 0.0,
                "eval_reward_std": np.std(eval_rewards) if eval_rewards else 0.0,
            }

            if eval_successes:
                success_rate = np.mean(eval_successes)
                eval_success_history.append(success_rate)
                eval_info["success_rate"] = success_rate
                eval_info["avg_success_rate"] = np.mean(eval_success_history)
                eval_info["distance_completed"] = np.mean(eval_dists) if eval_dists else 0.0
                eval_info["slack"] = np.mean(eval_slack) if eval_slack else 0.0
            if eval_checkpoints:
                eval_info["checkpoints_passed"] = np.mean(eval_checkpoints)
            save_checkpoint(agent, policy_folder, i)
            logger.log_eval(eval_info, i)
            logger.print_status(i, FLAGS.max_steps)

    # Print final training statistics
    save_checkpoint(agent, f"checkpoints/final_drq_ig", 1)
    training_duration = time.time() - training_start_time
    print(f"\nTraining completed in {training_duration/3600:.2f} hours")
    print(f"Logs saved to: {logger.log_dir}")

if __name__ == "__main__":
    app.run(main)