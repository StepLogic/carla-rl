from collections import deque
import os
import random
import ml_collections
import numpy as np
from absl import app, flags
from ml_collections import config_flags
from flax.training import checkpoints
from jaxrl2.agents import DrQLearner,PPOLearner
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.timelimit import TimeLimit
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from rlib_integration.carla_goal_env import CarlaGoalEnv
from src.configs.train_env_config import config as carla_config
from src.sac_lane_following import sac_config
os.environ['XLA_FLAGS']="--xla_gpu_enable_command_buffer="
# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", None, "Path to the checkpoint directory")
flags.DEFINE_integer("n_eval_episodes", 100, "Number of evaluation episodes")
flags.DEFINE_boolean("deterministic", True, "Whether to use deterministic actions")

def load_checkpoint(agent, checkpoint_path):
    """Load agent parameters from checkpoint."""
    state_dict = {
        'actor_params': agent._actor,
        'critic_params': agent._critic,
        'target_critic_params': agent._target_critic_params,
        # 'temp': agent._temp,
        # 'rng': agent._rng,
        # Add any other numerical state you need to save
    }
    state_dict = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_path,
        target=state_dict
    )

    # Update agent parameters
    # breakpoint()
    agent._actor = state_dict['actor_params']
    agent._critic = state_dict['critic_params'] 
    # agent._target_critic_params = state_dict['target_critic_params']
    # agent._temp = state_dict['temp']
    # agent._rng = state_dict['rng']
    
    return agent

def evaluate_policy(agent, env, n_eval_episodes=10, deterministic=True):
    """Evaluate the agent for n_eval_episodes."""
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    distance_completed = []
    slack_values = []
    
    for _ in range(n_eval_episodes):
        observation, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            target=3.0
            heading=np.pi
            vecs=observation["vector"]
            current_velocity=env.unwrapped.experiment.velocity
            current_heading=env.unwrapped.experiment.current_heading
            vecs[2] = np.clip(current_velocity/(target+1e-8), 0.0, 1.0)
            vecs[3]= np.clip(current_heading/(heading+1e-8),-1.0,1.0) 
            
            if deterministic:
                action = agent.eval_actions(observation)
            else:
                action = agent.sample_actions(observation)
                
            observation, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = done or truncated
            
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                # print(info)
                if "is_success" in info:
                    success_rate.append(float(info["is_success"]))
                if "distance_completed" in info:
                    distance_completed.append(float(info["distance_completed"]))
                if "slack" in info:
                    slack_values.append(float(info["slack"]))
    
    # Compute statistics
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
    }
    
    if success_rate:
        stats["success_rate"] = np.mean(success_rate)
    if distance_completed:
        stats["mean_distance"] = np.mean(distance_completed)
    if slack_values:
        stats["mean_slack"] = np.mean(slack_values)
    
    return stats

def main(_):
    # Create and wrap environment

    env = CarlaGoalEnv(carla_config["env_config"])
    env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
    # env = FrameStack(env=env, num_stack=1, stacking_key="goal")
    env = TimeLimit(env, max_episode_steps=2500)
    env = RecordEpisodeStatistics(env)
    
    # Initialize agent
    # kwargs = dict(FLAGS.config)
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

    agent = DrQLearner(
        0,
        env.observation_space.sample(),
        env.action_space.sample(),
        **sac_config
    )
    
    # Load checkpoint
    agent = load_checkpoint(agent, FLAGS.checkpoint_path)
    
    # Evaluate
    stats = evaluate_policy(
        agent,
        env,
        n_eval_episodes=FLAGS.n_eval_episodes,
        deterministic=FLAGS.deterministic
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    flags.mark_flag_as_required("checkpoint_path")
    app.run(main)