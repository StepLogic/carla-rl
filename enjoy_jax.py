from collections import deque
import os
import numpy as np
from absl import app, flags
from ml_collections import config_flags
from flax.training import checkpoints
from jaxrl2.agents import DrQLearner
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.timelimit import TimeLimit
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from rlib_integration.carla_goal_env import CarlaGoalEnv
from src.carla_eval import CarlaEvalEnv
from src.jax_experiments_goal import JAXGoalExperiments

# fix
os.environ['XLA_FLAGS']="--xla_gpu_enable_command_buffer="
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".20"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", None, "Path to the checkpoint directory")
flags.DEFINE_integer("n_eval_episodes", 100, "Number of evaluation episodes")
flags.DEFINE_boolean("deterministic", True, "Whether to use deterministic actions")
config_flags.DEFINE_config_file(
    "config",
    "/home/kojogyaase/Projects/Research/carla-rl/dependencies/jaxrl2/examples/configs/drq_default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def load_checkpoint(agent, checkpoint_path):
    """Load agent parameters from checkpoint."""
    state_dict = {
        'actor_params': agent._actor,
        'critic_params': agent._critic,
        'target_critic_params': agent._target_critic_params,
        'temp': agent._temp,
        'rng': agent._rng,
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
    agent._target_critic_params = state_dict['target_critic_params']
    agent._temp = state_dict['temp']
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
        stats["std_distance"] = np.std(distance_completed)
    if slack_values:
        stats["mean_slack"] = np.mean(slack_values)
    
    return stats

def main(_):
    # Create and wrap environment
    config = {
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
                "town": "Town01"
            },
            "experiment": {
                "type": JAXGoalExperiments,
                "hero": {
                    "blueprint": "vehicle.mercedes.coupe_2020",
                    "sensors": {
                        "collision": {"type": "sensor.other.collision"},
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
                        },
                        "imu": {"type": "sensor.other.imu"},
                        "lane_invasion": {"type": "sensor.other.lane_invasion"}
                    }
                },
                "background_activity": {
                    "n_vehicles": 0,
                    "n_walkers": 0,
                    "tm_hybrid_mode": True
                },
                "town": "Town02",
                "others": {
                    "framestack": 1,
                    "max_time_idle": 150,
                    "max_dist": 200,
                    "target_speed": 5.0
                }
            }
        }
    }
    
    env = CarlaGoalEnv(config["env_config"])
    # env = CarlaEvalEnv(config["env_config"])
    env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
    env = FrameStack(env=env, num_stack=1, stacking_key="goal")
    env = TimeLimit(env, max_episode_steps=2500)
    env = RecordEpisodeStatistics(env)
    
    # Initialize agent
    kwargs = dict(FLAGS.config)
    agent = DrQLearner(
        42,  # seed
        env.observation_space.sample(),
        env.action_space.sample(),
        **kwargs
    )
    
    # Load checkpoint
    agent = load_checkpoint(agent, FLAGS.checkpoint_path)
    
    # Evaluate
    stats = evaluate_policy(
        agent,
        env,
        n_eval_episodes=FLAGS.n_eval_episodes,
        deterministic=True
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