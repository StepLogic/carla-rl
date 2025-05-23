import os
import ml_collections
import numpy as np
from absl import app, flags
from flax.training import checkpoints
from jaxrl2.agents.resnet_agents import PixelResNetBCLearner
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit
import pickle
from collections import defaultdict
from leo_env import LeoEnv
import rospy

# Environment setup
os.environ['XLA_FLAGS'] = "--xla_gpu_enable_command_buffer="
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".30"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# BC configuration
config = ml_collections.ConfigDict()
config.actor_lr = 3e-4
config.hidden_dims = (256, 256)
config.cnn_features = (32, 64, 128, 256)
config.cnn_filters = (3, 3, 3, 3)
config.cnn_strides = (2, 2, 2, 2)
config.cnn_padding = "VALID"
config.latent_dim = 50
config.encoder = "d4pg"
config.dropout_rate = 0.2
bc_config = config.to_dict()

# SAC configuration
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

def filter_observations(observation):
    acceptable_keys = ["pixels", "vector"]
    return {k: observation[k] for k in acceptable_keys}

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", None, "Path to the checkpoint directory")
flags.DEFINE_enum('model', 'PixelResNetBCLearner', ['PixelResNetBCLearner', "PixelResNetDrQLearner"], 'Model to run')
flags.DEFINE_integer("n_eval_episodes", 5, "Number of evaluation episodes")
flags.DEFINE_boolean("deterministic", True, "Whether to use deterministic actions")
flags.DEFINE_string("map_dir", None, "Evaluation directory trajectory")
flags.DEFINE_float("target_speed", 3.0, "Target speed for the Leo robot")
flags.DEFINE_float("target_heading", None, "Target heading angle in radians")

def load_checkpoint(agent, checkpoint_path):
    """Load agent parameters from checkpoint."""
    state_dict = {
        'actor_params': agent._actor,
    }
    
    state_dict = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_path,
        target=state_dict
    )

    agent._actor = state_dict['actor_params']
    
    return agent

def eval_environment(agent, env, n_eval_episodes=1, deterministic=True):
    """Evaluate the agent for n_eval_episodes"""

    episode_rewards = []
    episode_lengths = []
    success_rate = []
    distance_completed = []
    slack_values = []
    data = defaultdict(list)

    # for episode in range(n_eval_episodes):
    print(f"Starting episode {1}/{n_eval_episodes}")
    observation, info = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    slack_actions = 100  # Initial slack period
    
    while not done:
        # Set target speed from flags
        target = FLAGS.target_speed
        
        # Update vector observations
        vecs = observation["vector"]
        current_velocity = env.unwrapped.speed if hasattr(env.unwrapped, 'speed') else 0.0
        current_heading = env.unwrapped.current_heading if hasattr(env.unwrapped, 'current_heading') else 0.0
        target_heading = FLAGS.target_heading if FLAGS.target_heading is not None else 0.0
        
        # Normalize velocity for neural network input
        vecs[2] = np.clip(current_velocity/(target+1e-8), 0.0, 5.1)
        
        # Add heading information if target_heading is specified
        if FLAGS.target_heading is not None:
            vecs[3] = np.cos(abs(current_heading - target_heading))
            
        observation["vector"] = vecs
        
        # Print current status
        print(f"Step: {episode_length}, Velocity: {current_velocity:.2f}, Heading: {np.rad2deg(current_heading):.2f}°", end="\r")
        
        # Initial slack period to stabilize robot
        if slack_actions > 0:
            observation, reward, done, truncated, info = env.step(np.array([0.0, 0.2]))
            slack_actions -= 1
        else:
            # Get action from policy
            action_dist = agent.action_dist(filter_observations(observation))
            action = action_dist.mode() if deterministic else action_dist.sample()
            
            # Take step in environment
            observation, reward, done, truncated, info = env.step(action)
        
        episode_reward += reward
        episode_length += 1
        done = done or truncated
        
        # Optional early termination for safety
        if episode_length > 2500:
            print("\nReached maximum episode length.")
            break
    
    # Record episode results
    print(f"\nEpisode {1} completed: Length={episode_length}, Reward={episode_reward:.4f}")
    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)
    
    if "is_success" in info:
        success_rate.append(float(info["is_success"]))
    else:
        success_rate.append(0)
        
    if "distance_completed" in info:
        distance_completed.append(float(info["distance_completed"]))
        
    if "slack" in info:
        slack_values.append(float(info["slack"]))

    # Compute statistics
    map_dir_parts = FLAGS.map_dir.split("/") if FLAGS.map_dir else ["default"]
    difficulty = map_dir_parts[-2] if len(map_dir_parts) > 1 else "default"
    name = map_dir_parts[-1] if len(map_dir_parts) > 1 else FLAGS.model
    
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
    }
    
    if success_rate:
        stats["std_success_rate"] = np.std(success_rate)
        stats["success_rate"] = np.mean(success_rate)
        stats["max_success_rate"] = np.max(success_rate)
        
    if distance_completed:
        stats["mean_distance"] = np.mean(distance_completed)
        stats["std_distance"] = np.std(distance_completed)
        
    if slack_values:
        stats["mean_slack"] = np.mean(slack_values)
        
    # Save all evaluation data
    data.update({
        "experiment_results": stats,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "success_rate": success_rate,
        "distance_completed": distance_completed,
    })
    
    # Create results directory
    path = f"results/{FLAGS.model}/{difficulty}"
    os.makedirs(path, exist_ok=True)
    
    # Save results
    results_file = f"{path}/{name}_test_results.pkl"
    with open(results_file, "wb") as f:
        pickle.dump(dict(data), f)
    print(f"Results saved to {results_file}")
    
    return stats

def main(_):
    # Initialize ROS node
    rospy.init_node("BC_EVALUATION", anonymous=False)
    
    # Create environment
    env = LeoEnv()
    env = TimeLimit(env, max_episode_steps=2500)
    env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
    env = RecordEpisodeStatistics(env)
    
    # Set ROS rate if available
    if hasattr(env, 'RATE'):
        rate = rospy.Rate(env.RATE)
    
    # Initialize agent
    if FLAGS.model == "DrQLearner":
        config = sac_config
    else:
        config = bc_config
        
    agent = globals()[FLAGS.model](
        0,  # seed
        filter_observations(env.observation_space.sample()),
        env.action_space.sample(),
        **config
    )
    
    # Load checkpoint
    agent = load_checkpoint(agent, FLAGS.checkpoint_path)
    
    # Print evaluation settings
    print("\nEvaluation Settings:")
    print("=" * 50)
    print(f"Model: {FLAGS.model}")
    print(f"Checkpoint: {FLAGS.checkpoint_path}")
    print(f"Episodes: {FLAGS.n_eval_episodes}")
    print(f"Target Speed: {FLAGS.target_speed}")
    if FLAGS.target_heading is not None:
        print(f"Target Heading: {np.rad2deg(FLAGS.target_heading):.2f}°")
    print("=" * 50)

    # Evaluate
    stats = eval_environment(
        agent,
        env,
        n_eval_episodes=FLAGS.n_eval_episodes,
        deterministic=FLAGS.deterministic
    )
    
    # Print final results
    print("\nEvaluation Results:")
    print("=" * 50)
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    flags.mark_flag_as_required("checkpoint_path")
    app.run(main)