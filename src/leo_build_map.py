import os
import random
import time
import cv2
import matplotlib.pyplot as plt
import ml_collections
import numpy as np
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
from leo_eval import LeoEvalEnv
from jaxrl2.agents import DrQLearner, PixelBCLearner, PixelResNetBCLearner,PixelResNetDrQLearner
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit
import pickle
from collections import deque
import rospy
from src.mapping.topological_map import TopologicalMap

# Set environment variables for better GPU performance
os.environ['XLA_FLAGS'] = "--xla_gpu_enable_command_buffer="
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".30"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", None, "Path to the checkpoint directory")
flags.DEFINE_enum('model', 'PixelResNetBCLearner', ['PixelResNetBCLearner', "PixelResNetDrQLearner"], 'Model to run')
flags.DEFINE_integer("n_eval_episodes", 5, "Number of evaluation episodes")
flags.DEFINE_boolean("deterministic", True, "Whether to use deterministic actions")
flags.DEFINE_string("map_dir", None, "Evaluation directory trajectory")
flags.DEFINE_float("target_speed", 1.0, "Target speed for the Leo robot")
flags.DEFINE_float("target_heading", None, "Target heading angle (in radians)")

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
config.target_entropy = None
config.backup_entropy = True
config.critic_reduction = "mean"
sac_config = config.to_dict()

class RealTimeVectorEMA:
    def __init__(self, window_size=100, alpha=0.1, vector_dim=3):
        self.window = deque(maxlen=window_size)
        self.alpha = alpha
        self.last_value = np.zeros(vector_dim)
        
    def update(self, new_vector):
        self.window.append(new_vector)
        self.last_value = self.alpha * new_vector + (1 - self.alpha) * self.last_value
        return self.last_value
    
    def threshold(self):
        return np.std(self.window) * 1.5
    
    def get_current(self):
        return self.last_value

def filter_observations(observation):
    acceptable_keys = ["pixels", "vector"]
    return {k: observation[k] for k in acceptable_keys}

def load_checkpoint(agent, checkpoint_path):
    """Load agent parameters from checkpoint."""
    if FLAGS.model != "PixelResNetDrQLearner":
        state_dict = {
            'actor_params': agent._actor,
        }
    else:
        state_dict = {
            'actor_params': agent._actor,
            'critic_params': agent._critic,
        }
    
    state_dict = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_path,
        target=state_dict
    )

    # Update agent parameters
    agent._actor = state_dict['actor_params']
    if 'critic_params' in state_dict and hasattr(agent, '_critic'):
        agent._critic = state_dict['critic_params'] 
    
    return agent

def map_environment(agent, env, deterministic=True):
    """Map the environment by running the agent for a single episode."""
    steps = 0
    log_stds = []
    trace_log_stds = []
    moving_average = []
    junctions = []
    images = []
    features = []
    heading_ar = []
    locations = []
    unit_vectors = []
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    distance_completed = []
    slack_values = []
    steps=0
    log_stds=[]
    trace_log_stds=[]
    moving_average=[]
    junctions=[]
    restarts=[]
    images=[]
    features=[]
    heading_ar=[]
    locations=[]
    unit_vectors=[]
    ema_filter = RealTimeVectorEMA(window_size=100, vector_dim=2)
    start = time.time()
    mapper = TopologicalMap() if 'TopologicalMap' in globals() else None
    
    # Single episode run
    print("Starting episode")
    observation, info = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    reward = 0
    slack_actions = 100
    truncated = False
    
    while not done:
        target = FLAGS.target_speed
        vecs = observation["vector"]
        
        # Update vector observations based on current robot state
        current_velocity = env.unwrapped.speed
        current_heading = env.unwrapped.current_heading
        target_heading = FLAGS.target_heading if FLAGS.target_heading is not None else np.pi
        
        # Normalize velocity and heading for neural network input
        vecs[2] = np.clip(current_velocity/(target+1e-8), 0.0, 1.0)
        vecs[3] = np.cos(np.pi)
        observation["vector"] = vecs
        
        # Get action from agent
        if deterministic:
            action = agent.eval_actions(filter_observations(observation))
        else:
            action = agent.sample_actions(filter_observations(observation))
        
        # Initial slack period to stabilize robot
        if slack_actions == 0:
            observation, reward, done, truncated, info = env.step(action)
        else:
            observation, reward, done, truncated, info = env.step(np.array([0.0, 0.2]))
            slack_actions -= 1
        
        observation["vector"] = vecs
        action_dist = agent.action_dist(filter_observations(observation))
        feature = agent.extract_features(filter_observations(observation))
        
        action = action_dist.mode()
        observation, reward, done, truncated, info = env.step(action)
        
        # Check if robot is at a junction (if applicable to LeoEnv)
        is_at_junction = False
        unit_vector = np.zeros(3)
        location = np.zeros(3)
        
        if hasattr(env.unwrapped, 'is_robot_at_junction'):
            is_at_junction, unit_vector, location = env.unwrapped.is_robot_at_junction()
        
        if is_at_junction:
            junctions.append(steps)
            
        locations.append(location)
        unit_vectors.append(unit_vector)
        
        # Track action distribution statistics
        std = np.array(action_dist.stddev())
        log_stds.append(std)
        trace_log_stds.append(np.sum(std**2))
        filtered_vector = ema_filter.update(np.sum(std**2))
        moving_average.append(filtered_vector)
        
        episode_reward += reward
        episode_length += 1
        done = done or truncated
        steps += 1
        
        # Print current status
        print(f"Step: {steps}, Reward: {reward:.4f}, Heading: {np.rad2deg(current_heading):.2f}°, Speed: {current_velocity:.2f}", end="\r")
        
        # Add observation to map if uncertainty threshold is exceeded
        if mapper is not None and np.any((ema_filter.get_current()-np.sum(std**2)) > ema_filter.threshold()) and not done:
            obs = (observation["pixels"][..., 0]*255).astype(np.uint8)
            heading_obs = env.unwrapped.current_heading
            images.append(obs)
            heading_ar.append(heading_obs)
            mapper.update(feature, heading_obs)
            features.append(feature)
            cv2.imwrite(f"leo_observation_{steps}.jpg", (observation["pixels"][..., 0]*255).astype(np.uint8))
        
        # Break if exceeding step limit (5000 steps)
        if steps > int(5e3):
            print("\nReached maximum step limit.")
            break
    
    end = time.time()
    difficulty = FLAGS.map_dir.split("/")[-2] if FLAGS.map_dir else "default"
    name = FLAGS.map_dir.split("/")[-1] if FLAGS.map_dir else FLAGS.model
    
    # Compute statistics
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "total_map_steps": steps,
        "number_of_restarts": len(restarts),
        "exploration_time": int(end-start)
    }
    
    if success_rate:
        stats["success_rate"] = np.mean(success_rate)
    if distance_completed:
        stats["mean_distance"] = np.mean(distance_completed)
    if slack_values:
        stats["mean_slack"] = np.mean(slack_values)
    
    # Save map data if applicable
    if mapper is not None:
        map_data = dict(images=images, heading=heading_ar, features=features, locations=locations)
        path = f"maps/{FLAGS.model}/{difficulty}"
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/{name}.pickle", 'wb') as handle:
            pickle.dump(map_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return stats

def main(_):
    # Initialize ROS node for Leo
    rospy.init_node("LEO_EVAL", anonymous=False)
    
    # Create environment
    env = L(target_speed=FLAGS.target_speed)
    # env = TimeLimit(env, max_episode_steps=2500)
    env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
    env = RecordEpisodeStatistics(env)
    
    # Set ROS rate
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
    
    # Set a start position if available in the environment
    if hasattr(env.unwrapped, 'set_start_position') and hasattr(env.unwrapped, 'get_random_position'):
        start_pos = env.unwrapped.get_random_position()
        env.unwrapped.set_start_position(start_pos)
        print(f"Starting from position: {start_pos}")
    
    # Run a single episode
    start_time = time.time()
    
    # Run the agent in the environment
    stats = map_environment(
        agent,
        env,
        deterministic=FLAGS.deterministic
    )
    
    # Save the results
    results_path = f"leo_results_{int(time.time())}.pkl"
    with open(results_path, 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    flags.mark_flag_as_required("checkpoint_path")
    app.run(main)