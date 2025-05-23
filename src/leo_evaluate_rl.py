import glob
import os
import cv2
import matplotlib.pyplot as plt
import ml_collections
import numpy as np
from absl import app, flags
from flax.training import checkpoints
from leo_eval import LeoEvalEnv
from jaxrl2.agents import DrQLearner, PixelBCLearner
from jaxrl2.agents.resnet_agents import PixelResNetBCLearner
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit
import pickle
from pyflann import *
from mapping.topological_map import TopologicalMap
import rospy
from collections import defaultdict, deque

from src.leo_env import LeoEnv

# Environment variables for performance
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

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", None, "Path to the checkpoint directory")
flags.DEFINE_enum('model', 'PixelResNetBCLearner', ['PixelResNetBCLearner', "PixelResNetDrQLearner"], 'Model to run')
flags.DEFINE_integer("n_eval_episodes", 5, "Number of evaluation episodes")
flags.DEFINE_boolean("deterministic", True, "Whether to use deterministic actions")
flags.DEFINE_string("map_dir", None, "Evaluation directory trajectory")
flags.DEFINE_float("target_speed", 1.0, "Target speed for the Leo robot")

def filter_observations(observation):
    acceptable_keys = ["pixels", "vector"]
    return {k: observation[k] for k in acceptable_keys}

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
        return np.std(self.window)
    
    def get_current(self):
        return self.last_value

def eval_environment(agent, env, n_eval_episodes=10, deterministic=True):
    """
    Evaluate the agent for n_eval_episodes in the Leo environment
    """
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    SPL = []
    SPL_per_skip_frame = []
    distance_completed = []
    slack_values = []
    data = defaultdict(lambda: [])
    mapper = TopologicalMap()
    mean_nodes = []
    
    # Load route information if available
    shortest_distance_along_path = 1e-8
    if FLAGS.map_dir and os.path.exists(f'{FLAGS.map_dir}/aux.pkl'):
        with open(f'{FLAGS.map_dir}/aux.pkl', 'rb') as handle:
            dataset = pickle.load(handle)
            start_position = dataset.get("start")
            goal_position = dataset.get("goal")
            
            if hasattr(env.unwrapped, 'set_start_position') and start_position is not None:
                env.unwrapped.set_start_position(start_position)
            if hasattr(env.unwrapped, 'set_destination_position') and goal_position is not None:
                env.unwrapped.set_destination_position(goal_position)
            
            # Calculate shortest path distance if available
            if "path_distance" in dataset:
                shortest_distance_along_path = dataset["path_distance"]
            elif "path" in dataset:
                # Calculate distance along the path
                path = dataset["path"]
                for i in range(1, len(path)):
                    shortest_distance_along_path += np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
            
            # Set a minimum value
            shortest_distance_along_path = max(shortest_distance_along_path, 0.1)
    
    # Load map data
    difficulty = FLAGS.map_dir.split("/")[-2] if FLAGS.map_dir and len(FLAGS.map_dir.split("/")) > 1 else "default"
    name = FLAGS.model
    path = f"maps/{name}/{difficulty}/*"
    
    for m in glob.glob(path):
        with open(m, 'rb') as handle:
            dataset = pickle.load(handle)
        features = dataset["features"]
        heading = dataset["heading"]
        
        for feat, head in zip(features, heading):
            mapper.update(feat, head)  # Build map
    
    # Evaluation loop
    # for _ in range(n_eval_episodes):
    observation, info = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    
    # If goal observation is available
    if "goal" in observation:
        goal_observation = dict(pixels=observation["goal"][..., None], vector=np.ones_like(observation["vector"]))
        feature = agent.extract_features(filter_observations(goal_observation))
        mapper.update(feature, 1e-8)  # Add goal to map
    
    feature = agent.extract_features(filter_observations(observation))
    subgoal = mapper.create_navigation_guide(len(mapper.des_nodes))
    (_, heading), (goal_reached, index) = subgoal(feature)
    
    # Main interaction loop
    while not (done or goal_reached):
        # Track nodes used for navigation
        if len(index) > 0 and not index[0] in mean_nodes:
            mean_nodes.append(index[0])
        
        # Update vector observations
        target = FLAGS.target_speed
        vecs = observation["vector"]
        current_velocity = env.unwrapped.speed if hasattr(env.unwrapped, 'speed') else 0.0
        current_heading = env.unwrapped.current_heading if hasattr(env.unwrapped, 'current_heading') else 0.0
        
        vecs[2] = np.clip(current_velocity/(target+1e-8), 0.0, 5.1)
        vecs[3] = np.cos(abs(current_heading - heading))
        observation["vector"] = vecs
        
        # Get action from policy
        action_dist = agent.action_dist(filter_observations(observation))
        feature = agent.extract_features(filter_observations(observation))
        action = action_dist.mode()
        
        # Take step in environment
        observation, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1
        done = done or truncated
        
        # Update heading from topological map
        (_, heading), (goal_reached, index) = subgoal(feature)
        
        # Handle episode end
        if done or goal_reached:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Record metrics
            if "is_success" in info:
                success_rate.append(float(info["is_success"]))
            else:
                success_rate.append(0)
            
            if "distance_completed" in info:
                agent_distance_completed = float(info["distance_completed"])
                distance_completed.append(agent_distance_completed)
                
                # Calculate SPL (Success weighted by Path Length)
                if "is_success" in info:
                    S = int(info["is_success"])
                    _spl = S * (shortest_distance_along_path) / max(shortest_distance_along_path, agent_distance_completed)
                    SPL.append(_spl)
                    print("==================SPL===============", _spl, SPL)
            
            if "slack" in info:
                slack_values.append(float(info["slack"]))

    # Compute statistics
    difficulty = FLAGS.map_dir.split("/")[-2] if FLAGS.map_dir and len(FLAGS.map_dir.split("/")) > 1 else "default"
    name = FLAGS.map_dir.split("/")[-1] if FLAGS.map_dir and len(FLAGS.map_dir.split("/")) > 1 else FLAGS.model
    
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "max_SPL": np.nan_to_num(np.mean(SPL), nan=0),
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
    
    # Record all data
    data.update({
        "experiment_results": stats,
        "SPLs": SPL_per_skip_frame,
        "nodes": mapper.heading_nodes.__len__(),
        "mean_nodes": len(mean_nodes)
    })
    
    # Save results
    path = f"results/{FLAGS.model}_ours/{difficulty}"
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/{name}_test_results.pkl", "wb") as f:
        pickle.dump(dict(data), f)
    return stats

def main(_):
    # Initialize ROS node
    rospy.init_node("LEO_EVAL", anonymous=False)
    
    # Create environment
    env = LeoEnv()
    env = TimeLimit(env, max_episode_steps=int(2e4))
    env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
    env = RecordEpisodeStatistics(env)
    
    # Set ROS rate
    rate = rospy.Rate(env.RATE)
    
    # Initialize agent
    if FLAGS.model == "PixelResNetDrQLearner":
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
    
    # Evaluate
    stats = eval_environment(
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