import os
import cv2
import ml_collections
import numpy as np
from absl import app, flags
from flax.training import checkpoints
from carla_eval import CarlaEvalEnv
from jaxrl2.agents import DrQLearner, PixelBCLearner
from jaxrl2.agents.resnet_agents import PixelResNetBCLearner
from rlib_integration.helper import ndarray_to_location
from rlib_integration.agent import GlobalRoutePlanner
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit
import pickle
from collections import defaultdict

# Environment setup
os.environ['XLA_FLAGS']="--xla_gpu_enable_command_buffer="
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".30"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

# BC configuration
config = ml_collections.ConfigDict()
config.actor_lr = 3e-4
config.hidden_dims = (256, 256)
config.cnn_features = (8, 16, 32, 64)
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
config.cnn_features = (8, 16, 32, 64)
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
flags.DEFINE_enum('model', 'DrQLearner', ['DrQLearner', 'PixelResNetBCLearner', "PixelBCLearner"], 'Model to run')
flags.DEFINE_integer("n_eval_episodes", 5, "Number of evaluation episodes")
flags.DEFINE_boolean("deterministic", True, "Whether to use deterministic actions")
flags.DEFINE_string("map_dir", None, "Evaluation directory trajectory")
flags.DEFINE_string("town", "Town01", "Town Name")

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

def eval_environment(agent, env, n_eval_episodes=10, deterministic=True):
    """Evaluate the agent for n_eval_episodes"""

    episode_rewards = []
    episode_lengths = []
    success_rate = []
    SPL = []
    SPL_per_skip_frame = []
    distance_completed = []
    slack_values = []
    truncate_steps = 0
    data = defaultdict(list)

    shortest_distance_along_road = 1e-8
    with open(f'{FLAGS.map_dir}/aux.pkl', 'rb') as handle:
        dataset = pickle.load(handle)
        start_location = ndarray_to_location(dataset["start"])
        goal_location = ndarray_to_location(dataset["goal"])
        env.unwrapped.set_start_transform(start_location)
        env.unwrapped.set_destination_transform(goal_location)

        route_plannner = GlobalRoutePlanner(env.unwrapped.core.map, 2.0)
        prev_waypoint = None
        try:
            trace = route_plannner.trace_route(start_location, goal_location)
            for wp, _ in trace:
                if prev_waypoint is None:
                    prev_waypoint = wp
                shortest_distance_along_road += prev_waypoint.transform.location.distance(wp.transform.location)
                prev_waypoint = wp
        except:
                if shortest_distance_along_road <= 2.0:
                    shortest_distance_along_road=start_location.distance(goal_location)
    termination_budget=10
    while termination_budget>0:
        observation, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        # heading = 0 + 1e-8
        while not done:
            truncate_steps += 1
            vecs=observation["vector"]
            vecs[3]=-1.0
            observation["vector"]=vecs
            action = agent.eval_actions(filter_observations(observation))
            observation, reward, done, truncated, info = env.step(action)

            # action = action_dist.mode()
            
            # observation, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = done or truncated
            
            if done:
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                if "is_success" in info:
                    success_rate.append(float(info["is_success"]))
                else:
                    success_rate.append(0)
                    
                if "distance_completed" in info:
                    distance_completed.append(float(info["distance_completed"]))

                if "is_success" in info:
                    termination_budget=0
                else:
                    termination_budget-=1
                env.unwrapped.set_start_transform(env.unwrapped.last_position.location)
                if "slack" in info:
                    slack_values.append(float(info["slack"]))

        # Compute statistics
    difficulty = FLAGS.map_dir.split("/")[-2]
    name = FLAGS.map_dir.split("/")[-1] or FLAGS.model
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "max_SPL": np.nan_to_num(np.mean(SPL), nan=0),
        "mean_distance_per_step": dataset.get("mean_distance_per_step", 1.0),
        "distance_completed":distance_completed,
        "shortest_distance_along_road":shortest_distance_along_road
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
        
    data.update({
        "experiment_results": stats,
        "SPLs": SPL_per_skip_frame,
    })
    
    path = f"results/{FLAGS.model}/{difficulty}"
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/{name}_test_results.pkl", "wb") as f:
        pickle.dump(dict(data), f)
        
    return stats

def main(_):
    env = CarlaEvalEnv(start_server=True, town=FLAGS.town)
    env = TimeLimit(env, max_episode_steps=int(2e4))
    env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
    env = RecordEpisodeStatistics(env)

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
        if not isinstance(value,list):
            print(f"{key}: {value:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    flags.mark_flag_as_required("checkpoint_path")
    app.run(main)