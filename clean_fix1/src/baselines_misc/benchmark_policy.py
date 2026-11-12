from collections import defaultdict, deque
import os
import pickle
import cv2
import numpy as np
from absl import app, flags
from ml_collections import config_flags
from flax.training import checkpoints
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.timelimit import TimeLimit
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from rlib_integration.agent import GlobalRoutePlanner
from rlib_integration.carla_goal_env import CarlaGoalEnv
from rlib_integration.helper import ndarray_to_location
from src.carla_eval import CarlaEvalEnv
from src.jax_experiments_goal import JAXGoalExperiments
from navigation_policies.baseline_policies.nomad_policy import NoMaD
from navigation_policies.baseline_policies.gnm_policy import GNM_Policy
from navigation_policies.baseline_policies.vint_policy import ViNT_Policy

from agent_wrapper import SetPointAgent
from custom_controller import VehiclePIDController


# task
# tune pid controller
# fix

os.environ['XLA_FLAGS']="--xla_gpu_enable_command_buffer="
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".20"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
# Define flags
FLAGS = flags.FLAGS
# flags.DEFINE_string("checkpoint_path", None, "Path to the checkpoint directory")
flags.DEFINE_enum('model', 'nomad', ['nomad', 'gnm','vint'], 'Model type')
flags.DEFINE_integer("n_eval_episodes", 10, "Number of evaluation episodes")
flags.DEFINE_boolean("deterministic", True, "Whether to use deterministic actions")
flags.DEFINE_string("map_dir", None, "Whether to use deterministic actions")
flags.DEFINE_string("town", "Town01", "Town Name")
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
    agent._actor = state_dict['actor_params']
    agent._critic = state_dict['critic_params'] 
    agent._target_critic_params = state_dict['target_critic_params']
    agent._temp = state_dict['temp']
    return agent

def evaluate_policy(model_type,env, n_eval_episodes=10, deterministic=True):
    """Evaluate the agent for n_eval_episodes."""
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    SPL = []
    SPL_per_skip_frame = []
    distance_completed = []
    slack_values = []
    skip_index=1
   
    data=defaultdict(lambda :[])
    models={
        "gnm":GNM_Policy,
        "nomad":NoMaD,
        "vint":ViNT_Policy
    }
    default_checkpointts={
        "gnm":"/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/gnm.pth",
        "nomad":"/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/nomad.pth",
        "vint":"/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/vint.pth"
    }
    MODEL=models[model_type]
    checkpoint_path=default_checkpointts[model_type]
    # https://github.com/carla-simulator/carla/issues/2832

    for i in range(n_eval_episodes):
        done = False
        episode_reward = 0
        episode_length = 0
        goal_location=None
        shortest_distance_along_road=1e-8
        if FLAGS.map_dir is None:
            if model_type != "nomad":
                raise ValueError("Only NoMaD can explore")
            agent = MODEL(ckpt_path=checkpoint_path,mode="explore")
        else:
            agent = MODEL(ckpt_path=checkpoint_path,mode="navigate",skip_index=skip_index ,map_dir=FLAGS.map_dir)
            # map_dir="/home/kojogyaase/Projects/Research/carla-rl/topomap"
            with open(f'{FLAGS.map_dir}/aux.pkl', 'rb') as handle:
                dataset=pickle.load(handle)
                # breakpoint()
                start_location=ndarray_to_location(dataset["start"])
                goal_location=ndarray_to_location(dataset["goal"])
                env.unwrapped.set_start_transform(start_location)
                route_plannner=GlobalRoutePlanner(env.unwrapped.core.map, 2.0)
        
                prev_waypoint=None
                trace=route_plannner.trace_route(start_location,goal_location)
                for wp,_ in trace:
                    if prev_waypoint is None:
                        prev_waypoint=wp
                    shortest_distance_along_road+=prev_waypoint.transform.location.distance(wp.transform.location)
                    prev_waypoint=wp
                assert shortest_distance_along_road>1.0
        # print("shortest_distance_along_roads is",shortest_distance_along_road)
        observation, info = env.reset()
        if not FLAGS.map_dir is None:
            goal=np.asarray(agent.topomap[agent.goal_node])
            print("Goal Node",agent.goal_node)
            # cv2.imwrite("goal.jpg",goal)
            env.unwrapped.set_goal(goal,0.0,goal_location)
     
        spAgent=SetPointAgent(env.unwrapped.core.hero)
        while not done:
            waypoints = np.array(agent.eval_action(observation["pixels"]))
            actions=spAgent.run_step(waypoints)
            # actions=spAgent.run_step(waypoints)
            # config["env_config"]["carla"]["timestep"]  
            observation, reward, done, truncated, info = env.step(actions)
            episode_reward += reward
            episode_length += 1
            done = done or truncated
            
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                # print(info)
                if "is_success" in info:
                    success_rate.append(float(info["is_success"]))
                    data[skip_index].append(float(info["is_success"]))
                else:
                    data[skip_index].append(0)
                    success_rate.append(0)
                if "distance_completed" in info :
                    distance_completed.append(float(info["distance_completed"]))
                if "distance_completed" in info:
                    # https://arxiv.org/pdf/1807.06757
                    agent_distance_completed=float(info["distance_completed"])
                    S=int(info.get("is_success",0)) #for timeouts where this key is not 
                    _spl=S*(shortest_distance_along_road)/max(shortest_distance_along_road,agent_distance_completed)
                    SPL.append(_spl)
                    # distance_completed.append()
                    print("==================SPL===============",_spl,SPL)
                    
                if "slack" in info:
                    slack_values.append(float(info["slack"]))
                # if i%5==0 and i!=0:
                #     skip_index+=2
                #     l=np.nan_to_num(np.mean(SPL),nan=0)
                #     SPL_per_skip_frame.append(l)
                #     # breakpoint()
                #     print("=================Skip Index==============",skip_index,l,SPL_per_skip_frame)
                #     SPL = []
    # breakpoint()
    # Compute statistics
    difficulty=FLAGS.map_dir.split("/")[-2]
    name=FLAGS.map_dir.split("/")[-1] or model_type
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "max_SPL":np.nan_to_num(np.mean(SPL),nan=0),
        "mean_distance_per_step":dataset.get("mean_distance_per_step",1.0)
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
        "experiment_results":stats,
        "SPLs":SPL_per_skip_frame,
    })
    path=f"results/{model_type}/{difficulty}"
    os.makedirs(path,exist_ok=True)
    with open(f"{path}/{name}_test_results.pkl", "wb") as f:
        pickle.dump(dict(data), f)
    return stats

def main(_):
    # Create and wrap environment
    config["env_config"]["carla"]["town"]=FLAGS.town
    env = CarlaEvalEnv(config["env_config"],use_rgb=True,image_size=96,start_server=False)
    env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
    env = FrameStack(env=env, num_stack=1, stacking_key="goal")
    env = TimeLimit(env, max_episode_steps=3500)
    env = RecordEpisodeStatistics(env)

    # Initialize agent
    # kwargs = dict(FLAGS.config)

    # Evaluate
    stats = evaluate_policy(
        FLAGS.model,
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
    # flags.mark_flag_as_required("checkpoint_path")
    app.run(main)