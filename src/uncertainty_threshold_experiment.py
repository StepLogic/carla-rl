from collections import defaultdict, deque
from functools import partial
import glob
import os
import pickle
import random
import carla
import cv2
import numpy as np
import shutil
from absl import app, flags
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.timelimit import TimeLimit
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from rlib_integration.agent import GlobalRoutePlanner
from rlib_integration.helper import ndarray_to_location
from carla_eval import CarlaEvalEnv
from navigation_policies.baseline_policies.nomad_policy import NoMaD
from navigation_policies.baseline_policies.gnm_policy import GNM_Policy
from navigation_policies.baseline_policies.gnm_policy_v2 import GNM_Policy_V2
from navigation_policies.baseline_policies.vint_policy import ViNT_Policy
from agent_wrapper import SetPointAgent


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
flags.DEFINE_integer("n_eval_episodes", 5, "Number of evaluation episodes")
flags.DEFINE_boolean("deterministic", True, "Whether to use deterministic actions")
flags.DEFINE_string("map_dir", None, "Whether to use deterministic actions")
flags.DEFINE_string("town", "Town01", "Town Name")

def evaluate_policy(env:CarlaEvalEnv,experiment:str,model_type:str,map_dir:str,origin=None,destination=None,locations=None,similarity_matrix=None,threshold=None):
    """Evaluate the agent for n_eval_episodes."""
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    SPL = []
    SPL_per_skip_frame = []
    distance_completed = []
    slack_values = []
    skip_index=1
    visited_locations=[origin]
    data=defaultdict(lambda :[])
    models={
        "gnm":GNM_Policy,
        "nomad":NoMaD,
        "vint":ViNT_Policy,
        "gnm_v2":GNM_Policy_V2
    }
    default_checkpoints={
        "gnm":"/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/gnm.pth",
        "gnm_v2":"/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/gnm_modified.pth",
        "nomad":"/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/nomad.pth",
        "vint":"/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/vint.pth"
    }
    MODEL=models[model_type]
    checkpoint_path=default_checkpoints[model_type]
    # https://github.com/carla-simulator/carla/issues/2832

    termination_budget=5
    while termination_budget>0:
        done = False
        episode_reward = 0
        episode_length = 0
        goal_location=None
        shortest_distance_along_road=1e-8
        print("Map directory",map_dir)
        if map_dir is None:
            if model_type != "nomad":
                raise ValueError("Only NoMaD can explore")
            agent = MODEL(ckpt_path=checkpoint_path,mode="explore")
        else:
            agent = MODEL(ckpt_path=checkpoint_path,mode="navigate",skip_index=skip_index ,map_dir=map_dir,threshold=threshold,similarity_matrix=similarity_matrix)
            print(f"Using {len(agent.topomap)} Nodes")
            # map_dir="/home/kojogyaase/Projects/Research/carla-rl/topomap"
            shortest_distance_along_road=None
            if origin is None or destination is None:
                with open(f'{map_dir}/aux.pkl', 'rb') as handle:
                    dataset=pickle.load(handle)
                    origin=dataset["start"]
                    destination=dataset["goal"]
            # 
            # else:
                # origin,destination=carla.Location(x=origin[0],y=origin[1],z=origin[2]),carla.Location(x=destination[0],y=destination[1],z=destination[2])

            env.unwrapped.set_start_transform(ndarray_to_location(origin))
            env.unwrapped.set_destination_transform(ndarray_to_location(destination))
            route_plannner=GlobalRoutePlanner(env.unwrapped.core.map, 2.0)
            prev_waypoint=None
            trace=route_plannner.trace_route(ndarray_to_location(origin),ndarray_to_location(destination))
            for wp,_ in trace:
                if shortest_distance_along_road is None:
                    shortest_distance_along_road=0
                if prev_waypoint is None:
                    prev_waypoint=wp
                shortest_distance_along_road+=prev_waypoint.transform.location.distance(wp.transform.location)
                prev_waypoint=wp
            if shortest_distance_along_road <= 2.0:
                shortest_distance_along_road=ndarray_to_location(origin).distance(ndarray_to_location(destination))

        observation, info = env.reset()
        spAgent=SetPointAgent(env.unwrapped.core.hero)
        while not done:
            waypoints  = agent.eval_action_v2(observation["pixels"])
            if not locations is None:
                visited_locations.append(locations[agent.closest_node])
            actions=spAgent.run_step(waypoints) 
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
                    if shortest_distance_along_road is None:
                        shortest_distance_along_road=agent_distance_completed

                    _spl=S*(shortest_distance_along_road)/max(shortest_distance_along_road,agent_distance_completed)
                    SPL.append(_spl)
                    # print("==================SPL===============",_spl,SPL)
                    
                if "slack" in info:
                    slack_values.append(float(info["slack"]))
                if "is_success" in info:
                    termination_budget=0
                else:
                    termination_budget-=1
                env.unwrapped.set_start_transform(env.unwrapped.last_position.location)
    path_to_goal_length=None
    prev_waypoint=None
    trace=route_plannner.trace_route(env.unwrapped.last_position.location,ndarray_to_location(destination))
    for wp,_ in trace:
        if path_to_goal_length is None:
            path_to_goal_length=0
        if prev_waypoint is None:
            prev_waypoint=wp
        path_to_goal_length+=prev_waypoint.transform.location.distance(wp.transform.location)
        prev_waypoint=wp
    if path_to_goal_length <= 2.0:
        path_to_goal_length=env.unwrapped.last_position.location.distance(ndarray_to_location(destination))


    path_from_origin_length=None
    prev_waypoint=None
    trace=route_plannner.trace_route(ndarray_to_location(origin),env.unwrapped.last_position.location)
    for wp,_ in trace:
        if path_from_origin_length is None:
            path_from_origin_length=0
        if prev_waypoint is None:
            prev_waypoint=wp
        path_from_origin_length+=prev_waypoint.transform.location.distance(wp.transform.location)
        prev_waypoint=wp
    if path_from_origin_length <= 2.0:
        path_from_origin_length=ndarray_to_location(origin).distance(env.unwrapped.last_position.location)

    visited_locations.append(destination)
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "max_SPL":np.nan_to_num(np.mean(SPL),nan=0),
        # "mean_distance_per_step":dataset.get("mean_distance_per_step",1.0),
        "distance_completed":distance_completed,
        "nodes":len(agent.topomap),
        "shortest_distance_along_road":shortest_distance_along_road,
        "distance_from_origin":path_from_origin_length,
        "distance_to_goal":path_to_goal_length,
        "dist_in_efficience":((path_from_origin_length)+path_to_goal_length*10)/shortest_distance_along_road
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
    stats["locations"]=visited_locations
    data.update({
        "experiment_results":stats,
        "SPLs":SPL_per_skip_frame,
    })
    # criteria=map_dir.split("/")[-3]
    # difficulty=map_dir.split("/")[-2]
    # trajectory = map_dir.split("/")[-1]
    # path=f"results/{experiment}/{criteria}/{difficulty}/{trajectory}"
    # os.makedirs(path,exist_ok=True)
    # with open(f"{path}/{model_type}_test_results.pkl", "wb") as f:
    #     pickle.dump(dict(data), f)
    return stats


def run_agent():
    
    base="/home/robotlab/scratch/carla-rl/topomap"
    results_store="/home/robotlab/scratch/carla-rl/evaluation_v2/uncertainty"
    trajectories=iter([[1362, 1354],
    [627, 472],
    [741, 1495],
    [561, 244],
    ])
    with open(f"{base}/traj.pkl","rb") as pipe:
        traj_data=pickle.load(pipe)
    locations = traj_data["locations"]
    # origin=random.randint(0,len(locations))
    # start_location_pairs=[]
   
    # origin,destination=locations[origin],locations[destination]
    # max_len=int(abs(origin-destination)*5)
    # print(origin,destination)

    env = CarlaEvalEnv(start_server=False,town="Town02")
    env = TimeLimit(env, max_episode_steps=int(2e3))
    env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
    env = FrameStack(env=env, num_stack=1, stacking_key="goal")
    env = RecordEpisodeStatistics(env)

    for traj in range(4):
        # origin,destination=random.choices(list(range(0,len(locations))),k=2)
        origin,destination=next(trajectories)
        origin,destination=locations[origin],locations[destination]
        os.makedirs(f"{results_store}/{traj}",exist_ok=True)
        for fx in range(0,50,1):
            stats = evaluate_policy(env,"results","gnm",map_dir="/home/robotlab/scratch/carla-rl/topomap",origin=origin,destination=destination,similarity_matrix="/home/robotlab/scratch/carla-rl/cache/gnm/uncertainty_similarity.pth",threshold=fx/10,locations=locations)
            with open(f"{results_store}/{traj}/gnm_{fx}.pkl", "wb") as f:
                pickle.dump(stats, f)
  
run_agent()