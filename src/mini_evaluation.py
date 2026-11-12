from collections import defaultdict, deque
from functools import partial
import glob
import os
import pickle
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

def evaluate_policy(env:CarlaEvalEnv,experiment:str,model_type:str,map_dir:str,origin=None,destination=None):
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

    termination_budget=10
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
            agent = MODEL(ckpt_path=checkpoint_path,mode="navigate",skip_index=skip_index ,map_dir=map_dir)
            print(f"Using {len(agent.topomap)} Nodes")
            # map_dir="/home/kojogyaase/Projects/Research/carla-rl/topomap"
            shortest_distance_along_road=None
            if origin is None or destination is None:
                with open(f'{map_dir}/aux.pkl', 'rb') as handle:
                    dataset=pickle.load(handle)
                    origin=ndarray_to_location(dataset["start"])
                    destination=ndarray_to_location(dataset["goal"])
            
            env.unwrapped.set_start_transform(origin)
            env.unwrapped.set_destination_transform(destination)
            route_plannner=GlobalRoutePlanner(env.unwrapped.core.map, 2.0)
            prev_waypoint=None
            trace=route_plannner.trace_route(origin,destination)
            for wp,_ in trace:
                if shortest_distance_along_road is None:
                    shortest_distance_along_road=0
                if prev_waypoint is None:
                    prev_waypoint=wp
                shortest_distance_along_road+=prev_waypoint.transform.location.distance(wp.transform.location)
                prev_waypoint=wp
            if shortest_distance_along_road <= 2.0:
                shortest_distance_along_road=origin.distance(goal_location)

        observation, info = env.reset()
        spAgent=SetPointAgent(env.unwrapped.core.hero)
        while not done:
            waypoints  = agent.eval_action(observation["pixels"])
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
                    print("termination_budget",termination_budget)
                env.unwrapped.set_start_transform(env.unwrapped.last_position.location)
                
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "max_SPL":np.nan_to_num(np.mean(SPL),nan=0),
        # "mean_distance_per_step":dataset.get("mean_distance_per_step",1.0),
        "distance_completed":distance_completed,
        "nodes":len(agent.topomap),
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
        "experiment_results":stats,
        "SPLs":SPL_per_skip_frame,
    })
    # criteria=map_dir.split("/")[-3]
    # difficulty=map_dir.split("/")[-2]
    difficulty = map_dir.split("/")[-1]
    path=f"mini_results/{difficulty}"
    os.makedirs(path,exist_ok=True)
    with open(f"{path}/{model_type}_test_results.pkl", "wb") as f:
        pickle.dump(dict(data), f)
    return stats


def run_agent():
    env = CarlaEvalEnv(start_server=True,town="Town02")
    env = TimeLimit(env, max_episode_steps=int(1e3))
    env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
    env = FrameStack(env=env, num_stack=1, stacking_key="goal")
    env = RecordEpisodeStatistics(env)

    base="/home/robotlab/scratch/carla-rl"
    for experiment in ["experiment"]:
        wrapped_func=partial(evaluate_policy,env,experiment)
        for model_type in ["gnm","nomad","vint"]:
            model_wrapped_func=partial(wrapped_func,model_type)
            for difficulty  in os.listdir(f"{base}/{experiment}"):
                    # for trajectory  in os.listdir(f"{base}/{experiment}/{model_type}/{criterion}/{difficulty}"):
                        map_dir=f"{base}/{experiment}/{difficulty}"
                        # dense_map=f"/home/robotlab/scratch/carla-rl/evaluation_trajectory/gnm/dense/{difficulty}/{trajectory}"
                        # the goal image is not always present in the topological map
                        # goal_image = sorted(glob.glob(f"{dense_map}/*.jpg"), 
                        #      key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))[-1]
                        # shutil.copy(goal_image,f"{map_dir}")
                        # print(goal_image,map_dir)
                        with open(f'{map_dir}/aux.pkl', 'rb') as handle:
                            dataset=pickle.load(handle)
                            origin=ndarray_to_location(dataset["start"])
                            destination=ndarray_to_location(dataset["goal"])
                            print("origin",origin,destination)
                        # try:
                            model_wrapped_func(map_dir,origin=origin,destination=destination)
                        # except:
                            # breakpoint()
                        # run_agent_on_map()

run_agent()