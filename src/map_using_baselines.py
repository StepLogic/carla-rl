# %%
from collections import defaultdict, deque
from functools import partial
import glob
import os
import pickle
import carla
import cv2
import numpy as np
from absl import app, flags
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.timelimit import TimeLimit
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from rlib_integration.agent import GlobalRoutePlanner
from rlib_integration.helper import carla_location_to_np_array, ndarray_to_location
from carla_eval import CarlaEvalEnv
from navigation_policies.baseline_policies.nomad_policy import NoMaD
from navigation_policies.baseline_policies.gnm_policy import GNM_Policy
from navigation_policies.baseline_policies.vint_policy import ViNT_Policy
from agent_wrapper import SetPointAgent
from PIL import Image
from rlib_integration.agent import BasicAgent
os.environ['XLA_FLAGS']="--xla_gpu_enable_command_buffer="
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".20"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
# Define flags
# FLAGS = flags.FLAGS
# # flags.DEFINE_string("checkpoint_path", None, "Path to the checkpoint directory")
#  DEFINE_enum('model', 'nomad', ['nomad', 'gnm','vint'], 'Model type')
#  DEFINE_integer("n_eval_episodes", 5, "Number of evaluation episodes")
#  DEFINE_boolean("deterministic", True, "Whether to use deterministic actions")
# flags.DEFINE_string("map_dir", None, "Whether to use deterministic actions")
# flags.DEFINE_string("town", "Town01", "Town Name")


# %%

def map_environment(env,model_type="nomad",map_dir=None,mode=["dense","sparse_dist","sparse_actions"],difficulty="easy",threshold=0.1,n_eval_episodes=10, deterministic=True,origin=None,destination=None):
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    SPL = []
    SPL_per_skip_frame = []
    distance_completed = []
    slack_values = []
    skip_index=37
    data=defaultdict(lambda :[])
    locations=[]

    MODEL=None
    checkpoint_path=None
    agent=None

    if mode !="dense":
        model_type="nomad"
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

    # termination_budget=10
    # while termination_budget>0:
    done = False
    episode_reward = 0
    episode_length = 0
    goal_location=None
    shortest_distance_along_road=1e-8
    if mode !="dense":
        if map_dir is None:
            if model_type != "nomad":
                raise ValueError("Only NoMaD can explore")
            agent = MODEL(ckpt_path=checkpoint_path,mode="explore")
        else:
            agent = MODEL(ckpt_path=checkpoint_path,mode="navigate",map_dir=map_dir)
            print(f"Using {len(agent.topomap)} Nodes")
            # map_dir="/home/kojogyaase/Projects/Research/carla-rl/topomap"
        shortest_distance_along_road=None
        with open(f'{map_dir}/aux.pkl', 'rb') as handle:
            dataset=pickle.load(handle)
            origin=ndarray_to_location(dataset["start"])
            destination=ndarray_to_location(dataset["goal"])
    env.unwrapped.set_start_transform(origin)
    env.unwrapped.set_destination_transform(destination)
    observation, info = env.reset()


    # start_location = env.unwrapped.core.hero.get_transform().location
    # destination = env.unwrapped.core.destination

    expert_agent = BasicAgent(env.unwrapped.core.hero, target_speed=5.0)
    expert_agent.set_destination(destination)
    expert_agent.ignore_traffic_lights(True)
    expert_agent.ignore_stop_signs(True)

    save_dir=None
    keyframe=None
    start_location=None
    keyframe_count=0


    if mode =="dense":
        path=f"evaluation_trajectory/{mode}/{difficulty}/"
        os.makedirs(path, exist_ok=True)
        save_dir=path+str(len(os.listdir(path)))  
        os.makedirs(save_dir, exist_ok=True)  
    elif mode =="sparse_dist":
        base=map_dir.split("/")[-4]
        difficulty=map_dir.split("/")[-2]
        name=map_dir.split("/")[-1] 
        save_dir=f"{base}/{mode}_alpha_{threshold}/{difficulty}/{name}"
        os.makedirs(save_dir,exist_ok=True)
    else:
        pass
    while not done:
        control = expert_agent.run_step()
        actions = np.array([control.steer, control.throttle])
        # next_observation, reward, done, truncated, info = env.step(action)
        if mode == "sparse_dist":

            if keyframe is None:
                # breakpoint()
                keyframe=observation["pixels"]
                origin = carla_location_to_np_array(env.unwrapped.core.hero.get_transform().location)
            agent.eval_action(observation["pixels"])
            add_to_map,keyframe = agent.distance_to_observation(keyframe)
            if add_to_map:
                image = (keyframe[...,-1] * 255).astype(np.uint8)
                saved=cv2.imwrite(f"{save_dir}/{keyframe_count}.jpg",image)
                # print("Saved",saved)
                keyframe_count+=1
            # breakpoint()
        elif mode=="sparse_actions":
            pass
        else:
            obs=(observation["pixels"][...,0]*255).astype(np.uint8)
            cv2.imwrite(f"{save_dir}/{keyframe_count}.jpg",obs)
            keyframe_count+=1
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
                # distance_completed.append()
                print("==================SPL===============",_spl,SPL)
                
            if "slack" in info:
                slack_values.append(float(info["slack"]))
            if "is_success" in info:
                termination_budget=0
            else:
                termination_budget-=1



    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "max_SPL":np.nan_to_num(np.mean(SPL),nan=0),
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
        "experiment_results":stats,
        "SPLs":SPL_per_skip_frame,
    })

    if mode=="sparse_dist":
        with open(f"{save_dir}/aux.pkl", "wb") as f:
            pickle.dump(dict(
                start=origin,
                goal=carla_location_to_np_array(env.unwrapped.core.hero.get_transform().location),
                mean_distance_per_step=info.get("mean_distance_per_step",0.0)
            ), f)
    elif mode=="sparse_actions":
        pass
    else:
        with open(f"{save_dir}/aux.pkl", "wb") as f:
            pickle.dump(dict(start=carla_location_to_np_array(origin),
                            goal=carla_location_to_np_array(env.unwrapped.core.hero.get_transform().location),
                            mean_distance_per_step=info.get("mean_distance_per_step",0.0),
                            # locations=locations,
                            town="Town02"
                            ),f)
    return stats



# %%
# os.environ['CARLA_ROOT']="/home/kojogyaase/Apps/CARLA_0.9.15"
env = CarlaEvalEnv(start_server=True,town="Town02",image_size=96)
env = TimeLimit(env, max_episode_steps=int(2e4))
env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
env = FrameStack(env=env, num_stack=1, stacking_key="goal")
env = RecordEpisodeStatistics(env)

# %%
# Initialize agent
def run_agent():
    user_defined_trajectories=None
    with open("src/user_defined_trajectories.pkl", 'rb') as handle:
            user_defined_trajectories=pickle.load(handle)
    
    wrapped_func=partial(map_environment,env)
    if user_defined_trajectories:
        dense_mappting_func=partial(wrapped_func,mode="dense")
        for difficult,trajectories in user_defined_trajectories.items():
           for trajectory in trajectories.values():
               location=trajectory[0]["location"]
               origin=carla.Location(x=location[0],y=location[1],z=location[2])
               location=trajectory[-1]["location"]
               destination=carla.Location(x=location[0],y=location[1],z=location[2])
               dense_mappting_func(difficulty=difficult,origin=origin,destination=destination)
            #    wrapped_func
        sparse_dist_mappting_func=partial(wrapped_func,mode="sparse_dist")
        for difficult,trajectories in user_defined_trajectories.items():
           for ix,trajectory in enumerate(trajectories.values()):
               location=trajectory[0]["location"]
               origin=carla.Location(x=location[0],y=location[1],z=location[2])
               location=trajectory[-1]["location"]
               destination=carla.Location(x=location[0],y=location[1],z=location[2])
               sparse_dist_mappting_func(map_dir=f"evaluation_trajectory/dense/{difficult}/{ix}")
            #    wrapped_func
    # print("\nEvaluation Results:")
    # print("=" * 50)
    # for key, value in stats.items():
    #     if not isinstance(value,list):
    #         print(f"{key}: {value:.4f}")
    # print("=" * 50)
    # stats = map_environment(
    #     "nomad",
    #     env,
    #     # map_dir="/home/kojogyaase/Projects/Research/carla-rl/evaluation_trajectory/trajectories/0/",
    #     map_dir="/home/robotlab/scratch/carla-rl/evaluation_trajectory/hard/distance_heuristics_map/0",
    #     n_eval_episodes=10,
    #     deterministic=False,
    #     sparsify_map=False
    # )

    # print("\nEvaluation Results:")
    # print("=" * 50)
    # for key, value in stats.items():
    #     if not isinstance(value,list):
    #         print(f"{key}: {value:.4f}")
    # print("=" * 50)


# %%
# app.run(run_agent)
run_agent()


