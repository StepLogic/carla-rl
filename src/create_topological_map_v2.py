from datetime import datetime
import glob
import time
import cv2
import numpy as np
from tqdm import tqdm
import os
import pickle
from rlib_integration.agent import BasicAgent
# from train_online_pixels import CarlaGoalEnv,config,FrameStack,TimeLimit,RecordEpisodeStatistics,ReplayBuffer
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.timelimit import TimeLimit
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from rlib_integration.helper import carla_location_to_np_array
from carla_eval import CarlaEvalEnv
from PIL import Image
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
# from jax_mapping_experiment import JAXMappingExperiments
from carla_eval import CarlaEvalEnv

#!/usr/bin/env python

import carla
import random
import numpy as np
import networkx as nx
from collections import deque
import time

# Import the BasicAgent class from the provided code
# This assumes the BasicAgent class is defined in a file named 'agents.py'
# from agents import BasicAgent, GlobalRoutePlanner, RoadOption
# Since we're extending the provided code, we'll reuse the classes directly
#!/usr/bin/env python

import carla
import random
import numpy as np
import networkx as nx
from collections import deque
import time




def collect_basic_agent_data(origin,destination,difficulty="easy"):
    town="Town02"
    env = CarlaEvalEnv(use_rgb=True,town="Town02",start_server=True,max_dist=1000)
    env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
    env = FrameStack(env=env, num_stack=1, stacking_key="goal")
    env = TimeLimit(env, max_episode_steps=4500)
    env = RecordEpisodeStatistics(env)
    env.unwrapped.set_start_transform(origin)
    env.unwrapped.set_destination_transform(destination)
    
    # wmap=env.unwrapped.core.map
    

    # Main collection loop
    observation, info, done = *env.reset(), False
    collection_start_time = time.time()
    start_location = env.unwrapped.core.hero.get_transform().location
    # destination = env.unwrapped.core.destination
    # Initialize BasicAgent
    agent = BasicAgent(env.unwrapped.core.hero, target_speed=5.0)
    # agent = JunctionTrajectoryAgent(
    #         env.unwrapped.core.hero, 
    #         target_speed=30,
    #         min_junctions=min_junction,
    #         max_distance=max_dist
    #         )
    # agent.set_destination(env.unwrapped.core.destination.transform.location)
    agent.ignore_traffic_lights(True)
    agent.ignore_stop_signs(True)
    # data=[]
    # dataset_folder = os.path.join("topomap")

    step=0

    path=f"evaluation_trajectory/{difficulty}/"
    os.makedirs(path, exist_ok=True)
    map_dir=path+str(len(os.listdir(path)))  
    os.makedirs(map_dir, exist_ok=True)  
    # for i in tqdm(range(1, replay_buffer_size + 10)):
    while not agent.find_random_junction_trajectory(search_attempts=100, visualize=False):
        print("Planning")
    
    while not agent.done():
            obs=(observation["pixels"][...,0]*255).astype(np.uint8)
            Image.fromarray(obs).save(f"{map_dir}/{step}.jpg")
            control = agent.run_step()
            action = np.array([control.steer, control.throttle])
            next_observation, reward, done, truncated, info = env.step(action)
            done=done or agent.done()
            observation = next_observation
            step+=1
    # Save final buffer
    with open(f"{map_dir}/aux.pkl", "wb") as f:
        pickle.dump(dict(start=carla_location_to_np_array(start_location),
                         goal=carla_location_to_np_array(env.unwrapped.core.hero.get_transform().location),
                         mean_distance_per_step=info.get("mean_distance_per_step",0.0),
                         town="Town02"
                         ),f)
    collection_duration = time.time() - collection_start_time
    print(f"\nData collection completed in {collection_duration/3600:.2f} hours Distance Completed {info.get('distance_completed',0.0)}")
    print(f"Final dataset saved to:")

if __name__ == "__main__":
    number_trajectories_per_difficulty=5
    user_defined_trajectories=None
    with open("user_defined_trajectories.pkl", 'rb') as handle:
            user_defined_trajectories=pickle.load(handle)
    
    
    # difficulty=[
    #     # {
    #     #     "name":"easy",
    #     #     "town":"Town02",
    #     #     "min_junction":1,
    #     #     "max_dist":250
    #     # },
    #     # {
    #     #     "name":"medium",
    #     #     "town":"Town03",
    #     #      "min_junction":2,
    #     #     "max_dist":350
    #     # },
    #     {
    #         "name":"hard",
    #         "town":"Town05",
    #          "min_junction":3,
    #         "max_dist":450
    #     }
    # ]
    # for stage in difficulty:
    #     for i in range(number_trajectories_per_difficulty):
    if user_defined_trajectories:
        for difficult,trajectories in user_defined_trajectories.items():
            # print(trajectories)
            for trajectory in trajectories.values():
                # breakpoint()
                location=trajectory[0]["location"]
                origin=carla.Location(x=location[0],y=location[1],z=location[2])
                location=trajectory[-1]["location"]
                destination=carla.Location(x=location[0],y=location[1],z=location[2])
                collect_basic_agent_data(origin,destination,difficulty=difficult)