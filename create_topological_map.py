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
from src.carla_eval import CarlaEvalEnv
from PIL import Image
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
from src.jax_mapping_experiment import JAXMappingExperiments

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
                "town": "Town03"
            },
            "experiment": {
                "type": JAXMappingExperiments,
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
                    "max_dist": 600,
                    "target_speed": 5.0
                }
            }
        }
    }
# def teleport_agent
def collect_basic_agent_data(town="Town05",max_dist=600,difficulty="easy"):
    # Create environment
    # parser = argparse.ArgumentParser(description='Collect basic agent data')
    # parser.add_argument('town', help='Name of the town')
    # parser.add_argument('--agent-id', help='Agent ID (optional)')
    # Parse arguments
    # args = parser.parse_args()
    # Access the town name
    # town_name = args.town
    config["env_config"]["carla"]["town"]=town
    config["env_config"]["experiment"]["others"]["max_dist"]=max_dist
    env = CarlaEvalEnv(config["env_config"],use_rgb=True,image_size=96,start_server=False)
    env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
    env = FrameStack(env=env, num_stack=1, stacking_key="goal")
    env = TimeLimit(env, max_episode_steps=2500)
    env = RecordEpisodeStatistics(env)



    # Main collection loop
    observation, info, done = *env.reset(), False
    collection_start_time = time.time()
    start_location = env.unwrapped.core.hero.get_transform().location
    # destination = env.unwrapped.core.destination
    # Initialize BasicAgent
    agent = BasicAgent(env.unwrapped.core.hero, target_speed=5.0)
    agent.set_destination(env.unwrapped.core.destination.location)
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
    while not done:
        # if done:
        #     observation, info = env.reset()
        #     noise.reset()
        #     # Reinitialize BasicAgent for new episode
        #     agent = BasicAgent(env.unwrapped.core.hero, target_speed=0.5)
        #     agent.set_destination(env.unwrapped.core.destination)
        #     agent.ignore_traffic_lights(True)
        #     agent.ignore_stop_signs(True)
        obs=(observation["pixels"][...,0]*255).astype(np.uint8)
        # obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
        Image.fromarray(obs).save(f"{map_dir}/{step}.jpg")
        # heading= observation["vector"][-2]
        # mapper.update(obs,heading)
        # breakpoint()
        # cv2.imwrite(f"topological_map/{step}.jpg")
        # cv2.imwrite(f"topomap/{step}.jpg",obs)
        # Get action from BasicAgent
        control = agent.run_step()
        action = np.array([control.steer, control.throttle])
        
        # # Add noise and clip
        # action = np.clip(action + noise(), -1, 1)
        # action = np.array([
        #     np.clip(action[0], -1.0, 1.0),  # steer
        #     np.clip(action[1], 0.0, 1.0)    # throttle
        # ])

        next_observation, reward, done, truncated, info = env.step(action)
        done=done or agent.done()
        # Handle episode termination
        # mask = 1.0 if not done and not truncated else 0.0
        observation = next_observation
        step+=1
    # Save final buffer
    with open(f"{map_dir}/aux.pkl", "wb") as f:
        pickle.dump(dict(start=carla_location_to_np_array(start_location),
                         goal=carla_location_to_np_array(env.unwrapped.core.hero.get_transform().location),
                         mean_distance_per_step=info.get("mean_distance_per_step",0.0),
                         town=config["env_config"]["carla"]["town"]
                         ),f)
    collection_duration = time.time() - collection_start_time
    print(f"\nData collection completed in {collection_duration/3600:.2f} hours Distance Completed {info.get('distance_completed',0.0)}")
    print(f"Final dataset saved to:")

if __name__ == "__main__":
    number_trajectories_per_difficulty=5
    difficulty=[
        # {
        #     "name":"easy",
        #     "town":"Town01",
        #     "max_dist":250
        # },
        # {
        #     "name":"medium",
        #     "town":"Town02",
        #     "max_dist":350
        # },
        {
            "name":"hard",
            "town":"Town05",
            "max_dist":450
        }
    ]
    for stage in difficulty:
        for i in range(number_trajectories_per_difficulty):
            collect_basic_agent_data(stage["town"],stage["max_dist"],stage["name"])