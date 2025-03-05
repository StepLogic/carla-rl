from copy import copy
from datetime import datetime
import glob
import itertools
import math
import random
import time
from jaxrl2.data.replay_buffer import ReplayBuffer
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit
import numpy as np
from rlib_integration.carla_goal_env import CarlaGoalEnv
from tqdm import tqdm
import os
import pickle
from rlib_integration.agent import BasicAgent
# from train_online_pixels import CarlaGoalEnv,config,FrameStack,TimeLimit,RecordEpisodeStatistics,ReplayBuffer
from src.configs.train_env_config import config
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
import carla
import argparse
def random_shift(observation,next_observation,action):
    observation["pixels"]=np.fliplr(observation["pixels"][...,0])[...,None]
    next_observation["pixels"]=np.fliplr(next_observation["pixels"][...,0])[...,None]
    action[0]=-action[0]
    return observation,next_observation,action

def random_perturb(env):
    # observation["pixels"]=np.fliplr(observation["pixels"][...,0])[...,None]
    # next_observation["pixels"]=np.fliplr(next_observation["pixels"][...,0])[...,None]
    perturb_steering_list=[1.0,-1.0]
    # _, _, _, _, _ = env.step([random.choice(perturb_steering_list),0.5])
    observation, _, _, _, _ = env.step([random.choice(perturb_steering_list),0.5])
    # action[0]=-action[0]
    return observation


def is_agent_at_junction(env):
    env=env.unwrapped
    wp =env.core.map.get_waypoint(env.hero.get_transform().location,project_to_road=True)
    return wp.is_junction

def add_random_impulse(env):
    impulse_strength = 80000  # Base strength of the impulse
    env = env.unwrapped  # Unwrap the environment if necessary

    # Generate a random Y-axis impulse between negative and positive values
    y_impulse = random.uniform(-impulse_strength, impulse_strength)

    # Ensure physics simulation is enabled for the vehicle
    env.hero.set_simulate_physics(True)

    # Get the vehicle's rotation (yaw in degrees)
    rotation = env.hero.get_transform().rotation
    yaw = math.radians(rotation.yaw)  # Convert yaw to radians

    # Transform the local Y-axis force to the world coordinate system
    # Local Y-axis force: (0, y_impulse, 0)
    # World coordinate system:
    # X_world = -sin(yaw) * Y_local
    # Y_world = cos(yaw) * Y_local
    x_force = -math.sin(yaw) * y_impulse
    y_force = math.cos(yaw) * y_impulse

    # Apply the force in the world coordinate system
    env.hero.add_force(carla.Vector3D(x_force, y_force, 0))
     
def collect_basic_agent_data(replay_buffer_size=int(1e3)):
    # Create environment

    parser = argparse.ArgumentParser(description='Collect basic agent data')
    parser.add_argument('town', help='Name of the town', default="Town01")
    # parser.add_argument('--agent-id', help='Agent ID (optional)')
    # Parse arguments
    args = parser.parse_args()
    # Access the town name
    town_name = args.town
    #do not use 01,02,05
    env=None
    # towns=['Town04',"Town03",""]
    towns=itertools.cycle(["Town07","Town03","Town06","Town04"])
    def reset_env():
        nonlocal env
        if not env is None:
             env.close()
        config["env_config"]["carla"]["town"]=next(towns)
        config["env_config"]["carla"]["start_server"]=False
        env = CarlaGoalEnv(config["env_config"])
        env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
        # env = FrameStack(env=env, num_stack=1, stacking_key="goal")
        env = TimeLimit(env, max_episode_steps=4500)
        env = RecordEpisodeStatistics(env)
        return env
    def reset_agent(env):
            # Initialize BasicAgent
            # print("hell",env.unwrapped.experiment.target_speed)
            agent = BasicAgent(env.unwrapped.core.hero, target_speed=env.unwrapped.experiment.target_speed)
            try:
                agent.set_destination(env.unwrapped.core.destination.transform.location)
            except:
                # env=reset_env()
                env.reset()
                agent=reset_agent(env)
                 
            agent.ignore_traffic_lights(True)
            agent.ignore_stop_signs(True)
            # data=[]
            # do some recursion


            return agent
    env=reset_env()

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(
        env.observation_space, 
        env.action_space,
        capacity=int(1e6)
    )

    # Initialize noise for exploration
    action_dim = 2
    mean = np.zeros(1)
    sigma = 2 * np.ones(1)
    noise = OrnsteinUhlenbeckActionNoise(mean=mean, sigma=sigma)

    # Main collection loop
    observation, info, done = *env.reset(), False
    collection_start_time = time.time()
    agent=reset_agent(env=env)

    epidsodes_per_env=int(replay_buffer_size/4)
    switch_env=False
    for i in tqdm(range(1, replay_buffer_size*4 + 10)):
        if not switch_env:
            switch_env=i%epidsodes_per_env == 0
        if done:
            if switch_env:
                 env=reset_env()
                 switch_env=False
            observation, info = env.reset()
            noise.reset()
            agent=reset_agent(env=env)
        # Get action from BasicAgent
        rand_key=random.randint(0,1)
        # if rand_key==1:
        #      add_random_impulse(env)

    
        vecs=observation["vector"]
        env_target_speed=env.unwrapped.experiment.target_speed
        target=np.clip(float(env_target_speed-noise().item()),0,env_target_speed+2)
        # print(target)
        current_velocity=env.unwrapped.experiment.velocity
        # current_heading=env.unwrapped.experiment.current_heading
        vecs[2] = np.clip(current_velocity/(target+1e-8), 0.0, 5.1)
        # vecs[3]= np.clip(current_heading/(heading+1e-8),-5.1,5.1) 
        agent.set_target_speed(target)
        control = agent.run_step()
        action = np.array([control.steer,control.throttle])
        action = np.nan_to_num(action)
        
        # Add noise and clip
        # action = np.clip(action + noise(),
        #     env.action_space.low,
        #     env.action_space.high)
        # action = np.array([
        #     env.action_space.low,  # steer
        #     env.action_space.high  # throttle
        # ])

        next_observation, reward, done, truncated, info = env.step(action)
        
        # Handle episode termination
        mask = 1.0 if not done and not truncated else 0.0
        done = done or agent.done()

        # if  agent.done():
        #      reward+=10
        # breakpoint()
        # copy_observation,copy_next_observation,copy_action=random_shift(observation,next_observation,action)
        # replay_buffer.insert(
        #     dict(
        #         observations=copy_observation,
        #         actions=copy_action,
        #         rewards=reward,
        #         masks=mask,
        #         dones=done,
        #         next_observations=copy_next_observation,
        #     )
        # )
        # oversample junction entries
        if is_agent_at_junction(env):
             for _ in range(5):
                  replay_buffer.insert(
                    dict(
                        observations=observation,
                        actions=action,
                        rewards=reward,
                        masks=mask,
                        dones=done,
                        next_observations=next_observation,
                    )
                )
        else: 
            if random.randint(0,5)==1:
                replay_buffer.insert(
                    dict(
                        observations=observation,
                        actions=action,
                        rewards=reward,
                        masks=mask,
                        dones=done,
                        next_observations=next_observation,
                    )
                )
        observation=next_observation
        if rand_key==1:
            observation = random_perturb(env)
        # # Save buffer periodically
        # if i % 10000 == 0:
        #     dataset_folder = os.path.join("datasets")
        #     os.makedirs(dataset_folder, exist_ok=True)
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     dataset_file = os.path.join(dataset_folder, f"basic_agent_data_{timestamp}.pkl")
        #     with open(dataset_file, "wb") as f:
        #         pickle.dump(replay_buffer, f)
        #     print(f"\nSaved dataset to: {dataset_file}")
    # Save final buffer
    dataset_folder = os.path.join("datasets")
    os.makedirs(dataset_folder, exist_ok=True)
    count=len(glob.glob(f"{dataset_folder}/*.pkl"))
    final_dataset_file = os.path.join(dataset_folder, f"goal_condition_{town_name}_data_{count}.pkl")
    with open(final_dataset_file, "wb") as f:
        pickle.dump(replay_buffer, f)
    
    collection_duration = time.time() - collection_start_time
    print(f"\nData collection completed in {collection_duration/3600:.2f} hours")
    print(f"Final dataset saved to: {final_dataset_file} Size: {replay_buffer._size}")

if __name__ == "__main__":
    collect_basic_agent_data()