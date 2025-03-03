from copy import copy
from datetime import datetime
import glob
import random
import time
from jaxrl2.data.replay_buffer import ReplayBuffer,VariableCapacityBuffer
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
import argparse
def collect_basic_agent_data(replay_buffer_size=int(2e5)):
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
    def reset_env():
        nonlocal env
        if not env is None:
             env.close()
        config["env_config"]["carla"]["town"]=town_name
        # config["env_config"]["carla"]["start_server"]=False
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
                #   do some  recursion
                env=reset_env()
                reset_agent(env)
            
            agent.ignore_traffic_lights(True)
            agent.ignore_stop_signs(True)
            # data=[]
            return agent
    env=reset_env()

    # Initialize replay buffer
    replay_buffer = VariableCapacityBuffer(
        env.observation_space, 
        env.action_space
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

    # epidsodes_per_env=int(replay_buffer_size/len(towns))
    switch_env=False
    for i in tqdm(range(1, replay_buffer_size + 10)):
        # if not switch_env:
        #     switch_env=i%epidsodes_per_env
        if done:
            if switch_env:
                 env=reset_env()
                 switch_env=False
            observation, info = env.reset()
            noise.reset()
            agent=reset_agent(env=env)
        # Get action from BasicAgent
    
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
        action = np.array([control.steer, control.throttle])
        
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
        if  agent.done():
             reward+=10
        #complement_image
        observation_copy=copy(observation)
        next_observation_copy=copy(next_observation)
        observation_copy["pixels"]=np.fliplr(observation_copy["pixels"][:,:,:,0])[...,None]
        next_observation_copy["pixels"]=np.fliplr(next_observation_copy["pixels"][:,:,:,0])[...,None]
        action_copy=np.array([-control.steer, control.throttle])
        replay_buffer.insert(
            dict(
                observations=observation_copy,
                actions=action_copy,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation_copy,
            )
        )
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
        
        # Store transition
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
        
        observation = next_observation
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