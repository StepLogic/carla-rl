from datetime import datetime
import glob
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
def collect_basic_agent_data(town="Town05",replay_buffer_size=10000):
    # Create environment

    parser = argparse.ArgumentParser(description='Collect basic agent data')
    parser.add_argument('town', help='Name of the town')
    # parser.add_argument('--agent-id', help='Agent ID (optional)')
    # Parse arguments
    args = parser.parse_args()
    # Access the town name
    town_name = args.town
    #do not use 01,02,05
    config["env_config"]["carla"]["town"]=town_name
    config["env_config"]["carla"]["start_server"]=False
    env = CarlaGoalEnv(config["env_config"])
    env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
    # env = FrameStack(env=env, num_stack=1, stacking_key="goal")
    env = TimeLimit(env, max_episode_steps=2500)
    env = RecordEpisodeStatistics(env)

    # Initialize replay buffer
    replay_buffer = VariableCapacityBuffer(
        env.observation_space, 
        env.action_space
    )

    # Initialize noise for exploration
    action_dim = 2
    mean = np.zeros(action_dim)
    sigma = 0.2 * np.ones(action_dim)
    noise = OrnsteinUhlenbeckActionNoise(mean=mean, sigma=sigma)

    # Main collection loop
    observation, info, done = *env.reset(), False
    collection_start_time = time.time()
    
    # Initialize BasicAgent
    agent = BasicAgent(env.unwrapped.core.hero, target_speed=env.unwrapped.experiment.target_speed)
    # agent.set_destination(env.unwrapped.core.destination.transform.location)
    agent.ignore_traffic_lights(True)
    agent.ignore_stop_signs(True)
    # data=[]

    for i in tqdm(range(1, replay_buffer_size + 10)):
        if done:
            observation, info = env.reset()
            noise.reset()
            # Reinitialize BasicAgent for new episode
            agent = BasicAgent(env.unwrapped.core.hero, target_speed=env.unwrapped.experiment.target_speed)
            # agent.set_destination(env.unwrapped.core.destination.transform.location)
            agent.ignore_traffic_lights(True)
            agent.ignore_stop_signs(True)

        # Get action from BasicAgent
        control = agent.run_step()
        action = np.array([control.steer, control.throttle])
        
        # Add noise and clip
        # action = np.clip(action + noise(), -1, 1)
        # action = np.array([
        #     np.clip(action[0], -1.0, 1.0),  # steer
        #     np.clip(action[1], 0.0, 1.0)    # throttle
        # ])

        next_observation, reward, done, truncated, info = env.step(action)
        
        # Handle episode termination
        mask = 1.0 if not done and not truncated else 0.0
            
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
    final_dataset_file = os.path.join(dataset_folder, f"goal_condition_{town}_data_{count}.pkl")
    with open(final_dataset_file, "wb") as f:
        pickle.dump(replay_buffer, f)
    
    collection_duration = time.time() - collection_start_time
    print(f"\nData collection completed in {collection_duration/3600:.2f} hours")
    print(f"Final dataset saved to: {final_dataset_file} Size: {replay_buffer._size}")

if __name__ == "__main__":
    collect_basic_agent_data()