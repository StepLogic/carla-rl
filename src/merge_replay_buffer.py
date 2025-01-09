from datetime import datetime
import time
import numpy as np
from tqdm import tqdm
import os
import pickle
from rlib_integration.agent import BasicAgent
from train_online_pixels import CarlaGoalEnv,config,FrameStack,TimeLimit,RecordEpisodeStatistics,ReplayBuffer
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
def collect_basic_agent_data(max_steps=100000, replay_buffer_size=100000):
    # Create environment
    env = CarlaGoalEnv(config["env_config"])
    env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
    env = FrameStack(env=env, num_stack=1, stacking_key="goal")
    env = TimeLimit(env, max_episode_steps=2500)
    env = RecordEpisodeStatistics(env)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(
        env.observation_space, 
        env.action_space, 
        replay_buffer_size
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
    agent = BasicAgent(env.unwrapped.core.hero, target_speed=1)
    agent.ignore_traffic_lights(True)
    agent.ignore_stop_signs(True)

    for i in tqdm(range(1, max_steps + 1)):
        if done:
            observation, info = env.reset()
            noise.reset()
            # Reinitialize BasicAgent for new episode
            agent = BasicAgent(env.unwrapped.core.hero, target_speed=1)
            agent.ignore_traffic_lights(True)
            agent.ignore_stop_signs(True)

        # Get action from BasicAgent
        control = agent.run_step()
        action = np.array([control.steer, control.throttle])
        
        # Add noise and clip
        action = np.clip(action + noise(), -1, 1)
        action = np.array([
            np.clip(action[0], -1.0, 1.0),  # steer
            np.clip(action[1], 0.0, 1.0)    # throttle
        ])

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
    final_dataset_file = os.path.join(dataset_folder, "basic_agent_data_final.pkl")
    with open(final_dataset_file, "wb") as f:
        pickle.dump(replay_buffer, f)
    
    collection_duration = time.time() - collection_start_time
    print(f"\nData collection completed in {collection_duration/3600:.2f} hours")
    print(f"Final dataset saved to: {final_dataset_file}")

if __name__ == "__main__":
    # data=[]
    collect_basic_agent_data()
    # with open("replay_buffer.pickle", "wb") as f:
    #     pickle.dump(replay_buffer, f)
    