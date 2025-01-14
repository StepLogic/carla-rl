import os
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
from src.carla_eval import CarlaEvalEnv
from src.jax_experiments_goal import JAXGoalExperiments
from src.jax_mapping_experiment import JAXMappingExperiments
from jaxrl2.agents import DrQLearner
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit
import pickle
from pyflann import *

from topological_map import TopologicalMap
# fix
os.environ['XLA_FLAGS']="--xla_gpu_enable_command_buffer="
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".30"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", None, "Path to the checkpoint directory")
flags.DEFINE_integer("n_eval_episodes", 100, "Number of evaluation episodes")
flags.DEFINE_boolean("deterministic", True, "Whether to use deterministic actions")
config_flags.DEFINE_config_file(
    "config",
    "/home/kojogyaase/Projects/Research/carla-rl/dependencies/jaxrl2/examples/configs/drq_default.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

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
    # breakpoint()
    agent._actor = state_dict['actor_params']
    agent._critic = state_dict['critic_params'] 
    agent._target_critic_params = state_dict['target_critic_params']
    agent._temp = state_dict['temp']
    # agent._rng = state_dict['rng']
    
    return agent
# def save_image_heading(self,step,image,heading):
    # res = cv2.imwrite(f"{step}.jpg",image)
    # if()

# def ema(s, n):
#     # """
#     # returns an n period exponential moving average for
#     # the time series s

#     # s is a list ordered from oldest (index 0) to most
#     # recent (index -1)
#     # n is an integer

#     # returns a numeric array of the exponential
#     # moving average
#     # """
#     s = np.array(s)
#     ema = []
#     j = 1
#     sma = sum(s[:n]) / n
#     multiplier = 2 / float(1 + n)
#     ema.append(sma)
#     ema.append(( (s[n] - sma) * multiplier) + sma)
#     #now calculate the rest of the values
#     for i in s[n+1:]:
#         tmp = ( (i - ema[j]) * multiplier) + ema[j]
#         j = j + 1
#         ema.append(tmp)
#     return ema

import numpy as np
from collections import deque

class RealTimeVectorEMA:
    def __init__(self, window_size=100, alpha=0.1, vector_dim=3):
        self.window = deque(maxlen=window_size)
        self.alpha = alpha
        self.last_value = np.zeros(vector_dim)
        
    def update(self, new_vector):
        self.window.append(new_vector)
        self.last_value = self.alpha * new_vector + (1 - self.alpha) * self.last_value
        return self.last_value
    def threshold(self):
        return np.std(self.window)
    def get_current(self):
        return self.last_value
class Mapper:
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.des_nodes=[]
        self.image_node=[]
        self.heading_nodes=[]
        self.flann = FLANN()
        # # find the keypoints and descriptors with SIFT
        # kp1, des1 = sift.detectAndCompute(image,None)
        # kp2, des2 = sift.detectAndCompute(goal_image,None)
        
    def update(self, image_obs,heading_obs):
        _, des1 = self.sift.detectAndCompute(image_obs,None)
        self.image_node.append(image_obs)
        self.des_nodes.append(des1)
        self.heading_nodes.append(heading_obs)

    def select_subgoal(self,image_obs):
        result, dists = flann.nn(self.des_nodes, image_obs, 5, algorithm="kmeans", branching=32, iterations=7, checks=16)
        nn=result[np.argmin(dists)]
        return self.image_node[nn],self.heading_nodes[nn]
        
    


        


def map_environment(agent:DrQLearner, env, n_eval_episodes=10, deterministic=True):
    """Evaluate the agent for n_eval_episodes."""
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    distance_completed = []
    slack_values = []
    steps=0
    log_stds=[]
    moving_average=[]
    junctions=[]
    restarts=[]
    images=[]
    heading_ar=[]
    # filter=StreamingMovingAverage(window_size=100)
    ema_filter = RealTimeVectorEMA(window_size=100, vector_dim=2)
    # Real-time updates
    mapper=TopologicalMap()
    for _ in range(n_eval_episodes):
        observation, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        while not done:
            action_dist=agent.action_dist(observation)
            # if deterministic:
            action = action_dist.mode()
            observation, reward, done, truncated, info = env.step(action)
            if env.unwrapped.is_agent_at_junction():
                junctions.append(steps)
            std=np.array(action_dist.log_std())
            log_stds.append(std)
            # moving_average.append(filter.process(np.array(action_dist.log_std())))
            filtered_vector = ema_filter.update(std)
            moving_average.append(filtered_vector)
            episode_reward += reward
            episode_length += 1
            done = done or truncated
            steps+=1
            if np.any((ema_filter.get_current()-std)>ema_filter.threshold()):
                #add image to map
                obs=(observation["pixels"][...,0]*255).astype(np.uint8)
                heading= observation["vector"][-2]
                images.append(obs)
                heading_ar.append(heading)
                mapper.update(obs,heading)
                cv2.imwrite(f"sample_map/{steps}.jpg",np.vstack([(observation["pixels"][...,0]*255).astype(np.uint8),(observation["goal"][...,0]*255).astype(np.uint8)
                                                                 ]))
                
            
            if done:
                restarts.append(steps)
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                # print(info)
                if "is_success" in info:
                    success_rate.append(float(info["is_success"]))
                if "distance_completed" in info:
                    distance_completed.append(float(info["distance_completed"]))
                if "slack" in info:
                    slack_values.append(float(info["slack"]))
    
    # Compute statistics
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "total_map_steps":steps
    }
    
    if success_rate:
        stats["success_rate"] = np.mean(success_rate)
    if distance_completed:
        stats["mean_distance"] = np.mean(distance_completed)
    if slack_values:
        stats["mean_slack"] = np.mean(slack_values)
        
    plt.plot(np.array(log_stds)[:,0], color='blue',linestyle = 'dotted')
    plt.plot(np.array(log_stds)[:,1], color='red',linestyle = 'dotted') 

    plt.plot(np.array(moving_average)[:,0], color='blue' )
    plt.plot(np.array(moving_average)[:,1], color='red') 

    # plt.plot(moving_average, color='red') 
    for k in junctions:
        plt.axvline(x=k, color='g',ls="--")
    for k in restarts:
        plt.axvline(x=k, color='m',ls="--")
    # plt.hlines(x=junctions, ymin=np.min(log_stds), ymax=np.max(log_stds), colors='green', ls=':', lw=2, label='Junctions')
    plt.legend(loc="upper left")
    plt.savefig("uncertainty_profile_at_junctions.pdf")
    # plt.legend()
    a=dict(log_stds=log_stds,junctions=junctions)
    with open('uncertainty_profile_at_junctions.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
    a=dict(images=images,heading=heading_ar)
    with open('map.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return stats,mapper

def navigate(agent:DrQLearner, env:CarlaEvalEnv, mapper:TopologicalMap,n_eval_episodes=10):
    """Evaluate the agent for n_eval_episodes."""
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    distance_completed = []
    slack_values = []
    steps=0
    log_stds=[]
    moving_average=[]
    junctions=[]
    restarts=[]
    # filter=StreamingMovingAverage(window_size=100)
    ema_filter = RealTimeVectorEMA(window_size=100, vector_dim=2)
    # Real-time updates
    # mapper=TopologicalMap()

    for _ in range(n_eval_episodes):
        observation, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        #select goal
        # breakpoint()
        goal_idx=random.randint(0,len(mapper.image_node)-1)
        obs=(observation["pixels"][...,0]*255).astype(np.uint8)
        subgoal=mapper.create_navigation_guide(obs,goal_idx)
        while not done:
            goal,done=subgoal(obs)
            if not done and not goal is None:
                env.unwrapped.set_goal(goal[0],goal[1])
            
            action_dist=agent.action_dist(observation)
            # if deterministic:
            action = action_dist.mode()
            observation, reward, done, truncated, info = env.step(action)
            if env.unwrapped.is_agent_at_junction():
                junctions.append(steps)
            std=np.array(action_dist.log_std())
            log_stds.append(std)
            # moving_average.append(filter.process(np.array(action_dist.log_std())))
            filtered_vector = ema_filter.update(std)
            moving_average.append(filtered_vector)
            episode_reward += reward
            episode_length += 1
            done = done or truncated
            steps+=1
            # if np.any((ema_filter.get_current()-std)>ema_filter.threshold()):
            #     #add image to map
            #     obs=(observation["pixels"][...,0]*255).astype(np.uint8)
            #     heading= observation["vector"][-2]
            #     mapper.update(obs,heading)
            #     cv2.imwrite(f"sample_map/{steps}.jpg",np.vstack([(observation["goal"][...,0]*255).astype(np.uint8)
            #                                                      ,obs]))
                
            
            if done:
                restarts.append(steps)
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                # print(info)
                if "is_success" in info:
                    success_rate.append(float(info["is_success"]))
                if "distance_completed" in info:
                    distance_completed.append(float(info["distance_completed"]))
                if "slack" in info:
                    slack_values.append(float(info["slack"]))
    
    # Compute statistics
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "total_map_steps":steps
    }
    
    if success_rate:
        stats["success_rate"] = np.mean(success_rate)
    if distance_completed:
        stats["mean_distance"] = np.mean(distance_completed)
    if slack_values:
        stats["mean_slack"] = np.mean(slack_values)
    plt.plot(np.array(log_stds)[:,0], color='blue',linestyle = 'dotted')
    plt.plot(np.array(log_stds)[:,1], color='red',linestyle = 'dotted') 

    plt.plot(np.array(moving_average)[:,0], color='blue' )
    plt.plot(np.array(moving_average)[:,1], color='red') 

    # plt.plot(moving_average, color='red') 
    for k in junctions:
        plt.axvline(x=k, color='g',ls="--")
    for k in restarts:
        plt.axvline(x=k, color='m',ls="--")
    # plt.hlines(x=junctions, ymin=np.min(log_stds), ymax=np.max(log_stds), colors='green', ls=':', lw=2, label='Junctions')
    plt.legend(loc="upper left")
    plt.savefig("uncertainty_profile_at_junctions.pdf")
    # plt.legend()
    a=dict(log_stds=log_stds,junctions=junctions)
    with open('uncertainty_profile_at_junctions.pickle', 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return stats,mapper

def main(_):
    # Create and wrap environment
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
                    "max_dist": 200,
                    "target_speed": 5.0
                }
            }
        }
    }
    
    env = CarlaEvalEnv(config["env_config"])
    env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
    env = FrameStack(env=env, num_stack=1, stacking_key="goal")
    env = TimeLimit(env, max_episode_steps=2500)
    env = RecordEpisodeStatistics(env)
    
    # Initialize agent
    kwargs = dict(FLAGS.config)
    agent = DrQLearner(
        42,  # seed
        env.observation_space.sample(),
        env.action_space.sample(),
        **kwargs
    )
    
    # Load checkpoint
    agent = load_checkpoint(agent, FLAGS.checkpoint_path)
    
    # Evaluate
    stats,mapper = map_environment(
        agent,
        env,
        n_eval_episodes=FLAGS.n_eval_episodes,
        deterministic=FLAGS.deterministic
    )
    stats = navigate(
        agent,
        env,
        mapper,
        n_eval_episodes=FLAGS.n_eval_episodes,
        # deterministic=FLAGS.deterministic
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    flags.mark_flag_as_required("checkpoint_path")
    app.run(main)