import os
import random
import time
import cv2
import matplotlib.pyplot as plt
import ml_collections
import numpy as np
from absl import app, flags
from flax.training import checkpoints
from ml_collections import config_flags
from src.carla_eval import CarlaEvalEnv
from src.sac_lane_following import sac_config
# from src.bc_lane_following import bc_config
from jaxrl2.agents import DrQLearner,PixelBCLearner
from jaxrl2.agents.resnet_agents import PixelResNetBCLearner
from rlib_integration.helper import ndarray_to_location
from rlib_integration.agent import GlobalRoutePlanner
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit
import pickle
from pyflann import *
from src.mapping.topological_map import TopologicalMap
# fix
os.environ['XLA_FLAGS']="--xla_gpu_enable_command_buffer="
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".30"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"


config = ml_collections.ConfigDict()
config.actor_lr = 3e-4
config.hidden_dims = (256, 256)
config.cnn_features = (32, 64, 128, 256)
config.cnn_filters = (3, 3, 3, 3)
config.cnn_strides = (2, 2, 2, 2)
config.cnn_padding = "VALID"
config.latent_dim = 50
config.encoder = "d4pg"
bc_config = config.to_dict()



config = ml_collections.ConfigDict()
config.actor_lr = 3e-4
config.critic_lr = 3e-4
config.temp_lr = 3e-4
config.hidden_dims = (256, 256)
config.cnn_features = (32, 64, 128, 256)
config.cnn_filters = (3, 3, 3, 3)
config.cnn_strides = (2, 2, 2, 2)
config.cnn_padding = "VALID"
config.latent_dim = 50
config.encoder = "d4pg"
config.discount = 0.98
config.tau = 0.005
config.init_temperature = 1.0
config.target_entropy = None
config.backup_entropy = True
config.critic_reduction = "mean"
sac_config = config.to_dict()



# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_path", None, "Path to the checkpoint directory")
flags.DEFINE_enum('model', 'DrQLearner', ['DrQLearner', 'PixelResNetBCLearner'], 'Model to run')
flags.DEFINE_integer("n_eval_episodes", 10, "Number of evaluation episodes")
flags.DEFINE_boolean("deterministic", True, "Whether to use deterministic actions")
flags.DEFINE_string("map_dir", None, "Evaluation directory trajectory")
flags.DEFINE_string("town", "Town01", "Town Name")
def load_checkpoint(agent, checkpoint_path):
    """Load agent parameters from checkpoint."""
    state_dict = {
        'actor_params': agent._actor,
        # 'critic_params': agent._critic,
        # 'target_critic_params': agent._target_critic_params,
        # 'temp': agent._temp,
        # 'rng': agent._rng,
        # Add any other numerical state you need to save
    }
    # breakpoint()
    state_dict = checkpoints.restore_checkpoint(
        ckpt_dir=checkpoint_path,
        target=state_dict
    )

    # Update agent parameters
    # breakpoint()
    agent._actor = state_dict['actor_params']
    # agent._critic = state_dict['critic_params'] 
    # agent._target_critic_params = state_dict['target_critic_params']
    # agent._temp = state_dict['temp']
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
from collections import defaultdict, deque

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
        
    


        


def eval_environment(agent:DrQLearner, env, n_eval_episodes=10, deterministic=True):
    """
    
    Evaluate the agent for n_eval_episodes
   
    """

    episode_rewards = []
    episode_lengths = []
    success_rate = []
    SPL = []
    SPL_per_skip_frame = []
    distance_completed = []
    slack_values = []
    truncate_steps=0
    data=defaultdict(lambda :[])

    #
    shortest_distance_along_road=1e-8
    with open(f'{FLAGS.map_dir}/aux.pkl', 'rb') as handle:
            dataset=pickle.load(handle)
            start_location=ndarray_to_location(dataset["start"])
            goal_location=ndarray_to_location(dataset["goal"])
            env.unwrapped.set_start_transform(start_location)
            env.unwrapped.set_destination_transform(goal_location)

            route_plannner=GlobalRoutePlanner(env.unwrapped.core.map, 2.0)
    
            prev_waypoint=None
            trace=route_plannner.trace_route(start_location,goal_location)
            for wp,_ in trace:
                if prev_waypoint is None:
                    prev_waypoint=wp
                shortest_distance_along_road+=prev_waypoint.transform.location.distance(wp.transform.location)
                prev_waypoint=wp
            # assert shortest_distance_along_road>1.0

    for _ in range(n_eval_episodes):
        observation, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        # heading=random.choice([0,np.pi/2,np.pi,2/3*np.pi,2*np.pi])
        heading=0+1e-8
        while not done:
            truncate_steps+=1
            target=5.0
            vecs=observation["vector"]
            current_velocity=env.unwrapped.experiment.velocity
            # current_heading=env.unwrapped.experiment.current_heading
            # breakpoint()
            # if (current_heading - heading)<np.deg2rad(10):
            #         truncate_steps=int(1e5) #break loop
            #         print(np.rad2deg(current_heading),np.rad2deg(heading))
            vecs[2] = np.clip(current_velocity/(target+1e-8), 0.0, 5.1)
            # vecs[3]= np.clip(current_heading/(current_heading+1e-8),-5.1,5.1) 
            # vecs[4]= np.clip(heading/np.pi,-5.1,5.1) 
            observation["vector"]=vecs
            action_dist=agent.action_dist(observation)
            feature=agent.extract_features(observation)
            # breakpoint()
            # if deterministic:
            action = action_dist.mode()
            # print(action)
            observation, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = done or truncated
            
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                # print(info)
                if "is_success" in info:
                    success_rate.append(float(info["is_success"]))
                else:
                    success_rate.append(0)
                if "distance_completed" in info :
                    distance_completed.append(float(info["distance_completed"]))
                if "distance_completed" in info:
                    # https://arxiv.org/pdf/1807.06757
                    agent_distance_completed=float(info["distance_completed"])
                    S=int(info.get("is_success",0)) #for timeouts where this key is not 
                    _spl=S*(shortest_distance_along_road)/max(shortest_distance_along_road,agent_distance_completed)
                    SPL.append(_spl)
                    # distance_completed.append()
                    print("==================SPL===============",_spl,SPL)
                    
                if "slack" in info:
                    slack_values.append(float(info["slack"]))
                # if i%5==0 and i!=0:
                #     skip_index+=2
                #     l=np.nan_to_num(np.mean(SPL),nan=0)
                #     SPL_per_skip_frame.append(l)
                #     # breakpoint()
                #     print("=================Skip Index==============",skip_index,l,SPL_per_skip_frame)
                #     SPL = []
        # breakpoint()
        # Compute statistics
        difficulty=FLAGS.map_dir.split("/")[-2]
        name=FLAGS.map_dir.split("/")[-1] or FLAGS.model
        stats = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "max_SPL":np.nan_to_num(np.mean(SPL),nan=0),
            "mean_distance_per_step":dataset.get("mean_distance_per_step",1.0)
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
        path=f"results/{FLAGS.model}/{difficulty}"
        os.makedirs(path,exist_ok=True)
        with open(f"{path}/{name}_test_results.pkl", "wb") as f:
            pickle.dump(dict(data), f)
        return stats

def main(_):
   
    # config["env_config"]["carla"]["town"]=town_name
    # config["env_config"]["carla"]["start_server"]=False
    env = CarlaEvalEnv(start_server=False,town=FLAGS.town)
    env = TimeLimit(env, max_episode_steps=2500)
    env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
    env = RecordEpisodeStatistics(env)

    # Initialize agent
    if FLAGS.model == "DrQLearner":
        config=sac_config
    else:
        config=bc_config
    agent = globals()[FLAGS.model](
        0,  # seed
        env.observation_space.sample(),
        env.action_space.sample(),
        **config
    )
    
    # Load checkpoint
    agent = load_checkpoint(agent, FLAGS.checkpoint_path)

    # Evaluate
    stats = eval_environment(
        agent,
        env,
        n_eval_episodes=FLAGS.n_eval_episodes,
        deterministic=FLAGS.deterministic
    )
    # stats = navigate(
    #     agent,
    #     env,
    #     mapper,
    #     n_eval_episodes=FLAGS.n_eval_episodes,
    #     # deterministic=FLAGS.deterministic
    # )
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    flags.mark_flag_as_required("checkpoint_path")
    app.run(main)