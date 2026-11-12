# Modified from https://github.com/carla-simulator/rllib-integration/blob/main/dqn_example/dqn_experiment.py

from collections import defaultdict
import math
import os
import random
import numpy as np
from gymnasium.spaces import Box, Dict
import torch
import torchvision.transforms as T
import carla
import torch.nn as nn
import torchvision.models as models  # <-- ADDED THIS

from rlib_integration.base_experiment import BaseExperiment
from rlib_integration.helper import post_process_image, carla_location_to_np_array

def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


class STBL3Experiment(BaseExperiment):
    def __init__(self, config={}):
        super().__init__(config)  # Creates a self.config with the experiment configuration

        self.frame_stack = self.config["others"]["framestack"]
        self.max_time_idle = self.config["others"]["max_time_idle"]
        self.max_dist = self.config["others"]["max_dist"]
        self.target_speed = self.config["others"]["target_speed"]
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]
        self.last_action = None
        # control variables
        self.max_steer = 1.0
        self.max_throttle = 1.0
        self.max_angle_deviation=np.pi
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.current_heading=0.0
        self.trajectories = None
        self.velocity=0.0
        self.info=dict()
        self.rewards=[]
        self.image_size=84 # This will be resized by the transform
        self.episode_num=1
        self.max_episodes=10000
        self.explore_mode=False
        self.distance_travelled = 0.0
        self.distance_completed = 0.04
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_mode=False
        self.obs_img=None
        # --- MODIFICATION: Load Shallow MobileNetV2 ---
        # 1. Load pretrained MobileNetV2
        print("Loading pre-trained MobileNetV2 model...")
        self.model = models.mobilenet_v3_small(pretrained=True)
        
        # 2. Use the first 4 blocks of the 'features' sequential
        self.model = self.model.features[:4]
        
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode
        
        # 3. Define the correct image transform (same as ResNet)
        # This assumes the input 'image' from post_process_image is a numpy array
        self.image_transform = T.Compose([
            T.ToPILImage(), # Convert numpy array to PIL Image
            T.Resize((84, 84)), # MobileNetV2 expects 224x224
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # --- END MODIFICATION ---

    def _cache_waypoints(self,world) -> None:
            env_map = world.get_map()
            waypoints = env_map.generate_waypoints(distance=2)
            trajectories = defaultdict(list)
            for wpt in waypoints:
                trajectories[f"{wpt.road_id}-{wpt.lane_id}"].append(wpt)
            self.trajectories = sorted([traj for traj in trajectories.values() if len(traj) > 3], 
                                        key=len, reverse=True)

    def reset(self,env):
        """Called at the beginning and each time the simulation is reset"""

        # Ending variables
        self.time_idle = 0
        self.time_episode = 0
        self.done_time_idle = False
        self.done_falling = False
        self.done_dist = False

        # hero variables
        self.last_location = None
        self.last_velocity = 0
        self.distance_travelled = 0.0
        self.distance_completed = 0.0
        self.last_heading = None

        # Sensor stack
        self.prev_vec_0 = None
        self.prev_vec_1 = None
        self.prev_vec_2 = None
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None
        self.obs_img=None

        # control variables
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.steer = 0.0
        self.throttle = 0.0
        self.target_speed = 6.0
        self.velocity=0.0
        self.current_heading=0.0
        self.explore_mode=random.randint(0,2)==1
        self.info=dict()
        if self.trajectories is None:
            self._cache_waypoints(env.world)

    def get_observation_space(self):
        # --- MODIFICATION: Update observation space shape ---
        # Shallow MobileNetV2 (features[:4]) output is [B, 24, 56, 56]
        # The flattened shape is 24 * 56 * 56 = 75264
        image_space= Box(
                low=-np.inf,
                high=np.inf,
                shape=(2904,), # Shallow MobileNetV2 flattened feature map
                dtype=np.float32
            )
        # --- END MODIFICATION ---

        vec_space = Box(
            low=-5.1,
            high=5.1,
            shape=(3 * self.frame_stack,),
            dtype=np.float32,
        )

        return Dict({"state":image_space, "vector":vec_space})

    def get_action_space(self):
        """Returns the continuous action space for steering and throttle"""
        return Box(
            low=np.array([-self.max_steer, 0.0]),  # [steering, throttle/brake]
            high=np.array([self.max_steer, 1.0]),
            dtype=np.float32
        )

    def compute_action(self, action):
        """Convert continuous actions to CARLA vehicle controls"""
        steer, throttle_brake = action

        action = carla.VehicleControl()
        action.steer = float(np.clip(steer, -self.max_steer, self.max_steer))
        self.steer=action.steer
        self.throttle=action.throttle
        
        if throttle_brake >= 0:
            action.throttle = float(np.clip(throttle_brake, 0.0, self.max_throttle))
            action.brake = 0.0
        else:
            action.throttle = 0.0
            action.brake = float(np.clip(throttle_brake, 0.0, 1.0))
            
        self.last_action = action
        action.reverse = False
        action.hand_brake = False
        self.steer=action.steer
        self.throttle=action.throttle
        return action

    def get_observation(self, sensor_data, core):
        """Function to do all the post processing of observations (sensor data)."""
        vecs = self.get_vec_obs(sensor_data, core)
        images = self.get_img_obs(sensor_data, core)
        return {"state":images, "vector":vecs}, self.info

    def get_vec_obs(self, sensor_data, core):
        heading=sensor_data["goal_heading"][-1][-1]
        imu=sensor_data["imu"][-1][-1]
        vec = np.zeros(3)
        vec[0] = self.steer / self.max_steer
        vec[1] = self.throttle / self.max_throttle
        hero = core.hero
        wp=core.map.get_waypoint(hero.get_transform().location,project_to_road=True) 

        if self.target_speed == 0:
            vec[2] = 0.0 if self.get_speed(hero) < 0.1 else 1.0  # Only 0 if almost stopped
        else:
            vec[2] = np.clip(self.get_speed(hero)/self.target_speed, 0.0, 1.0)
        
        if self.prev_vec_0 is None:
            self.prev_vec_0 = vec
            self.prev_vec_1 = self.prev_vec_0
            self.prev_vec_2 = self.prev_vec_1

        vecs = vec

        if self.frame_stack >= 2:
            vecs = np.concatenate([self.prev_vec_0, vecs], axis=0)
        if self.frame_stack >= 3 and vecs is not None:
            vecs = np.concatenate([self.prev_vec_1, vecs], axis=0)
        if self.frame_stack >= 4 and vecs is not None:
            vecs = np.concatenate([self.prev_vec_2, vecs], axis=0)

        self.prev_vec_2 = self.prev_vec_1
        self.prev_vec_1 = self.prev_vec_0
        self.prev_vec_0 = vec
        self.velocity=self.get_speed(hero)
        self.current_heading=imu
        vecs=np.nan_to_num(vecs,nan=1e-8)
        return vecs

    def get_img_obs(self, sensor_data, core):
        image = post_process_image(sensor_data['rgb'][1], normalized = False,crop=False, grayscale = False,image_size=self.image_size)
        self.obs_img=image
        if self.prev_image_0 is None:
            self.prev_image_0 = image
            self.prev_image_1 = self.prev_image_0
            self.prev_image_2 = self.prev_image_1

        images = image

        if self.frame_stack >= 2:
            images = np.concatenate([self.prev_image_0, images], axis=2)
        if self.frame_stack >= 3 and images is not None:
            images = np.concatenate([self.prev_image_1, images], axis=2)
        if self.frame_stack >= 4 and images is not None:
            images = np.concatenate([self.prev_image_2, images], axis=2)

        self.prev_image_2 = self.prev_image_1
        self.prev_image_1 = self.prev_image_0
        self.prev_image_0 = image
        
        img_tensor = self.image_transform(image.copy()).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # This self.model() call now uses the shallow MobileNetV2 features
            latent_vector = self.model(img_tensor)
            # The output shape will be [1, 24, 56, 56]
        
        # This flatten operation will correctly turn [1, 24, 56, 56] into (75264,)
        return  latent_vector.squeeze(0).cpu().numpy().flatten()
    
    def get_speed(self, hero):
        """Computes the speed of the hero vehicle in Km/h"""
        vel = hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2)

    def get_done_status(self, sensor_data, core):
        """Returns whether or not the experiment has to end"""
        self.info=dict()
        hero = core.hero
       
        self.done_time_idle = self.max_time_idle < self.time_idle
        if self.get_speed(hero) > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1
        self.time_episode += 1
               
        hero_velocity = self.get_speed(hero)
        wp=core.map.get_waypoint(hero.get_transform().location,project_to_road=False) 
        self.done_falling = False
        self.diff_lane = 'lane_invasion' in sensor_data.keys() or  wp is None if not self.eval_mode else False
        self.collision = 'collision' in sensor_data.keys()
    
        self.done_speed= False
        self.done_dist = self.distance_travelled>=205
        
        done=self.done_falling or self.done_dist or self.diff_lane or self.collision or self.done_dist or self.done_speed
        self.info.update(dict(is_success=self.done_dist,
                             distance_travelled=self.distance_travelled,
                             goal_directed_distance=self.distance_completed,
                             max_reward=0,
                             min_reward=0,
                             mean_reward=0
                             ))
        if len(self.rewards)>0:
                    self.info.update(dict(
                             max_reward=np.max(self.rewards),
                             min_reward=np.min(self.rewards),
                             mean_reward=np.mean(self.rewards)))
        if done:
            self.info.update(is_success=self.done_dist,
                             distance_completed=self.distance_travelled,
                             max_reward=np.max(self.rewards),
                             min_reward=np.min(self.rewards),
                             mean_reward=np.mean(self.rewards))
        return done

    def compute_reward(self, sensor_data, core):
        """
        Compute the reward for RL training based on vehicle performance.
        """
        # Define reward component weights
        WEIGHTS = {
            'speed': 0.6,
            'direction': 0.6,
            # 'action_smoothness': 0,
            'time_penalty': 0.09,
            'collision_penalty': 2.0,
            'lane_departure_penalty': 2.0,
            'goal_reached_bonus':2.0
        }
        
        # Get vehicle and sensor data
        hero = core.hero
        target_heading = np.nan_to_num(sensor_data["goal_heading"][-1][-1])
        current_heading = np.nan_to_num(sensor_data["imu"][-1][-1])
        hero_location = hero.get_location()
        hero_velocity = self.get_speed(hero)
        
        if not hasattr(self, 'last_location') or self.last_location is None:
            self.last_location = hero_location
            self.distance_travelled = 0.0
            self.distance_completed=0.0
        
        delta_distance = float(np.sqrt(np.square(hero_location.x - self.last_location.x) + 
                            np.square(hero_location.y - self.last_location.y)))
        
        heading_diff = abs(current_heading - target_heading)
        wp=core.map.get_waypoint(hero.get_transform().location,project_to_road=True) 
        
        if self.explore_mode:
            direction_factor = 0
        else:
            if not wp is None and wp.is_junction:
                direction_factor=np.cos(heading_diff)
            else:
                direction_factor=np.cos(0)
            
        forward_distance = max(delta_distance * direction_factor, 0)
        self.distance_travelled += delta_distance
        self.distance_completed +=forward_distance
        
        self.last_location = hero_location
        
        # 1. Speed component
        speed_factor=hero_velocity / self.target_speed
        normalized_speed = np.clip(speed_factor, 0, 1.0)
        speed_reward = normalized_speed
        
        # 2. Direction component
        direction_reward = direction_factor
        
        # 3. Action smoothness component
        action_smoothness_reward = abs(self.steer - self.prev_steer)
        
        reward = -1e-3 - action_smoothness_reward
        
        if speed_factor<1.0:
            reward +=  delta_distance
        
        # reward += 
        
        if self.done_speed:
            reward -= WEIGHTS['collision_penalty']
            
        if self.collision:
            reward -= WEIGHTS['collision_penalty']
            
        if self.diff_lane:
            reward -= WEIGHTS['lane_departure_penalty']
        
        if self.done_dist:
            # print("Done")
            reward += WEIGHTS['goal_reached_bonus']
        
        reward = np.nan_to_num(reward, nan=0.0)
        
        self.info.update({
            'speed_reward': speed_reward,
            'direction_reward': direction_reward,
            'action_smoothness_reward': action_smoothness_reward,
            'delta_distance': delta_distance,
            'forward_distance': forward_distance,
            'total_reward': reward
        })
        
        self.rewards.append(reward)
        
        self.prev_steer = self.steer
        self.prev_throttle =self.throttle
        return reward

# --- CONFIG (Unchanged) ---
config = {
    "framework": "torch",
    "num_workers": 1,
    "num_gpus_per_worker": 1,
    "num_cpus_per_worker": 3,
    "rollout_fragment_length": 16,
    "timesteps_per_iteration": 2000,
    "train_batch_size": 16,
    "learning_starts": 5000,
    "buffer_size": 15000,
    "lr": 0.0003,
    "exploration_config": {
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.0,
        "final_epsilon": 0.1,
        "epsilon_timesteps": 50000
    },
    "env_config": {
        "carla": {
            "host": "localhost",
            "timeout": 50.0,
            "timestep": 0.1,
            "retries_on_error": 25,
            "resolution_x": 600,
            "resolution_y": 600,
            "quality_level": "Low",
            "enable_map_assets": True,
            "enable_rendering": True,
            "show_display": True,
            "town":"Town01"
        },
        "experiment": {
            "type":STBL3Experiment,
            "hero": {
                "blueprint": ["vehicle.micro.microlino"],
                "sensors": {
                    "collision": {
                        "type": "sensor.other.collision"
                    },
                    "rgb": {
                        "type": "sensor.camera.rgb",
                        "image_size_x": 448, # This is the raw sensor size
                        "image_size_y": 448, # This is the raw sensor size
                        "transform": "1.9, 0.0, 1.7, 0.0, -15.0, 0.0"
                    },
                    
                    "lane_invasion": {
                        "type": "sensor.other.lane_invasion"
                    },
                    "imu":{
                        "type":"sensor.other.imu"
                    },
                    "goal_heading":{
                        "type":"sensor.goal.heading"
                    },
                },
            },
            "background_activity": {
                "n_vehicles": 0,
                "n_walkers": 0,
                "tm_hybrid_mode": True
            },
            "weather": "CloudySunset",
            "others": {
                "framestack": 1,
                "max_time_idle": 600,
                "max_dist": 200,
                "target_speed": 5.0
            }
        }
    }
}