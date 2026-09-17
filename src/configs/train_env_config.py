
# Modified from https://github.com/carla-simulator/rllib-integration/blob/main/dqn_example/dqn_experiment.py

from collections import defaultdict
import math
import random
import numpy as np
from gymnasium.spaces import Box, Dict

import carla

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
        self.image_size=64
        self.episode_num=1
        self.max_episodes=100000
        self.explore_mode=False
        self.distance_travelled = 0.0
        self.distance_completed = 0.0
        self.delta_angle=0.0
    def _cache_waypoints(self,world) -> None:
            env_map = world.get_map()
            waypoints = env_map.generate_waypoints(distance=2)
            trajectories = defaultdict(list)
            for wpt in waypoints:
                trajectories[f"{wpt.road_id}-{wpt.lane_id}"].append(wpt)
            self.trajectories = sorted([traj for traj in trajectories.values() if len(traj) > 3], 
                                        key=len, reverse=True)
            # self.max_curriculum_steps = len(self.trajectories)
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

        # control variables
        # self.max_steer = 0.5
        # self.max_throttle = 0.6
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.steer = 0.0
        self.throttle = 0.0
        # self.target_speed = 6.0
        self.velocity=0.0
        self.current_heading=0.0
        self.delta_angle=0.0
        # if self.episode_num < self.max_episodes * 0.3:  # First 30% of training
            # Higher chance of exploration in early training
        # self.explore_mode = random.random() < 0.3
        self.explore_mode=not self.explore_mode
        # Modify the explore_mode probability to decrease more rapidly with training progress
        # self.explore_mode = random.random() < max(0.8 * (1 - self.episode_num / (self.max_episodes * 0.2)), 0.3)
        # self.explore_mode=False
        # else:  # Later training
        #     self.explore_mode = random.random() < 0.3
        if self.explore_mode:
            print(f"Episode {self.episode_num}: Free exploration mode (no target heading)")
        else:
            print(f"Episode {self.episode_num}: Target-following mode")
        self.target_speed = random.uniform(5.0,10.0)
        self.info=dict()
        if self.trajectories is None:
            self._cache_waypoints(env.core.world)

    # def get_action_space(self):
    #     """Returns the action space, in this case, a discrete space"""
    #     return Discrete(len(self.get_actions()))

    def get_observation_space(self):
        image_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.image_size, self.image_size, 3,),
            dtype=np.float32,
        )
        
        vec_space = Box(
            low=-5.1,
            high=5.1,
            shape=(4 * self.frame_stack,),
            dtype=np.float32,
        )

        return Dict({"pixels":image_space, "vector":vec_space})

    def get_action_space(self):
        """Returns the continuous action space for steering and throttle"""
        return Box(
            low=np.array([-self.max_steer, -1.0]),  # [steering, throttle/brake]
            high=np.array([self.max_steer, 1.0]),
            dtype=np.float32
        )

    def compute_action(self, action):
        """Convert continuous actions to CARLA vehicle controls"""
        steer, throttle_brake = action

        action = carla.VehicleControl()
        # Smooth steering using previous value
        # steer=steer+self.steer
        # throttle_brake=throttle_brake+self.throttle
        # # Smooth steering using previous value

        action.steer = float(np.clip(steer, -self.max_steer, self.max_steer))
        self.steer=action.steer
        self.throttle=action.throttle
        # Handle throttle and brake separately
        
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

    # Remove the get_actions method since we're using continuous actions
    def get_observation(self, sensor_data, core):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """
        vecs = self.get_vec_obs(sensor_data, core)
        images = self.get_img_obs(sensor_data, core)
        return {"pixels":images, "vector":vecs}, self.info

    def get_vec_obs(self, sensor_data, core):
        # breakpoint()
        heading=sensor_data["goal_heading"][-1][-1]
        imu=sensor_data["imu"][-1][-1]
        vec = np.zeros(4)
        vec[0] = self.prev_steer / self.max_steer
        vec[1] = self.prev_throttle / self.max_throttle
        hero = core.hero
        wp=core.map.get_waypoint(hero.get_transform().location,project_to_road=True) 
        vec[2] = np.clip(self.get_speed(hero)/self.target_speed, 0.0, 1.0)

        current_forward_vector = carla_location_to_np_array(core.hero.get_transform().get_forward_vector())
        correct_forward_vector = carla_location_to_np_array(core.destination.transform.get_forward_vector())
        self.delta_angle=np.dot(current_forward_vector, correct_forward_vector)
        vec[3] = self.delta_angle

        if self.explore_mode:
            vec[3] = -1.0

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
        image = post_process_image(sensor_data['rgb'][1], normalized = True,crop=False, grayscale = False,image_size=self.image_size)

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

        return images
    
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
        # marker_location=sensor_data["goal_heading"][-1][0]
        wp=core.map.get_waypoint(hero.get_transform().location,project_to_road=False) 
        # self.done_dist = self.distance_travelled>200
        self.done_falling = False
        self.diff_lane = 'lane_invasion' in sensor_data.keys() or wp is None
        # self.diff_lane=False
        self.collision = 'collision' in sensor_data.keys()
    
        self.done_speed = (hero_velocity/(self.target_speed)) > 1.0
        self.done_speed=False
        if core.destination.transform.location.distance(hero.get_transform().location) <1.0:
            core.extend_goal()
        self.done_dist= self.distance_travelled>=200
        # self.done_dist = self.distance_travelled>=100
        done=self.done_falling or self.done_dist or self.diff_lane or self.collision or self.done_speed
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
            self.episode_num+=1
        #     self.info.update(is_success=self.done_dist,
        #                      distance_completed=self.distance_travelled,
        #                      max_reward=np.max(self.rewards),
        #                      min_reward=np.min(self.rewards),
        #                      mean_reward=np.mean(self.rewards))
            # print(self.distance_travelled)
        return done
    def compute_reward(self, sensor_data, core):
        """
        Compute the reward for RL training based on vehicle performance.
        
        Args:
            sensor_data: Dictionary containing sensor readings
            core: Core simulation object containing hero vehicle and map
            
        Returns:
            float: Calculated reward value
        """
        # Define reward component weights
        WEIGHTS = {
            'speed': 0.3,
            'direction': 0.1,
            'action_smoothness': 0.0,
            'time_penalty': 0.01,
            'speed_penalty': 1.0,
            'collision_penalty': 1.0,  # Still the largest penalty but scaled down
            'lane_departure_penalty': 1.0,
            'goal_reached_bonus': 1.0,
            "progress":0.4,
            "multipliers":10
        }
        
        
        # Get vehicle and sensor dat1
        hero = core.hero
        target_heading = np.nan_to_num(sensor_data["goal_heading"][-1][-1])
        current_heading = np.nan_to_num(sensor_data["imu"][-1][-1])
        hero_location = hero.get_location()
        hero_velocity = self.get_speed(hero)
        
        # Initialize tracking variables if needed
        if not hasattr(self, 'last_location') or self.last_location is None:
            self.last_location = hero_location
            self.distance_travelled = 0.0
            self.distance_completed=0.0
        
        # Compute distance moved since last step
        delta_distance = float(np.sqrt(np.square(hero_location.x - self.last_location.x) + 
                            np.square(hero_location.y - self.last_location.y)))
        
      # Calculate direction alignment using unit vectors and dot product
        # Convert headings to unit vectors
        # target_heading_vector = np.array([np.cos(target_heading), np.sin(target_heading), 0.0])
        # current_heading_vector = np.array([np.cos(current_heading), np.sin(current_heading), 0.0])

        # Calculate direction factor using dot product of unit vectors
        # This gives cosine of angle between vectors

        direction_factor = self.delta_angle
        # print(current_heading,target_heading,np.rad2deg(np.acos(direction_factor)),direction_factor)
        # Update distance metrics
        # print("Angle factor",self.delta_angle)

        # Update tracking variables for next step
        self.last_location = hero_location
        
        # For road alignment
        wp = core.map.get_waypoint(hero.get_transform().location, project_to_road=True)
        # if wp is not None and wp.is_junction:
        #     # In junctions, use the heading-based direction factor
        #     pass  # Already calculated above
        # else:
        #     # On regular roads, use the road's forward vector
        #     current_forward_vector = carla_location_to_np_array(core.hero.get_transform().get_forward_vector())
        #     correct_forward_vector = carla_location_to_np_array(wp.transform.get_forward_vector())
            
        #     # Normalize vectors to ensure they're unit vectors
        #     current_forward_vector = current_forward_vector / np.linalg.norm(current_forward_vector)
        #     correct_forward_vector = correct_forward_vector / np.linalg.norm(correct_forward_vector)
            
        #     # Calculate direction factor using dot product
        #     direction_factor = np.dot(current_forward_vector, correct_forward_vector)


        # print("Dire",direction_factor)
        # 1. Speed component: how close to target speed
        speed_factor = hero_velocity / self.target_speed
        speed_weight=WEIGHTS['speed']
        normalized_speed = speed_factor
        speed_reward = normalized_speed
        speed_reward=np.exp(-abs(hero_velocity-self.target_speed))
        # print(-abs(hero_velocity-self.target_speed)**2,-abs(hero_velocity-self.target_speed))
        forward_distance = delta_distance * direction_factor
        self.distance_travelled += delta_distance
        self.distance_completed += forward_distance
        # 2. Direction component: alignment with target heading
        # direction_reward = max(direction_factor, 0)  # Only reward positive alignment
        if self.explore_mode:
            direction_factor = 0.0
            speed_weight+=WEIGHTS['direction']
        # if hero_velocity<self.target_speed:
        #     print(delta_distance,speed_reward,direction_factor)
        direction_reward=direction_factor
        # 3. Action smoothness component (steer is set elsewhere)
        action_smoothness_reward = -abs(self.steer)  # Penalize large steering actions
        # print(WEIGHTS['action_smoothness'] * action_smoothness_reward)
        # ===== Combine Reward Components =====
        reward = 0.0
     
        # Add positive components
        # if speed_factor<1.0:
        reward += speed_weight * speed_reward
        
        # Add direction reward only in non-exploration mode
        # if not self.explore_mode:
        reward += WEIGHTS['direction'] * direction_reward
        
        # Add action smoothness
        reward += WEIGHTS['action_smoothness'] * action_smoothness_reward
        
        # Apply time penalty to encourage efficiency
        reward -= WEIGHTS['time_penalty']
        
        # Apply penalties
        if self.done_speed or self.collision or self.diff_lane:  # Vehicle stopped when it shouldn't
            # print("oVer speeding")
            reward -= WEIGHTS['collision_penalty']
            
        # if self.collision:   # Vehicle collided with something
        #     # print("Collision")
        #     reward -= WEIGHTS['collision_penalty']
            
        # if self.diff_lane:   # Vehicle left its lane
        #     # print("Diff/ lane")
        #     reward -= WEIGHTS['lane_departure_penalty']
        # if hero_velocity/
        # Add bonus for reaching target distance
        if self.done_dist:   # Reached target distance
            print("Done")
            reward += WEIGHTS['goal_reached_bonus']
        reward=reward*WEIGHTS["multipliers"]
        # Ensure reward is numerically stable
        # print(reward,WEIGHTS["time_penalty"]*10)
        if math.isnan(reward):
            breakpoint()
        # reward = np.nan_to_num(reward, nan=0.0)
        
        # Store info for logging
        self.info.update({
            'speed_reward': WEIGHTS['speed'] * speed_reward,
            'direction_reward': WEIGHTS['direction'] * direction_reward,
            'action_smoothness_reward': action_smoothness_reward,
            'delta_distance': delta_distance,
            'forward_distance': forward_distance,
            'total_reward': reward
        })
        
        # Store reward for logging
        self.rewards.append(reward)
        
        # Update previous actions for potential future smoothness calculation
        self.prev_steer = self.steer
        self.prev_throttle =self.throttle
        return reward

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
                        "image_size_x": 64,
                        "image_size_y": 64,
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
                # "spawn_points": [
                #     "-115.60, -207.60, 11.02, -0.00, -0.01, -179.92",  # tl_l
                #     "-243.70, 71.50, 11.98, -0.00, -0.08, 90.81",      # bl_d
                #     "76.60, 202.10, 1.00, -0.00, 0.00, 0.29",          # br_r
                #     "210.20, -50.30, 1.00, -0.00, 0.00, -90.67",       # ur_u
                #     "-114.30, 191.00, 9.27, -0.00, 0.61, 179.97",      # bl_l
                #     "-233.00, -51.50, 11.00, -0.00, 0.00, -90.84",     # tl_u
                #     "83.90, -190.30, 1.00, -0.00, 0.00, -0.24",        # tr_r
                #     "189.30, 88.20, 1.00, -0.00, 0.00, 90.66"          # br_d
                # ]
            },
            "background_activity": {
                "n_vehicles": 0,
                "n_walkers": 0,
                "tm_hybrid_mode": True
            },
            # "town": "Town05",
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