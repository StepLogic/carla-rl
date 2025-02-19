
# Modified from https://github.com/carla-simulator/rllib-integration/blob/main/dqn_example/dqn_experiment.py

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
        self.velocity=0.0
        self.info=dict()
        self.rewards=[]
        self.image_size=64

    def reset(self,*arg,**kwargs):
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
        self.target_speed = 8.0
        self.velocity=0.0
        self.current_heading=0.0
        # self.target_speed = random.uniform(5.0,10.0)
        self.info=dict()

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
        action.steer = float(np.clip(steer, -self.max_steer, self.max_steer))
        self.steer=action.steer
        self.throttle=action.throttle
        # Handle throttle and brake separately
        
        if throttle_brake >= 0:
            action.throttle = float(np.clip(throttle_brake, 0.0, self.max_throttle))
            action.brake = 0.0
        else:
            action.throttle = 0.0
            action.brake = float(np.clip(-throttle_brake, 0.0, 1.0))
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
        vec[0] = self.steer / self.max_steer
        vec[1] = self.throttle / self.max_throttle
        hero = core.hero
        vec[2] = np.clip(self.get_speed(hero)/(self.target_speed+1e-8), 0.0, 5.1)
        # vec[3] = self.time_idle / self.max_time_idle
        vec[3]= np.clip(imu/(heading+1e-8),-5.1,5.1) 
        # vec[4]= np.clip(heading/np.pi,-5.1,5.1) 
        if self.prev_vec_0 is None:
            self.prev_vec_0 = vec
            self.prev_vec_1 = self.prev_vec_0
            self.prev_vec_2 = self.prev_vec_1

        vecs = vec  #add gaussian noise

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
        hero = core.hero
        self.info=dict()
        self.done_time_idle = self.max_time_idle < self.time_idle
        if self.get_speed(hero) > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1
        self.time_episode += 1
               
        hero_velocity = self.get_speed(hero)
        # marker_location=sensor_data["goal_heading"][-1][0]
        wp=core.map.get_waypoint(hero.get_transform().location,project_to_road=False) 
        self.done_dist = self.distance_travelled>200
        self.done_falling = hero.get_location().z < -0.5
        self.diff_lane = 'lane_invasion' in sensor_data.keys()
        self.collision = 'collision' in sensor_data.keys()
        self.done_speed=(hero_velocity/(self.target_speed+1e-8)) > 5.0
        done=self.done_falling or self.done_dist or self.diff_lane or self.collision
        self.info.update(dict(is_success=self.done_dist,
                             distance_completed=self.distance_travelled,
                             max_reward=0,
                             min_reward=0,
                             mean_reward=0
                             ))
        if len(self.rewards)>0:
                    self.info.update(dict(
                             max_reward=np.max(self.rewards),
                             min_reward=np.min(self.rewards),
                             mean_reward=np.mean(self.rewards)))
        # if done:
        #     self.info.update(is_success=self.done_dist,
        #                      distance_completed=self.distance_travelled,
        #                      max_reward=np.max(self.rewards),
        #                      min_reward=np.min(self.rewards),
        #                      mean_reward=np.mean(self.rewards))
            # print(self.distance_travelled)
        return done

    def compute_reward(self, sensor_data, core):
        hero = core.hero
        heading=sensor_data["goal_heading"][-1][-1]
        imu=sensor_data["imu"][-1][-1]
        # delta_heading=np.clip(abs(imu-heading),0,np.pi)
        # angle_factor=max(1-min(delta_heading/self.max_angle_deviation,1.0),1e-3)
        # heading=np.nan_to_num(math.cos(delta_heading),0)
        # Hero-related variables
        hero_location = hero.get_location()
        hero_velocity = self.get_speed(hero)

        # Initialize last location
        if self.last_location == None:
            self.last_location = hero_location
        if self.last_heading == None:
            self.last_heading = heading
        # Compute deltas
        delta_distance = float(np.sqrt(np.square(hero_location.x - self.last_location.x) + \
                            np.square(hero_location.y - self.last_location.y)))
        

        # distance_travelled=self.distance_travelled+(delta_distance)
        # Update variables
        self.last_location = hero_location
        self.last_velocity = hero_velocity
        self.distance_travelled += max(delta_distance*np.cos(abs(imu-heading)),0)

        # Reward if going forward
        # if hero_velocity < self.target_speed  and hero_velocity > 1.0:
        #     reward = delta_distance
        # # print(np.rad2deg(imu -heading),np.rad2deg(heading),np.rad2deg(imu),np.exp(-abs(imu-heading))*0.5 )
        
        # # reward = min(abs(self.target_speed-hero_velocity))
        # else:
        #     reward = -1e-2
        # print(reward)
        # if hero_velocity < self.target_speed and hero_velocity < self.target_speed:
        #     # print(heading/5)
        #     reward += heading*1e-3
        # reward = 1e-2*((self.target_speed-hero_velocity)**2 + (imu-heading)**2 + 1e-1*np.sum(np.array([self.prev_steer,self.prev_throttle]-np.array([self.steer,self.throttle])))**2)

        # max_speed_error = self.target_speed**2
        # max_heading_error = heading**2
        # max_action_error = np.sum(self.get_action_space().low-self.get_action_space().high)**2
        
        # min_reward = 1e-2 * ((max_speed_error+1e-8) + (max_heading_error+1e-8)+1e-1*max_action_error)
        # max_reward = 0

        # Normalize to [0,1]
        # reward = (reward - min_reward) / (max_reward - min_reward) #scale # or 
        # reward = -reward/min_reward
        # reward=-1.0 + np.exp(-(imu-heading)**2) + .4*np.exp(-(self.target_speed-hero_velocity)**2)+0.1*np.exp(-np.sum(np.array([self.prev_steer,self.prev_throttle]-np.array([self.steer,self.throttle])))**2)
        # reward=-1e-3
        # Normalize target speed error to [0.2, 1.0] to avoid being too lenient
        # target_speed_error = np.clip( / self.target_speed, -1.0, 1.0)
        speed_factor=np.exp(-(hero_velocity-self.target_speed)**2)

        # Normalize heading error to [-1.0, 1.0] to allow for larger corrections
        heading_factor = np.exp(-(imu-heading)**2)
        # heading_error = np.clip(heading_error, -1.0, 1.0)

        # Calculate smooth action penalty to encourage smoother control inputs
        action_factor = np.exp(-np.sum(abs(self.prev_steer - self.steer) + abs(self.prev_throttle - self.throttle)))

        # Base reward combines speed error, heading error, and smooth action
        # reward = target_speed_error * (heading_error + smooth_action)
        # reward = target_speed_error*(0.8+heading_error+0.2*smooth_action) + self.distance_travelled/200
        wp=core.map.get_waypoint(hero.get_transform().location,project_to_road=False) 
        # Only penalize heading when it's significantly off or at intersections
        # heading_factor = np.exp(-((imu-heading)**2)) 
        # heading_weight = 1.0 if  wp.is_junction else 0.0
        reward=  2.0*speed_factor + 0.1*action_factor + heading_factor
        # print(speed_factor,self.target_speed)
        # reward=-1e-3
        # if hero_velocity<self.target_speed:
        #     reward += delta_distance
        # else:
        #     reward+=0
        # reward=  target_speed_error*0.5 + heading_error + smooth_action*0.1
        # print(reward,target_speed_error,heading_error)
        # Penalize falling, collisions, lane invasions, and excessive speed
        if self.collision:
            # print(f'Collision Smooth={smooth_action:3f} Dist={self.distance_travelled:3f} Target_S={self.target_speed:.4f} Vel={hero_velocity:.4f} R={reward:.4f} Err={target_speed_error:.4f} H_Err={heading_error:.4f}')
            reward += -10
        if self.diff_lane:
            # print(f'Lane Invasion  Smooth={smooth_action:3f} Dist={self.distance_travelled:3f} Target_S={self.target_speed:.4f} Vel={hero_velocity:.4f} R={reward:.4f} Err={target_speed_error:.4f} H_Err={heading_error:.4f}')
            reward += -10
        if self.done_speed:
            # print(f'Too fast Smooth={smooth_action:3f} Dist={self.distance_travelled:3f} Ratio={hero_velocity/self.target_speed:.3f} Target_S={self.target_speed:.3f} Vel={hero_velocity:.3f} R={reward:.4f} Err={target_speed_error:.4f} H_Err={heading_error:.4f}')
            reward += -10
        # Reward for reaching the target distance
        if self.done_dist:
            # print(f"Max Dist Smooth={smooth_action:3f} Dist={self.distance_travelled:3f}")
            reward += 10

        # Scale the reward to a reasonable range (no need for *10)
        # reward = np.clip(reward, -2.0, 2.0)
        # reward*=10
# 
        # Store the reward for logging or analysis
        self.rewards.append(reward)

        # Update previous actions for smoothness calculation
        self.prev_steer = self.steer
        self.prev_throttle = self.throttle

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
            "timeout": 20.0,
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
                "blueprint": "vehicle.mercedes.coupe_2020",
                "sensors": {
                    "collision": {
                        "type": "sensor.other.collision"
                    },
                    "rgb": {
                        "type": "sensor.camera.rgb",
                        "image_size_x": 300,
                        "image_size_y": 300,
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