# Modified from https://github.com/carla-simulator/rllib-integration/blob/main/dqn_example/dqn_experiment.py

from collections import defaultdict
import math
import numpy as np
from gymnasium.spaces import Box, Dict

import carla

from vision_rl.rllib_integration.base_experiment import BaseExperiment
from vision_rl.rllib_integration.helper import carla_location_to_np_array, post_process_image


class STB3HERGoalExperiment(BaseExperiment):
    def __init__(self, config={}):
        super().__init__(config)  # Creates a self.config with the experiment configuration

        self.frame_stack = self.config["others"]["framestack"]
        self.max_time_idle = self.config["others"]["max_time_idle"]
        self.max_dist = self.config["others"]["max_dist"]
        self.target_speed = self.config["others"]["target_speed"]
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]
        self.last_action = None
        # control variables
        self.max_steer = 0.5
        self.max_throttle = 0.6
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.achieved_goal=None
        self.trajectories=None
        self.large_deviation=False
        self.info=dict()

    def _cache_waypoints(self,world) -> None:
        env_map = world.get_map()
        waypoints = env_map.generate_waypoints(distance=2)
        trajectories = defaultdict(list)
        for wpt in waypoints:
            trajectories[f"{wpt.road_id}-{wpt.lane_id}"].append(wpt)
        self.trajectories = [traj for traj in trajectories.values() if len(traj) > 3]

    def reset(self,core):
        """Called at the beginning and each time the simulation is reset"""

        # Ending variables
        self.time_idle = 0
        self.time_episode = 0
        self.done_time_idle = False
        self.done_falling = False
        self.done_dist = False
        # self.step=0

        # hero variables
        self.last_location = None
        self.last_velocity = 0
        self.distance_travelled = 0.0

        # Sensor stack
        self.prev_vec_0 = None
        self.prev_vec_1 = None
        self.prev_vec_2 = None
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None

        # control variables
        self.max_steer = 0.5
        self.max_throttle = 0.6
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.info=dict()
        if self.trajectories is None:
            self._cache_waypoints(core.core.world)


    # def get_action_space(self):
    #     """Returns the action space, in this case, a discrete space"""
    #     return Discrete(len(self.get_actions()))

    def get_observation_space(self):
        image_space = Box(
            low=0,
            high=255,
            shape=(84, 84,3),
            dtype=np.uint8,
        )
        
        vec_space = Box(
            low=-5.1,
            high=5.1,
            # shape=(4 * self.frame_stack,),
            shape=(4,),
            dtype=np.float32,
        )
        loc = Box(
            low=-5.1,
            high=5.1,
            # shape=(4 * self.frame_stack,),
            shape=(3,),
            dtype=np.float32,
        )

        return Dict({"observation":image_space,"states":vec_space,
                     "goal":image_space,
                     "achieved_goal":loc,
                     "desired_goal":loc,})


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
        action.steer = float(np.clip(self.prev_steer + steer, -self.max_steer, self.max_steer))
        
        # Handle throttle and brake separately
        if throttle_brake >= 0:
            action.throttle = float(np.clip(throttle_brake, 0.0, self.max_throttle))
            action.brake = 0.0
        else:
            action.throttle = 0.0
            action.brake = float(np.clip(-throttle_brake, 0.0, 1.0))

        action.reverse = False
        action.hand_brake = False

        self.last_action = action
        self.prev_steer = action.steer
        self.prev_throttle = action.throttle

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
        images,goal = self.get_img_obs(sensor_data, core)
        self.achieved_goal=images
        hero_location = core.hero.get_location()
        self.info.update(dict(obs=carla_location_to_np_array(hero_location),goal=sensor_data['goal'][1][-1]))
        return {"observation":images,"states":vecs,"goal":goal,"achieved_goal":carla_location_to_np_array(hero_location),"desired_goal":sensor_data['goal'][1][-1]}, self.info

    def get_vec_obs(self, sensor_data, core):
        vec = np.zeros(4)
        vec[0] = self.prev_steer / self.max_steer
        vec[1] = self.prev_throttle / self.max_throttle
        
        hero = core.hero
        vec[2] = np.clip(self.get_speed(hero)/self.target_speed, 0.0, 1.0)

        vec[3] = self.time_idle / self.max_time_idle

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
        return vecs

    def get_img_obs(self, sensor_data, core):

        image = post_process_image(sensor_data['rgb'][1], normalized = False, grayscale = False)
        # breakpoint()
        goal = post_process_image(sensor_data['goal'][1][0], normalized = False, grayscale = False)

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

        return images,goal
    def get_speed(self, hero):
        """Computes the speed of the hero vehicle in Km/h"""
        vel = hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def get_done_status(self, sensor_data, core):
        """Returns whether or not the experiment has to end"""
        
        hero = core.hero
        self.done_time_idle = self.max_time_idle < self.time_idle
        if self.get_speed(hero) > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1
        self.time_episode += 1
        # dist=sensor_data['goal'][1][1]
        goal_loc=sensor_data['goal'][1][-1]
        hero_location = hero.get_location()
        hero_location=carla_location_to_np_array(hero_location)
        dist=np.linalg.norm(goal_loc - hero_location)
        self.done_dist = self.distance_travelled > self.max_dist
        self.done_falling = hero.get_location().z < -0.5
        self.diff_lane = 'lane_invasion' in sensor_data.keys()
        self.collision = 'collision' in sensor_data.keys()
        done=self.done_time_idle or self.done_falling or self.done_dist or self.diff_lane or self.collision or dist<=3.0
        # if dist:
        # print(dist)
        if done:
            self.info.update(dict(is_success=int(dist<=3.0),distance_to_goal=self.distance_travelled,collision=int(self.collision)))
        return done

    def compute_reward(self, sensor_data, core):
        hero = core.hero

        goal_loc=sensor_data['goal'][1][-1]
        dist=sensor_data['goal'][1][1]
        # Hero-related variables
        hero_location = hero.get_location()
        hero_velocity = self.get_speed(hero)

        # Initialize last location
        if self.last_location == None:
            self.last_location = hero_location
        transform = hero.get_transform()
        # Compute deltas
        # delta_distance = float(np.sqrt(np.square(hero_location.x - self.last_location.x) + np.square(hero_location.y - self.last_location.y)))
        delta_loc=carla_location_to_np_array(self.last_location)-carla_location_to_np_array(hero_location)
        displacement=np.dot(delta_loc,goal_loc/np.linalg.norm(goal_loc))
        location = np.array([transform.location.x, transform.location.y])
        f, d_f = core.spline(location)
        d_to_lane = np.linalg.norm(f - location)
        # max_dev = hero.bounding_box.extent.y * 2
        # Update variables
        self.last_location = hero_location
        self.last_velocity = hero_velocity

        # Reward if going forward
        # reward=displacement+np.exp(-d_to_lane)
        
        if hero_velocity < self.target_speed:
            reward=displacement
        else:
            reward = 0.0
        if self.done_falling:
            reward += -1.0
        if self.done_dist:
        #     print("Max dist travelled")
            reward += 1.0
        if self.done_time_idle:
        #     print("Done idle")
            reward += -1.0
        if self.collision:
        #     # print('collision')
            reward += -1.0
        if self.diff_lane:
            reward += -1.0
        if dist<=1.5:
            reward += 1.0

        return reward*10
    def compute_hinsight_rewards(self,            
            achieved_goal,
            desired_goal,
            info
            ):
        dist = np.linalg.norm(np.subtract(achieved_goal ,desired_goal),axis=-1)
        # breakpoint()
        return np.where(dist<=3.0,10,0)

        
    
