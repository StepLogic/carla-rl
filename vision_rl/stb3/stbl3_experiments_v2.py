from collections import defaultdict
# import gym.spaces
from gymnasium.spaces import Box
import carla
from vision_rl.rllib_integration.base_experiment import BaseExperiment
from vision_rl.rllib_integration.helper import post_process_image,carla_location_to_np_array
import gymnasium as gym
import math
# import warnings
import numpy as np
from scipy.interpolate import splprep, splev

def compute_heading(location1, location2):
    dx = location2[0] - location1[0]
    dy = location2[1] - location1[1]
    heading_rad = math.atan2(dy, dx)
    heading_deg = math.degrees(heading_rad)
    if heading_deg < 0:
        heading_deg += 360
    return heading_deg

def make_curve(points):
    t = np.linspace(0, 1, len(points))
    tckp, u = splprep(points.T, u=t,k=3)
    return lambda t:np.array(splev(t,tckp)),tckp

def derivative_curve(tckp):
    return lambda t:np.array(splev(t,tckp,der=1))

def get_curve(points):
    indx = np.argsort(points[:,0])
    points = points[indx]
    f_t,tckp=make_curve(points)
    f_prime_t=derivative_curve(tckp)
    t_samples = np.linspace(0, 1, 1000)
    curve_points = np.array([f_t(t) for t in t_samples])
    def curve(x):
        distances = np.linalg.norm(curve_points - x, axis=1)
        closest_idx = np.argmin(distances)
        t = t_samples[closest_idx]
        return f_t(t),f_prime_t(t)
    return curve

class STBL3GoalExperiment(BaseExperiment):
    def __init__(self,config={}):
        super().__init__(config)
        self.frame_stack = self.config["others"]["framestack"]
        self.max_time_idle = 150  # Reduced from 300
        self.max_dist = 100  # Added reasonable distance limit
        self.target_speed = self.config["others"]["target_speed"]
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]
        self.last_action = None
        self.max_steer = 0.5
        self.max_throttle = 0.6
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.trajectories = None
        self.large_deviation = False
        self.info = dict()
        self.heading = None
        self.total_distance = None
        self.curriculum_step = 1
        self.max_curriculum_steps = 5

    def _cache_waypoints(self,world) -> None:
            env_map = world.get_map()
            waypoints = env_map.generate_waypoints(distance=2)
            trajectories = defaultdict(list)
            for wpt in waypoints:
                trajectories[f"{wpt.road_id}-{wpt.lane_id}"].append(wpt)
            self.trajectories = sorted([traj for traj in trajectories.values() if len(traj) > 3], 
                                        key=len, reverse=True)
            self.max_curriculum_steps = len(self.trajectories)
    def reset(self,core):
        self.time_idle = 0
        self.time_episode = 0
        self.done_time_idle = False
        self.done_falling = False
        self.done_dist = False
        self.large_deviation = False

        self.last_location = None
        self.last_velocity = 0
        self.distance_travelled = 0.0
        self.last_distance_to_goal = None

        self.prev_vec_0 = None
        self.prev_vec_1 = None
        self.prev_vec_2 = None
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None

        self.max_steer = 1.0
        self.max_throttle = 1.0
        self.prev_steer = 0.0
        self.prev_reward=None
        self.prev_throttle = 0.0
        self.heading = None
        self.total_distance = None
        
        if self.trajectories is None:
            self._cache_waypoints(core.core.world)

    def get_observation_space(self):
        image_space = Box(
            low=-1.0,
            high=1.0,
            shape=(84, 84, self.frame_stack,),
            dtype=np.float32,
        )
        goal_image_space = Box(
            low=-1.0,
            high=1.0,
            shape=(84, 84,1),
            dtype=np.float32,
        )
        vec_space = Box(
            low=-5.1,
            high=5.1,
            shape=(6 * self.frame_stack,),
            dtype=np.float32,
        )
        return gym.spaces.Dict({"image":image_space,"goal":goal_image_space, "vector":vec_space})

    def get_action_space(self):
        return Box(
            low=np.array([-self.max_steer, -self.max_throttle]),
            high=np.array([self.max_steer, self.max_throttle]),
            dtype=np.float32
        )

    def compute_action(self, action):
        steer, throttle_brake = action
        action = carla.VehicleControl()
        action.steer = float(np.clip(self.prev_steer + steer, -self.max_steer, self.max_steer))
        
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

    def get_observation(self, sensor_data, core):
        if self.total_distance is None:
            hero = core.hero
            goal_location = sensor_data['goal'][1][-1]
            self.total_distance = np.linalg.norm(goal_location[:2]-carla_location_to_np_array(hero.get_transform().location)[:2])
        vecs = self.get_vec_obs(sensor_data, core)
        images, goal = self.get_img_obs(sensor_data, core)
        return {"image":images, "goal":goal, "vector":vecs}, None

    def get_vec_obs(self, sensor_data, core):
        imu = sensor_data['imu'][1]
        if self.heading is None:
            heading = np.deg2rad(compute_heading(carla_location_to_np_array(core.hero.get_location()),sensor_data['goal'][1][-1]))
            self.heading = (imu[-1] + heading) % (2*np.pi)

        vec = np.zeros(6)
        vec[0] = self.prev_steer / self.max_steer
        vec[1] = self.prev_throttle / self.max_throttle
        hero = core.hero
        vec[2] = np.clip(self.get_speed(hero)/self.target_speed, 0.0, 1.0)
        vec[3] = self.time_idle / self.max_time_idle
        vec[4] = imu[-1]/np.pi
        vec[5] = self.heading/np.pi

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
        image = post_process_image(sensor_data['rgb'][1], crop=False, normalized=True, grayscale=True)
        goal = post_process_image(sensor_data['goal'][1][0], crop=False, normalized=True, grayscale=True)

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

        return images, goal
    
    def get_speed(self, hero):
        vel = hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def get_done_status(self, sensor_data, core):
        hero = core.hero
        self.info = dict()
        self.done_time_idle = self.max_time_idle < self.time_idle
        if self.get_speed(hero) > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1
        self.time_episode += 1
        wp=core.map.get_waypoint(hero.get_transform().location)
        goal_location = sensor_data['goal'][1][-1]
        distance_to_goal = np.linalg.norm(goal_location[:2]-carla_location_to_np_array(hero.get_transform().location)[:2])
        self.done_falling = hero.get_location().z < -0.5
        self.diff_lane = 'lane_invasion' in sensor_data.keys()
        self.collision = 'collision' in sensor_data.keys()


        done = (self.done_time_idle or self.done_falling or self.diff_lane or 
                self.collision or distance_to_goal <= 1.5 or wp is None)
        # done = distance_to_goal <= 1.5
        if done:
            self.info = dict(
                is_success=distance_to_goal <= 1.5,
                distance_to_goal=self.last_distance_to_goal
            )
        return done

    def compute_reward(self, sensor_data, core):
        hero = core.hero
        goal_loc = sensor_data['goal'][1][-1]
        hero_location = hero.get_location()
        hero_velocity = self.get_speed(hero)
        # hero_velocity=np.dot(carla_location_to_np_array(hero.get_velocity()),goal_loc/np.linalg.norm(goal_loc))
        distance_to_goal = np.linalg.norm(goal_loc-carla_location_to_np_array(hero.get_transform().location))
        displacement=np.dot(carla_location_to_np_array(hero.get_transform().location),goal_loc/np.linalg.norm(goal_loc))
        
        # print(self.total_distance,distance_to_goal)
        # displ=np.dot(carla_location_to_np_array(hero.get_velocity()),goal_location/np.linalg.norm(goal_location))
        if self.last_location is None:
            self.last_location = hero_location
            self.last_distance_to_goal = distance_to_goal
        if self.prev_reward is None:
            self.prev_reward=displacement
            
        delta_distance = float(np.sqrt(np.square(hero_location.x - self.last_location.x) + \
                            np.square(hero_location.y - self.last_location.y)))
        self.distance_travelled += delta_distance
        wp=core.map.get_waypoint(hero.get_transform().location)
        # Dense progress reward
        # reward = min(distance_to_goal/self.total_distance,1.0)
        
        # # Speed matching reward
        # speed_reward = -min(abs(self.target_speed-hero_velocity)/self.target_speed,1.0)
        # reward += speed_reward
        
        # Heading alignment reward
        # heading = sensor_data['imu'][1][-1]
        # heading_diff = abs(self.heading - heading)
        # heading_reward = -heading_diff/(2*np.pi)
        # reward += 0.1 * heading_reward

        # Update distance traveled
        delta_distance = float(np.sqrt(np.square(hero_location.x - self.last_location.x) + \
                            np.square(hero_location.y - self.last_location.y)))
        self.distance_travelled += delta_distance
        reward=displacement

        if hero_velocity < self.target_speed:
        #     # print(heading-self.heading)
        #     # reward = delta_distance -min(abs(self.target_speed-hero_velocity)/self.target_speed,1.0) - (1-min(distance_to_goal/self.total_distance,1.0))
        #     reward = hero_velocity*3.6/self.target_speed
                step_reward=self.prev_reward-reward - -0.1
        else:
            step_reward = -0.1
        # # Terminal rewards/penalties
        if self.done_falling or self.collision or self.done_time_idle or self.diff_lane or wp is None:
            step_reward += -1.0
        if distance_to_goal <= 2.5:
            step_reward += 1.0
        #     if self.curriculum_step < self.max_curriculum_steps:
        #         self.curriculum_step += 1
        # Update tracking variables
        self.last_location = hero_location
        self.last_velocity = hero_velocity
        self.last_distance_to_goal = distance_to_goal
        # print(step_reward,self.prev_reward,reward)
        self.prev_reward=reward
        
        
        return step_reward