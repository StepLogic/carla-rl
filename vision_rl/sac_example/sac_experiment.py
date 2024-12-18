import math
import numpy as np
from gym.spaces import Box, Tuple
import carla
from vision_rl.rllib_integration.base_experiment import BaseExperiment
from vision_rl.rllib_integration.helper import post_process_image

class SACExperiment(BaseExperiment):
    def __init__(self, config={}):
        super().__init__(config)
        self.frame_stack = self.config["others"]["framestack"]
        self.max_time_idle = self.config["others"]["max_time_idle"]
        self.max_dist = self.config["others"]["max_dist"]
        self.target_speed = self.config["others"]["target_speed"]
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]
        self.last_action = None
        self.max_steer=0.5


    def reset(self):
        self.time_idle = 0
        self.time_episode = 0
        self.done_time_idle = False
        self.done_falling = False
        self.done_dist = False
        self.last_location = None
        self.last_velocity = 0
        self.distance_travelled = 0.0
        self.prev_vec_0 = None
        self.prev_vec_1 = None
        self.prev_vec_2 = None
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None
        self.max_steer = 0.5
        self.max_throttle = 0.6
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.max_steer=0.5

    def get_action_space(self):
        return Box(
            low=np.array([-self.max_steer, -1.0]),
            high=np.array([self.max_steer, 1.0]),
            dtype=np.float32
        )

    def get_observation_space(self):
        image_space = Box(
            low=-1.0,
            high=1.0,
            shape=(84, 84, self.frame_stack,),
            dtype=np.float32,
        )
        vec_space = Box(
            low=-5.1,
            high=5.1,
            shape=(4 * self.frame_stack,),
            dtype=np.float32,
        )
        return Tuple((image_space, vec_space))

    def compute_action(self, action):
        steer, throttle_brake = action
        vehicle_control = carla.VehicleControl()
        vehicle_control.steer = float(np.clip(steer, -self.max_steer, self.max_steer))
        
        if throttle_brake >= 0:
            vehicle_control.throttle = float(np.clip(throttle_brake, 0.0, self.max_throttle))
            vehicle_control.brake = 0.0
        else:
            vehicle_control.throttle = 0.0
            vehicle_control.brake = float(np.clip(-throttle_brake, 0.0, 1.0))

        vehicle_control.reverse = False
        vehicle_control.hand_brake = False

        self.last_action = vehicle_control
        self.prev_steer = vehicle_control.steer
        self.prev_throttle = vehicle_control.throttle

        return vehicle_control

    def get_observation(self, sensor_data, core):
        vecs = self.get_vec_obs(sensor_data, core)
        images = self.get_img_obs(sensor_data, core)
        return (images, vecs), {}

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
        image = post_process_image(sensor_data['rgb'][1], normalized=True, grayscale=True)

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
        vel = hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def get_done_status(self, sensor_data, core):
        hero = core.hero
        self.done_time_idle = self.max_time_idle < self.time_idle
        if self.get_speed(hero) > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1
        self.time_episode += 1
        self.done_dist = self.distance_travelled > self.max_dist
        self.done_falling = hero.get_location().z < -0.5
        self.diff_lane = 'lane_invasion' in sensor_data.keys()
        self.collision = 'collision' in sensor_data.keys()
        return self.done_time_idle or self.done_falling or self.done_dist or self.diff_lane or self.collision

    def compute_reward(self, sensor_data, core):
        hero = core.hero
        hero_location = hero.get_location()
        hero_velocity = self.get_speed(hero)

        if self.last_location is None:
            self.last_location = hero_location

        delta_distance = float(np.sqrt(np.square(hero_location.x - self.last_location.x) + 
                            np.square(hero_location.y - self.last_location.y)))
        self.distance_travelled += delta_distance

        self.last_location = hero_location
        self.last_velocity = hero_velocity

        # Speed matching reward
        speed_reward = -abs(hero_velocity - self.target_speed) / self.target_speed
        
        # Progress reward
        progress_reward = delta_distance * 2.0
        
        # Action smoothness penalty
        if self.last_action is not None:
            smoothness_penalty = -abs(self.last_action.steer) * 0.2
        else:
            smoothness_penalty = 0.0

        # Combine rewards
        reward = speed_reward + progress_reward + smoothness_penalty

        # Terminal rewards/penalties
        if self.done_falling:
            reward += -50.0
        if self.done_dist:
            reward += 100.0
        if self.done_time_idle:
            reward += -50.0
        if self.collision:
            reward += -50.0
        if self.diff_lane:
            reward += -25.0

        return reward