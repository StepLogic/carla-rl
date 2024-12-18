# Modified from https://github.com/carla-simulator/rllib-integration/blob/main/dqn_example/dqn_experiment.py

from collections import defaultdict
import gym.spaces
from gym.spaces import Box
import carla
from vision_rl.rllib_integration.base_experiment import BaseExperiment
from vision_rl.rllib_integration.helper import post_process_image,carla_location_to_np_array
import gym
import math
import warnings

warnings.filterwarnings('ignore', module='scipy')
# %%
from scipy.interpolate import splprep, splev
import numpy as np


def make_curve(points):
    t = np.linspace(0, 1, len(points))
    tckp, u = splprep(points.T, u=t,k=3 )
    return lambda t:np.array(splev(t,tckp)),tckp

def derivative_curve(tckp):
    return lambda t:np.array(splev(t,tckp,der=1))

def get_curve(points):
        indx = np.argsort(points[:,0])
        # print(points,points.shape)
        points = points[indx]
        f_t,tckp=make_curve(points)
        f_prime_t=derivative_curve(tckp)
        t_samples = np.linspace(0, 1, 1000)
        curve_points = np.array([f_t(t) for t in t_samples])
        def curve(x):
            # func=lambda v:np.dot((x-f_t(v[-1])).T,f_prime_t(v[-1]))
            # t_initial=0.5
            # result=root(func,[t_initial],method="anderson")
            # if result.success:
            #     t=result.x.squeeze()
            #     return f_t(t),f_prime_t(t)
            distances = np.linalg.norm(curve_points - x, axis=1)
            closest_idx = np.argmin(distances)
            t = t_samples[closest_idx]
            return f_t(t),f_prime_t(t)
        return curve




class STBL3GoalExperiment(BaseExperiment):
    def __init__(self,config={}):
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
        # self.world=config.get("world",Nones
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
        self.large_deviation=False

        # hero variables
        self.last_location = None
        self.last_velocity = 0
        self.distance_travelled = 0.0
        self.last_distance_to_goal=0.0

        # Sensor stack
        self.prev_vec_0 = None
        self.prev_vec_1 = None
        self.prev_vec_2 = None
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None

        # control variables
        self.max_steer = 1.0
        self.max_throttle = 1.0
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
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
            shape=(4 * self.frame_stack,),
            dtype=np.float32,
        )

        return gym.spaces.Dict({"image":image_space,"goal":goal_image_space, "vector":vec_space})

    def get_action_space(self):
        """Returns the continuous action space for steering and throttle"""
        return Box(
            low=np.array([-self.max_steer, -self.max_throttle]),  # [steering, throttle/brake]
            high=np.array([self.max_steer, self.max_throttle]),
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
        distance_to_goal=sensor_data['goal'][1][1]

        return {"image":images,"goal":goal, "vector":vecs}, None

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

        image = post_process_image(sensor_data['rgb'][1], normalized = True, grayscale = True)
        # breakpoint()
        goal = post_process_image(sensor_data['goal'][1][0], normalized = True, grayscale = True)

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
        self.info=dict()
        self.done_time_idle = self.max_time_idle < self.time_idle
        if self.get_speed(hero) > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1
        self.time_episode += 1
        distance_to_goal=sensor_data['goal'][1][1]
        self.done_dist = self.distance_travelled > self.max_dist
        self.done_falling = hero.get_location().z < -0.5
        self.diff_lane = 'lane_invasion' in sensor_data.keys()
        self.collision = 'collision' in sensor_data.keys()
        # done=self.done_time_idle or self.done_falling or self.done_dist or self.diff_lane or self.collision or distance_to_goal<=1.5 or self.large_deviation
        done=distance_to_goal<=1.5
        print(distance_to_goal)
        if done:
            self.info=dict(is_success=distance_to_goal<=1.5,distance_to_goal=self.last_distance_to_goal)
        return  done

    def compute_reward(self, sensor_data, core):
        hero = core.hero
        distance_to_goal=sensor_data['goal'][1][1]
        goal_loc=sensor_data['goal'][1][-1]
        # Hero-related variables
        hero_location = hero.get_location()
        hero_velocity = self.get_speed(hero)

        # Initialize last location
        if self.last_location == None:
            self.last_location = hero_location

        # Compute deltas
        # delta_distance = float(np.sqrt(np.square(hero_location.x - self.last_location.x) + np.square(hero_location.y - self.last_location.y)))
        delta_loc=carla_location_to_np_array(self.last_location)-carla_location_to_np_array(hero_location)
        displacement=np.dot(delta_loc,goal_loc/np.linalg.norm(goal_loc))
        # self.distance_travelled += delta_distance

        # Update variables
        # self.last_location = hero_location
        # self.last_velocity = hero_velocity
        reward=0.0
        # if self.last_distance_to_goal==0.0:
        #     reward=self.last_distance_to_goal-distance_to_goal

        # # Initialize last location
        # if self.last_location == None:
        #     self.last_location = hero_location

        # # Compute deltas
        delta_distance = float(np.sqrt(np.square(hero_location.x - self.last_location.x) + \
                            np.square(hero_location.y - self.last_location.y)))
        self.distance_travelled += delta_distance
        transform = hero.get_transform()
        # distance_to_goal = np.linalg.norm(
        #     carla_location_to_np_array(transform.location) - 
        #     carla_location_to_np_array(self.destination.location)
        # )
        
        # Calculate lane deviation
        location = np.array([transform.location.x, transform.location.y])
        f, d_f = core.spline(location)
        d_to_lane = np.linalg.norm(f - location)
        max_dev = hero.bounding_box.extent.y * 2
        self.large_deviation= self.distance_travelled>1.0 and d_to_lane>max_dev
        
        # Calculate alignment
        # forward = np.array([
        #     transform.get_forward_vector().x,
        #     transform.get_forward_vector().y
        # ])
        # # d_f_norm = np.linalg.norm(d_f)
        # cos_alpha_t = np.dot(forward, d_f / d_f_norm) if d_f_norm > 0 else 0

        # Check completion
        # done = distance_to_goal <= 2.0
        
        # Calculate reward
        # reward = delta_distance
        # Update variables
        self.last_location = hero_location
        self.last_velocity = hero_velocity

        # Reward if going forward
        # if hero_velocity < self.target_speed:
            # reward = abs(displacement)
        # else:
            # reward = 0.0
        # print(reward)
        reward=-1.0
        if self.done_falling:
            reward += -1.0
            # print("Falling")
        if self.done_dist:
            # print("Max dist travelled")
            # reward += 1.0
            pass
        if self.done_time_idle:
            # print("Done idle")
            reward += -1.0
        if self.collision:
            # print('collision')
            reward += -1.0
        if self.diff_lane:
            # print("Lane Invasion")
            reward += -1.0
        if distance_to_goal<=1.5:
            # reward+=1.0
            pass
        if self.large_deviation:
            reward += -1.0
            # print("Short path deviation")
        self.last_distance_to_goal=distance_to_goal
        # print(distance_to_goal)
        return reward*10
    

# # %%
# def draw_waypoints(world, waypoints, z=0.5, lifetime=30.0, color=(255,0,0)):
#     """
#     Draw a list of waypoints at a certain height given in z.

#         :param world: carla.world object
#         :param waypoints: list or iterable container with the waypoints to draw
#         :param z: height in meters
#     """
#     for wpt in waypoints:
#         wpt_t = wpt.transform
#         begin = wpt_t.location + carla.Location(z=z)
#         angle = math.radians(wpt_t.rotation.yaw)
#         end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
#         world.debug.draw_hud_arrow(begin, end, color=carla.Color(*color), arrow_size=0.3, life_time=lifetime)
#         # world.debug.draw_string(begin,f"{wpt.road_id} {wpt.lane_id} {wpt.section_id}", color=carla.Color(*color), life_time=lifetime)

# # %%
# def draw_spline(world, waypoints, z=0.5, lifetime=30.0, color=(255,0,0)):
#     for wpt in waypoints:
#         world.debug.draw_hud_point(wpt, color=carla.Color(*color), size=0.3, life_time=lifetime)
