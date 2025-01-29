import carla
import cv2
import gymnasium as gym
import math
import numpy as np
from collections import defaultdict
from collections import deque
from gymnasium.spaces import Box
from scipy.interpolate import splprep, splev
from rlib_integration.base_experiment import BaseExperiment
from rlib_integration.helper import post_process_image, carla_location_to_np_array
import skimage
from scipy.spatial.distance import euclidean

# from matplotlib import pyplot as plt
# MIN_MATCH_COUNT = 10

def relative_compute_heading(location1, location2):
    dx = location2[0] - location1[0]
    dy = location2[1] - location1[1]
    heading_rad = math.atan2(dy, dx)
    heading_deg = math.degrees(heading_rad)
    if heading_deg < 0:
        heading_deg += 360
    return heading_deg

def absolute_heading(location):
    dx = location[0]
    dy = location[1]
    heading_rad = math.atan2(dy, dx)
    heading_deg = math.degrees(heading_rad)
    if heading_deg < 0:
        heading_deg += 360
    return heading_deg+np.random.normal(0,1)

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

class JAXMappingExperiments(BaseExperiment):
    def __init__(self,config={},is_rgb=False,image_size=32):
        super().__init__(config)
        self.frame_stack = self.config["others"]["framestack"]
        self.max_time_idle = self.config["others"]["max_time_idle"]
        self.max_dist = self.config["others"]["max_dist"]
        self.target_speed = self.config["others"]["target_speed"]
        self.is_rgb=is_rgb
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
        self.max_curriculum_steps = 10
        self.goal_threshold=2.5
        self.running_success_rate=deque(maxlen=100)
        self.distance_travelled_toward_goal=0
        self.done_goal=False
        self.image_size=image_size
        self.prev_reward=0.0
        self.origin=None
        self.goal_image=None
        self.distances=[]


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
        self.done_goal=False

        self.last_location = None
        self.last_velocity = 0
        self.distance_travelled = 0.0
        self.last_distance_to_goal = None
        self.distance_travelled_toward_goal=0

        self.prev_vec_0 = None
        self.prev_vec_1 = None
        self.prev_vec_2 = None
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None

        # self.max_steer = 1.0
        # self.max_throttle = 1.0
        self.prev_steer = 0.0
        self.prev_reward=None
        self.prev_throttle = 0.0
        self.heading = None
        self.total_distance = None
        self.goal_features=None     
        self.match_features=0.0  
        self.goal_image=None
        self.user_defined_location=None
        if self.trajectories is None:
            self._cache_waypoints(core.core.world)

    def get_observation_space(self):
        channel=1
        if self.is_rgb:
            channel=3
        image_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.image_size, self.image_size,channel,),
            dtype=np.float32,
        )
        goal_image_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.image_size, self.image_size,channel),
            dtype=np.float32,
        )
        vec_space = Box(
            low=-5.1,
            high=5.1,
            shape=(6 * self.frame_stack,),
            dtype=np.float32,
        )
        return gym.spaces.Dict({"pixels":image_space,"goal":goal_image_space, "vector":vec_space})

    def get_action_space(self):
        return Box(
            low=np.array([-self.max_steer, -self.max_throttle]),
            high=np.array([self.max_steer, self.max_throttle]),
            dtype=np.float32
        )


    def compute_action(self, action):
        steer, throttle_brake = action
        # print(steer,throttle_brake)
        action = carla.VehicleControl()
        action.steer = float(np.clip(steer, -self.max_steer, self.max_steer))
        # throttle_brake=self.prev_throttle+throttle_brake
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
        vecs = self.get_vec_obs(sensor_data, core)
        images, goal = self.get_img_obs(sensor_data, core)
        return {"pixels":images, "goal":goal, "vector":vecs}, self.info
    
    def set_goal(self,goal_image,heading,location=None):
        self.goal_image=goal_image
        self.heading=heading
        self.user_defined_location=location

    def get_vec_obs(self, sensor_data, core):
        imu = sensor_data['imu'][1]
        self.heading = sensor_data['goal'][1][-1]
        # breakpoint()
        if self.heading is None:
            self.heading = sensor_data['goal'][1][-1]
        # self.heading =  np.deg2rad(absolute_heading(sensor_data['goal'][1][-1]))

        vec = np.zeros(6)
        vec[0] = self.prev_steer / self.max_steer
        vec[1] = self.prev_throttle / self.max_throttle
        hero = core.hero
        vec[2] = np.clip(self.get_speed(hero)/self.target_speed, 0.0, 1.0)
        vec[3] = self.time_idle / self.max_time_idle
        vec[4] = imu[-1]/np.pi
        vec[5] = self.heading/np.pi
        # print("compass",np.rad2deg(imu[-1]),np.rad2deg(self.heading))
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
        image = post_process_image(sensor_data['rgb'][1], crop=False, normalized=True, grayscale=not self.is_rgb,image_size=self.image_size)
        goal =  post_process_image(sensor_data['goal'][1][0], crop=False, normalized=True, grayscale=not self.is_rgb,image_size=self.image_size) if self.goal_image  is None else  self.goal_image 
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
        bbox = hero.bounding_box
        self.goal_threshold = max(bbox.extent.x, bbox.extent.y, bbox.extent.z)*1.5
        self.info = dict()
        self.done_time_idle = self.max_time_idle < self.time_idle
        if self.get_speed(hero) > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1
        self.time_episode += 1
        
        wp=core.map.get_waypoint(hero.get_transform().location,project_to_road=False)
        goal_location = sensor_data['goal'][1][-2]
        distance_to_goal = np.linalg.norm(goal_location[:2]-carla_location_to_np_array(hero.get_transform().location)[:2])
        self.done_falling = hero.get_location().z < -0.5
        self.diff_lane = 'lane_invasion' in sensor_data.keys() or wp is None
        self.collision = 'collision' in sensor_data.keys()
        # breakpoint()
        if not self.goal_image is None:
            self.done_goal = self.check_goal_reached(sensor_data['rgb'][1],sensor_data['goal'][1][0] if self.goal_image  is None else self.goal_image,carla_location_to_np_array(self.user_defined_location),carla_location_to_np_array(hero.get_transform().location))
        else:
            self.done_goal=False
            # self.check_goal_reached(core,hero,goal_location,self.goal_image)
        # image = post_process_image(sensor_data['rgb'][1], crop=False, normalized=True, grayscale=True,image_size=self.image_size)
        # goal = post_process_image(sensor_data['goal'][1][0], crop=False, normalized=True, grayscale=True,image_size=self.image_size)

        # self.done_goal = self.check_goal_reached(sensor_data['rgb'][1],sensor_data['goal'][1][0] if self.goal_image  is None else self.goal_image ) 

        done = (self.done_time_idle or self.done_goal  or self.done_falling or self.diff_lane or 
                self.collision)
        # done = distance_to_goal <= 1.5
        # if done:
        self.info = dict(
                distance_completed=self.distance_travelled,
                slack=distance_to_goal,
                mean_distance_per_step=np.mean(self.distances)
            )
        if done:
            wp=core.map.get_waypoint(hero.get_transform().location,project_to_road=True)
            self.origin=wp
            self.info.update(dict(
                is_success=self.done_goal,
            ))
            self.running_success_rate.append(float(self.done_goal))
            if np.mean(self.running_success_rate)>0.5:
                self.running_success_rate=deque(maxlen=100)
                self.curriculum_step+=1
        return done
    # def check_goal_reached(self,image,goal_image):
    #     image=post_process_image(image, crop=False, normalized=False, grayscale=False,image_size=self.image_size)
    #     # h, w,c = image.shape
    #     # breakpoint()
    #     image=skimage.feature.hog(image,channel_axis=-1)
    #     goal_image=skimage.feature.hog(goal_image,channel_axis=-1)
    #     # diff = cv2.subtract(image, goal_image)
    #     # err = np.sum(diff**2)
    #     # mse = err/(float(h*w*c))
    #     # print("mse",mse)
    #     # breakpoint()

    #     dist=euclidean(image,goal_image)
    #     # print("mse",dist)e
    #     return dist<6.1
    def check_goal_reached(self,image,goal_image,goal_location=None,hero_location=None):
            # breakpoint()
            # img1 = cv2.imread('Q/IMG_1192.JPG', 0)          # queryImage
            # img2 = cv2.imread('DB/IMG_1208-1000.jpg', 0) # trainImage
            if abs(self.distance_travelled) < self.goal_threshold*2:
                        return False
            dist=np.linalg.norm(goal_location[:2]-hero_location[:2])     
            # Initiate SIFT detector
            image=post_process_image(image, crop=False, normalized=False, grayscale=False,image_size=self.image_size)
            sift = cv2.xfeatures2d.SIFT_create()
            # find the keypoints and descriptors with SIFT
            
            kp1, des1 = sift.detectAndCompute(image,None)
            kp2, des2 = sift.detectAndCompute(goal_image,None)
            if des1 is None or len(kp2) < 2:
                return False
            
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   # or pass empty dictionary
             
            flann = cv2.FlannBasedMatcher(index_params,search_params)
             
            matches = flann.knnMatch(des1,des2,k=2)
            # print("matches",len(matches))
            # Need to draw only good matches, so create a mask
            matchesMask = [[0,0] for i in range(len(matches))]
            num_good_matches=0

            # ratio test as per Lowe's paper
            # Perform matching
            # matches = self.matcher.knnMatch(descriptors1, self.goal_descriptors, k=2)
            
            # Count good matches using numpy for speed
            # Convert matches to distance ratios
            # print("dist",dist)
            if len(matches) < 2:
                return False
                
            distances = np.array([[m.distance if m is not None else np.inf for m in match] 
                                for match in matches])
            good_matches = np.sum(distances[:, 0] < 0.001 * distances[:, 1])
            
            # draw_params = dict(matchColor = (0,255,0),
            #                    singlePointColor = (255,0,0),
            #                    matchesMask = matchesMask,
            #                    flags = cv2.DrawMatchesFlags_DEFAULT)
            match_percentage = good_matches / len(matches)
            # print(match_percentage)
            # if  match_percentage > 0.09:
            # img3 = cv2.drawMatchesKnn(image,kp1,goal_image,kp2,matches,None,**draw_params)             
            # cv2.imwrite("test.jpg",img3)
            # print("num_good_matches",match_percentage)
            self.match_features=match_percentage/0.09
            # print(match_percentage,dist)
            return match_percentage > 0.9 or dist<5.0


    # def check_goal_reached(self, core, hero, goal_location, distance_to_goal, distance_threshold=2.0, angle_threshold=45.0):
    #     """
    #     Check if goal is reached considering spatial, temporal, and waypoint-based conditions
        
    #     Args:
    #         core: CARLA core instance
    #         hero: Hero vehicle actor
    #         goal_location: Target location (numpy array [x,y,z])
    #         distance_to_goal: Pre-calculated direct distance to goal
    #         distance_threshold: Distance threshold for goal reaching
    #         angle_threshold: Angle threshold (not used currently)
    #     """
    #     # Basic validation
    #     if abs(self.distance_travelled) < self.goal_threshold*2:
    #         return False
            
        # hero_transform = hero.get_transform()
        # hero_location = hero_transform.location
        
    #     # # Calculate temporal distance (time to reach goal at current velocity)
    #     # hero_velocity = np.dot(
    #     #     carla_location_to_np_array(hero.get_velocity()),
    #     #     goal_location / (np.linalg.norm(goal_location) + 1e-8)  # Avoid division by zero
    #     # )
    #     # temporal_distance =  (distance_to_goal)/(hero_velocity+1e-8)
        
    #     # Convert goal location to CARLA format
    #     goal_location_carla = carla.Location(
    #         x=goal_location[0],
    #         y=goal_location[1],
    #         z=goal_location[2]
    #     )
        
    #     # # Get waypoints
    #     # try:
        # hero_waypoint = core.map.get_waypoint(hero_location)
        # goal_waypoint = core.map.get_waypoint(goal_location_carla)

        
    #     # If waypoints are valid, do detailed checks
    #     if hero_waypoint and goal_waypoint:
    #         # Check if on same road segment
    #         same_section = hero_waypoint.section_id == goal_waypoint.section_id
    #         same_road = hero_waypoint.road_id == goal_waypoint.road_id
    #         same_lane = hero_waypoint.lane_id == goal_waypoint.lane_id
            
    #         # Calculate distance along path
    #         distance_along_path = hero_waypoint.s - goal_waypoint.s
            
    #         # Check various thresholds
    #         within_path_threshold = abs(distance_along_path) < self.goal_threshold
    #         # within_direct_threshold = distance_to_goal < self.goal_threshold
    #         # within_temporal_threshold = temporal_distance < self.goal_threshold
    #         same_waypoint = hero_waypoint.id == goal_waypoint.id
            
    #         # Combined conditions
    #         road_condition = same_section and same_road and same_lane and within_path_threshold
    #         return (road_condition or same_waypoint)
    #     else:
    #         # Fallback to simpler checks if waypoints are invalid
    #         return (distance_to_goal < self.goal_threshold)
    # def check_goal_reached(self,hero, goal_location, distance_threshold=2.0, angle_threshold=45.0):
    #         """
    #         Check if vehicle has reached goal based on:
    #         1. Distance to goal is within threshold
    #         2. Vehicle is facing roughly the right direction
    #         3. Vehicle speed is low enough (optional)
    #         """
    #         # Get current location and compute distance
    #         hero_transform = hero.get_transform()
    #         hero_location = hero_transform.location
    #         distance_to_goal = np.linalg.norm(
    #             goal_location[:2] - carla_location_to_np_array(hero_location)[:2]
    #         )
            
    #         # Get vehicle's forward vector
    #         forward_vector = hero_transform.get_forward_vector()
    #         forward = np.array([forward_vector.x, forward_vector.y])
    #         forward = forward / np.linalg.norm(forward)
            
    #         # Get direction to goal
    #         hero_pos = carla_location_to_np_array(hero_location)[:2]
    #         goal_pos = goal_location[:2]
    #         to_goal = goal_pos - hero_pos
    #         if np.linalg.norm(to_goal) > 0:
    #             to_goal = to_goal / np.linalg.norm(to_goal)
            
    #         # Compute angle between forward vector and goal direction
    #         angle = np.arccos(np.clip(np.dot(forward, to_goal), -1.0, 1.0))
    #         angle_deg = np.degrees(angle)
            
    #         # Optional: Check vehicle speed
    #         velocity = hero.get_velocity()
    #         speed = np.linalg.norm([velocity.x, velocity.y])
            
    #         # Check all conditions
    #         distance_ok = distance_to_goal < self.goal_threshold
    #         angle_ok = angle_deg < angle_threshold
    #         speed_ok = speed < 0.1  # Optional speed check
    #         return distance_ok and angle_ok , speed_ok


    def compute_reward(self, sensor_data, core):
        hero = core.hero
        goal_loc = sensor_data['goal'][1][-2]
        hero_location = hero.get_location()
        hero_velocity = self.get_speed(hero)
        distance_to_goal = np.linalg.norm(goal_loc[:2]-carla_location_to_np_array(hero_location)[:2])
        if self.total_distance is None:
            self.total_distance = np.linalg.norm(goal_loc[:2]-carla_location_to_np_array(hero_location)[:2])
            assert np.allclose(distance_to_goal,self.total_distance)
        # hero_velocity=np.dot(carla_location_to_np_array(hero.get_velocity()),goal_loc/np.linalg.norm(goal_loc))
        
      
        # print(self.total_distance,distance_to_goal)
        # displ=np.dot(carla_location_to_np_array(hero.get_velocity()),goal_location/np.linalg.norm(goal_location))
        if self.last_location is None:
            self.last_location = hero_location
            self.last_distance_to_goal = distance_to_goal
        
        displacement=np.dot(carla_location_to_np_array(hero_location)-carla_location_to_np_array(self.last_location),goal_loc/np.linalg.norm(goal_loc))
        self.distance_travelled_toward_goal-=displacement
        delta_distance = float(np.sqrt(np.square(hero_location.x - self.last_location.x) + \
                            np.square(hero_location.y - self.last_location.y)))
        self.distance_travelled += delta_distance
        imu = sensor_data['imu'][1]
        self.heading = sensor_data['goal'][1][-1]
        # print("delta_distance",delta_distance)
        self.distances.append(delta_distance)
        # print("compass",np.rad2deg(imu[-1]),np.rad2deg(self.heading))
        # vehicle_transform = hero.get_transform()
        # vehicle_yaw = vehicle_transform.rotation.yaw

        # Get waypoint's yaw
        # waypoint = core.map.get_waypoint(hero_location)

        # Dense progress reward
        # reward = min(distance_to_goal/self.total_distance,1.0)
        # heading_diff = min(abs(self.heading - imu[-1]), 2*np.pi - abs(self.heading - imu[-1]))
        # heading_reward = np.cos(heading_diff)  # Peaks at 1 when aligned, -1 when opposite
        
        # reward=heading_reward + self.match_features
        # # Speed matching reward
        # speed_reward = -min(abs(self.target_speed-hero_velocity)/self.target_speed,1.0)
        # reward += speed_reward
        
        # Heading alignment reward
        # heading = sensor_data['imu'][1][-1]
        # heading_diff = abs(self.heading - heading)
        # heading_reward = -heading_diff/(2*np.pi)
        # reward += 0.1 * heading_reward

        # # Update distance traveled
        # delta_distance = float(np.sqrt(np.square(hero_location.x - self.last_location.x) + \
        #                     np.square(hero_location.y - self.last_location.y)))
        # self.distance_travelled += delta_distance
        # reward=displacement
        # yaw_diff_rad=0.0
        # if not waypoint is None:
        #         waypoint_yaw = waypoint.transform.rotation.yaw
        #         yaw_diff = (vehicle_yaw - waypoint_yaw) % 360.0
        #         if yaw_diff > 180:
        #             yaw_diff -= 360.0
        #         yaw_diff_rad = np.deg2rad(yaw_diff)
        # reward = 1-(distance_to_goal/self.total_distance) # Base step penalty
        # if self.prev_reward is None:
        #     self.prev_reward=reward
        # step_reward=self.prev_reward-reward
        # self.prev_reward=reward
        # Reward for velocity
        reward=0.0
        if hero_velocity < self.target_speed:
            # if self.heading
            # reward += np.cos(imu[-1]-self.heading)*delta_distance
            reward += delta_distance 
            # re   
        else:
            reward -= 0.0  # Optional penalty for exceeding target speed
        # print(f"Goal {self.done_goal} Lane {self.diff_lane}")
        # Goal reward
        # print(reward,distance_to_goal,self.total_distance)
        if self.done_goal:
            print(f"Goal reached :travelled {self.distance_travelled}")
            reward += 1.0 
        elif self.done_falling or self.collision or self.done_time_idle or self.diff_lane:
            print(f"Truncated :travelled {self.distance_travelled} idle:{self.done_time_idle} falling :{self.done_falling} diff {self.diff_lane} or collision {self.collision}")
            reward += -1.0
        # Scale the reward
        self.last_location = hero_location
        self.last_distance_to_goal = distance_to_goal
        return reward*10