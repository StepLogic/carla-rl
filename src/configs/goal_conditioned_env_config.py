
# # Modified from https://github.com/carla-simulator/rllib-integration/blob/main/dqn_example/dqn_experiment.py

# # from typing import Dict
# import carla
# import cv2
# import gymnasium as gym
# import math
# import numpy as np
# from collections import defaultdict
# from collections import deque
# from gymnasium.spaces import Box,Dict
# from scipy.interpolate import splprep, splev
# from rlib_integration.base_experiment import BaseExperiment
# from rlib_integration.helper import post_process_image, carla_location_to_np_array
# import skimage
# from scipy.spatial.distance import euclidean

# # from matplotlib import pyplot as plt
# # MIN_MATCH_COUNT = 10

# def relative_compute_heading(location1, location2):
#     dx = location2[0] - location1[0]
#     dy = location2[1] - location1[1]
#     heading_rad = math.atan2(dy, dx)
#     heading_deg = math.degrees(heading_rad)
#     if heading_deg < 0:
#         heading_deg += 360
#     return heading_deg

# def absolute_heading(location):
#     dx = location[0]
#     dy = location[1]
#     heading_rad = math.atan2(dy, dx)
#     heading_deg = math.degrees(heading_rad)
#     if heading_deg < 0:
#         heading_deg += 360
#     return heading_deg+np.random.normal(0,1)

# def make_curve(points):
#     t = np.linspace(0, 1, len(points))
#     tckp, u = splprep(points.T, u=t,k=3)
#     return lambda t:np.array(splev(t,tckp)),tckp

# def derivative_curve(tckp):
#     return lambda t:np.array(splev(t,tckp,der=1))

# def get_curve(points):
#     indx = np.argsort(points[:,0])
#     points = points[indx]
#     f_t,tckp=make_curve(points)
#     f_prime_t=derivative_curve(tckp)
#     t_samples = np.linspace(0, 1, 1000)
#     curve_points = np.array([f_t(t) for t in t_samples])
#     def curve(x):
#         distances = np.linalg.norm(curve_points - x, axis=1)
#         closest_idx = np.argmin(distances)
#         t = t_samples[closest_idx]
#         return f_t(t),f_prime_t(t)
#     return curve

# class JAXMappingExperiments(BaseExperiment):
#     def __init__(self,config={},is_rgb=False,image_size=32):
#         super().__init__(config)
#         self.frame_stack = self.config["others"]["framestack"]
#         self.max_time_idle = self.config["others"]["max_time_idle"]
#         self.max_dist = self.config["others"]["max_dist"]
#         self.target_speed = self.config["others"]["target_speed"]
#         self.is_rgb=is_rgb
#         self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]
#         self.last_action = None
#         self.max_steer = 0.5
#         self.max_throttle = 0.6
#         self.prev_steer = 0.0
#         self.prev_throttle = 0.0
#         self.trajectories = None
#         self.large_deviation = False
#         self.info = dict()
#         self.heading = None
#         self.total_distance = None
#         self.curriculum_step = 1
#         self.max_curriculum_steps = 10
#         self.goal_threshold=2.5
#         self.running_success_rate=deque(maxlen=100)
#         self.distance_travelled_toward_goal=0
#         self.done_goal=False
#         self.image_size=image_size
#         self.prev_reward=0.0
#         self.origin=None
#         self.goal_image=None
#         self.distances=[]


#     # def _cache_waypoints(self,world) -> None:
#     #         env_map = world.get_map()
#     #         waypoints = env_map.generate_waypoints(distance=2)
#     #         trajectories = defaultdict(list)
#     #         for wpt in waypoints:
#     #             trajectories[f"{wpt.road_id}-{wpt.lane_id}"].append(wpt)
#     #         self.trajectories = sorted([traj for traj in trajectories.values() if len(traj) > 3], 
#     #                                     key=len, reverse=True)
#     #         self.max_curriculum_steps = len(self.trajectories)
#     def reset(self,core):
#         self.time_idle = 0
#         self.time_episode = 0
#         self.done_time_idle = False
#         self.done_falling = False
#         self.done_dist = False
#         self.large_deviation = False
#         self.done_goal=False

#         self.last_location = None
#         self.last_velocity = 0
#         self.distance_travelled = 0.0
#         self.last_distance_to_goal = None
#         self.distance_travelled_toward_goal=0

#         self.prev_vec_0 = None
#         self.prev_vec_1 = None
#         self.prev_vec_2 = None
#         self.prev_image_0 = None
#         self.prev_image_1 = None
#         self.prev_image_2 = None

#         # self.max_steer = 1.0
#         # self.max_throttle = 1.0
#         self.prev_steer = 0.0
#         self.prev_reward=None
#         self.prev_throttle = 0.0
#         self.heading = None
#         self.total_distance = None
#         self.goal_features=None     
#         self.match_features=0.0  
#         self.goal_image=None
#         self.user_defined_location=None
#         # if self.trajectories is None:
#         #     self._cache_waypoints(core.core.world)

#     def get_observation_space(self):
#         image_space = Box(
#             low=-1.0,
#             high=1.0,
#             shape=(84, 84, self.frame_stack,),
#             dtype=np.float32,
#         )
        
#         vec_space = Box(
#             low=-5.1,
#             high=5.1,
#             shape=(4 * self.frame_stack,),
#             dtype=np.float32,
#         )

#         return Dict({"pixels":image_space, "vector":vec_space})

#     def get_action_space(self):
#         """Returns the continuous action space for steering and throttle"""
#         return Box(
#             low=np.array([-self.max_steer, -1.0]),  # [steering, throttle/brake]
#             high=np.array([self.max_steer, 1.0]),
#             dtype=np.float32
#         )

#     def compute_action(self, action):
#         """Convert continuous actions to CARLA vehicle controls"""
#         steer, throttle_brake = action

#         action = carla.VehicleControl()
#         # Smooth steering using previous value
#         action.steer = float(np.clip(steer, -self.max_steer, self.max_steer))
#         self.steer=action.steer
#         self.throttle=action.throttle
#         # Handle throttle and brake separately
        
#         if throttle_brake >= 0:
#             action.throttle = float(np.clip(throttle_brake, 0.0, self.max_throttle))
#             action.brake = 0.0
#         else:
#             action.throttle = 0.0
#             action.brake = float(np.clip(-throttle_brake, 0.0, 1.0))
#         self.last_action = action
#         action.reverse = False
#         action.hand_brake = False
#         self.steer=action.steer
#         self.throttle=action.throttle
#         return action

#     # Remove the get_actions method since we're using continuous actions
#     def get_observation(self, sensor_data, core):
#         """Function to do all the post processing of observations (sensor data).

#         :param sensor_data: dictionary {sensor_name: sensor_data}

#         Should return a tuple or list with two items, the processed observations,
#         as well as a variable with additional information about such observation.
#         The information variable can be empty
#         """
#         vecs = self.get_vec_obs(sensor_data, core)
#         images = self.get_img_obs(sensor_data, core)
#         return {"pixels":images, "vector":vecs}, self.info
#     def get_speed(self, hero):
#         """Computes the speed of the hero vehicle in Km/h"""
#         vel = hero.get_velocity()
#         return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

#     def get_vec_obs(self, sensor_data, core):
#         # breakpoint()
#         heading=sensor_data["goal_heading"][-1][-1]
#         imu=sensor_data["imu"][-1][-1]
#         vec = np.zeros(4)
#         vec[0] = self.prev_steer / self.max_steer
#         vec[1] = self.prev_throttle / self.max_throttle
#         hero = core.hero
#         vec[2] = np.clip(self.get_speed(hero)/(self.target_speed+1e-8), 0.0, 1.0)
#         # vec[3] = self.time_idle / self.max_time_idle
#         vec[3]= np.clip(imu/(heading+1e-8),-1.0,1.0) 
#         if self.prev_vec_0 is None:
#             self.prev_vec_0 = vec
#             self.prev_vec_1 = self.prev_vec_0
#             self.prev_vec_2 = self.prev_vec_1

#         vecs = vec  #add gaussian noise

#         if self.frame_stack >= 2:
#             vecs = np.concatenate([self.prev_vec_0, vecs], axis=0)
#         if self.frame_stack >= 3 and vecs is not None:
#             vecs = np.concatenate([self.prev_vec_1, vecs], axis=0)
#         if self.frame_stack >= 4 and vecs is not None:
#             vecs = np.concatenate([self.prev_vec_2, vecs], axis=0)

#         self.prev_vec_2 = self.prev_vec_1
#         self.prev_vec_1 = self.prev_vec_0
#         self.prev_vec_0 = vec
#         self.velocity=self.get_speed(hero)
#         self.current_heading=imu
#         return vecs

#     def get_img_obs(self, sensor_data, core):
#         image = post_process_image(sensor_data['rgb'][1], normalized = True,crop=False, grayscale = True)

#         if self.prev_image_0 is None:
#             self.prev_image_0 = image
#             self.prev_image_1 = self.prev_image_0
#             self.prev_image_2 = self.prev_image_1

#         images = image

#         if self.frame_stack >= 2:
#             images = np.concatenate([self.prev_image_0, images], axis=2)
#         if self.frame_stack >= 3 and images is not None:
#             images = np.concatenate([self.prev_image_1, images], axis=2)
#         if self.frame_stack >= 4 and images is not None:
#             images = np.concatenate([self.prev_image_2, images], axis=2)

#         self.prev_image_2 = self.prev_image_1
#         self.prev_image_1 = self.prev_image_0
#         self.prev_image_0 = image

#         return images
    
#     def get_done_status(self, sensor_data, core):
#         hero = core.hero
#         bbox = hero.bounding_box
#         self.goal_threshold = max(bbox.extent.x, bbox.extent.y, bbox.extent.z)*1.5
#         self.info = dict()
#         self.done_time_idle = self.max_time_idle < self.time_idle
#         if self.get_speed(hero) > 1.0:
#             self.time_idle = 0
#         else:
#             self.time_idle += 1
#         self.time_episode += 1
        
#         wp=core.map.get_waypoint(hero.get_transform().location,project_to_road=False)
#         goal_location = sensor_data['goal'][1][-2]
#         distance_to_goal = np.linalg.norm(goal_location[:2]-carla_location_to_np_array(hero.get_transform().location)[:2])
#         self.done_falling = hero.get_location().z < -0.5
#         self.diff_lane = 'lane_invasion' in sensor_data.keys() or wp is None
#         self.collision = 'collision' in sensor_data.keys()
#         # breakpoint()
#         if not self.goal_image is None:
#             self.done_goal = self.check_goal_reached(sensor_data['rgb'][1],sensor_data['goal'][1][0] if self.goal_image  is None else self.goal_image,carla_location_to_np_array(self.user_defined_location),carla_location_to_np_array(hero.get_transform().location))
#         else:
#             self.done_goal=False
#             # self.check_goal_reached(core,hero,goal_location,self.goal_image)
#         # image = post_process_image(sensor_data['rgb'][1], crop=False, normalized=True, grayscale=True,image_size=self.image_size)
#         # goal = post_process_image(sensor_data['goal'][1][0], crop=False, normalized=True, grayscale=True,image_size=self.image_size)

#         # self.done_goal = self.check_goal_reached(sensor_data['rgb'][1],sensor_data['goal'][1][0] if self.goal_image  is None else self.goal_image ) 

#         done = (self.done_time_idle or self.done_goal  or self.done_falling or self.diff_lane or 
#                 self.collision)
#         # done = distance_to_goal <= 1.5
#         # if done:
#         self.info = dict(
#                 distance_completed=self.distance_travelled,
#                 slack=distance_to_goal,
#                 mean_distance_per_step=np.mean(self.distances)
#             )
#         if done:
#             wp=core.map.get_waypoint(hero.get_transform().location,project_to_road=True)
#             self.origin=wp
#             self.info.update(dict(
#                 is_success=self.done_goal,
#             ))
#             self.running_success_rate.append(float(self.done_goal))
#             if np.mean(self.running_success_rate)>0.5:
#                 self.running_success_rate=deque(maxlen=100)
#                 self.curriculum_step+=1
#         return done
#     # def check_goal_reached(self,image,goal_image):
#     #     image=post_process_image(image, crop=False, normalized=False, grayscale=False,image_size=self.image_size)
#     #     # h, w,c = image.shape
#     #     # breakpoint()
#     #     image=skimage.feature.hog(image,channel_axis=-1)
#     #     goal_image=skimage.feature.hog(goal_image,channel_axis=-1)
#     #     # diff = cv2.subtract(image, goal_image)
#     #     # err = np.sum(diff**2)
#     #     # mse = err/(float(h*w*c))
#     #     # print("mse",mse)
#     #     # breakpoint()

#     #     dist=euclidean(image,goal_image)
#     #     # print("mse",dist)e
#     #     return dist<6.1
#     def check_goal_reached(self,image,goal_image,goal_location=None,hero_location=None):
#             # breakpoint()
#             # img1 = cv2.imread('Q/IMG_1192.JPG', 0)          # queryImage
#             # img2 = cv2.imread('DB/IMG_1208-1000.jpg', 0) # trainImage
#             if abs(self.distance_travelled) < self.goal_threshold*2:
#                         return False
#             dist=np.linalg.norm(goal_location[:2]-hero_location[:2])     
#             # Initiate SIFT detector
#             image=post_process_image(image, crop=False, normalized=False, grayscale=False,image_size=self.image_size)
#             sift = cv2.xfeatures2d.SIFT_create()
#             # find the keypoints and descriptors with SIFT
            
#             kp1, des1 = sift.detectAndCompute(image,None)
#             kp2, des2 = sift.detectAndCompute(goal_image,None)
#             if des1 is None or len(kp2) < 2:
#                 return False
            
#             FLANN_INDEX_KDTREE = 1
#             index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#             search_params = dict(checks=50)   # or pass empty dictionary
             
#             flann = cv2.FlannBasedMatcher(index_params,search_params)
             
#             matches = flann.knnMatch(des1,des2,k=2)
#             # print("matches",len(matches))
#             # Need to draw only good matches, so create a mask
#             matchesMask = [[0,0] for i in range(len(matches))]
#             num_good_matches=0

#             # ratio test as per Lowe's paper
#             # Perform matching
#             # matches = self.matcher.knnMatch(descriptors1, self.goal_descriptors, k=2)
            
#             # Count good matches using numpy for speed
#             # Convert matches to distance ratios
#             # print("dist",dist)
#             if len(matches) < 2:
#                 return False
                
#             distances = np.array([[m.distance if m is not None else np.inf for m in match] 
#                                 for match in matches])
#             good_matches = np.sum(distances[:, 0] < 0.001 * distances[:, 1])
            
#             # draw_params = dict(matchColor = (0,255,0),
#             #                    singlePointColor = (255,0,0),
#             #                    matchesMask = matchesMask,
#             #                    flags = cv2.DrawMatchesFlags_DEFAULT)
#             match_percentage = good_matches / len(matches)
#             # print(match_percentage)
#             # if  match_percentage > 0.09:
#             # img3 = cv2.drawMatchesKnn(image,kp1,goal_image,kp2,matches,None,**draw_params)             
#             # cv2.imwrite("test.jpg",img3)
#             # print("num_good_matches",match_percentage)
#             self.match_features=match_percentage/0.09
#             # print(match_percentage,dist)
#             return match_percentage > 0.9 or dist<5.0


#     # def check_goal_reached(self, core, hero, goal_location, distance_to_goal, distance_threshold=2.0, angle_threshold=45.0):
#     #     """
#     #     Check if goal is reached considering spatial, temporal, and waypoint-based conditions
        
#     #     Args:
#     #         core: CARLA core instance
#     #         hero: Hero vehicle actor
#     #         goal_location: Target location (numpy array [x,y,z])
#     #         distance_to_goal: Pre-calculated direct distance to goal
#     #         distance_threshold: Distance threshold for goal reaching
#     #         angle_threshold: Angle threshold (not used currently)
#     #     """
#     #     # Basic validation
#     #     if abs(self.distance_travelled) < self.goal_threshold*2:
#     #         return False
            
#         # hero_transform = hero.get_transform()
#         # hero_location = hero_transform.location
        
#     #     # # Calculate temporal distance (time to reach goal at current velocity)
#     #     # hero_velocity = np.dot(
#     #     #     carla_location_to_np_array(hero.get_velocity()),
#     #     #     goal_location / (np.linalg.norm(goal_location) + 1e-8)  # Avoid division by zero
#     #     # )
#     #     # temporal_distance =  (distance_to_goal)/(hero_velocity+1e-8)
        
#     #     # Convert goal location to CARLA format
#     #     goal_location_carla = carla.Location(
#     #         x=goal_location[0],
#     #         y=goal_location[1],
#     #         z=goal_location[2]
#     #     )
        
#     #     # # Get waypoints
#     #     # try:
#         # hero_waypoint = core.map.get_waypoint(hero_location)
#         # goal_waypoint = core.map.get_waypoint(goal_location_carla)

        
#     #     # If waypoints are valid, do detailed checks
#     #     if hero_waypoint and goal_waypoint:
#     #         # Check if on same road segment
#     #         same_section = hero_waypoint.section_id == goal_waypoint.section_id
#     #         same_road = hero_waypoint.road_id == goal_waypoint.road_id
#     #         same_lane = hero_waypoint.lane_id == goal_waypoint.lane_id
            
#     #         # Calculate distance along path
#     #         distance_along_path = hero_waypoint.s - goal_waypoint.s
            
#     #         # Check various thresholds
#     #         within_path_threshold = abs(distance_along_path) < self.goal_threshold
#     #         # within_direct_threshold = distance_to_goal < self.goal_threshold
#     #         # within_temporal_threshold = temporal_distance < self.goal_threshold
#     #         same_waypoint = hero_waypoint.id == goal_waypoint.id
            
#     #         # Combined conditions
#     #         road_condition = same_section and same_road and same_lane and within_path_threshold
#     #         return (road_condition or same_waypoint)
#     #     else:
#     #         # Fallback to simpler checks if waypoints are invalid
#     #         return (distance_to_goal < self.goal_threshold)
#     # def check_goal_reached(self,hero, goal_location, distance_threshold=2.0, angle_threshold=45.0):
#     #         """
#     #         Check if vehicle has reached goal based on:
#     #         1. Distance to goal is within threshold
#     #         2. Vehicle is facing roughly the right direction
#     #         3. Vehicle speed is low enough (optional)
#     #         """
#     #         # Get current location and compute distance
#     #         hero_transform = hero.get_transform()
#     #         hero_location = hero_transform.location
#     #         distance_to_goal = np.linalg.norm(
#     #             goal_location[:2] - carla_location_to_np_array(hero_location)[:2]
#     #         )
            
#     #         # Get vehicle's forward vector
#     #         forward_vector = hero_transform.get_forward_vector()
#     #         forward = np.array([forward_vector.x, forward_vector.y])
#     #         forward = forward / np.linalg.norm(forward)
            
#     #         # Get direction to goal
#     #         hero_pos = carla_location_to_np_array(hero_location)[:2]
#     #         goal_pos = goal_location[:2]
#     #         to_goal = goal_pos - hero_pos
#     #         if np.linalg.norm(to_goal) > 0:
#     #             to_goal = to_goal / np.linalg.norm(to_goal)
            
#     #         # Compute angle between forward vector and goal direction
#     #         angle = np.arccos(np.clip(np.dot(forward, to_goal), -1.0, 1.0))
#     #         angle_deg = np.degrees(angle)
            
#     #         # Optional: Check vehicle speed
#     #         velocity = hero.get_velocity()
#     #         speed = np.linalg.norm([velocity.x, velocity.y])
            
#     #         # Check all conditions
#     #         distance_ok = distance_to_goal < self.goal_threshold
#     #         angle_ok = angle_deg < angle_threshold
#     #         speed_ok = speed < 0.1  # Optional speed check
#     #         return distance_ok and angle_ok , speed_ok


#     def compute_reward(self, sensor_data, core):
#         hero = core.hero
#         goal_loc = sensor_data['goal'][1][-2]
#         hero_location = hero.get_location()
#         hero_velocity = self.get_speed(hero)
#         distance_to_goal = np.linalg.norm(goal_loc[:2]-carla_location_to_np_array(hero_location)[:2])
#         if self.total_distance is None:
#             self.total_distance = np.linalg.norm(goal_loc[:2]-carla_location_to_np_array(hero_location)[:2])
#             assert np.allclose(distance_to_goal,self.total_distance)
#         # hero_velocity=np.dot(carla_location_to_np_array(hero.get_velocity()),goal_loc/np.linalg.norm(goal_loc))
        
      
#         # print(self.total_distance,distance_to_goal)
#         # displ=np.dot(carla_location_to_np_array(hero.get_velocity()),goal_location/np.linalg.norm(goal_location))
#         if self.last_location is None:
#             self.last_location = hero_location
#             self.last_distance_to_goal = distance_to_goal
        
#         displacement=np.dot(carla_location_to_np_array(hero_location)-carla_location_to_np_array(self.last_location),goal_loc/np.linalg.norm(goal_loc))
#         self.distance_travelled_toward_goal-=displacement
#         delta_distance = float(np.sqrt(np.square(hero_location.x - self.last_location.x) + \
#                             np.square(hero_location.y - self.last_location.y)))
#         self.distance_travelled += delta_distance
#         imu = sensor_data['imu'][1]
#         self.heading = sensor_data['goal'][1][-1]
#         # print("delta_distance",delta_distance)
#         self.distances.append(delta_distance)
#         # print("compass",np.rad2deg(imu[-1]),np.rad2deg(self.heading))
#         # vehicle_transform = hero.get_transform()
#         # vehicle_yaw = vehicle_transform.rotation.yaw

#         # Get waypoint's yaw
#         # waypoint = core.map.get_waypoint(hero_location)

#         # Dense progress reward
#         # reward = min(distance_to_goal/self.total_distance,1.0)
#         # heading_diff = min(abs(self.heading - imu[-1]), 2*np.pi - abs(self.heading - imu[-1]))
#         # heading_reward = np.cos(heading_diff)  # Peaks at 1 when aligned, -1 when opposite
        
#         # reward=heading_reward + self.match_features
#         # # Speed matching reward
#         # speed_reward = -min(abs(self.target_speed-hero_velocity)/self.target_speed,1.0)
#         # reward += speed_reward
        
#         # Heading alignment reward
#         # heading = sensor_data['imu'][1][-1]
#         # heading_diff = abs(self.heading - heading)
#         # heading_reward = -heading_diff/(2*np.pi)
#         # reward += 0.1 * heading_reward

#         # # Update distance traveled
#         # delta_distance = float(np.sqrt(np.square(hero_location.x - self.last_location.x) + \
#         #                     np.square(hero_location.y - self.last_location.y)))
#         # self.distance_travelled += delta_distance
#         # reward=displacement
#         # yaw_diff_rad=0.0
#         # if not waypoint is None:
#         #         waypoint_yaw = waypoint.transform.rotation.yaw
#         #         yaw_diff = (vehicle_yaw - waypoint_yaw) % 360.0
#         #         if yaw_diff > 180:
#         #             yaw_diff -= 360.0
#         #         yaw_diff_rad = np.deg2rad(yaw_diff)
#         # reward = 1-(distance_to_goal/self.total_distance) # Base step penalty
#         # if self.prev_reward is None:
#         #     self.prev_reward=reward
#         # step_reward=self.prev_reward-reward
#         # self.prev_reward=reward
#         # Reward for velocity
#         reward=0.0
#         if hero_velocity < self.target_speed:
#             # if self.heading
#             # reward += np.cos(imu[-1]-self.heading)*delta_distance
#             reward -= displacement 
#             # re   
#         # else:
#             # reward -= 0.0  # Optional penalty for exceeding target speed
#         # print(f"Goal {self.done_goal} Lane {self.diff_lane}")
#         # Goal reward
#         # print(reward,distance_to_goal,self.total_distance)
#         if self.done_goal:
#             print(f"Goal reached :travelled {self.distance_travelled}")
#             reward += 1.0 
#         elif self.done_falling or self.collision or self.done_time_idle or self.diff_lane:
#             print(f"Truncated :travelled {self.distance_travelled} idle:{self.done_time_idle} falling :{self.done_falling} diff {self.diff_lane} or collision {self.collision}")
#             reward += -1.0
#         # Scale the reward
#         self.last_location = hero_location
#         self.last_distance_to_goal = distance_to_goal
#         return reward*10
    


# config = {
    
#     "env_config": {
#         "carla": {
#             "host": "localhost",
#             "timeout": 20.0,
#             "timestep": 0.1,
#             "retries_on_error": 25,
#             "resolution_x": 600,
#             "resolution_y": 600,
#             "quality_level": "Low",
#             "enable_map_assets": True,
#             "enable_rendering": True,
#             "show_display": True,
#             "town":"Town01"
#         },
#         "experiment": {
#             "type":JAXMappingExperiments,
#             "hero": {
#                 "blueprint": "vehicle.mercedes.coupe_2020",
#                 "sensors": {
#                     "collision": {
#                         "type": "sensor.other.collision"
#                     },
#                     "rgb": {
#                         "type": "sensor.camera.rgb",
#                         "image_size_x": 300,
#                         "image_size_y": 300,
#                         "transform": "1.9, 0.0, 1.7, 0.0, -15.0, 0.0"
#                     },
#                     "lane_invasion": {
#                         "type": "sensor.other.lane_invasion"
#                     },
#                     "imu":{
#                         "type":"sensor.other.imu"
#                     },
#                     "goal_heading":{
#                         "type":"sensor.goal.heading"
#                     },
                    # "goal":{
                    #     "type":"sensor.goal"
                    # },
#                 },
#                 # "spawn_points": [
#                 #     "-115.60, -207.60, 11.02, -0.00, -0.01, -179.92",  # tl_l
#                 #     "-243.70, 71.50, 11.98, -0.00, -0.08, 90.81",      # bl_d
#                 #     "76.60, 202.10, 1.00, -0.00, 0.00, 0.29",          # br_r
#                 #     "210.20, -50.30, 1.00, -0.00, 0.00, -90.67",       # ur_u
#                 #     "-114.30, 191.00, 9.27, -0.00, 0.61, 179.97",      # bl_l
#                 #     "-233.00, -51.50, 11.00, -0.00, 0.00, -90.84",     # tl_u
#                 #     "83.90, -190.30, 1.00, -0.00, 0.00, -0.24",        # tr_r
#                 #     "189.30, 88.20, 1.00, -0.00, 0.00, 90.66"          # br_d
#                 # ]
#             },
#             "background_activity": {
#                 "n_vehicles": 0,
#                 "n_walkers": 0,
#                 "tm_hybrid_mode": True
#             },
#             # "town": "Town05",
#             "weather": "CloudySunset",
#             "others": {
#                 "framestack": 1,
#                 "max_time_idle": 600,
#                 "max_dist": 200,
#                 "target_speed": 5.0
#             }
#         }
#     }
# }



# Modified from https://github.com/carla-simulator/rllib-integration/blob/main/dqn_example/dqn_experiment.py

from collections import defaultdict, deque
import math
import os
import random
import numpy as np
from gymnasium.spaces import Box, Dict

import carla

import torch
import torchvision.transforms as T
import carla
import torch.nn as nn 
from dune.model.dune import build_encoder
from dune.model.model_utils import extra_repr
from rlib_integration.base_experiment import BaseExperiment
from rlib_integration.helper import post_process_image, carla_location_to_np_array

class DUNE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        # proj: nn.Module,
        apply_last_enc_norm: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        # self.proj = proj
        self.apply_last_enc_norm = apply_last_enc_norm

    def extra_repr(self) -> str:
        return extra_repr(self)

    @property
    def patch_size(self) -> int:
        return self.encoder.patch_size  # type: ignore

    def get_encoder_output(
        self, image: torch.Tensor, concat_cls_patch=True
    ):
        out_dict = self.encoder(image, apply_norm=self.apply_last_enc_norm)

        enc_out = out_dict
        if concat_cls_patch:
            enc_out = torch.cat(
                [
                    out_dict["x_norm_clstoken"].unsqueeze(1),
                    out_dict["x_norm_patchtokens"],
                ],
                dim=1,
            )
        # print(enc_out.shape)
        return enc_out

def build_student_from_args(args):
    encoder = build_encoder(args.image_size, args.enc_args)
    model = DUNE(encoder)
    return model


def load_student_from_checkpoint(ckpt_fname, ckpt_key="model"):
    assert os.path.isfile(ckpt_fname), ckpt_fname
    ckpt = torch.load(ckpt_fname, "cpu", weights_only=False)
    model = build_student_from_args(ckpt["args"])
    state_dict = ckpt.get(ckpt_key, ckpt)
    state_dict = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }
    state_dict_np={}
    for  k,v in state_dict.items():
        if not "projector" in k and not "projectors" in k and not "teacher_norms" in k:
            state_dict_np[k]=v

    model.load_state_dict(state_dict_np)
   
    return model



import numpy as np
import torch
import carla
import math
from collections import defaultdict
from gymnasium.spaces import Box, Dict
import torchvision.transforms as T


import numpy as np
import torch
import carla
import math
from collections import defaultdict
from gymnasium.spaces import Box, Dict
import torchvision.transforms as T


class JaxGoalConditionedExperiment(BaseExperiment):
    def __init__(self, config={}, is_rgb=False, image_size=()):
        super().__init__(config)
        
        # Config parameters
        self.frame_stack = self.config["others"]["framestack"]
        self.max_time_idle = self.config["others"]["max_time_idle"]
        self.max_dist = self.config["others"]["max_dist"]
        self.target_speed = self.config["others"]["target_speed"]
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]
        
        # Control variables
        self.max_steer = 0.5
        self.max_throttle = 0.6
        self.max_angle_deviation = np.pi
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.steer = 0.0
        self.throttle = 0.0
        
        # State variables
        self.current_heading = 0.0
        self.velocity = 0.0
        self.relative_goal_loc = np.zeros((3,))
        self.last_action = None
        self.info = dict()
        self.rewards = []
        self.origin = None
        self.image_size = 70
        self.vecs=np.zeros(4)
        
        # Vision model setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_student_from_checkpoint(
            "/home/robotlab/scratch/carla-rl/dune_checkpoints/dune_vitbase14_448.pth"
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.image_transform = T.Compose([
            T.ToTensor(),
            T.Resize((70, 70), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # HER (Hindsight Experience Replay) buffers
        self.relabel_pairs = []
        self.current_trajectory = []
        self.current_episode_data = []
        self.trajectories = None
        self.prev_distance = None

    def reset(self, core, *arg, **kwargs):
        """Called at the beginning and each time the simulation is reset"""
        # Ending variables
        self.time_idle = 0
        self.time_episode = 0
        self.done_time_idle = False
        self.done_falling = False
        self.done_goal = False
        self.vecs=np.zeros(4)
        # Hero variables
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

        # Control variables
        self.max_steer = 1.0
        self.max_throttle = 1.0
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.steer = 0.0
        self.throttle = 0.0
        self.target_speed = 12.0
        self.velocity = 0.0
        self.current_heading = 0.0
        self.relative_goal_loc = np.zeros((3,))
        self.info = dict()
        
        # HER buffers
        self.current_trajectory = []
        self.current_episode_data = []
        self.rewards = []
        self.prev_distance=None
        if self.trajectories is None:
            self._cache_waypoints(core.world)
        core.trajectories=self.trajectories

    def _cache_waypoints(self, world) -> None:
        """Cache waypoint trajectories for curriculum learning"""
        env_map = world.get_map()
        waypoints = env_map.generate_waypoints(distance=0.1)
        trajectories = defaultdict(list)
        
        for wpt in waypoints:
            trajectories[f"{wpt.road_id}-{wpt.lane_id}"].append(wpt)
        
        self.trajectories = sorted(
            [traj for traj in trajectories.values() if len(traj) > 3], 
            key=len, 
            reverse=True
        )
        # self.trajectories=[traj for traj in trajectories.values() if 3 < len(traj) <= 10]1
        self.max_curriculum_steps = len(self.trajectories)

    def get_observation_space(self):
        """Returns the observation space"""
        image_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 * 26 * 768,),  # 2 images (current + goal) * DUNE ViT-Base tokens
            dtype=np.float32
        )
        
        relative_goal = Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32,
        )
        
        return Dict({
            "state_obs": image_space,
            "relative_goal_loc": relative_goal
        })

    def get_action_space(self):
        """Returns the continuous action space for steering and throttle"""
        return Box(
            low=np.array([-self.max_steer, 0]),
            high=np.array([self.max_steer, 1.0]),
            dtype=np.float32
        )

    def compute_action(self, action):
        """Convert continuous actions to CARLA vehicle controls"""
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
        self.steer = vehicle_control.steer
        self.throttle = vehicle_control.throttle
        
        return vehicle_control

    def get_observation(self, sensor_data, core):
        """Process observations from sensor data"""
        vecs = self.get_vec_obs(sensor_data, core)
        images = self.get_img_obs(sensor_data, core)
        
        obs = {
            "state_obs": images,
            "relative_goal_loc": self.relative_goal_loc
        }
        
        # Store for HER
        if hasattr(self, '_current_image_latent') and hasattr(self, '_current_goal_latent'):
            self.current_episode_data.append({
                'observation': {
                    'state_obs': obs['state_obs'].copy(),
                    'relative_goal_loc': obs['relative_goal_loc'].copy()
                },
                'image_latent': self._current_image_latent.copy(),
                'goal_latent': self._current_goal_latent.copy(),
                'vec_obs': vecs.copy() if vecs is not None else None
            })
        
        return obs, self.info

    def get_vec_obs(self, sensor_data, core):
        """Get vector observations"""
        heading = sensor_data["goal"][-1][-1]
        imu = sensor_data["imu"][-1][-1]
        hero = core.hero
        
        vec = np.zeros(4)
        vec[0] = self.prev_steer / self.max_steer
        vec[1] = self.prev_throttle / self.max_throttle
        vec[2] = np.clip(self.get_speed(hero) / (self.target_speed + 1e-8), 0.0, 1.0)
        vec[3] = np.clip(imu / (heading + 1e-8), -1.0, 1.0)
        
        # Frame stacking
        if self.prev_vec_0 is None:
            self.prev_vec_0 = vec
            self.prev_vec_1 = vec
            self.prev_vec_2 = vec

        vecs = vec
        if self.frame_stack >= 2:
            vecs = np.concatenate([self.prev_vec_0, vecs], axis=0)
        if self.frame_stack >= 3:
            vecs = np.concatenate([self.prev_vec_1, vecs], axis=0)
        if self.frame_stack >= 4:
            vecs = np.concatenate([self.prev_vec_2, vecs], axis=0)

        # Update history
        self.prev_vec_2 = self.prev_vec_1
        self.prev_vec_1 = self.prev_vec_0
        self.prev_vec_0 = vec
        self.vecs=vecs
        # Update state
        self.velocity = self.get_speed(hero)
        self.current_heading = imu
        return vecs

    def get_img_obs(self, sensor_data, core):
        """Get image observations using vision transformer encoder"""
        image = post_process_image(
            sensor_data['rgb'][1], 
            normalized=True, 
            crop=False, 
            grayscale=False, 
            image_size=self.image_size
        )
        goal = post_process_image(
            sensor_data['goal'][1][0], 
            normalized=False, 
            crop=False, 
            grayscale=False, 
            image_size=self.image_size
        )

        # Frame stacking
        if self.prev_image_0 is None:
            self.prev_image_0 = image
            self.prev_image_1 = image
            self.prev_image_2 = image

        images = image
        if self.frame_stack >= 2:
            images = np.concatenate([self.prev_image_0, images], axis=2)
        if self.frame_stack >= 3:
            images = np.concatenate([self.prev_image_1, images], axis=2)
        if self.frame_stack >= 4:
            images = np.concatenate([self.prev_image_2, images], axis=2)

        # Update history
        self.prev_image_2 = self.prev_image_1
        self.prev_image_1 = self.prev_image_0
        self.prev_image_0 = image
        
        # Encode images efficiently (memory-optimized)
        torch.cuda.empty_cache()
        latents = []
        
        for img in [image, goal]:
            img_tensor = self.image_transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                latent = self.model.get_encoder_output(img_tensor)
            latents.append(latent.squeeze(0).cpu())
            del img_tensor, latent
        
        torch.cuda.empty_cache()
        
        # Extract and store individual latents for HER
        image_latent = latents[0].flatten().numpy()
        goal_latent = latents[1].flatten().numpy()
        
        self._current_image_latent = image_latent
        self._current_goal_latent = goal_latent
        
        return np.concatenate([image_latent, goal_latent], axis=0)
    
    def get_speed(self, hero):
        """Computes the speed of the hero vehicle in Km/h"""
        vel = hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    # def get_done_status(self, sensor_data, core):
    #     """Returns whether the episode should end"""
    #     hero = core.hero
    #     self.info = dict()
        
    #     # Update idle time
    #     self.done_time_idle = self.max_time_idle < self.time_idle
    #     if self.get_speed(hero) > 1.0:
    #         self.time_idle = 0
    #     else:
    #         self.time_idle += 1
    #     self.time_episode += 1
        
    #     # Goal distance
    #     goal_location = sensor_data['goal'][1][-2]
    #     hero_location = carla_location_to_np_array(hero.get_transform().location)
    #     loc=carla_location_to_np_array(core.destination.location)
    #     delta=goal_location[:2] - hero_location[:2]
    #     self.relative_goal_loc = np.array([*delta.tolist(),self.current_heading])
    #     # breakpoint()
    #     distance_to_goal = np.linalg.norm(delta)
    #     goal_location = sensor_data['goal'][1][-2]
    #     hero_location = carla_location_to_np_array(hero.get_transform().location)
    #     loc=carla_location_to_np_array(core.destination.location)
    #     print(loc,goal_location)
    #     # Terminal conditions
    #     wp = core.map.get_waypoint(hero.get_transform().location, project_to_road=False)
    #     self.done_goal = distance_to_goal < 4.0
    #     self.done_falling = hero.get_location().z < -0.5
    #     self.diff_lane = 'lane_invasion' in sensor_data.keys() or wp is None
    #     self.collision = 'collision' in sensor_data.keys()
        
    #     # Store trajectory step for HER (including action)
    #     trajectory_step = {
    #         'vehicle_location': hero_location.copy(),
    #         'goal_location': goal_location.copy(),
    #         'relative_goal_loc': self.relative_goal_loc.copy(),
    #         'distance_to_goal': distance_to_goal,
    #         'collision': self.collision,
    #         'lane_invasion': self.diff_lane,
    #         'done_goal': self.done_goal,
    #         'done_falling': self.done_falling,
    #         'velocity': self.velocity,
    #         'heading': self.current_heading,
    #         'timestep': self.time_episode,
    #         'reward': None,  # Will be filled in compute_reward
    #         'action': np.array([self.steer, self.throttle])  # Store current action
    #     }
    #     self.current_trajectory.append(trajectory_step)
        
    #     # Update info
    #     done = self.done_goal
    #     self.info.update({
    #         'is_success': self.done_goal,
    #         'distance_completed': self.distance_travelled
    #     })
        
    #     if len(self.rewards) > 0:
    #         self.info.update({
    #             'max_reward': np.max(self.rewards),
    #             'min_reward': np.min(self.rewards),
    #             'mean_reward': np.mean(self.rewards)
    #         })
        
    #     # Truncation
    #     trunc = self.done_falling or self.diff_lane or self.collision
    #     if trunc:
    #         self.info.update({'truncate': True})
        
    #     # Process HER relabeling when episode ends
    #     if done or trunc:
    #         self.her_label(done, trunc)
        
    #     return done
    def lane_following_get_observation_space(self):
        image_space= Box(
                low=-np.inf,
                high=np.inf,
                shape=(26*768,), # DUNE ViT-Base patch token output
                dtype=np.float32
            )
        vec_space = Box(
            low=-5.1,
            high=5.1,
            shape=(4 * self.frame_stack,),
            dtype=np.float32,
        )

        return Dict({"state":image_space, "vector":vec_space})
    
    def get_lane_vec_obs(self):
        vecs=np.nan_to_num(self.vecs,nan=1e-8)
        return vecs
    
    def get_lane_observations(self):
        return {"state":self._current_image_latent, "vector":self.vecs}
    

    def get_done_status(self, sensor_data, core):
        """Returns whether the episode should end"""
        hero = core.hero
        self.info = dict()
        
        # Update idle time
        self.done_time_idle = self.max_time_idle < self.time_idle
        if self.get_speed(hero) > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1
        self.time_episode += 1
        
        # Goal distance
        goal_location = sensor_data['goal'][1][-2]
        hero_location = carla_location_to_np_array(hero.get_transform().location)
        delta = goal_location[:2] - hero_location[:2]
        
        # FIXED: Include angle to goal, not current heading
        angle_to_goal = np.arctan2(delta[1], delta[0])
        self.relative_goal_loc = np.array([delta[0], delta[1], angle_to_goal])
        
        distance_to_goal = core.destination.location.distance(hero.get_transform().location)
        goal_location = sensor_data['goal'][1][-2]
        hero_location = carla_location_to_np_array(hero.get_transform().location)
        loc=carla_location_to_np_array(core.destination.location)
        # print(loc,goal_location,distance_to_goal)
        # Terminal conditions
        wp = core.map.get_waypoint(hero.get_transform().location, project_to_road=False)
        self.done_goal = distance_to_goal < 4.0
        self.done_falling = hero.get_location().z < -0.5
        self.diff_lane = 'lane_invasion' in sensor_data.keys() or wp is None
        self.collision = 'collision' in sensor_data.keys()
        # print(distance_to_goal)
        # FIXED: Store action in trajectory
        trajectory_step = {
            'vehicle_location': hero_location.copy(),
            'goal_location': goal_location.copy(),
            'relative_goal_loc': self.relative_goal_loc.copy(),
            'distance_to_goal': distance_to_goal,
            'collision': self.collision,
            'lane_invasion': self.diff_lane,
            'done_goal': self.done_goal,
            'done_falling': self.done_falling,
            'velocity': self.velocity,
            'heading': self.current_heading,
            'timestep': self.time_episode,
            'reward': None,
            'action': np.array([self.steer, self.throttle])  # Current action
        }
        self.current_trajectory.append(trajectory_step)
        
        # Update info
        done = self.done_goal
        self.info.update({
            'is_success': self.done_goal,
            'distance_completed': self.distance_travelled,
            'distance_to_goal': distance_to_goal
        })
        
        if len(self.rewards) > 0:
            self.info.update({
                'max_reward': np.max(self.rewards),
                'min_reward': np.min(self.rewards),
                'mean_reward': np.mean(self.rewards)
            })
        
        # Truncation
        trunc = self.done_falling or self.diff_lane or self.collision
        if trunc:
            self.info.update({'truncate': True})
        
        # Process HER relabeling when episode ends
        if done or trunc:
            self.her_label(done, trunc)
        
        return done


    def compute_reward(self, sensor_data, core):
        """Compute reward for current step"""
        hero = core.hero
        hero_location = carla_location_to_np_array(hero.get_transform().location)
        hero_velocity = self.get_speed(hero)
        goal_location = sensor_data['goal'][1][-2]
        
        # Initialize last location
        if self.last_location is None:
            self.last_location = hero_location
        
        # Compute distance to goal
        distance_to_goal =  core.destination.location.distance(hero.get_transform().location)
        # distance_to_goal = np.linalg.norm(delta)
        
        if self.prev_distance is None:
            self.prev_distance = distance_to_goal
        
        # Compute reward
        reward = -1 # Living penalty
        reward += (self.prev_distance - distance_to_goal)  # Progress reward
        
        # if self.done_goal:
        #     reward += 10.0
        
        self.rewards.append(reward)
        
        # FIXED: Store reward AND distance in trajectory
        if len(self.current_trajectory) > 0:
            self.current_trajectory[-1]['reward'] = reward
            self.current_trajectory[-1]['distance_to_goal'] = distance_to_goal
        
        # Update control history
        self.prev_steer = self.steer
        self.prev_throttle = self.throttle
        self.prev_distance = distance_to_goal
        self.last_location = hero_location
        self.last_velocity = hero_velocity
        
        return reward


    def her_label(self, done, truncated):
        """
        Hindsight Experience Replay relabeling.
        Creates replay buffer compatible transitions with relabeled goals.
        """
        if len(self.current_trajectory) < 2 or len(self.current_episode_data) < 2:
            self.current_trajectory = []
            self.current_episode_data = []
            return
        
        k = 4  # Number of future states for relabeling
        
        for t in range(len(self.current_trajectory) - 1):
            current_step = self.current_trajectory[t]
            current_data = self.current_episode_data[t]
            next_step = self.current_trajectory[t + 1]
            next_data = self.current_episode_data[t + 1]
            
            # Sample future states
            future_indices = range(t + 1, min(t + 1 + k, len(self.current_trajectory)))
            
            for future_idx in future_indices:
                future_step = self.current_trajectory[future_idx]
                future_data = self.current_episode_data[future_idx]
                
                # FIXED: Compute new relative goal with angle to goal
                delta_current = future_step['vehicle_location'][:2] - current_step['vehicle_location'][:2]
                angle_to_goal_current = np.arctan2(delta_current[1], delta_current[0])
                new_relative_goal = np.array([
                    delta_current[0], 
                    delta_current[1], 
                    angle_to_goal_current
                ])
                distance_to_new_goal = np.linalg.norm(delta_current)
                
                # Compute next step relative goal
                delta_next = future_step['vehicle_location'][:2] - next_step['vehicle_location'][:2]
                angle_to_goal_next = np.arctan2(delta_next[1], delta_next[0])
                next_relative_goal = np.array([
                    delta_next[0],
                    delta_next[1],
                    angle_to_goal_next
                ])
                distance_to_new_goal_next = np.linalg.norm(delta_next)
                
                # Create relabeled observations
                relabeled_state_obs = np.concatenate([
                    current_data['image_latent'],
                    future_data['image_latent']  # Future state as goal
                ], axis=0)
                
                relabeled_observation = {
                    'state_obs': relabeled_state_obs,
                    'relative_goal_loc': new_relative_goal
                }
                
                relabeled_next_state_obs = np.concatenate([
                    next_data['image_latent'],
                    future_data['image_latent']
                ], axis=0)
                
                relabeled_next_observation = {
                    'state_obs': relabeled_next_state_obs,
                    'relative_goal_loc': next_relative_goal
                }
                
                # FIXED: Recompute reward based on new goal
                relabeled_reward = -1 # Living penalty
                # relabeled_reward += (distance_to_new_goal - distance_to_new_goal_next)  # Progress toward NEW goal
                
                # Check if new goal is reached
                relabeled_done = distance_to_new_goal < 4.0
                if relabeled_done:
                    relabeled_reward += 10.0  # Goal bonus
                
                # Compute mask
                relabeled_truncated = (
                    current_step['collision'] or 
                    current_step['lane_invasion'] or 
                    current_step['done_falling']
                )
                mask = 0.0 if (relabeled_done or relabeled_truncated) else 1.0
                
                # Store replay buffer compatible transition
                her_transition = {
                    'observations': relabeled_observation,
                    'actions': current_step['action'],  # Action is now guaranteed to exist
                    'rewards': relabeled_reward,
                    'masks': mask,
                    'dones': relabeled_done,
                    'next_observations': relabeled_next_observation,
                }
                
                self.relabel_pairs.append(her_transition)
        
        # Clear buffers
        self.current_trajectory = []
        self.current_episode_data = []
        
        # Limit buffer size
        max_buffer = 10000
        if len(self.relabel_pairs) > max_buffer:
            self.relabel_pairs = self.relabel_pairs[-max_buffer:]

    def get_her_batch(self, batch_size=32):
        """Sample a batch of HER relabeled experiences for replay buffer"""
        if len(self.relabel_pairs) < batch_size:
            return None
        
        indices = np.random.choice(
            len(self.relabel_pairs), 
            size=batch_size, 
            replace=False
        )
        return [self.relabel_pairs[i] for i in indices]

    def sample_her_transitions(self, n_samples=1):
        """
        Sample HER transitions in replay buffer format.
        Returns: dict with keys matching replay_buffer.insert() format
        """
        if len(self.relabel_pairs) == 0:
            return None
        
        if n_samples == 1:
            return self.relabel_pairs[np.random.randint(len(self.relabel_pairs))]
        
        indices = np.random.choice(
            len(self.relabel_pairs), 
            size=min(n_samples, len(self.relabel_pairs)), 
            replace=False
        )
        return [self.relabel_pairs[i] for i in indices]
    

class HERWithNegativeSampling:
    """
    Enhanced HER implementation with negative sampling.
    Samples observations and goals from different trajectories to teach the agent
    to distinguish between correct and incorrect goal pairings.
    """
    
    def __init__(self, negative_sampling_ratio=0.5, max_buffer_size=10000):
        """
        Args:
            negative_sampling_ratio: Proportion of negative samples (0.0 to 1.0)
            max_buffer_size: Maximum size of the replay buffer
        """
        self.relabel_pairs = []
        self.current_trajectory = []
        self.current_episode_data = []
        self.completed_trajectories = deque(maxlen=100)  # Store past trajectories
        self.negative_sampling_ratio = negative_sampling_ratio
        self.max_buffer_size = max_buffer_size
        self.negative_reward_penalty = -10.0  # Large penalty for wrong goal
        
    def her_label(self, done, truncated, k=4):
        """
        Enhanced HER labeling with negative sampling.
        
        Args:
            done: Whether episode completed successfully
            truncated: Whether episode was truncated
            k: Number of future states for positive relabeling
        """
        if len(self.current_trajectory) < 2 or len(self.current_episode_data) < 2:
            self.current_trajectory = []
            self.current_episode_data = []
            return
        
        # Store completed trajectory for negative sampling
        self.completed_trajectories.append({
            'trajectory': self.current_trajectory.copy(),
            'episode_data': self.current_episode_data.copy()
        })
        
        # Generate positive samples (standard HER)
        self._generate_positive_samples(k)
        
        # Generate negative samples (from different trajectories)
        if len(self.completed_trajectories) >= 2:
            self._generate_negative_samples()
        
        # Clear current buffers
        self.current_trajectory = []
        self.current_episode_data = []
        
        # Limit buffer size
        if len(self.relabel_pairs) > self.max_buffer_size:
            self.relabel_pairs = self.relabel_pairs[-self.max_buffer_size:]
    
    def _generate_positive_samples(self, k):
        """Generate positive HER samples (standard approach)"""
        for t in range(len(self.current_trajectory) - 1):
            current_step = self.current_trajectory[t]
            current_data = self.current_episode_data[t]
            next_step = self.current_trajectory[t + 1]
            next_data = self.current_episode_data[t + 1]
            
            # Sample future states from same trajectory
            future_indices = range(t + 1, min(t + 1 + k, len(self.current_trajectory)))
            
            for future_idx in future_indices:
                future_step = self.current_trajectory[future_idx]
                future_data = self.current_episode_data[future_idx]
                
                # Compute relative goal with angle
                delta_current = future_step['vehicle_location'][:2] - current_step['vehicle_location'][:2]
                angle_to_goal_current = np.arctan2(delta_current[1], delta_current[0])
                new_relative_goal = np.array([
                    delta_current[0], 
                    delta_current[1], 
                    angle_to_goal_current
                ])
                distance_to_new_goal = np.linalg.norm(delta_current)
                
                # Compute next step relative goal
                delta_next = future_step['vehicle_location'][:2] - next_step['vehicle_location'][:2]
                angle_to_goal_next = np.arctan2(delta_next[1], delta_next[0])
                next_relative_goal = np.array([
                    delta_next[0],
                    delta_next[1],
                    angle_to_goal_next
                ])
                distance_to_new_goal_next = np.linalg.norm(delta_next)
                
                # Create relabeled observations
                relabeled_state_obs = np.concatenate([
                    current_data['image_latent'],
                    future_data['image_latent']
                ], axis=0)
                
                relabeled_observation = {
                    'state_obs': relabeled_state_obs,
                    'relative_goal_loc': new_relative_goal
                }
                
                relabeled_next_state_obs = np.concatenate([
                    next_data['image_latent'],
                    future_data['image_latent']
                ], axis=0)
                
                relabeled_next_observation = {
                    'state_obs': relabeled_next_state_obs,
                    'relative_goal_loc': next_relative_goal
                }
                
                # Compute reward
                relabeled_reward = -1  # Living penalty
                relabeled_reward += (distance_to_new_goal - distance_to_new_goal_next)
                
                relabeled_done = distance_to_new_goal_next < 4.0
                if relabeled_done:
                    relabeled_reward += 10.0
                
                # Compute mask
                relabeled_truncated = (
                    current_step['collision'] or 
                    current_step['lane_invasion'] or 
                    current_step['done_falling']
                )
                mask = 0.0 if (relabeled_done or relabeled_truncated) else 1.0
                
                # Store transition
                her_transition = {
                    'observations': relabeled_observation,
                    'actions': current_step['action'],
                    'rewards': relabeled_reward,
                    'masks': mask,
                    'dones': relabeled_done,
                    'next_observations': relabeled_next_observation,
                    'is_negative': False  # Mark as positive sample
                }
                
                self.relabel_pairs.append(her_transition)
    
    def _generate_negative_samples(self):
        """
        Generate negative samples by pairing observations from one trajectory
        with goals from a different trajectory.
        """
        current_traj_idx = len(self.completed_trajectories) - 1
        current_traj_data = self.completed_trajectories[current_traj_idx]
        
        # Calculate number of negative samples to generate
        num_positives = len(current_traj_data['trajectory']) - 1
        num_negatives = int(num_positives * self.negative_sampling_ratio)
        
        for _ in range(num_negatives):
            # Sample a random step from current trajectory
            t = np.random.randint(0, len(current_traj_data['trajectory']) - 1)
            current_step = current_traj_data['trajectory'][t]
            current_data = current_traj_data['episode_data'][t]
            next_step = current_traj_data['trajectory'][t + 1]
            next_data = current_traj_data['episode_data'][t + 1]
            
            # Sample a goal from a DIFFERENT trajectory
            other_traj_idx = np.random.choice(
                [i for i in range(len(self.completed_trajectories)) if i != current_traj_idx]
            )
            other_traj_data = self.completed_trajectories[other_traj_idx]
            
            # Sample random goal state from other trajectory
            goal_idx = np.random.randint(0, len(other_traj_data['trajectory']))
            goal_step = other_traj_data['trajectory'][goal_idx]
            goal_data = other_traj_data['episode_data'][goal_idx]
            
            # Compute relative goal (will be incorrect/unrelated)
            delta_current = goal_step['vehicle_location'][:2] - current_step['vehicle_location'][:2]
            angle_to_goal_current = np.arctan2(delta_current[1], delta_current[0])
            negative_relative_goal = np.array([
                delta_current[0], 
                delta_current[1], 
                angle_to_goal_current
            ])
            distance_to_negative_goal = np.linalg.norm(delta_current)
            
            # Compute next step relative goal
            delta_next = goal_step['vehicle_location'][:2] - next_step['vehicle_location'][:2]
            angle_to_goal_next = np.arctan2(delta_next[1], delta_next[0])
            next_negative_relative_goal = np.array([
                delta_next[0],
                delta_next[1],
                angle_to_goal_next
            ])
            distance_to_negative_goal_next = np.linalg.norm(delta_next)
            
            # Create negative sample observations
            negative_state_obs = np.concatenate([
                current_data['image_latent'],
                goal_data['image_latent']  # Goal from different trajectory
            ], axis=0)
            
            negative_observation = {
                'state_obs': negative_state_obs,
                'relative_goal_loc': negative_relative_goal
            }
            
            negative_next_state_obs = np.concatenate([
                next_data['image_latent'],
                goal_data['image_latent']
            ], axis=0)
            
            negative_next_observation = {
                'state_obs': negative_next_state_obs,
                'relative_goal_loc': next_negative_relative_goal
            }
            
            # Large negative reward for pursuing wrong goal
            negative_reward = self.negative_reward_penalty
            
            # Check if by chance the negative goal was reached (very unlikely)
            negative_done = distance_to_negative_goal_next < 4.0
            
            # Compute mask
            negative_truncated = (
                current_step['collision'] or 
                current_step['lane_invasion'] or 
                current_step['done_falling']
            )
            mask = 0.0 if (negative_done or negative_truncated) else 1.0
            
            # Store negative transition
            negative_transition = {
                'observations': negative_observation,
                'actions': current_step['action'],
                'rewards': negative_reward,
                'masks': mask,
                'dones': negative_done,
                'next_observations': negative_next_observation,
                'is_negative': True  # Mark as negative sample
            }
            
            self.relabel_pairs.append(negative_transition)
    
    def get_statistics(self):
        """Get statistics about positive/negative samples in buffer"""
        if len(self.relabel_pairs) == 0:
            return {
                'total': 0,
                'positive': 0,
                'negative': 0,
                'negative_ratio': 0.0
            }
        
        negative_count = sum(1 for t in self.relabel_pairs if t.get('is_negative', False))
        positive_count = len(self.relabel_pairs) - negative_count
        
        return {
            'total': len(self.relabel_pairs),
            'positive': positive_count,
            'negative': negative_count,
            'negative_ratio': negative_count / len(self.relabel_pairs)
        }

# Integration with your experiment class
class JaxGoalConditionedExperimentWithNegativeSampling(JaxGoalConditionedExperiment):
    """
    Extended experiment class with negative sampling support
    """
    
    def __init__(self, config={}, is_rgb=False, image_size=()):
        super().__init__(config, is_rgb, image_size)
        
        # Replace standard HER with negative sampling version
        self.her_sampler = HERWithNegativeSampling(
            negative_sampling_ratio=0.5,  # 50% negative samples
            max_buffer_size=10000
        )
    
    def reset(self, core, *args, **kwargs):
        """Reset with HER sampler"""
        super().reset(core, *args, **kwargs)
        
        # Keep reference to trajectories
        if self.trajectories is None:
            self._cache_waypoints(core.world)
        core.trajectories = self.trajectories
    
    def her_label(self, done, truncated):
        """Use HER sampler's labeling with negative samples"""
        # Transfer data to HER sampler
        self.her_sampler.current_trajectory = self.current_trajectory
        self.her_sampler.current_episode_data = self.current_episode_data
        
        # Generate positive and negative samples
        self.her_sampler.her_label(done, truncated, k=4)
        
        # Update local reference
        self.relabel_pairs = self.her_sampler.relabel_pairs
        
        # Clear local buffers
        self.current_trajectory = []
        self.current_episode_data = []
        
        # Print statistics periodically
        if len(self.relabel_pairs) % 100 == 0:
            stats = self.her_sampler.get_statistics()
            print(f"HER Buffer Stats - Total: {stats['total']}, "
                  f"Positive: {stats['positive']}, Negative: {stats['negative']}, "
                  f"Negative Ratio: {stats['negative_ratio']:.2f}")
    
    def sample_her_transitions(self, n_samples=1):
        """Sample from buffer with both positive and negative samples"""
        if len(self.relabel_pairs) == 0:
            return None
        
        if n_samples == 1:
            return self.relabel_pairs[np.random.randint(len(self.relabel_pairs))]
        
        indices = np.random.choice(
            len(self.relabel_pairs), 
            size=min(n_samples, len(self.relabel_pairs)), 
            replace=False
        )
        return [self.relabel_pairs[i] for i in indices]
    
    def get_her_batch(self, batch_size=32):
        """Sample a batch ensuring mix of positive and negative samples"""
        if len(self.relabel_pairs) < batch_size:
            return None
        
        # Optional: Ensure balanced sampling
        positive_indices = [i for i, t in enumerate(self.relabel_pairs) 
                          if not t.get('is_negative', False)]
        negative_indices = [i for i, t in enumerate(self.relabel_pairs) 
                          if t.get('is_negative', False)]
        
        if len(positive_indices) > 0 and len(negative_indices) > 0:
            # Sample half from each
            n_positive = min(batch_size // 2, len(positive_indices))
            n_negative = min(batch_size - n_positive, len(negative_indices))
            
            selected_indices = (
                list(np.random.choice(positive_indices, size=n_positive, replace=False)) +
                list(np.random.choice(negative_indices, size=n_negative, replace=False))
            )
        else:
            # Fallback to random sampling
            selected_indices = np.random.choice(
                len(self.relabel_pairs), 
                size=batch_size, 
                replace=False
            )
        
        return [self.relabel_pairs[i] for i in selected_indices]

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
            "type":JaxGoalConditionedExperimentWithNegativeSampling,
            "hero": {
                "blueprint": "vehicle.mercedes.coupe_2020",
                "is_goal_env":True,
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
                    # "goal_heading":{
                    #     "type":"sensor.goal.heading"
                    # },
                      "goal":{
                        "type":"sensor.goal",
                          "image_size_x": 112,
                        "image_size_y": 112,
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