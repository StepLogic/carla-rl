#!/usr/bin/env python3
from datetime import datetime
import gymnasium as gym
import rospy 
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image,Imu,LaserScan
from geometry_msgs.msg import Twist,Vector3
import queue
from gymnasium.spaces import Dict,Box
import numpy as np
import time
IMAGE_TOPIC="/camera/image_raw"
IMU_TOPIC="/imu/data_raw"
ROBOT_CMD_TOPIC="/cmd_vel"
LIDAR_TOPIC="/scan"
RATE = 60
def ros_vector3_to_np_array(msg):
    return np.array([msg.x,-1*msg.y,msg.z])

import numpy as np
from scipy.signal import filtfilt, butter

def estimate_orientation(a, w, angle,dt, alpha=0.9, g_ref=(0., 0., 1.), theta_min=1e-6, highpass=.01, lowpass=.05):
    """
    Source:https://gist.github.com/phausamann/721fa3df0f8ef6f4f6f24b86fdde53c0
    """

    g_ref = np.array(g_ref)
    w = filtfilt(*butter(5, highpass, btype='high'), w, axis=0)
    w[np.linalg.norm(w, axis=1) < theta_min] = 0
    a = filtfilt(*butter(5, lowpass, btype='low'), a, axis=0)
    angle = (1-alpha)*(angle + w * dt) + (alpha)*(a)

    return angle
class LeoEnv(gym.Env):
    def __init__(self):
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC,Image,self.image_callback)
        self.lidar_sub = rospy.Subscriber(LIDAR_TOPIC,LaserScan,self.lidar_callback)
        self.imu_sub = rospy.Subscriber(IMU_TOPIC,Imu,self.imu_calback)
        self.robot_cmd =rospy.Publisher(ROBOT_CMD_TOPIC, Twist, queue_size=10)
        self.bridge = CvBridge()
        self.image_queue=queue.Queue() #define queue
        self.vector_queue=queue.Queue() #define quue for imu
        self.collision_queue=queue.Queue() #define quue for imu
        # self.current_timestamp=datetime.now()
        self.current_velocity = 0
        self.current_heading = 0
        self.target_speed=2.0
        self.max_steer=1.0
        self.image_size=64
        self.theta=np.zeros((3,))
        self.v=np.zeros((3,))
        self.previous_actions=np.zeros((2)) #steer ,throttle
        self.heading=0
        self.idle_count=0
        self.collision_threshold=None #0.5m
        self.time_episode =0
        self.velocities=[]
        self.dts=[]
        self.headings=[]
        self.distance_travelled=0

        self.action=None
        self.collision=None
        self.done_fast=None
        self.image=None
        self.vector=None

        self.info=dict()
        self.rewards=[]
        self.offsets=[]
        self.ranges=[]
        # self.vector=None
        image_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.image_size, self.image_size, 3,),
            dtype=np.float32,
        )
        
        vec_space = Box(
            low=-5.1,
            high=5.1,
            shape=(4,),
            dtype=np.float32,
        )

        self.observation_space=Dict({"pixels":image_space, "vector":vec_space})

        self.action_space = Box(
            low=np.array([-self.max_steer, -1.0]),  # [steering, throttle/brake]
            high=np.array([self.max_steer, 1.0]),
            dtype=np.float32
        )
        self.rate = rospy.Rate(RATE)
        self.RATE=RATE

    def get_observation(self,reset=False):
        # image=self.observation_space["pixels"].sample()

        # vector=self.observation_space["vector"].sample()
        image=None
        vector=None

        timeout=int(1e2)
        # for _ in range(int(timeout)):
        #     # print(self.vector_queue.qsize())
        #     vector=self.vector_queue.get()
        #     if not vector is None:
        #         # self.vector_queue.queue.clear()
        #         break
        #     # if vector.header.timestamp==self.current_timestamp:
        #     #     break
        # # timeout=0
        # for _ in range(int(timeout)):
        #     image=self.image_queue.get()
        #     if not image is None:
        #         # self.image_queue.queue.clear()
        #         break


        # self.collision=None
        # for _ in range(int(timeout)):
        #     self.collision=self.collision_queue.get() 
        #     if not self.collision is None:
        #         # self.collision_queue.queue.clear()
        #         # self.ranges=[]
        #         break
        
        #     if reset:
        #         break

        return dict(pixels=self.image,vector=self.vector)
    
    def reset(self,*args,**kwargs):
        #zero robot measurements
        self.velocities=[]
        self.dts=[]
        self.headings=[]
        
        self.action=None
        self.collision=None
        self.done_fast=None
        self.image=None
        self.vector=None
        self.ranges=[0.2]
        time.sleep(10.0)
        print("Reset Robot Please!!!!!!")
        self.offsets=[np.mean(self.velocities),np.mean(self.headings)]
        self.collision_threshold=np.max(self.ranges)
        self.ranges=[]
        self.velocities=[]
        self.dts=[]
        self.headings=[]
        self.image_queue.queue.clear()
        self.vector_queue.queue.clear()
        time.sleep(1.0)
        print("Awaiting observations!!!!!!")
        observation=self.get_observation(reset=True)
        self.time_episode =0
        self.distance_travelled=0
        self.rewards=[]
        self.current_heading=0.0
        self.action=np.zeros(2)
        self.info=dict()
        self.speed=0.0
        self.heading=np.random.uniform(1e-8,2*np.pi,)+self.offsets[-1]
        self.target_speed=2.0+self.offsets[0]
        return observation,self.info
            
    def image_callback(self,image):
        # if image.header.timestamp >= self.current_timestamp: #look up
        #         return
        # try:
        image = self.bridge.imgmsg_to_cv2(image, "rgb8")
        image=cv2.resize(image,(self.image_size,self.image_size))
        # self.image_queue.put(image)
        self.image=image
        # except Exception as e:
        #     print(e)
    def lidar_callback(self,scan):
        # if image.header.timestamp >= self.current_timestamp: #look up
        #         return
        # try:
        
        # print(scan)

        angles = np.arange(
            scan.angle_min,
            scan.angle_max + scan.angle_increment,
            scan.angle_increment
        )
        
        # Select indices where:
        # 1. Angle is either < 90° or > 270° (convert to radians)
        # 2. Range values are greater than minimum range
        selected_indices = np.ravel(np.argwhere(
            np.logical_and(
                np.logical_or(
                    angles < np.pi/2,  # Less than 90 degrees
                    angles > np.pi*(3/2)  # Greater than 270 degrees
                ),
                np.array(scan.ranges) > scan.range_min
            )
        ))
        # print(selected_indices)
        
        # Update ranges using only the selected indices
        ranges=np.array(scan.ranges)
        
    
        self.ranges.append(np.max(ranges[selected_indices]))
        
        # Check for collision based on range threshold
        if self.collision_threshold is None or not self.collision:
            # self.collision_queue.put(False)
            # if not self.co
            # if not self.collision:
                self.collision=False
        else:
            # self.collision_queue.put(
                self.collision=np.min(ranges[selected_indices]) > self.collision_threshold
        # )
        #     print(e)
            
    def imu_calback(self,imu):
        dt=1/self.RATE
        # imu=imu.data
        vec=self.observation_space["vector"].sample()
        # if dt > int(1e3): #look up
        #         return
        # try:
        # print(imu.linear_acceleration)
        w=ros_vector3_to_np_array(imu.angular_velocity)
        accel=ros_vector3_to_np_array(imu.linear_acceleration)
        self.theta=estimate_orientation(accel,w,self.theta,dt)
        accel[2]=0
        # accel[1]=-1*accel[1]
        self.v=self.v+accel*dt
        # print(accel,self.v)
        # print(self.v,np.linalg.norm(self.v))
        self.velocities.append(self.v[0]) # on forward velocity
        self.headings.append(self.theta[-1])
        # breakpoint()
        # print(self.velocities)
        self.speed=np.mean(self.velocities)
        self.current_heading=np.mean(self.headings)
        if self.offsets:
            self.current_heading-=self.offsets[-1]
            self.speed-=self.offsets[0]
 
        # print(self.speed,self.current_heading)
        vec[0]=self.previous_actions[0]/self.max_steer
        vec[1]=self.previous_actions[1]/self.max_steer
        vec[2]=np.clip(self.speed/(self.target_speed+1.0),0,5.1)
        vec[3]=np.clip(self.current_heading/(self.heading+1.0),-5.1,5.1)
        self.vector=np.nan_to_num(vec,nan=0)
        self.vector_queue.put(vec)
        self.dts.append(dt)

        self.distance_travelled+=abs(self.v[0]*self.current_heading)
        # except Exception as e:
        #     print(e)
    def step(self,action:np.ndarray):
        # action[0]=np.clip(action[0]+self.previous_actions[0],-1.0,1.0)
        # action[1]=np.clip(action[1]+self.previous_actions[1],-1.0,1.0)
        self.action=action
        vel_msg = Twist()
        vel_msg.linear.x = np.clip(action[1],0.0,1.0)
        vel_msg.angular.z = action[0]
        # print(vel_msg)
        self.robot_cmd.publish(vel_msg)
        observation=self.get_observation()
        done=self.compute_done()
        reward=self.compute_reward()
        self.rate.sleep()
        return observation, reward, done,False,self.info     
    def compute_done(self):
        self.info=dict()
        # self.collision = self.collision_threshold < self.idle_count
        # if np.linalg.norm(self.v) > 1e-4:
        #     self.idle_count = 0
        # else:
        #     self.idle_count += 1
        self.time_episode += 1
               
        # hero_velocity = self.get_speed(hero)
        # marker_location=sensor_data["goal_heading"][-1][0]
        # wp=core.map.get_waypoint(hero.get_transform().location,project_to_road=False) 
        self.done_dist = False
        ratio_speed=(self.speed/(self.target_speed+1e-8))
        self.done_speed=ratio_speed > 5.1

        # print("Speed Ratio",self.heading,
        # self.current_heading,
        # self.collision_threshold,
        # self.collision
        # ,self.done_speed,
        # self.speed,
        # self.target_speed,
        # (self.speed/(self.target_speed+1e-8)))
        # self.done_falling = hero.get_location().z < -0.5
        # self.diff_lane = 'lane_invasion' in sensor_data.keys()
        # self.collision = 'collision' in sensor_data.keys()
        # self.done_speed=(hero_velocity/(self.target_speed+1e-8)) > 5.0
        done= self.done_dist or self.collision or self.done_speed
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
    def compute_reward(self):
        # hero = core.hero
        heading=self.heading
        imu=self.current_heading


        hero_velocity = self.speed


        speed_factor=np.exp(-(hero_velocity-self.target_speed)**2)

        # Normalize heading error to [-1.0, 1.0] to allow for larger corrections
        heading_factor = np.exp(-(imu-heading)**2)
        # heading_error = np.clip(heading_error, -1.0, 1.0)

        # Calculate smooth action penalty to encourage smoother control inputs
        action_factor = np.exp(-np.sum((self.previous_actions-self.action)**2))


        reward=  2.0*speed_factor + 0.5*action_factor + heading_factor
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
            print(f'Collision Dist={self.distance_travelled:3f} Target_S={self.target_speed:.4f} Vel={hero_velocity:.4f} R={reward:.4f} ')
            reward += -10
        if self.done_speed:
            # print(f'Too fast Smooth={smooth_action:3f} Dist={self.distance_travelled:3f} Ratio={hero_velocity/self.target_speed:.3f} Target_S={self.target_speed:.3f} Vel={hero_velocity:.3f} R={reward:.4f} Err={target_speed_error:.4f} H_Err={heading_error:.4f}')
            print(f'Too fast Dist={self.distance_travelled:3f} Target_S={self.target_speed:.4f} Vel={hero_velocity:.4f} R={reward:.4f} ')
            reward += -10
        # Reward for reaching the target distance
        if self.done_dist:
            # print(f"Max Dist Smooth={smooth_action:3f} Dist={self.distance_travelled:3f}")
            print(f'Max Dist={self.distance_travelled:3f} Target_S={self.target_speed:.4f} Vel={hero_velocity:.4f} R={reward:.4f} ')
            reward += 10

        # Scale the reward to a reasonable range (no need for *10)
        # reward = np.clip(reward, -2.0, 2.0)
        # reward*=10
# 
        # Store the reward for logging or analysis
        self.rewards.append(reward)

        # Update previous actions for smoothness calculation
        self.previous_actions=self.action
        reward=np.nan_to_num(reward,nan=0)
        return reward
        
