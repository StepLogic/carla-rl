#!/usr/bin/env python3
import random
import numpy as np
import rospy
import cv2
import time
import queue
import keyboard
from collections import deque
from datetime import datetime
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation
from gymnasium import Env
from gymnasium.spaces import Dict, Box
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# Constants
IMAGE_TOPIC = "/camera/image_raw"
IMU_TOPIC = "/imu"
ROBOT_CMD_TOPIC = "/cmd_vel"
WHEEL_ODOMETRY_TOPIC = "/wheel_odom_with_covariance"
RATE = 120

class LeoEnv(Env):
    def __init__(self,target_heading=None):
        # Initialize ROS subscribers and publishers
        self.bridge = CvBridge()
        self.image_queue = queue.Queue()
        self.vector_queue = queue.Queue()
        self.collision_queue = queue.Queue()
        
        # Set up subscribers
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback)
        self.wheel_sub = rospy.Subscriber(WHEEL_ODOMETRY_TOPIC, Odometry, self.wheel_callback)
        self.imu_sub = rospy.Subscriber(IMU_TOPIC, Imu, self.imu_callback)
        self.robot_cmd = rospy.Publisher(ROBOT_CMD_TOPIC, Twist, queue_size=10)
        
        # Robot state variables
        self.current_velocity = 0
        self.current_heading = 0
        self.target_speed = 0.35
        self.max_steer = 1.0
        self.image_size = 64
        self.previous_actions = np.zeros(2)  # [steer, throttle]
        if not target_heading is None:
            self.heading = target_heading
        else:
            self.heading = 0
        self.collision_threshold = None
        self.accelerations = deque(maxlen=100)
        self.time_episode = 0
        self.distance_travelled = 0
        self.speed = 0.0
        
        # Current observation state
        self.action = None
        self.collision = None
        self.done_fast = None
        self.image = None
        self.vector = None
        self.orientation = None
        
        # Environment info
        self.info = dict()
        self.rewards = []
        self.offsets = [0, 0]
        self.ranges = []
        self.goal_done = False
        
        # Define observation and action spaces
        image_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.image_size, self.image_size, 3),
            dtype=np.float32,
        )
        
        vec_space = Box(
            low=-5.1,
            high=5.1,
            shape=(4,),
            dtype=np.float32,
        )
        
        self.observation_space = Dict({"pixels": image_space, "vector": vec_space})
        self.action_space = Box(
            low=np.array([-self.max_steer, -1.0]),  # [steering, throttle/brake]
            high=np.array([self.max_steer, 1.0]),
            dtype=np.float32
        )
        
        # Set up ROS rate
        self.rate = rospy.Rate(RATE)
        self.RATE = RATE
        
        # Register keyboard callbacks
        keyboard.on_press_key("c", self.set_collision_callback)
        keyboard.on_press_key("g", self.set_goal_reached_callback)
    
    def set_collision_callback(self, *args):
        print("Collision detected by user")
        self.collision = True
    
    def set_goal_reached_callback(self, *args):
        print("Goal reached signaled by user")
        self.goal_done = True
    
    def get_observation(self, reset=False):
        """Get current observation from robot sensors"""
        vec = self.observation_space["vector"].sample()
        
        if self.orientation:
            # try:
                quat = np.array([self.orientation.x, self.orientation.y, 
                                 self.orientation.z, self.orientation.w])
                payload = Rotation.from_quat(quat)
                
                euler = payload.as_euler('xyz', degrees=False)
                
                self.current_heading = euler[2]
                # print(self.current_heading)
                # Normalize and clip the vector components
                vec[0] = self.previous_actions[0] / self.max_steer
                vec[1] = self.previous_actions[1] / self.max_steer
                vec[2] = np.clip(self.speed / (self.target_speed + 1e-8), 0, 5.1)
                vec[3] = np.clip(self.current_heading / (self.heading + 1e-10), -5.1, 5.1)
            # except:
            #     # Fall back if rotation conversion fails
            #     # print("quaternion",self.)
            #     pass
        # print("",self.orientation)
        self.vector = np.nan_to_num(vec, nan=0)
        self.vector_queue.put(vec)
        return dict(pixels=self.image, vector=self.vector)
    
    def reset(self, *args, **kwargs):
        """Reset the environment to initial state"""
        # Reset state variables
        self.velocities = []
        self.rewards = []
        self.action = None
        self.collision = None
        self.done_fast = None
        self.image = None
        self.vector = None
        self.ranges = [0.2]
        self.offsets = None
        self.goal_done = False
        self.orientation = None
        
        # Pause to allow manual robot reset
        time.sleep(10.0)
        print("Reset Robot Please!!!!!!")
        
        # Clear queues and reset measurements
        self.image_queue.queue.clear()
        self.vector_queue.queue.clear()
        time.sleep(1.0)
        
        # Initialize episode state
        self.time_episode = 0
        self.distance_travelled = 0
        # self.current_heading = 0.0
        self.action = np.zeros(2)
        self.info = dict()
        # self.speed = 0.0
        self.heading = random.choice([1e-8,np.pi/2,np.pi/4,np.pi/6,np.pi/8])
        self.target_speed = 0.35
        
        # Get initial observation
        observation = self.get_observation(reset=True)
        return observation, self.info
    
    def image_callback(self, image):
        """Process incoming camera images"""
        try:
            img = self.bridge.imgmsg_to_cv2(image, "rgb8")
            img = cv2.resize(img, (self.image_size, self.image_size))
            self.image = img/255
        except Exception as e:
            print(f"Image processing error: {e}")
    
    def wheel_callback(self, wheel):
        """Process wheel odometry data"""
        self.speed = wheel.twist.twist.linear.x
    
    def imu_callback(self, imu):
        """Process IMU data"""
        self.orientation = imu.orientation
    
    def step(self, action):
        """Take a step in the environment with the given action"""
        self.action = action
        
        # Create and publish velocity command
        vel_msg = Twist()
        vel_msg.linear.x = action[1]
        vel_msg.angular.z = action[0]
        self.robot_cmd.publish(vel_msg)
        
        # Get updated observation and compute reward/done
        observation = self.get_observation()
        done = self.compute_done()
        reward = self.compute_reward()
        
        # Sleep to maintain rate
        self.rate.sleep()
        return observation, reward, done, False, self.info
    
    def compute_done(self):
        """Determine if episode is done"""
        self.info = dict()
        self.time_episode += 1
        
        # Check terminal conditions
        self.done_dist = False
        ratio_speed = (self.speed/(self.target_speed+1e-8))
        self.done_speed = ratio_speed > 1.0
        
        # Combine terminal conditions
        done = self.done_dist or self.collision or self.done_speed or self.goal_done
        
        # Update info dictionary
        self.info.update(dict(
            is_success=self.goal_done,
            distance_completed=self.distance_travelled,
            max_reward=0,
            min_reward=0,
            mean_reward=0
        ))
        
        if len(self.rewards) > 0:
            self.info.update(dict(
                max_reward=np.max(self.rewards),
                min_reward=np.min(self.rewards),
                mean_reward=np.mean(self.rewards)
            ))
        
        return done
    
    def compute_reward(self):
        """Compute reward based on current state"""
        # Define reward weights
        WEIGHTS = {
            'speed': 0.6,
            'direction': 0.2,
            'action_smoothness': 0.05,
            'time_penalty': 0.09,
            'collision_penalty': 2.0,
            'lane_departure_penalty': 2.0,
            'goal_reached_bonus': 2.0
        }
        
        # Calculate reward components
        heading_diff = abs(self.heading - self.current_heading)
        direction_factor = np.cos(heading_diff)
        
        speed_factor = self.speed / self.target_speed
        normalized_speed = np.clip(speed_factor, 0, 1.0)
        speed_reward = normalized_speed
        
        direction_reward = max(direction_factor, 0)  # Only reward positive alignment
        action_smoothness_reward = -abs(self.action[0])  # Penalize large steering actions
        
        # Combine reward components
        reward = 0.0
        
        if speed_factor < 1.0:
            reward += WEIGHTS['speed'] * speed_reward
        
        reward += WEIGHTS['direction'] * direction_reward
        reward += WEIGHTS['action_smoothness'] * action_smoothness_reward
        reward -= WEIGHTS['time_penalty']
        if self.goal_done:
            reward += WEIGHTS['goal_reached_bonus']
        if self.collision:
            reward -= WEIGHTS['collision_penalty']
        # Ensure reward is numerically stable
        reward = np.nan_to_num(reward, nan=0.0)
        
        # Store reward for statistics
        self.rewards.append(reward)
        
        # Update previous actions for next step
        self.previous_actions = self.action
        
        return reward