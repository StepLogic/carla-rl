# Modified from https://github.com/carla-simulator/rllib-integration/blob/main/dqn_example/dqn_experiment.py

from collections import defaultdict
import math
import os
import random
import cv2
import numpy as np
from gymnasium.spaces import Box, Dict
import torch
import torchvision.transforms as T
import carla
import torch.nn as nn
import torchvision.models as models
from scipy.spatial.transform import Rotation as R
from rlib_integration.base_experiment import BaseExperiment
from rlib_integration.helper import post_process_image, carla_location_to_np_array


def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    return ((angle + np.pi) % (2 * np.pi)) - np.pi

class RLDataAugmentation:
    """
    Data augmentation for RL with MobileNetV3.
    Separate transforms for training and evaluation.
    """
    def __init__(self, image_size=84, aug_strength='medium', training=True):
        self.image_size = image_size
        self.training = training
        
        # Base transforms (always applied)
        base_transforms = [
            T.ToPILImage(),
        ]
        
        # Augmentation transforms (only during training)
        if training and aug_strength != 'none':
            if aug_strength == 'light':
                aug_transforms = [
                    T.Resize((image_size, image_size)),
                    T.ColorJitter(brightness=0.1, contrast=0.1),
                ]
            elif aug_strength == 'medium':
                aug_transforms = [
                    T.RandomResizedCrop(
                        image_size,
                        scale=(0.92, 1.0),  # Slight crop
                        ratio=(0.95, 1.05)
                    ),
                    T.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.15,
                        hue=0.03
                    ),
                ]
            elif aug_strength == 'strong':
                aug_transforms = [
                    T.RandomResizedCrop(
                        image_size,
                        scale=(0.85, 1.0),
                        ratio=(0.9, 1.1)
                    ),
                    T.ColorJitter(
                        brightness=0.3,
                        contrast=0.3,
                        saturation=0.2,
                        hue=0.05
                    ),
                    T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),
                ]
            else:
                aug_transforms = [T.Resize((image_size, image_size))]
        else:
            # No augmentation for evaluation
            aug_transforms = [T.Resize((image_size, image_size))]
        
        # Final transforms (always applied)
        final_transforms = [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        self.transform = T.Compose(base_transforms + aug_transforms + final_transforms)
    
    def __call__(self, image):
        return self.transform(image)
    
class STBL3Experiment(BaseExperiment):
    def __init__(self, config={}):
        super().__init__(config)

        # Configuration
        self.frame_stack = self.config["others"]["framestack"]
        self.max_time_idle = self.config["others"]["max_time_idle"]
        self.max_dist = self.config["others"]["max_dist"]
        self.target_speed = self.config["others"]["target_speed"]
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]
    
        # Control variables
        self.max_steer = 1.0
        self.max_throttle = 1.0
        self.max_angle_deviation = np.pi
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.steer = 0.0
        self.throttle = 0.0
        self.last_action = None
        
        # State variables
        self.current_heading = 0.0
        self.velocity = 0.0
        self.dot = 0.0
        self.distance_travelled = 0.0
        self.distance_completed = 0.0
        self.last_distance = None
        self.last_location = None
        self.last_velocity = 0
        
        # Episode variables
        self.trajectories = None
        self.info = dict()
        self.rewards = []
        self.episode_num = 1
        self.max_episodes = 10000
        self.explore_mode = False
        self.passed_chckp = False
        self.eval_mode = False
        self.mask_goal= False
        
        # Observation variables
        self.image_size = 84
        self.obs_img = None
        self.prev_vec_0 = None
        self.prev_vec_1 = None
        self.prev_vec_2 = None
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None
        self.masking_curriculum_step = 0  # Track training progress
        self.goal_mask_probability = 0.0  # Start at 0%
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load MobileNetV3 Small model
        print("Loading pre-trained MobileNetV3-Small model...")
        self.model = models.mobilenet_v3_small(pretrained=True)
        self.model = self.model.features[:4]  # Use first 4 blocks
        self.model.to(self.device)
        self.model.eval()
        
        # Image transform
        self.image_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((84, 84)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # self.image_transform = RLDataAugmentation()

    def _cache_waypoints(self, world) -> None:
        """Cache waypoints for trajectory planning."""
        env_map = world.get_map()
        waypoints = env_map.generate_waypoints(distance=2)
        trajectories = defaultdict(list)
        for wpt in waypoints:
            trajectories[f"{wpt.road_id}-{wpt.lane_id}"].append(wpt)
        self.trajectories = sorted(
            [traj for traj in trajectories.values() if len(traj) > 3],
            key=len,
            reverse=True
        )

    def reset(self, env):
        """Called at the beginning and each time the simulation is reset."""
        # Ending variables
        self.time_idle = 0
        self.time_episode = 0
        self.done_time_idle = False
        self.done_falling = False
        self.done_dist = False
        self.done_wrong_direction = False
        self.done_speed = False
        self.checkpoint_count=0

        # Hero variables
        self.last_location = None
        self.last_velocity = 0
        self.distance_travelled = 0.0
        self.distance_completed = 0.0
        self.last_heading = None
        self.last_distance = None

        # Sensor stack
        self.prev_vec_0 = None
        self.prev_vec_1 = None
        self.prev_vec_2 = None
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None
        self.obs_img = None

        # Control variables
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.steer = 0.0
        self.throttle = 0.0
        self.velocity = 0.0
        self.current_heading = 0.0
        self.dot = 0.0
        self.heading_vec_2d = np.zeros((2,))
        # Episode variables
        # self.target_speed = 6.0
        self.explore_mode = random.randint(0, 1) == 1
        self.passed_chckp = False
        self.info = dict()
        self.rewards = []
        
        if self.trajectories is None:
            self._cache_waypoints(env.world)

    def get_observation_space(self):
        """Define observation space with image features and vector state."""
        # MobileNetV3-Small features[:4] flattened output
        image_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(2904,),
            dtype=np.float32
        )

        # Vector space: [steering, throttle, speed, heading_x, heading_y] * frame_stack
        vec_space = Box(
            low=-5.1,
            high=5.1,
            shape=(7 * self.frame_stack,),
            dtype=np.float32,
        )

        return Dict({"state": image_space, "vector": vec_space})

    def get_action_space(self):
        """Returns the continuous action space for steering and throttle."""
        return Box(
            low=np.array([-self.max_steer, 0.0]),
            high=np.array([self.max_steer, 1.0]),
            dtype=np.float32
        )

    def compute_action(self, action,core):
        """Convert continuous actions to CARLA vehicle controls."""
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
        
        self.steer = vehicle_control.steer
        self.throttle = vehicle_control.throttle
        self.last_action = vehicle_control
        
        return vehicle_control

    def get_observation(self, sensor_data, core):
        """Process sensor data into observation."""
        vecs = self.get_vec_obs(sensor_data, core)
        images = self.get_img_obs(sensor_data, core)
        return {"state": images, "vector": vecs}, self.info

    def get_vec_obs(self, sensor_data, core):
        """Extract vector observations from sensors."""
        hero = core.hero
        # self.mask_goal =random.randint(0,1) == 1
        vec = np.zeros(7)
        vec[0] = self.steer / self.max_steer
        vec[1] = self.throttle / self.max_throttle
        
        # Calculate direction to goal
        hero_transform = hero.get_transform()
        hero_location = hero_transform.location
        goal_location = core.destination.location
        
        # Vector to goal in world frame
        heading_vec_world = carla_location_to_np_array(goal_location) - carla_location_to_np_array(hero_location)
        self.heading_vec_2d=heading_vec_world/(np.linalg.norm(heading_vec_world) + 1e-8)
        # Transform goal vector to agent's local frame
        world_to_agent_matrix = np.array(hero_transform.get_inverse_matrix())
        heading_vec_local = world_to_agent_matrix[:3, :3] @ heading_vec_world
        
        # In agent's local frame:
        # - Forward is +X axis: [1, 0, 0]
        # - Left is +Y axis: [0, 1, 0]
        # - Up is +Z axis: [0, 0, 1]
        
        # Normalize heading vector (2D for driving)
        heading_vec_2d = heading_vec_local[:2]  # Only X and Y (ignore Z)
        heading_vec_2d = heading_vec_2d / (np.linalg.norm(heading_vec_2d) + 1e-8)
        
        # Forward direction in local frame is simply [1, 0]
        forward_vec_2d = np.array([1.0, 0.0])
        
        # Dot product for alignment: 1 = forward, -1 = backward
        dot_product = np.dot(heading_vec_2d, forward_vec_2d)
        self.dot = dot_product
        
        # Cross product to determine left/right
        # In 2D: cross = x1*y2 - y1*x2
        # Positive = goal is to the LEFT
        # Negative = goal is to the RIGHT
        cross_product = heading_vec_2d[0] * forward_vec_2d[1] - heading_vec_2d[1] * forward_vec_2d[0]
        # Simplified: cross_product = -heading_vec_2d[1] (since forward is [1,0])
        
        # Speed normalized to target
        if self.target_speed == 0:
            vec[2] = 0.0 if self.get_speed(hero) < 0.1 else 1.0
        else:
            vec[2] = np.clip(self.get_speed(hero) / self.target_speed, 0.0, 1.0)
        
        # Heading vector components in agent's local frame
        vec[3] = heading_vec_2d[0]  # Forward/backward component
        vec[4] = heading_vec_2d[1]  # Left/right component
        vec[5] = dot_product        # Alignment score
        vec[6] = 1.0 
        # vec[6] = cross_product
        wp = core.map.get_waypoint(hero.get_transform().location, project_to_road=False)
        # if wp is not None and wp.is_junction:
        self.goal_mask_probability = min(0.5, self.masking_curriculum_step / 500000)
        mask_goal = random.random() < self.goal_mask_probability
        if random.random() < 0.3:
            vec[3] = -1.0
            vec[4] = -1.0
            vec[5] = -1.0
            vec[6] = 0.0

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

        self.prev_vec_2 = self.prev_vec_1
        self.prev_vec_1 = self.prev_vec_0
        self.prev_vec_0 = vec
        
        self.velocity = self.get_speed(hero)
        vecs = np.nan_to_num(vecs, nan=0.0)
        
        return vecs

    def get_img_obs(self, sensor_data, core):
        """Extract image observations and encode with MobileNetV3."""
        image = post_process_image(
            sensor_data['rgb'][1],
            normalized=False,
            crop=False,
            grayscale=False,
            image_size=self.image_size
        )

        # cv2.imwrite("obs.jpg",image)

        self.obs_img = image
        
        # Frame stacking (kept for consistency, though only using current frame)
        if self.prev_image_0 is None:
            self.prev_image_0 = image
            self.prev_image_1 = image
            self.prev_image_2 = image

        self.prev_image_2 = self.prev_image_1
        self.prev_image_1 = self.prev_image_0
        self.prev_image_0 = image
        
        # Transform and encode with model
        img_tensor = self.image_transform(image.copy()).unsqueeze(0).to(self.device)

        with torch.no_grad():
            latent_vector = self.model(img_tensor)
        
        return latent_vector.squeeze(0).cpu().numpy().flatten()
    
    def get_speed(self, hero):
        """Computes the speed of the hero vehicle in Km/h."""
        vel = hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2)

    def get_done_status(self, sensor_data, core):
        """Returns whether or not the episode should end."""
        self.info = dict()
        hero = core.hero
        hero_transform = hero.get_transform()
        hero_location = hero_transform.location
        # Check idle time
        self.done_time_idle = self.max_time_idle < self.time_idle
        if self.get_speed(hero) > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1
        self.time_episode += 1
            
        # Get current waypoint
        wp = core.map.get_waypoint(hero.get_transform().location, project_to_road=False)
        
        # Check failure conditions
        self.done_falling = False
        # 'lane_invasion' in sensor_data.keys() or
        # self.diff_lane =  wp is None
        self.diff_lane= 'lane_invasion' in sensor_data.keys() or wp is None
        self.collision = 'collision' in sensor_data.keys()
        self.done_dist = self.distance_completed>=200
        # Check distance to goal and resample if too close
        distance_to_goal = core.destination.location.distance(hero.get_transform().location)


        # Check success condition
        # self.done_dist = self.distance_travelled >= max(50,50*round(self.time_episode/4e4))
        if distance_to_goal < 2:  # Resample when within 2m of goal (for 4-8m range)
            core.resample_goal()  # Uses default 4-8m range
            self.passed_chckp=True
        else:
            self.passed_chckp = False

        # Check wrong direction at junction
        j_lax=0
        self.done_wrong_direction = False
        if wp is not None and wp.is_junction and not self.eval_mode: 
            self.done_wrong_direction = self.dot < -0.1

            # self.passed_chckp = not self.done_wrong_direction
            if self.done_wrong_direction:
                print(f"Wrong direction at junction: alignment={self.dot:.3f} ({np.rad2deg(np.arccos(np.clip(self.dot, -1, 1))):.1f}°)")
        
        # Speed-based termination (currently disabled)
        self.done_speed = False
        wp = core.map.get_waypoint(hero_location, project_to_road=True)
    
        # Calculate Cross Track Error (distance from lane center)
        # This acts as our "Lane Invasion" detector
        dist_to_center = hero_location.distance(wp.transform.location)
        lane_width = wp.lane_width
        
        # Threshold check: If car's center moves beyond half lane width (+ buffer for car width)
        # Standard car width is ~2.0m, so we allow some deviation.
        # print(dist_to_center)
        j_lax=0
        if wp.is_junction:
                j_lax=0.5


        is_lane_invaded = dist_to_center > (1.0 + j_lax)
        # self.diff_lane = self.diff_lane or  is_lane_invaded
        # Combine termination conditions
        done = (
            self.done_falling or
            self.done_dist or
            self.diff_lane or
            self.collision or
            self.done_speed 
            or self.done_wrong_direction
        )
        
        # Update info
        self.info.update(dict(
            is_success=self.done_dist,
            distance_travelled=self.distance_completed,
            goal_directed_distance=self.distance_completed,
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
        
        if done:
            self.info.update(
                is_success=self.done_dist,
                distance_completed=self.distance_travelled,
                mean_reward=np.mean(self.rewards) if len(self.rewards) > 0 else 0
            )
        
        return done

    def compute_reward(self, sensor_data, core):
        """
        Compute the reward for RL training based on vehicle performance.
        
        Detects lane invasion via geometric waypoint analysis to bypass 
        CARLA sensor failures at junctions.
        """
        # Reward weights
        WEIGHTS = {
            'collision_penalty': 5.0,
            'lane_departure_penalty': 5.0,
            'goal_reached_bonus': 25.0,
            'wrong_junction_penalty': 5.0,
            'base_penalty': 0.1,
            'smoothness_floor': 0.1,
            'cte_penalty_weight': 0.5  # Penalty for drifting from center
        }
        
        # 1. Get vehicle state
        hero = core.hero
        hero_transform = hero.get_transform()
        hero_location = hero_transform.location
        hero_velocity = self.get_speed(hero)
        
        # 2. GEOMETRIC LANE DETECTION (Junction Robust)
        # Get the nearest waypoint on the road center
        wp = core.map.get_waypoint(hero_location, project_to_road=True)

        # 3. PROGRESS TRACKING
        if self.last_distance is None:
            self.last_distance = core.destination.location.distance(hero_location)
        
        if self.last_location is None:
            self.last_location = hero_location
            displacement_vector = np.array([0.0, 0.0])
            delta_distance = 0.0
        else:
            # Displacement since last frame
            displacement_vector = np.array([
                hero_location.x - self.last_location.x,
                hero_location.y - self.last_location.y
            ])
            
            # Progress toward the final goal
            current_distance = core.destination.location.distance(hero_location)
            delta_distance = self.last_distance - current_distance
            self.last_distance = current_distance

        # Calculate raw distance traveled
        raw_distance = float(np.linalg.norm(displacement_vector))
        
        # 4. PROJECTED PROGRESS (Road-aligned)
        projected_distance = 0.0
        if wp is not None and raw_distance > 0:
            forward_vector = wp.transform.get_forward_vector()
            forward_2d = np.array([forward_vector.x, forward_vector.y])
            # Project movement onto the lane's forward direction
            projected_distance = float(np.dot(displacement_vector, forward_2d))

        # Update cumulative distance
        self.distance_travelled += projected_distance
        self.distance_completed += max(0, projected_distance) 
        self.last_location = hero_location
        
        # 5. REWARD CALCULATION
        speed_factor = hero_velocity / self.target_speed if self.target_speed > 0 else 0
        steer_change = abs(self.steer - self.prev_steer)
        throttle_change = abs(self.throttle - self.prev_throttle)
        
        # Base reward and smoothness penalty
        reward = -WEIGHTS['base_penalty'] - (steer_change * 0.5)
        
        # Progress Reward: Only when moving at a productive speed
        if 0.1 < speed_factor < 1.05:
            reward += projected_distance + delta_distance
        
        # 6. PENALTIES & BONUSES
        # Continuous penalty for being off-center (encourages staying in lane)
        # reward -= (dist_to_center * WEIGHTS['cte_penalty_weight'])

        if self.passed_chckp:
            reward += 1.0
            
        if self.done_wrong_direction:
            reward -= WEIGHTS['wrong_junction_penalty']
            
        if self.collision or self.done_speed:
            reward -= WEIGHTS['collision_penalty']
            
        if self.diff_lane:
            # This is triggered by our waypoint geometric check
            print("Lane departure")
            reward -= WEIGHTS['lane_departure_penalty']
            
        if self.done_dist:
            reward += WEIGHTS['goal_reached_bonus']

        # 7. LOGGING & CLEANUP
        reward = float(np.nan_to_num(reward, nan=0.0))
        
        self.info.update({
            'speed_factor': np.clip(speed_factor, 0, 1.0),
            # 'dist_to_center': dist_to_center,
            'projected_distance': projected_distance,
            'total_reward': reward
        })
        
        self.rewards.append(reward)
        self.prev_steer = self.steer
        self.prev_throttle = self.throttle
        
        return reward
# Configuration
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
            "town": "Town01"
        },
        "experiment": {
            "type": STBL3Experiment,
            "hero": {
                "blueprint": ["vehicle.micro.microlino"],
                "sensors": {
                    "collision": {
                        "type": "sensor.other.collision"
                    },
                    "rgb": {
                        "type": "sensor.camera.rgb",
                        "image_size_x": 748,
                        "image_size_y": 348,
                        "fov":110,
                        "transform": "1.9, 0.0, 1.7, 0.0, -15.0, 0.0"
                    },
                    "lane_invasion": {
                        "type": "sensor.other.lane_invasion"
                    },
                    "imu": {
                        "type": "sensor.other.imu"
                    },
                    "goal_heading": {
                        "type": "sensor.goal.heading"
                    },
                },
            },
            "background_activity": {
                "n_vehicles": 0,
                "n_walkers": 0,
                "tm_hybrid_mode": True
            },
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