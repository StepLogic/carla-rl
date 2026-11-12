# Modified from https://github.com/carla-simulator/rllib-integration/blob/main/dqn_example/dqn_experiment.py

from collections import defaultdict, deque
import math
import os
import random
import numpy as np
from gymnasium.spaces import Box, Dict
import cv2
import carla

import torch
import torchvision.transforms as T
import torch.nn as nn 
import torchvision.models as models

from rlib_integration.base_experiment import BaseExperiment
from rlib_integration.helper import post_process_image, carla_location_to_np_array


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
        self.image_size = 84
        self.vecs = np.zeros(4)
        
        # Vision model setup - MobileNetV3 Small
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading pre-trained MobileNetV3 Small model...")
        
        # Load MobileNetV3 Small and use first 4 blocks
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.model = mobilenet.features[:4]
        
        self.model.to(self.device)
        self.model.eval()
        
        self.image_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((84, 84)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Calculate feature dimension for MobileNetV3 Small features[:4]
        # Output shape: [B, 24, 42, 42] -> flattened: 24 * 42 * 42 = 42336
        self.feature_dim = 10584
        
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
        self.vecs = np.zeros(4)
        
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
        self.relative_goal_loc = None
        self.info = dict()
        
        # HER buffers
        self.current_trajectory = []
        self.current_episode_data = []
        self.rewards = []
        self.prev_distance = None
        
        if self.trajectories is None:
            self._cache_waypoints(core.world)
        core.trajectories = self.trajectories

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
        self.max_curriculum_steps = len(self.trajectories)

    def get_observation_space(self):
        """Returns the observation space"""
        # 2 images (current + goal) * feature_dim
        image_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 * 10584,),
            dtype=np.float32
        )
        
        relative_goal = Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32,
        )
        
        vec_space = Box(
            low=-5.1,
            high=5.1,
            shape=(3 * self.frame_stack,),
            dtype=np.float32,
        )
        
        return Dict({
            "state_obs": image_space,
            "relative_goal_loc": relative_goal,
            "vector": vec_space
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
        while self.relative_goal_loc is None:
            # core.step(np.zeros(2,2))
            sensor_data = core.tick(None)
            vecs = self.get_vec_obs(sensor_data, core)
        obs = {
            "state_obs": images,
            "relative_goal_loc": self.relative_goal_loc,
            "vector": self.vecs
        }
        
        # Store for HER
        if hasattr(self, '_current_image_latent') and hasattr(self, '_current_goal_latent'):
            self.current_episode_data.append({
                'observation': {
                    'state_obs': obs['state_obs'].copy(),
                    'relative_goal_loc': obs['relative_goal_loc'].copy(),
                    'vector': obs['vector'].copy()
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
        
        vec = np.zeros(3)
        vec[0] = self.prev_steer / self.max_steer
        vec[1] = self.prev_throttle / self.max_throttle
        vec[2] = np.clip(self.get_speed(hero) / (self.target_speed + 1e-8), 0.0, 1.0)
        # vec[3] = np.clip(imu / (heading + 1e-8), -1.0, 1.0)
        goal_location = carla_location_to_np_array(core.destination.location)
        hero_location = carla_location_to_np_array(hero.get_transform().location)
        delta = goal_location[:2] - hero_location[:2]
        
        # Include angle to goal
        angle_to_goal = np.arctan2(delta[1], delta[0])
        self.relative_goal_loc = np.array([delta[0], delta[1], angle_to_goal])
        
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
        self.vecs = vecs
        
        # Update state
        self.velocity = self.get_speed(hero)
        self.current_heading = imu
        return vecs

    def get_img_obs(self, sensor_data, core):
        """Get image observations using MobileNetV3 encoder"""
        image = post_process_image(
            sensor_data['rgb'][1], 
            normalized=False,
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
        # breakpoint()
        # cv2.imwrite("goal.jpg",goal)
        # cv2.imwrite("obs.jpg",image)

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
            img_tensor = self.image_transform(img.copy()).unsqueeze(0).to(self.device)
            with torch.no_grad():
                latent = self.model(img_tensor)
                # breakpoint()
            latents.append(latent.squeeze(0).cpu().numpy().flatten())
            del img_tensor, latent
        
        torch.cuda.empty_cache()
        
        # Extract and store individual latents for HER
        image_latent = latents[0]
        goal_latent = latents[1]
        
        self._current_image_latent = image_latent
        self._current_goal_latent = goal_latent
        
        return np.concatenate([image_latent, goal_latent], axis=0)
    
    def get_speed(self, hero):
        """Computes the speed of the hero vehicle in Km/h"""
        vel = hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2)

    def lane_following_get_observation_space(self):
        image_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(10584,),
            dtype=np.float32
        )
        vec_space = Box(
            low=-5.1,
            high=5.1,
            shape=(3 * self.frame_stack,),
            dtype=np.float32,
        )

        return Dict({"state": image_space, "vector": vec_space})
    
    def get_lane_vec_obs(self):
        vecs = np.nan_to_num(self.vecs, nan=1e-8)
        return vecs
    
    def get_lane_observations(self):
        return {"state": self._current_image_latent, "vector": self.vecs}

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
        goal_location = carla_location_to_np_array(core.destination.location)
        hero_location = carla_location_to_np_array(hero.get_transform().location)
        delta = goal_location[:2] - hero_location[:2]
        
        # Include angle to goal
        angle_to_goal = np.arctan2(delta[1], delta[0])
        self.relative_goal_loc = np.array([delta[0], delta[1], angle_to_goal])
        
        distance_to_goal = core.destination.location.distance(hero.get_transform().location)
     
        compute_distance = np.linalg.norm(delta)
        
        assert abs(distance_to_goal - compute_distance) < 5
        
        # Check terminal conditions BEFORE storing trajectory
        wp = core.map.get_waypoint(hero.get_transform().location, project_to_road=False)
        self.done_goal = distance_to_goal < 4.0
        self.done_falling = hero.get_location().z < -0.5
        self.diff_lane = 'lane_invasion' in sensor_data.keys() or wp is None
        self.collision = 'collision' in sensor_data.keys()
        
        # Store trajectory step ONLY if not already terminated
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
            'action': np.array([self.steer, self.throttle]),
            'is_valid': True  # Mark as valid state
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
        """Compute reward for current step with proper shaping"""
        hero = core.hero
        hero_location = carla_location_to_np_array(hero.get_transform().location)
        hero_velocity = self.get_speed(hero)
        goal_location = sensor_data['goal'][1][-2]
        
        # Initialize last location
        if self.last_location is None:
            self.last_location = hero_location
        delta_distance = np.linalg.norm(hero_location-self.last_location)
        # Compute distance to goal
        distance_to_goal = core.destination.location.distance(hero.get_transform().location)
        
        if self.prev_distance is None:
            self.prev_distance = distance_to_goal
        
        # Compute reward with progress and penalties
        reward = 0.0
        self.distance_travelled += delta_distance
        # Progress reward: reward for getting closer to goal
        progress = self.prev_distance - distance_to_goal
        reward += progress * 1.0  # Scale progress reward
        
        # Small time penalty to encourage efficiency
        reward -= 0.1
        
        # Velocity reward: encourage maintaining reasonable speed
        target_velocity = min(self.target_speed, distance_to_goal * 0.5)  # Slow down near goal
        velocity_error = abs(hero_velocity - target_velocity)
        reward -= velocity_error * 0.01
        
        # Goal reached: large positive reward
        if self.done_goal:
            reward += 100.0
        
        # Penalties for failures (applied when truncation occurs)
        if self.collision:
            reward -= 10.0
        if self.diff_lane:
            reward -= 5.0
        if self.done_falling:
            reward -= 10.0
        
        self.rewards.append(reward)
        
        # Store reward in trajectory
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
        Hindsight Experience Replay relabeling with validity checks.
        Creates replay buffer compatible transitions with relabeled goals.
        Only samples from valid, non-terminated states.
        """
        if len(self.current_trajectory) < 2 or len(self.current_episode_data) < 2:
            self.current_trajectory = []
            self.current_episode_data = []
            return
        
        # Filter out invalid states (those with collisions, lane invasions, or falling)
        valid_indices = []
        for i, step in enumerate(self.current_trajectory):
            if not (step['collision'] or step['lane_invasion'] or step['done_falling']):
                valid_indices.append(i)
        
        # Need at least 2 valid states for HER
        if len(valid_indices) < 2:
            self.current_trajectory = []
            self.current_episode_data = []
            return
        
        k = 4  # Number of future states for relabeling
        
        # Only iterate over valid states
        for t in valid_indices[:-1]:  # Exclude last state as it has no next state
            current_step = self.current_trajectory[t]
            current_data = self.current_episode_data[t]
            
            # Find valid next state
            next_t = t + 1
            while next_t < len(self.current_trajectory):
                next_step = self.current_trajectory[next_t]
                if not (next_step['collision'] or next_step['lane_invasion'] or next_step['done_falling']):
                    break
                next_t += 1
            
            # If no valid next state found, skip
            if next_t >= len(self.current_trajectory):
                continue
            
            next_data = self.current_episode_data[next_t]
            
            # Sample future valid states only
            future_valid_indices = [i for i in valid_indices if i > t and i < min(t + k + 1, len(self.current_trajectory))]
            
            if len(future_valid_indices) == 0:
                continue
            
            for future_idx in future_valid_indices:
                future_step = self.current_trajectory[future_idx]
                future_data = self.current_episode_data[future_idx]
                
                # Compute new relative goal with angle to goal
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
                
                # Create relabeled observations with ALL observation space keys
                relabeled_state_obs = np.concatenate([
                    current_data['image_latent'],
                    future_data['image_latent']
                ], axis=0)
                
                relabeled_observation = {
                    'state_obs': relabeled_state_obs,
                    'relative_goal_loc': new_relative_goal,
                    'vector': current_data['vec_obs']  # Include vector observations
                }
                
                relabeled_next_state_obs = np.concatenate([
                    next_data['image_latent'],
                    future_data['image_latent']
                ], axis=0)
                
                relabeled_next_observation = {
                    'state_obs': relabeled_next_state_obs,
                    'relative_goal_loc': next_relative_goal,
                    'vector': next_data['vec_obs']  # Include vector observations
                }
                
                # Recompute reward based on new goal
                relabeled_reward = 0.0
                
                # Progress reward
                prev_dist = distance_to_new_goal
                curr_dist = distance_to_new_goal_next
                progress = prev_dist - curr_dist
                relabeled_reward += progress * 1.0
                
                # Time penalty
                relabeled_reward -= 0.1
                
                # Velocity reward
                target_velocity = min(self.target_speed, distance_to_new_goal * 0.5)
                velocity_error = abs(current_step['velocity'] - target_velocity)
                relabeled_reward -= velocity_error * 0.01
                
                # Check if new goal is reached (within threshold)
                relabeled_done = distance_to_new_goal_next < 4.0
                if relabeled_done:
                    relabeled_reward += 100.0
                
                # Store replay buffer compatible transition
                # Use standard RL buffer format: observations, actions, rewards, next_observations, dones
                her_transition = {
                    'observations': relabeled_observation,
                    'actions': current_step['action'],
                    'rewards': relabeled_reward,
                    'next_observations': relabeled_next_observation,
                    "masks":1 - int(relabeled_done),
                    'dones': relabeled_done,
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
        """Sample HER transitions in replay buffer format"""
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
            "timeout": 20.0,
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
            "type": JaxGoalConditionedExperiment,
            "hero": {
                "blueprint": "vehicle.mercedes.coupe_2020",
                "is_goal_env": True,
                "sensors": {
                    "collision": {
                        "type": "sensor.other.collision"
                    },
                    "rgb": {
                        "type": "sensor.camera.rgb",
                        "image_size_x": 448,
                        "image_size_y": 448,
                        "transform": "1.9, 0.0, 1.7, 0.0, -15.0, 0.0"
                    },
                    "lane_invasion": {
                        "type": "sensor.other.lane_invasion"
                    },
                    "imu": {
                        "type": "sensor.other.imu"
                    },
                    "goal": {
                        "type": "sensor.goal",
                        "image_size_x": 112,
                        "image_size_y": 112,
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