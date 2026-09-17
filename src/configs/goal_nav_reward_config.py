
# Goal-Directed Navigation with Collision and Lane Invasion Terminal Conditions

import math
import numpy as np
from gymnasium.spaces import Box, Dict
import carla
from rlib_integration.base_experiment import BaseExperiment
from rlib_integration.helper import post_process_image, carla_location_to_np_array


class GoalNavRewardExperiment(BaseExperiment):
    """
    Goal-Directed Navigation Experiment with:
    - Progress-based reward (distance to goal reduction)
    - Collision as terminal condition
    - Lane invasion as terminal condition
    - Sparse goal reached bonus
    """

    def __init__(self, config={}, is_rgb=False, image_size=64):
        super().__init__(config)
        self.frame_stack = self.config["others"]["framestack"]
        self.max_time_idle = self.config["others"]["max_time_idle"]
        self.target_speed = self.config["others"]["target_speed"]
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]
        self.last_action = None
        self.max_steer = 1.0
        self.max_throttle = 1.0
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.trajectories = None
        self.info = dict()
        self.rewards = []
        self.image_size = image_size
        self.is_rgb = is_rgb
        self.eval_mode = False
        self.map_mode = False
        self.wp = None

        # Goal-directed specific
        self.goal_threshold = 8.0  # meters to consider goal reached
        self.prev_distance_to_goal = None
        self.distance_travelled = 0.0
        self.time_idle = 0
        self.time_episode = 0
        self.origin = None
        self.destination = None
        self.lane_invasion_count = 0

    def reset(self, *arg, **kwargs):
        """Reset tracking variables"""
        self.time_idle = 0
        self.time_episode = 0
        self.last_location = None
        self.prev_distance_to_goal = None
        self.distance_travelled = 0.0
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.steer = 0.0
        self.throttle = 0.0
        self.rewards = []
        self.lane_invasion_count = 0
        self.done_wrong_direction = False
        self.dot = 0.0

        # Sensor stack reset
        self.prev_vec_0 = None
        self.prev_vec_1 = None
        self.prev_vec_2 = None
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None

    def get_observation_space(self):
        image_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.image_size, self.image_size, 3,),
            dtype=np.float32,
        )

        vec_space = Box(
            low=-5.1,
            high=5.1,
            shape=(4 * self.frame_stack,),
            dtype=np.float32,
        )

        return Dict({"pixels": image_space, "vector": vec_space, "goal": image_space})

    def get_action_space(self):
        """Returns the continuous action space for steering and throttle"""
        return Box(
            low=np.array([-self.max_steer, 0.4]),  # Throttle in [-1, 1], negative = brake
            high=np.array([self.max_steer, 1.0]),
            dtype=np.float32
        )

    def compute_action(self, action, core=None):
        """Convert continuous actions to CARLA vehicle controls"""
        steer, throttle_brake = action

        action = carla.VehicleControl()
        action.steer = float(np.clip(steer, -self.max_steer, self.max_steer))
        self.steer = action.steer

        if throttle_brake >= 0:
            action.throttle = float(np.clip(throttle_brake, 0.0, self.max_throttle))
            action.brake = 0.0
        else:
            action.throttle = 0.0
            action.brake = float(np.clip(-throttle_brake, 0.0, 1.0))

        self.throttle = action.throttle
        self.last_action = action
        action.reverse = False
        action.hand_brake = False
        return action

    def get_observation(self, sensor_data, core):
        """Process sensor data to get observation"""
        vecs = self.get_vec_obs(sensor_data, core)
        images, goal = self.get_img_obs(sensor_data, core)

        # For HER: achieved_goal is current observation (what we've achieved)
        # This will be used to relabel goals in hindsight
        # Extract first frame and ensure CHW format (3, H, W)
        if images.shape[2] >= 3:  # HWC format with stacked frames
            # Extract first 3 channels (most recent frame)
            first_frame = images[:, :, :3]
            # Convert to CHW format
            self._last_observation = np.transpose(first_frame, (2, 0, 1))
        else:
            self._last_observation = images.copy()

        return {"pixels": images, "vector": vecs, "goal": goal}, self.info

    def get_vec_obs(self, sensor_data, core):
        """Get vector observations"""
        heading = sensor_data["goal_heading"][-1][-1]
        imu = sensor_data["imu"][-1][-1]
        vec = np.zeros(4)
        vec[0] = self.prev_steer / self.max_steer
        vec[1] = self.prev_throttle / self.max_throttle
        hero = core.hero
        vec[2] = np.clip(self.get_speed(hero) / self.target_speed, 0.0, 1.0)

        # Direction to goal
        current_forward_vector = carla_location_to_np_array(core.hero.get_transform().get_forward_vector())
        if isinstance(core.destination, carla.Transform):
            correct_forward_vector = carla_location_to_np_array(core.destination.get_forward_vector())
        else:
            correct_forward_vector = carla_location_to_np_array(core.destination.transform.get_forward_vector())

        self.delta_angle = np.dot(current_forward_vector, correct_forward_vector)
        vec[3] = self.delta_angle

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
        """Get image observations"""
        image = post_process_image(sensor_data['rgb'][1], normalized=False, crop=False, grayscale=False, image_size=self.image_size)
        goal = post_process_image(sensor_data['goal'][1][0], normalized=False, crop=False, grayscale=False, image_size=self.image_size)

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
        """Compute speed in Km/h"""
        vel = hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2)

    def get_done_status(self, sensor_data, core):
        """Check terminal conditions"""
        hero = core.hero
        self.info = dict()

        # Update idle time
        # self.done_time_idle = self.max_time_idle < self.time_idle
        if self.get_speed(hero) > 4.0:
            self.time_idle = 0
        else:
            self.time_idle += 1
        self.time_episode += 1

        # Check waypoint
        wp = core.map.get_waypoint(hero.get_transform().location, project_to_road=False)

        # Terminal conditions
        self.collision = 'collision' in sensor_data.keys()
        if 'lane_invasion' in sensor_data.keys():
            self.lane_invasion_count += 1
        self.diff_lane = self.lane_invasion_count > 0 or wp is None
        self.done_falling = hero.get_location().z < -0.5

        # Wrong direction check at junctions (from train_env_config_v4)
        self.done_wrong_direction = False
        if wp is not None and wp.is_junction and not self.eval_mode:
            self.done_wrong_direction = self.dot < -0.1
            if self.done_wrong_direction:
                print(f"Wrong direction at junction: alignment={self.dot:.3f} ({np.rad2deg(np.arccos(np.clip(self.dot, -1, 1))):.1f}°)")

        goal_location = core.destination.location

        hero_location = hero.get_transform().location
        distance_to_goal = hero_location.distance(goal_location)
        self.done_goal = distance_to_goal < self.goal_threshold
        # print(distance_to_goal)
        # Terminal condition: driving 200 meters (distance goal)
        self.done_distance_goal = self.distance_travelled >= 200.0
    
        # Done if any terminal condition met
        done = self.collision or self.diff_lane or self.done_falling or self.done_goal or self.done_distance_goal or self.done_wrong_direction or self.time_idle>self.max_time_idle

        # Get current observation as achieved goal for HER
        achieved_goal = self._last_observation if hasattr(self, '_last_observation') else None

        self.info.update(dict(
            is_success=self.done_goal or self.done_distance_goal,
            distance_to_goal=distance_to_goal,
            collision=self.collision,
            lane_invasion=self.diff_lane,
            distance_completed=self.distance_travelled,
            achieved_goal=achieved_goal,  # For HER
            done_distance_goal=self.done_distance_goal,
            wrong_direction_at_junction=self.done_wrong_direction,
        ))

        return done

    def compute_reward(self, sensor_data, core):
        """
        Compute reward based on train_env_config_v4.py:
        - Rewards progress along road direction (projected onto waypoint forward vector)
        - Penalizes collisions, lane departures, and erratic driving
        """
        # Reward weights
        WEIGHTS = {
            'collision_penalty': 3.0,
            'lane_departure_penalty': 5.0,
            'goal_reached_bonus': 5.0,
            'wrong_junction_penalty': 3.0,
            'base_penalty': 0.1,
            'smoothness_floor': 0.1,
        }

        # Get vehicle state
        hero = core.hero
        hero_location = hero.get_location()
        hero_velocity = self.get_speed(hero)

        # Get waypoint for road direction
        wp = core.map.get_waypoint(
            hero.get_transform().location,
            project_to_road=True
        )

        # Initialize tracking variables on first call
        if self.prev_distance_to_goal is None:
            self.prev_distance_to_goal = core.destination.location.distance(hero_location)

        if self.last_location is None:
            self.last_location = hero_location
            displacement_vector = np.array([0.0, 0.0])
            delta_distance = 0.0
        else:
            # Calculate displacement from last position
            displacement_vector = np.array([
                hero_location.x - self.last_location.x,
                hero_location.y - self.last_location.y
            ])

            # Calculate distance progress toward goal
            current_distance = core.destination.location.distance(hero_location)
            delta_distance = self.prev_distance_to_goal - current_distance
            self.prev_distance_to_goal = current_distance

        # Calculate raw distance traveled
        raw_distance = float(np.linalg.norm(displacement_vector))

        # Project displacement onto road direction (waypoint forward vector)
        projected_distance = 0.0

        if wp is not None and raw_distance > 0:
            # Get the forward vector from the waypoint's transform
            forward_vector = wp.transform.get_forward_vector()

            # Convert to 2D numpy array (ignore z-component for ground plane)
            forward_2d = np.array([forward_vector.x, forward_vector.y])

            # Project displacement onto road direction
            projected_distance = float(np.dot(displacement_vector, forward_2d))

        # Update cumulative distances
        self.distance_travelled += projected_distance

        # Calculate speed factor (ratio of current speed to target)
        speed_factor = hero_velocity / self.target_speed if self.target_speed > 0 else 0

        # Action smoothness penalties
        steer_change = abs(self.steer - self.prev_steer)
        action_smoothness_penalty = steer_change

        # Initialize reward with base penalty + steering smoothness
        reward = -WEIGHTS['base_penalty'] - steer_change * 0.5

        # Progress reward: only reward when moving at reasonable speed (10% to 100% of target)
        # Use one signal to avoid double-counting: delta_distance for goal-directed,
        # projected_distance for exploration (when no clear goal direction).
        if 0.1 < speed_factor < 1.0:
            if self.prev_distance_to_goal is not None and self.last_location is not None:
                # Goal-directed: reward reducing distance to goal
                reward += delta_distance
            else:
                # Fallback: reward road-aligned forward progress
                reward += projected_distance


        # Apply penalties for bad behavior
        if self.done_wrong_direction:
            reward -= WEIGHTS['wrong_junction_penalty']
            print(f"Wrong direction at junction! Distance completed: {self.distance_travelled:.2f}m")

        if self.collision:
            reward -= WEIGHTS['collision_penalty']

        if self.diff_lane or self.time_idle>self.max_time_idle:
            reward -= WEIGHTS['lane_departure_penalty']

        # Goal reached bonus
        if self.done_goal:
            reward += WEIGHTS['goal_reached_bonus']
            print(f"Goal reached! Distance: {self.prev_distance_to_goal:.2f}m")

        # Distance goal bonus (200m traveled)
        if self.distance_travelled >= 200.0:
            reward += WEIGHTS['goal_reached_bonus']
            print(f"Distance goal reached! Travelled: {self.distance_travelled:.2f}m")

        # Handle NaN values (safety check)
        reward = float(np.nan_to_num(reward, nan=0.0))

        # Update tracking
        self.last_location = hero_location

        # Update info for logging
        self.info.update({
            'speed_factor': np.clip(speed_factor, 0, 1.0),
            'action_smoothness_penalty': action_smoothness_penalty,
            'delta_distance': delta_distance,
            'projected_distance': projected_distance,
            'raw_distance': raw_distance,
            'total_reward': reward
        })

        self.rewards.append(reward)

        # Update previous action values for next smoothness calculation
        self.prev_steer = self.steer
        self.prev_throttle = self.throttle

        return reward


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
            "type": GoalNavRewardExperiment,
            "hero": {
                "blueprint": "vehicle.micro.microlino",
                "sensors": {
                    "collision": {
                        "type": "sensor.other.collision"
                    },
                    "rgb": {
                        "type": "sensor.camera.rgb",
                        "image_size_x": 748,
                        "image_size_y": 348,
                        "fov": 110,
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
                    "goal": {
                        "type": "sensor.goal"
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
                "target_speed": 13.0
            }
        }
    }
}
