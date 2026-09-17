# Evaluation config for image-goal model (no compass)
# Derived from train_env_config_v4_image_goal.py
# Changes: eval_mode=True, no wrong-direction termination,
#          terminal conditions = collision + lane_invasion + goal_reached only

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
import torchvision.models as models
from scipy.spatial.transform import Rotation as R
from rlib_integration.base_experiment import BaseExperiment
from rlib_integration.helper import post_process_image, carla_location_to_np_array


class STBL3EvalExperiment(BaseExperiment):
    """Image-goal evaluation experiment: terminates only on collision, lane invasion, or goal reached."""

    def __init__(self, config={}):
        super().__init__(config)

        self.frame_stack = self.config["others"]["framestack"]
        self.max_time_idle = self.config["others"]["max_time_idle"]
        self.max_dist = self.config["others"]["max_dist"]
        self.target_speed = self.config["others"]["target_speed"]
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]

        self.max_steer = 1.0
        self.max_throttle = 1.0
        self.max_angle_deviation = np.pi
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.steer = 0.0
        self.throttle = 0.0
        self.last_action = None

        self.current_heading = 0.0
        self.velocity = 0.0
        self.dot = 0.0
        self.distance_travelled = 0.0
        self.distance_completed = 0.0
        self.last_distance = None
        self.last_location = None
        self.last_velocity = 0
        self.wp=None

        self.trajectories = None
        self.info = dict()
        self.rewards = []
        self.episode_num = 1
        self.max_episodes = 10000
        self.explore_mode = False
        self.passed_chckp = False
        self.eval_mode = True  # Always eval mode
        self.mask_goal = False
        self.map_mode = False
        self.image_size = 84
        self.obs_img = None
        self.obs_img_vlad = None
        self.goal_img_raw = None
        self.goal_img_vlad = None
        self.prev_vec_0 = None
        self.prev_vec_1 = None
        self.prev_vec_2 = None
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading pre-trained MobileNetV3-Small model...")
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.model = self.model.features[:4]
        self.model.to(self.device)
        self.model.eval()

        self.image_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((84, 84)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _cache_waypoints(self, world) -> None:
        env_map = world.get_map()
        waypoints = env_map.generate_waypoints(distance=2)
        trajectories = defaultdict(list)
        for wpt in waypoints:
            trajectories[f"{wpt.road_id}-{wpt.lane_id}"].append(wpt)
        self.trajectories = sorted(
            [traj for traj in trajectories.values() if len(traj) > 3],
            key=len, reverse=True
        )

    def reset(self, env):
        self.time_idle = 0
        self.time_episode = 0
        self.done_time_idle = False
        self.done_falling = False
        self.done_dist = False
        self.done_wrong_direction = False
        self.done_speed = False
        self.checkpoint_count = 0

        self.last_location = None
        self.last_velocity = 0
        self.distance_travelled = 0.0
        self.distance_completed = 0.0
        self.last_heading = None
        self.last_distance = None

        self.prev_vec_0 = None
        self.prev_vec_1 = None
        self.prev_vec_2 = None
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None
        self.obs_img = None

        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.steer = 0.0
        self.throttle = 0.0
        self.velocity = 0.0
        self.current_heading = 0.0
        self.dot = 0.0
        self.heading_vec_2d = np.zeros((2,))

        self.explore_mode = False
        self.passed_chckp = False
        self.info = dict()
        self.rewards = []

        if self.trajectories is None:
            self._cache_waypoints(env.world)

    def get_observation_space(self):
        image_space = Box(low=-np.inf, high=np.inf, shape=(2904,), dtype=np.float32)
        goal_image_space = Box(low=-np.inf, high=np.inf, shape=(2904,), dtype=np.float32)
        vec_space = Box(low=-5.1, high=5.1, shape=(4 * self.frame_stack,), dtype=np.float32)
        return Dict({"state": image_space, "goal": goal_image_space, "vector": vec_space})

    def get_action_space(self):
        return Box(
            low=np.array([-self.max_steer, 0.0]),
            high=np.array([self.max_steer, 1.0]),
            dtype=np.float32
        )

    def compute_action(self, action, core):
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
        vecs = self.get_vec_obs(sensor_data, core)
        images = self.get_img_obs(sensor_data, core)
        goal_images = self.get_goal_img_obs(sensor_data, core)
        if self.mask_goal:
            goal_images = np.zeros_like(goal_images)
        return {"state": images, "goal": goal_images, "vector": vecs}, self.info

    def get_vec_obs(self, sensor_data, core):
        hero = core.hero
        vec = np.zeros(4)
        vec[0] = self.steer / self.max_steer
        vec[1] = self.throttle / self.max_throttle

        hero_transform = hero.get_transform()
        hero_location = hero_transform.location
        goal_location = core.destination.location
        heading_vec_world = carla_location_to_np_array(goal_location) - carla_location_to_np_array(hero_location)
        self.heading_vec_2d = heading_vec_world / (np.linalg.norm(heading_vec_world) + 1e-8)
        world_to_agent_matrix = np.array(hero_transform.get_inverse_matrix())
        heading_vec_local = world_to_agent_matrix[:3, :3] @ heading_vec_world
        heading_vec_2d = heading_vec_local[:2]
        heading_vec_2d = heading_vec_2d / (np.linalg.norm(heading_vec_2d) + 1e-8)
        forward_vec_2d = np.array([1.0, 0.0])
        self.dot = np.dot(heading_vec_2d, forward_vec_2d)

        if self.target_speed == 0:
            vec[2] = 0.0 if self.get_speed(hero) < 0.1 else 1.0
        else:
            vec[2] = np.clip(self.get_speed(hero) / self.target_speed, 0.0, 1.0)

        vec[3] = 0.0 if self.mask_goal else 1.0

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
        image = post_process_image(
            sensor_data['rgb'][1], normalized=False, crop=False, grayscale=False, image_size=self.image_size
        )
        self.obs_img = image
        self.obs_img_vlad = cv2.resize(sensor_data['rgb'][1], (256, 256), interpolation=cv2.INTER_AREA)
        if self.prev_image_0 is None:
            self.prev_image_0 = image
            self.prev_image_1 = image
            self.prev_image_2 = image
        self.prev_image_2 = self.prev_image_1
        self.prev_image_1 = self.prev_image_0
        self.prev_image_0 = image
        img_tensor = self.image_transform(image.copy()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent_vector = self.model(img_tensor)
        return latent_vector.squeeze(0).cpu().numpy().flatten()

    def get_goal_img_obs(self, sensor_data, core):
        goal_image = post_process_image(
            sensor_data['goal'][1][0], normalized=False, crop=False, grayscale=False, image_size=self.image_size
        )
        self.goal_img_raw = goal_image
        self.goal_img_vlad = cv2.resize(sensor_data['goal'][1][0], (256, 256), interpolation=cv2.INTER_AREA)
        if self.mask_goal:
            return np.zeros((2904,), dtype=np.float32)
        img_tensor = self.image_transform(goal_image.copy()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent_vector = self.model(img_tensor)
        return latent_vector.squeeze(0).cpu().numpy().flatten()

    def get_speed(self, hero):
        vel = hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2)

    def get_done_status(self, sensor_data, core):
        """Terminate on collision, lane invasion, or goal reached only. No wrong-direction termination."""
        self.info = dict()
        hero = core.hero
        self.passed_chckp = False

        # Check idle time
        self.done_time_idle = self.max_time_idle < self.time_idle
        if self.get_speed(hero) > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1
        self.time_episode += 1

        # Get current waypoint
        wp = core.map.get_waypoint(hero.get_transform().location, project_to_road=True)
        self.wp=wp

        wp = core.map.get_waypoint(hero.get_transform().location, project_to_road=False)
        # self.wp=wp
        # Check failure conditions
        self.done_falling = False
        self.diff_lane = 'lane_invasion' in sensor_data.keys()
        self.collision = 'collision' in sensor_data.keys()

        # Check goal reached
        goal_loc = carla_location_to_np_array(core.destination.location)
        hero_loc = carla_location_to_np_array(hero.get_transform().location)
        distance_to_goal = np.linalg.norm(goal_loc[:2] - hero_loc[:2])

        # Goal reached when within threshold distance
        self.done_dist = distance_to_goal < 2.0

        # No wrong-direction termination in eval
        self.done_wrong_direction = False

        # Speed-based termination (disabled)
        self.done_speed = False

        # Terminal conditions: collision, lane invasion, or goal reached
        done = self.collision or self.diff_lane or self.done_dist

        # Update info
        self.info.update(dict(
            is_success=self.done_dist,
            distance_travelled=self.distance_completed,
            distance_completed=self.distance_completed,
            checkpoints_passed=self.checkpoint_count,
            collision=self.collision,
            lane_invasion=self.diff_lane,
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
        """Reward function for evaluation — simpler, no wrong-direction penalty."""
        WEIGHTS = {
            'collision_penalty': 3.0,
            'lane_departure_penalty': 5.0,
            'goal_reached_bonus': 5.0,
            'base_penalty': 4e-3,
        }

        hero = core.hero
        hero_location = hero.get_location()
        hero_velocity = self.get_speed(hero)

        wp = core.map.get_waypoint(hero.get_transform().location, project_to_road=True)
        self.wp=wp
        if self.last_distance is None:
            goal_loc = carla_location_to_np_array(core.destination.location)
            self.last_distance = np.linalg.norm(goal_loc[:2] - carla_location_to_np_array(hero_location)[:2])

        if self.last_location is None:
            self.last_location = hero_location
            displacement_vector = np.array([0.0, 0.0])
            delta_distance = 0.0
        else:
            displacement_vector = np.array([
                hero_location.x - self.last_location.x,
                hero_location.y - self.last_location.y
            ])
            goal_loc = carla_location_to_np_array(core.destination.location)
            current_distance = np.linalg.norm(goal_loc[:2] - carla_location_to_np_array(hero_location)[:2])
            delta_distance = self.last_distance - current_distance
            self.last_distance = current_distance

        raw_distance = float(np.linalg.norm(displacement_vector))
        projected_distance = 0.0

        if wp is not None and raw_distance > 0:
            forward_vector = wp.transform.get_forward_vector()
            forward_2d = np.array([forward_vector.x, forward_vector.y])
            projected_distance = float(np.dot(displacement_vector, forward_2d))

        self.distance_travelled += projected_distance
        self.distance_completed += projected_distance
        self.last_location = hero_location

        speed_factor = hero_velocity / self.target_speed if self.target_speed > 0 else 0
        steer_change = abs(self.steer - self.prev_steer)

        reward = -WEIGHTS['base_penalty'] - steer_change

        if speed_factor < 1.0:
            reward += delta_distance + projected_distance

        if self.diff_lane:
            reward -= WEIGHTS['lane_departure_penalty']
        if self.collision:
            reward -= WEIGHTS['collision_penalty']
        if self.done_dist:
            reward += WEIGHTS['goal_reached_bonus']

        reward = float(np.nan_to_num(reward, nan=0.0))

        self.info.update({
            'speed_factor': np.clip(speed_factor, 0, 1.0),
            'heading_alignment': self.dot,
            'delta_distance': delta_distance,
            'projected_distance': projected_distance,
            'total_reward': reward
        })

        self.rewards.append(reward)
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
            "timeout": 50.0,
            "timestep": 0.1,
            "retries_on_error": 25,
            "resolution_x": 600,
            "resolution_y": 600,
            "quality_level": "Low",
            "enable_map_assets": True,
            "enable_rendering": True,
            "show_display": True,
            "town": "Town02"
        },
        "experiment": {
            "type": STBL3EvalExperiment,
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