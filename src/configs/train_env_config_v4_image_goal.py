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
    def __init__(self, image_size=None, aug_strength='medium', training=True):
        image_size = int(os.environ.get("ENV_IMAGE_SIZE", 84)) if image_size is None else image_size
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
    
# When set, crossing a lane marking or leaving the road WEDGES the vehicle:
# it stops responding to actions for the rest of the episode. See the long note
# in get_done_status for why this is a dynamics change rather than a reward
# penalty.
LANE_WEDGE = os.environ.get("LANE_WEDGE", "0") == "1"
# Same default and same meaning as core_state.py's read of it, so the wedge and
# the reward's lane signal cannot disagree about what counts as a violation.
# 1 = any crossing is fatal; 0 = only illegal markings (Solid/Curb/Grass) are,
# which lets a route-following agent cross a dashed line to turn at a junction.
# Publish remaining ALONG-ROAD distance to the goal, for potential-based
# shaping in the reward wrapper. Off by default: it costs a route plan per goal
# and every existing run script is written without it.
ROUTE_POTENTIAL = os.environ.get("ROUTE_POTENTIAL", "0") == "1"
# How many route plans to log on startup, so the route-vs-straight-line
# ratio is visible as a distribution rather than a single sample.
ROUTE_LOG_N = int(os.environ.get("ROUTE_LOG_N", 20))
LANE_STRICT = os.environ.get("LANE_STRICT", "1") == "1"
# Whether `off_road` (get_waypoint(project_to_road=False) is None) wedges.
#
# WHY IT IS WORTH TURNING OFF. The test is strict: it demands the hero's
# transform origin lie inside a Driving-lane polygon, and returns None the
# moment it does not. Junction interiors are where that is least reliable --
# lane polygons meet with gaps there, and a vehicle mid-turn can fall between
# them while driving the route correctly. It is the same false-positive
# mechanism as wedging on a Broken marking, in a different signal.
#
# WHAT IT COSTS. off_road is the ONLY signal that catches leaving the road
# where nothing is painted -- a kerb-hop onto grass or a median never fires the
# lane sensor. With this off, such an excursion costs nothing but time, and
# under a minimum-time objective a shortcut across grass is REWARDED if it is
# shorter. The rate stays in the [Why] breakdown either way, so the trade is
# measurable rather than invisible.
WEDGE_ON_OFF_ROAD = os.environ.get("WEDGE_ON_OFF_ROAD", "1") == "1"


class STBL3Experiment(BaseExperiment):
    def __init__(self, config={}):
        super().__init__(config)

        # Configuration
        self.frame_stack = self.config["others"]["framestack"]
        self.max_time_idle = self.config["others"]["max_time_idle"]
        self.max_dist = self.config["others"]["max_dist"]
        # TARGET_SPEED overridable, because the config's 5.0 km/h is wrong for
        # this vehicle. MEASURED holding the throttle cap (0.6) for 266 steps:
        # the car settles at a 22.4 km/h median, so a 5 km/h target gates the
        # distance reward OFF on 86.1% of steps and the agent is paid almost
        # nothing for driving correctly. 5.0 came from the upstream repo with a
        # different blueprint; it is not a property of the task.
        self.target_speed = float(os.environ.get(
            "TARGET_SPEED", self.config["others"]["target_speed"]))
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]
        
        # Control variables
        # ABSOLUTE limits. Under INCREMENTAL_ACTIONS these default to the
        # upstream repo's values (steer +/-0.5, throttle 0-0.6), which is what
        # keeps the car at the 5 km/h target_speed this config was calibrated
        # for. Under absolute control they stay at 1.0 so existing runs are
        # bit-identical.
        self.incremental_actions = os.environ.get("INCREMENTAL_ACTIONS", "0") == "1"
        if self.incremental_actions:
            self.max_steer = float(os.environ.get("MAX_STEER", 0.5))
            self.max_throttle = float(os.environ.get("MAX_THROTTLE", 0.6))
        else:
            self.max_steer = float(os.environ.get("MAX_STEER", 1.0))
            self.max_throttle = float(os.environ.get("MAX_THROTTLE", 1.0))
        # One action = one step of this size on the control.
        self.steer_step = float(os.environ.get("STEER_STEP", 0.01))
        self.throttle_step = float(os.environ.get("THROTTLE_STEP", 0.01))
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
        # ENV-VAR GATED, DEFAULT 84. The sparse-reward VAE runs raise this to
        # 120 because the lane markings the VAE has to segment do not survive a
        # squash to 84 (748x348 -> 84x84 smears a marking a few pixels wide
        # into sub-pixel contrast). Left at 84 by default so the CORL 2026
        # dense-reward experiments, which share this config, are unchanged.
        self.image_size = int(os.environ.get("ENV_IMAGE_SIZE", 84))
        # ENV_SKIP_IMAGE_FEATURES: return a correctly-shaped ZERO vector from
        # get_img_obs / get_goal_img_obs instead of running MobileNetV3 over
        # the frame. DEFAULT OFF, so every existing caller is unchanged.
        #
        # WHY IT EXISTS. torch.cuda.is_available() is False on this machine
        # (the installed build wants a newer driver than is present), so both
        # passes ran on the CPU for EVERY simulation step: measured 17.3 of 32
        # cores in the policy process against 3.5 for the simulator, with the
        # GPU at 0% and 701 MiB of 12282 -- the run was CPU-bound on them.
        #
        # WHY IT IS SAFE. Both results are DISCARDED by the gated stack.
        # VAEGoalWrapper._observation builds its observation from info["rgb"]
        # (core_state.py sets it from experiment.obs_img, assigned BELOW the
        # skip point), the goal fields, raw_obs["vector"] and info["mode"]; it
        # never reads the "state" or "goal" keys these methods return. The
        # frame the VAE encodes, and the frame stack, are untouched.
        # Pinned by tests/test_skip_discarded_image_features.py. Do NOT set
        # this for any arm whose policy actually consumes the image features.
        self.skip_image_features = (
            os.environ.get("ENV_SKIP_IMAGE_FEATURES", "0") == "1")
        self.obs_img = None
        self.prev_vec_0 = None
        self.prev_vec_1 = None
        self.prev_vec_2 = None
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None
        
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
            T.Resize((self.image_size, self.image_size)),
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
        self.wedged = False
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

    def _feature_dim(self):
        """Flattened MobileNet feature width for the CURRENT image_size.

        This was hardcoded to 2904, which is 11 x 11 x 24 -- the shape the
        truncated MobileNetV3-Small emits for an 84x84 input. That is not a
        constant of the model, it is a constant of that one resolution: at
        120x120 the same stack emits 15 x 15 x 24 = 5400, so the declared
        observation space described an array the env would never produce.
        Measured from a dummy forward pass so it cannot drift from image_size
        again.
        """
        if getattr(self, "_feat_dim_cache", None) is None:
            with torch.no_grad():
                probe = torch.zeros(1, 3, self.image_size, self.image_size,
                                    device=self.device)
                self._feat_dim_cache = int(
                    self.model(probe).squeeze(0).cpu().numpy().flatten().shape[0])
            print(f"[Env] image_size={self.image_size} -> "
                  f"feature_dim={self._feat_dim_cache}")
        return self._feat_dim_cache

    def get_observation_space(self):
        """Define observation space with image features, goal image, and vector state."""
        feat_dim = self._feature_dim()
        image_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(feat_dim,),
            dtype=np.float32
        )

        goal_image_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(feat_dim,),
            dtype=np.float32
        )

        vec_space = Box(
            low=-5.1,
            high=5.1,
            shape=(4 * self.frame_stack,),
            dtype=np.float32,
        )

        return Dict({"state": image_space, "goal": goal_image_space, "vector": vec_space})

    def get_action_space(self):
        """Action space. ABSOLUTE control, or a DELTA under INCREMENTAL_ACTIONS.

        Under incremental control the action is a normalised RATE, not a
        control value, so both dimensions span [-1, 1] and the physical limits
        live in compute_action instead.

        Reusing the absolute bounds here was a real bug, caught by a smoke test
        printing the space: with max_steer=0.5 the space came out as
        Box([-0.5, 0.0], [0.5, 1.0]), which meant (a) the steer delta was capped
        at half a step and (b) the THROTTLE DELTA COULD NEVER BE NEGATIVE, so
        the car could accelerate and never slow down.
        """
        if getattr(self, "incremental_actions", False):
            return Box(low=np.array([-1.0, -1.0], np.float32),
                       high=np.array([1.0, 1.0], np.float32),
                       dtype=np.float32)
        # The throttle bound is max_throttle, not a hardcoded 1.0. compute_action
        # clips to max_throttle anyway, so a wider space just lets the policy
        # spend probability mass on actions the env silently truncates. With
        # the default max_throttle=1.0 this is identical to the old space, so
        # existing runs are unaffected.
        return Box(
            low=np.array([-self.max_steer, 0.0], np.float32),
            high=np.array([self.max_steer, self.max_throttle], np.float32),
            dtype=np.float32
        )

    def compute_action(self, action,core):
        """Action -> CARLA VehicleControl, absolute or INCREMENTAL.

        INCREMENTAL_ACTIONS=1 makes the action a DELTA on the current control
        rather than the control itself, which is how the upstream
        nisheeth-golakiya/carla-rl this config was forked from does it: its
        Discrete(4) actions are +/-0.01 steer clipped to +/-0.5 and +/-0.01
        throttle clipped to 0-0.6.

        THAT DIFFERENCE IS WHY THE ABSOLUTE POLICY DRIVES BADLY. Writing steer
        and throttle straight from the action every step lets the network go
        from 0 to full throttle in one 0.05 s tick. MEASURED on the trained
        goal-conditioned agent: p95 |acceleration| 196 m/s^2, 79.5% of steps
        above the 4 m/s^2 "harsh" threshold, steering reversing on 32.6% of
        steps, speeds to 123 km/h in a town. Under the incremental space those
        numbers are not reachable at all -- the bound is on the control's rate
        of change, so smoothness is structural rather than something the reward
        has to buy.

        Off by default so the existing goal-navigation runs are unchanged.
        """
        a0, a1 = float(action[0]), float(action[1])

        vehicle_control = carla.VehicleControl()

        # WEDGED: the action is discarded entirely and the car is held still.
        # Full brake rather than merely zero throttle, so it does not coast on
        # for another twenty metres after crossing the line -- "wedged" should
        # mean stationary, not drifting.
        if getattr(self, "wedged", False):
            vehicle_control.steer = 0.0
            vehicle_control.throttle = 0.0
            vehicle_control.brake = 1.0
            vehicle_control.hand_brake = True
            vehicle_control.reverse = False
            self.steer = 0.0
            self.throttle = 0.0
            self.last_action = vehicle_control
            return vehicle_control
        if self.incremental_actions:
            # a in [-1, 1] scales one step of size STEER_STEP / THROTTLE_STEP.
            steer = self.steer + self.steer_step * np.clip(a0, -1.0, 1.0)
            throttle = self.throttle + self.throttle_step * np.clip(a1, -1.0, 1.0)
            vehicle_control.steer = float(np.clip(steer, -self.max_steer, self.max_steer))
            vehicle_control.throttle = float(np.clip(throttle, 0.0, self.max_throttle))
            vehicle_control.brake = 0.0
            self.steer = vehicle_control.steer
            self.throttle = vehicle_control.throttle
            self.last_action = vehicle_control
            vehicle_control.reverse = False
            vehicle_control.hand_brake = False
            return vehicle_control

        steer, throttle_brake = a0, a1
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

    def _route_remaining(self, core):
        """Metres left ALONG THE ROAD to the goal, or None if unavailable.

        Lazily built because it needs core.map, which does not exist when the
        experiment is constructed. Replans whenever the destination moves --
        which it does on every arrival, since the env resamples the goal the
        moment it is reached, and a stale route would keep measuring progress
        toward a goal that is no longer there.

        None, NOT 0.0, when there is no route. Zero reads as "arrived" and
        would pay out the whole potential in one step; None tells the reward
        wrapper to skip shaping for this step, which is always safe.
        """
        if not ROUTE_POTENTIAL:
            return None
        try:
            goal = core.destination.location
            hero = core.hero.get_transform().location
            if getattr(self, "_route_pot", None) is None:
                from vae_nav.route_potential import RoutePotential
                self._route_pot = RoutePotential(core.map)
                self._route_goal = None
            key = (round(goal.x, 2), round(goal.y, 2))
            if key != getattr(self, "_route_goal", None):
                length = self._route_pot.reset(hero, goal)
                self._route_goal = key
                # ONE-SHOT PROOF THAT SHAPING IS LIVE. A planner that silently
                # returns None disables the shaping and is INVISIBLE in every
                # metric -- shaping only redistributes credit, so a run with it
                # switched off looks like a run where it did not help. Print the
                # first plan's outcome once so that cannot go unnoticed.
                n = getattr(self, "_route_announced", 0)
                if n < ROUTE_LOG_N:
                    self._route_announced = n + 1
                    err = self._route_pot.plan_error
                    import math as _m
                    eu = _m.hypot(goal.x - hero.x, goal.y - hero.y)
                    # ROUTE vs STRAIGHT LINE. The goal is origin.next(d), a
                    # forward walk of 20-50 m ALONG the lane, so the route
                    # should be about the same. A ratio far above 1 means
                    # trace_route is not following the path the car will drive
                    # -- and the monotone pointer would then stall near the
                    # route's start and pay the shaping in one late spike
                    # instead of per metre. Logged for the first few plans so
                    # the distribution is visible, not one sample.
                    print(f"[Route] plan {n + 1}: {length:.1f} m road / "
                          f"{eu:.1f} m straight  ratio={length / max(eu, 1e-6):.2f}"
                          + (f"  ERROR={err}" if err else ""), flush=True)
            return self._route_pot.remaining((hero.x, hero.y))
        except Exception as e:  # noqa: BLE001
            # Shaping is optional by construction, so a planner failure must
            # degrade to "no shaping" rather than kill the episode -- but it
            # must not do so silently.
            if not getattr(self, "_route_announced", False):
                self._route_announced = True
                print(f"[Route] DISABLED -- {type(e).__name__}: {e}", flush=True)
            return None

    def get_observation(self, sensor_data, core):
        """Process sensor data into observation."""
        vecs = self.get_vec_obs(sensor_data, core)
        images = self.get_img_obs(sensor_data, core)
        goal_images = self.get_goal_img_obs(sensor_data, core)

        if self.mask_goal:
            goal_images = np.zeros_like(goal_images)

        return {"state": images, "goal": goal_images, "vector": vecs}, self.info

    def get_vec_obs(self, sensor_data, core):
        """Extract vector observations from sensors."""
        hero = core.hero
        vec = np.zeros(4)
        vec[0] = self.steer / self.max_steer
        vec[1] = self.throttle / self.max_throttle

        # Calculate direction to goal (kept for done_status/reward, not in observation)
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

        # Speed normalized to target
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

        # AFTER obs_img and the frame stack: those are the real outputs of this
        # method for the gated stack, and they must be identical either way.
        if getattr(self, "skip_image_features", False):
            return np.zeros((self._feature_dim(),), dtype=np.float32)

        # Transform and encode with model
        img_tensor = self.image_transform(image.copy()).unsqueeze(0).to(self.device)

        with torch.no_grad():
            latent_vector = self.model(img_tensor)
        
        return latent_vector.squeeze(0).cpu().numpy().flatten()

    def get_goal_img_obs(self, sensor_data, core):
        """Extract goal image observation and encode with MobileNetV3."""
        goal_image = post_process_image(
            sensor_data['goal'][1][0],
            normalized=False,
            crop=False,
            grayscale=False,
            image_size=self.image_size
        )
        # cv2.imwrite("goal.jpg",goal_image)
        if self.mask_goal or getattr(self, "skip_image_features", False):
            return np.zeros((self._feature_dim()),dtype=np.float32)
        img_tensor = self.image_transform(goal_image.copy()).unsqueeze(0).to(self.device)

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
        self.passed_chckp = False
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
        self.diff_lane= 'lane_invasion' in sensor_data.keys()

        # WEDGE ON LANE CROSSING. Once the vehicle crosses a marking it stops
        # responding to actions for the rest of the episode -- compute_action
        # returns a full-brake control whatever the policy asks for.
        #
        # This is a DYNAMICS change, not a reward change, and that distinction
        # is the point. Under the minimum-time reward the critic's value IS the
        # cost-to-go, so a reward penalty for lane crossing would corrupt that
        # reading -- V(s) would stop meaning "steps to the goal" and start
        # meaning "steps to the goal, plus a fine". Wedging instead makes the
        # state genuinely far from the goal: a stuck car really does need the
        # rest of the episode and never arrives, so V(s) stays an honest
        # distance estimate and the SORB graph can use it as an edge weight.
        #
        # The cost is therefore paid in TIME, exactly like every other failure
        # here: a wedge at step t costs -0.1 * (1000 - t) and the episode ends
        # at the step limit having never arrived.
        #
        # There is a one-step lag by construction: carla_env.step calls
        # compute_action BEFORE get_done_status, so a crossing detected at step
        # t first freezes the control at step t+1. That is the correct
        # ordering -- you cross the line, then you are stuck.
        #
        # The test itself sits below, AFTER off_road is computed -- reading it
        # here would use the previous step's value.
        # OFF-ROAD. get_waypoint(project_to_road=False) returns None when the
        # hero is on no drivable lane at all -- up a kerb, on a median, on
        # grass. That is a lane-keeping failure the lane_invasion SENSOR can
        # miss entirely: the sensor fires on CROSSING a painted marking, so a
        # car that leaves the road where no marking is painted, or that is
        # already off-road at spawn, never triggers it and drives on happily
        # outside the lane for the rest of the episode.
        #
        # The check existed here and was commented out. Kept ADDITIVE -- exposed
        # as its own attribute, NOT folded into diff_lane -- because diff_lane
        # feeds the dense reward the CORL 2026 experiments run on, and widening
        # it would silently change those episodes' termination behaviour.
        # core_state.py ORs this into the sparse path's lane_invasion.
        self.off_road = wp is None

        # ADDITIVE: record WHICH markings were crossed. diff_lane above is left
        # exactly as it was so the dense-reward CORL experiments are unaffected;
        # this only exposes detail they already discard. The sparse-reward
        # wrapper needs it because 'lane_invasion' fires on ANY crossing --
        # including a legal dashed centre line at a junction, which a
        # route-following expert cannot avoid.
        #
        # MOVED ABOVE THE WEDGE. It used to sit below, which meant the wedge had
        # no marking information available and could only read the raw flag.
        self.lane_marking_types = []
        if self.diff_lane:
            try:
                self.lane_marking_types = [
                    str(m.type) for m in sensor_data['lane_invasion'][1][1]]
            except Exception:
                self.lane_marking_types = []

        # THE WEDGE NOW ROUTES THROUGH THE SAME FILTER AS core_state's
        # lane_invasion, INSTEAD OF BYPASSING IT.
        #
        # THE BUG THIS FIXES. core_state.py gates the lane signal on
        # LANE_STRICT: strict reads the raw sensor event, non-strict reads
        # `lane_invasion_illegal`, which drops legal Broken-line crossings. Its
        # comment offers LANE_STRICT=0 as the remedy "if this starves learning"
        # -- but the wedge read `self.diff_lane` directly, upstream of that
        # gate, so LANE_STRICT=0 could not reach it and the documented escape
        # hatch did nothing. The wedge fired on every crossing regardless.
        #
        # WHY IT MATTERS HERE. JUNCTION_GOAL_FRAC=0.6 deliberately routes 60% of
        # episodes through a junction, where crossing a dashed line is not just
        # legal but REQUIRED to make the turn. MEASURED after the wedge began
        # terminating episodes: lane_any=0.52 at ~80 steps, against off_road at
        # 0.02 -- and core_state's own comment records an earlier run that
        # "ended 90% of its episodes on lane_any at ~44-70 steps".
        #
        # DEFAULT BEHAVIOUR IS UNCHANGED: at LANE_STRICT=1 this is still
        # `diff_lane or off_road`, exactly as before. Only LANE_STRICT=0 now
        # does what it always claimed to.
        #
        # off_road is ORed in unconditionally under both settings, because there
        # is no such thing as a legal excursion onto no lane at all.
        _illegal_marks = {"Solid", "SolidSolid", "SolidBroken", "Curb", "Grass"}
        # AN UNPAINTED BOUNDARY IS NOT A LANE VIOLATION.
        #
        # LaneMarkingType.NONE means the sensor fired where NOTHING IS PAINTED
        # -- a road edge, a segment transition, an unmarked lane boundary. There
        # is no lane discipline to violate at a line that does not exist, so it
        # can never be a driving fault, wherever it happens.
        #
        # MEASURED, and it is why this run could not work. Over 347 oracle
        # episodes at 100 m goals, the SCRIPTED EXPERT -- GlobalRoutePlanner
        # route, PID control, competent by construction -- crossed a line on
        # essentially every episode and arrived on ZERO of its last 100:
        #
        #   window        lane_any/orc     NONE@road     oracle success
        #     0-400           0.51            0.69
        #   400-800           0.99            0.81
        #   800-1200          0.94            0.81          0.000 (n=100)
        #  1200-1600          1.02            0.77
        #
        # NONE@road alone accounts for ~0.8 of episodes. Because the wedge is an
        # ABSORBING failure its probability compounds with route length, which
        # is why the same rules gave lane_any=0.24 at 20-31 m and ~0.85 at 100 m.
        # Treating NONE as fatal made the task unsolvable for ANY driver, not
        # just a learning one -- that is what the oracle 0.000 proves and what
        # the learned policy alone could never have shown.
        _unpainted_marks = {"NONE"}
        _marks = [t.split(".")[-1] for t in self.lane_marking_types]
        # Fail CLOSED on no marking information at all: absent data is not
        # evidence of a legal crossing.
        _crossed_illegal = bool(self.diff_lane and (
            not _marks or any(m in _illegal_marks for m in _marks)))
        # Only when EVERY reported marking is unpainted. A mixed Broken+NONE
        # crossing is judged as the Broken one -- the painted line is the real
        # boundary and the junction rule below decides it.
        _only_unpainted = bool(_marks) and all(m in _unpainted_marks
                                               for m in _marks)

        # THE EXEMPTION IS THE JUNCTION INTERIOR, NOT THE MARKING TYPE.
        #
        # Forgiving every Broken crossing is too broad: a dashed line crossed on
        # a straight road is the vehicle DRIFTING OUT OF LANE, which is exactly
        # the failure the wedge exists to punish. The same crossing inside a
        # junction is the turn itself. Same sensor event, same marking, opposite
        # meaning -- and only the location distinguishes them.
        #
        # project_to_road=True DELIBERATELY, unlike the off_road probe above.
        # That probe asks "is the hero on a lane at all" and must be allowed to
        # answer no; this asks "what part of the road network is the hero in",
        # where snapping to the nearest lane is what makes the answer reliable
        # in precisely the place the strict probe is least reliable.
        #
        # An illegal marking stays fatal even inside a junction: Curb and Grass
        # are physical departures rather than lane discipline, and no junction
        # geometry requires crossing them.
        _in_junction = False
        try:
            _jwp = core.map.get_waypoint(hero.get_transform().location,
                                         project_to_road=True)
            _in_junction = bool(_jwp is not None and _jwp.is_junction)
        except Exception:
            # Fail CLOSED: an unavailable junction test must not silently
            # forgive every crossing on the map.
            _in_junction = False
        self.in_junction = _in_junction

        if LANE_STRICT:
            # Unchanged: every crossing is fatal. Pinned by test so this fix
            # cannot alter the strict default.
            _lane_fatal = bool(self.diff_lane)
        elif _crossed_illegal:
            # Solid / Curb / Grass, or unknown markings. Fatal ANYWHERE --
            # no junction geometry requires crossing them.
            _lane_fatal = bool(self.diff_lane)
        elif _only_unpainted:
            # NONE everywhere: never fatal. See the block above.
            _lane_fatal = False
        else:
            # Broken (dashed): the turn inside a junction, drifting out of lane
            # outside one.
            _lane_fatal = bool(self.diff_lane and not _in_junction)
        # off_road is no longer unconditional -- see WEDGE_ON_OFF_ROAD. It still
        # reaches core_state and the [Why] breakdown as a RATE, so switching it
        # off here hides nothing; it only stops it freezing the vehicle.
        _off_road_fatal = WEDGE_ON_OFF_ROAD and self.off_road
        if LANE_WEDGE and (_lane_fatal or _off_road_fatal):
            self.wedged = True
        self.collision = 'collision' in sensor_data.keys()
        # DISTANCE CAP, env-var gated. Default 200 keeps the CORL dense-reward
        # experiments byte-identical; the long-horizon curriculum raises it.
        #
        # This is a hard episode terminator on distance TRAVELLED, and it
        # silently caps any curriculum whose goals approach it: a 200 m goal
        # under a 200 m cap ends the episode at the instant the goal becomes
        # reachable, and a CHAIN of such goals can never complete its second
        # leg. It must exceed chain_max_goals * max_goal_distance.
        self.done_dist = self.distance_completed >= float(
            os.environ.get("MAX_DISTANCE_M", 200))
        # Check distance to goal and resample if too close
        goal_loc = carla_location_to_np_array(core.destination.location)
        hero_loc = carla_location_to_np_array(hero.get_transform().location)
        distance_to_goal = np.linalg.norm(goal_loc[:2] - hero_loc[:2])
       
     
        
        # Check success condition
        # self.done_dist = self.distance_travelled >= max(50,50*round(self.time_episode/4e4))
        if distance_to_goal < 2:  # Resample when within 2m of goal (for 4-8m range)
            self.passed_chckp = True
            self.checkpoint_count += 1
            print("Checkppoint Passed")
            core.resample_goal()
        # WRONG-DIRECTION TERMINATION, env-var gated and OFF by default.
        #
        # It fires when heading alignment to the goal drops below -0.8 inside a
        # junction -- i.e. the car is pointing more than ~143 degrees away from
        # the goal. That is a reasonable guard on a straight road and a bad one
        # in a junction, which is exactly where it applies: turning through a
        # junction legitimately swings the heading away from the goal for
        # several steps before it comes back. Terminating there punishes the
        # manoeuvre the agent is meant to be learning, and it matters now that
        # JUNCTION_GOAL_FRAC routes most episodes through one.
        #
        # WRONG_DIR_TERMINAL=1 restores it.
        self.done_wrong_direction = False
        if (os.environ.get("WRONG_DIR_TERMINAL", "0") == "1"
                and wp is not None and wp.is_junction and not self.eval_mode):
            self.done_wrong_direction = self.dot < -0.8
            
            if self.done_wrong_direction:
                print(f"Wrong direction at junction: alignment={self.dot:.3f} ({np.rad2deg(np.arccos(np.clip(self.dot, -1, 1))):.1f}°)")
        
        # Speed-based termination (currently disabled)
        self.done_speed = False
        
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
            # PUBLISHED SO THE REWARD WRAPPER CAN END THE EPISODE EARLY.
            # `wedged` is set once and never cleared within an episode (see
            # reset and compute_action), so from the step it goes True the
            # vehicle is frozen under full brake for every remaining step. Its
            # future is therefore fully determined: it cannot reach the goal,
            # and the time it will burn is exactly the steps left on the clock.
            wedged=bool(getattr(self, "wedged", False)),
            # ALONG-ROAD metres left to the goal, or None when no route
            # exists. Phi(s) = -route_remaining_m; GatedReward differences it.
            route_remaining_m=self._route_remaining(core),
            # Whether the hero was inside a junction. The wedge's exemption
            # turns on this, so it has to be visible in the [Why] breakdown --
            # otherwise "crossings are forgiven correctly" is unfalsifiable.
            in_junction=bool(getattr(self, "in_junction", False)),
            distance_travelled=self.distance_completed,
            goal_directed_distances=self.distance_completed,
            checkpoints_passed=self.checkpoint_count,
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
        
        Rewards progress along the road direction (projected onto waypoint forward vector)
        and penalizes collisions, lane departures, and erratic driving.
        """
        # Reward weights
        WEIGHTS = {
            'collision_penalty': 3.0,
            'lane_departure_penalty': 5.0,
            'goal_reached_bonus': 5.0,
            'checkpoint_bonus': 2.0,
            'wrong_junction_penalty': 3.0,
            'base_penalty': 4e-3,
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
        if self.last_distance is None:
            goal_loc = carla_location_to_np_array(core.destination.location)
            self.last_distance = np.linalg.norm(goal_loc[:2] - carla_location_to_np_array(hero_location)[:2])

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
            goal_loc = carla_location_to_np_array(core.destination.location)
            current_distance = np.linalg.norm(goal_loc[:2] - carla_location_to_np_array(hero_location)[:2])
            delta_distance = self.last_distance - current_distance
            self.last_distance = current_distance
        
        # Calculate raw distance traveled
        raw_distance = float(np.linalg.norm(displacement_vector))
        # Project displacement onto road direction (waypoint forward vector)
        projected_distance = 0.0
        
        if wp is not None and raw_distance > 0:
            # Get the forward vector from the waypoint's transform
            forward_vector =core.destination.get_forward_vector() if wp.is_junction else wp.transform.get_forward_vector()
            
            # Convert to 2D numpy array (ignore z-component for ground plane)
            forward_2d = np.array([forward_vector.x, forward_vector.y])
            # forward_norm = np.linalg.norm(forward_2d)
            # print(forward_2d)e
            # if forward_norm > 1e-6:  # Avoid division by zero
            #     forward_2d_normalized = forward_2d / forward_norm
                
                # Project displacement onto road direction
                # Positive = moving forward along road, Negative = moving backward
            projected_distance = float(np.dot(displacement_vector, forward_2d))
            projected_distance = projected_distance
        # else:
        #      projected_distance = float(np.dot(displacement_vector, self.heading_vec_2d[:2]))
        
        # Update cumulative distances
        self.distance_travelled += projected_distance
        self.distance_completed += projected_distance # Only count forward progress
        self.last_location = hero_location
        
        # Calculate speed factor (ratio of current speed to target)
        speed_factor = hero_velocity / self.target_speed if self.target_speed > 0 else 0
        
        # Action smoothness penalties
        steer_change = abs(self.steer - self.prev_steer)
        throttle_change = abs(self.throttle - self.prev_throttle)
        action_smoothness_penalty = steer_change
        
        # Initialize reward with base penalty (encourages finishing quickly)
        reward = -WEIGHTS['base_penalty'] - steer_change
        
        # Main reward: distance progress along road direction
        # Only reward when moving at reasonable speed (10% to 100% of target)
        if  speed_factor < 1.0:
            # Combine projected distance (road-aligned) and goal progress
            reward += delta_distance + projected_distance
        
        # Apply penalties for bad behavior
        if self.done_wrong_direction:
            reward -= WEIGHTS['wrong_junction_penalty']
            print(f"Wrong Goal reached! Distance completed: {self.distance_completed:.2f}m")
        if self.done_speed:
            reward -= WEIGHTS['collision_penalty']
            
        if self.collision:
            reward -= WEIGHTS['collision_penalty']
            
        if self.diff_lane:
            reward -= WEIGHTS['lane_departure_penalty']
        
        # Goal reached bonus
        if self.done_dist:
            reward += WEIGHTS['goal_reached_bonus']
            print(f"Goal reached! Distance completed: {self.distance_completed:.2f}m")

        # Checkpoint bonus (each goal resample)
        if self.passed_chckp:
            reward += WEIGHTS['checkpoint_bonus']
        
        # Handle NaN values (safety check)
        reward = float(np.nan_to_num(reward, nan=0.0))
        
        # Update info for logging
        self.info.update({
            'speed_factor': np.clip(speed_factor, 0, 1.0),
            'heading_alignment': self.dot,
            'action_smoothness_penalty': action_smoothness_penalty,
            'throttle_smoothness_penalty': throttle_change,
            'delta_distance': delta_distance,
            'projected_distance': projected_distance,
            # THE LANE'S OWN DIRECTION, never the destination's.
            #
            # `projected_distance` above is NOT usable by a goal-free explorer,
            # because at a junction it projects onto
            # core.destination.get_forward_vector() -- the GOAL's heading. That
            # makes the upstream lane-following term secretly goal-conditioned
            # on exactly the 60% of episodes JUNCTION_GOAL_FRAC routes through a
            # junction. Publishing the waypoint's own forward vector lets a
            # goal-free wrapper recompute the projection against the road
            # instead. Additive: the upstream term is left untouched so the
            # goal-conditioned runs are unaffected.
            'wp_forward': (np.array([wp.transform.get_forward_vector().x,
                                     wp.transform.get_forward_vector().y],
                                    dtype=np.float64)
                           if wp is not None else np.zeros(2)),
            'raw_distance': raw_distance,
            'wrong_direction_at_junction': self.done_wrong_direction,
            'total_reward': reward
        })
        
        self.rewards.append(reward)
        
        # Update previous action values for next smoothness calculation
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