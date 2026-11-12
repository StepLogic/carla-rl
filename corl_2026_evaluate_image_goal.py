#!/usr/bin/env python
"""
CoRL 2026: Evaluate all policies using the same dense graph navigation approach.

All methods (DrQ-IG, NoMaD, GNM, ViNT) use the same dense topological map
from town02_map/ with SIFT+VLAD + FAISS. The graph provides:
  - Shortest path from start to goal node
  - Subgoal images at each node along the path

For DrQ-IG: goal image is overridden with mapper subgoal image at each step.
For baselines: topomap is replaced by the path images from the mapper.

Usage:
    python corl_2026_evaluate_image_goal.py --model all --town Town02 \
        --benchmark-dir corl-2026-results/trajectories \
        --map-dir corl-2026-results/town02_map \
        --output-dir corl-2026-results \
        --save-videos
"""
import os
import pickle
import json
import argparse
from datetime import datetime

import cv2
import numpy as np
import tqdm
import flax
from flax.training import checkpoints

os.environ['XLA_FLAGS'] = '--xla_gpu_enable_command_buffer='
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.20'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import torch
import torchvision.transforms as T
import torchvision.models as models

from rlib_integration.carla_env import CarlaEnv
from rlib_integration.agent import GlobalRoutePlanner
from rlib_integration.helper import ndarray_to_location, carla_location_to_np_array
from jaxrl2.agents import DrQLearner
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit
from src.configs.eval_env_config_image_goal import config as ig_config, STBL3EvalExperiment
from src.configs.drq_default import get_config
from src.agent_wrapper import SetPointAgent
from nomad_policy_dense import NoMaD_DenseMap
from gnm_policy_dense import GNM_Policy_DenseMap
from vint_policy_dense import ViNT_Policy_DenseMap
from eval_graph_builder_vlad_prebuilt import TopologicalGraphBuilder, GraphBuilderConfig, SIFTVLADConfig

flax.config.update('flax_use_orbax_checkpointing', True)

# ============================================================================
# DEFAULTS
# ============================================================================
TERMINATION_BUDGET = 2
MAX_EPISODE_STEPS = 3000
SEED = 42
SUBGOAL_REACH_DIST = 5.0  # meters to consider subgoal reached (GPS mode)
SUBGOAL_REACH_SIM = 0.6  # cosine similarity to consider subgoal reached (image mode)
REPLAN_MIN_PATH_LEN = 3  # replan if path has fewer nodes than this

MODEL_CHECKPOINTS = {
    "gnm": "/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/gnm.pth",
    "nomad": "/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/nomad.pth",
    "vint": "/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/vint.pth",
}

MODEL_CLASSES = {
    "gnm": GNM_Policy_DenseMap,
    "nomad": NoMaD_DenseMap,
    "vint": ViNT_Policy_DenseMap,
}

DRQ_IG_CHECKPOINT = "/home/robotlab/scratch/carla-rl/checkpoints/final_drq_ig/checkpoint_1"


# ============================================================================
# DENSE MAP NAVIGATOR
# ============================================================================

class DenseMapNavigator:
    """Unified navigator for all policies using the dense topological map."""

    def __init__(self, map_dir, localization_mode='gps'):
        self.map_dir = map_dir
        self.localization_mode = localization_mode
        self.mapper = None
        self.nodes_data = None
        self.positions = None
        self._load_map()

    def _load_map(self):
        """Load the dense topological map from pre-built artifacts."""
        graph_config = GraphBuilderConfig(feature_type='sift_vlad', use_gpu=True)
        self.mapper = TopologicalGraphBuilder(config=graph_config)
        self.mapper.load_from_directory(self.map_dir)

        # nodes_data: all original images with metadata (x, y, filename, etc.)
        self.nodes_data = self.mapper.nodes_data

        print(f"Loaded dense map: {len(self.mapper.nodes_data)} nodes, "
              f"{self.mapper.G.number_of_edges()} edges, "
              f"FAISS index: {self.mapper.all_features_normalized.shape[0]} features")

    def get_path_from_images(self, obs_img, goal_img, confidence_threshold=0.4):
        """Find shortest path using image-based localization.

        Uses SIFT+VLAD feature extraction and FAISS nearest-neighbor search
        to find start and goal nodes from camera observations.
        """
        start_idx, start_conf = self.mapper.find_nearest_node_from_image_with_confidence(
            obs_img, threshold=confidence_threshold
        )
        goal_idx, goal_conf = self.mapper.find_nearest_node_from_image_with_confidence(
            goal_img, threshold=confidence_threshold
        )

        print(f"  [Image loc] start_node={start_idx} (conf={start_conf:.3f}), "
              f"goal_node={goal_idx} (conf={goal_conf:.3f})")

        # Find path on full graph (all nodes, no index remapping needed)
        import networkx as nx
        try:
            path = nx.shortest_path(self.mapper.G, start_idx, goal_idx)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path = [start_idx, goal_idx]

        return path

    def get_path(self, start_location=None, goal_location=None, obs_img=None, goal_img=None):
        """Find shortest path on graph from start to goal.

        In GPS mode (default): uses location coordinates to find nearest nodes.
        In image mode: uses camera observations to find nearest nodes via SIFT+VLAD.
        """
        if self.localization_mode == 'image' and obs_img is not None and goal_img is not None:
            return self.get_path_from_images(obs_img, goal_img)

        # GPS mode: find nearest graph nodes by position over all nodes
        start_pos = np.array([start_location.x, start_location.y])
        goal_pos = np.array([goal_location.x, goal_location.y])

        import networkx as nx
        node_positions = np.array([[d['x'], d['y']] for d in self.nodes_data])

        start_idx = int(np.argmin(np.linalg.norm(node_positions - start_pos, axis=1)))
        goal_idx = int(np.argmin(np.linalg.norm(node_positions - goal_pos, axis=1)))

        print(f"  [GPS loc] start_node={start_idx}, goal_node={goal_idx}")

        try:
            path = nx.shortest_path(self.mapper.G, start_idx, goal_idx)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path = [start_idx, goal_idx]

        return path

    def get_subgoal_image(self, node_idx):
        """Get the RGB image for a map node."""
        images_dir = os.path.join(self.map_dir, "images")
        filename = self.nodes_data[node_idx]['filename']
        img_path = os.path.join(images_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_node_position(self, node_idx):
        """Get (x, y) position of a map node."""
        d = self.nodes_data[node_idx]
        return np.array([d['x'], d['y']])

    def extract_feature_from_image(self, image):
        """Extract SIFT+VLAD feature from an image (uniform interface)."""
        return self.mapper.extract_feature_from_image(image)

    def get_node_feature_normalized(self, node_idx):
        """Get L2-normalized feature vector for a node (uniform interface)."""
        return self.mapper.all_features_normalized[node_idx]


# ============================================================================
# DRQ IMAGE-GOAL EVALUATION
# ============================================================================

def load_drq_ig_agent(env, checkpoint_path, seed=42):
    """Load DrQ image-goal agent from checkpoint."""
    drq_config = get_config()
    agent = DrQLearner(
        seed,
        env.observation_space.sample(),
        env.action_space.sample(),
        **dict(drq_config)
    )
    state_dict = {
        'actor_params': agent._actor,
        'critic_params': agent._critic,
        'target_critic_params': agent._target_critic_params,
        'temp': agent._temp,
        'rng': agent._rng,
    }
    state_dict = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=state_dict)
    agent._actor = state_dict['actor_params']
    agent._critic = state_dict['critic_params']
    agent._target_critic_params = state_dict['target_critic_params']
    agent._temp = state_dict['temp']
    agent._rng = state_dict['rng']
    return agent


def evaluate_drq_ig_trajectory(env, agent, nav, trajectory_path, traj_id,
                                max_steps=MAX_EPISODE_STEPS, save_video=False):
    """Evaluate DrQ image-goal agent using dense map subgoals."""
    unwrapped_env = env.unwrapped

    # Load trajectory data
    aux_path = os.path.join(trajectory_path, "aux.pkl")
    with open(aux_path, "rb") as f:
        aux_data = pickle.load(f)

    start_location = ndarray_to_location(aux_data["start"])
    goal_location = ndarray_to_location(aux_data["goal"])

    unwrapped_env.set_start_transform(start_location)
    unwrapped_env.set_destination_transform(goal_location)

    # Compute shortest path distance
    route_planner = GlobalRoutePlanner(unwrapped_env.core.map, 2.0)
    try:
        trace = route_planner.trace_route(start_location, goal_location)
        shortest_distance = 0.0
        prev_wp = None
        for wp, _ in trace:
            if prev_wp is not None:
                shortest_distance += prev_wp.transform.location.distance(wp.transform.location)
            prev_wp = wp
        if shortest_distance <= 2.0:
            shortest_distance = start_location.distance(goal_location)
    except Exception:
        shortest_distance = start_location.distance(goal_location)

    # Get path on dense map — use image-based localization if in image mode
    if nav.localization_mode == 'image':
        # Need to reset env first to get observation images
        obs, info = env.reset()
        env.step([0.0, 0.0])
        obs_img = unwrapped_env.experiment.obs_img_vlad
        goal_img = unwrapped_env.experiment.goal_img_vlad
        path = nav.get_path(obs_img=obs_img, goal_img=goal_img)
        # Reset for the actual evaluation loop
        # (env.reset will be called again in the termination loop below)
    else:
        path = nav.get_path(start_location, goal_location)
    current_path_idx = 1  # Start navigating toward the second node (first is start)

    # Setup MobileNetV3 for goal image encoding (same as training)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    mobilenet = mobilenet.features[:4].to(device).eval()
    img_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((84, 84)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def encode_image(img):
        """Encode an RGB image using MobileNetV3 features[:4]."""
        if img is None:
            return np.zeros(2904, dtype=np.float32)
        img_tensor = img_transform(img.copy()).unsqueeze(0).to(device)
        with torch.no_grad():
            latent = mobilenet(img_tensor)
        return latent.squeeze(0).cpu().numpy().flatten()

    # Evaluation
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    SPL = []
    distance_completed = []
    agent_trajectory = []
    all_frames = []
    termination_budget = TERMINATION_BUDGET

    while termination_budget > 0:
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        traj_positions = []
        current_path_idx = 1  # Reset path index on retry

        # Replan until path has enough nodes or goal reached
        replan_attempts = 0
        while len(path) < REPLAN_MIN_PATH_LEN and replan_attempts < 10:
            replan_attempts += 1
            print(f"  [Replan] Path too short ({len(path)} nodes), replanning with image (attempt {replan_attempts})")
            obs_img = unwrapped_env.experiment.obs_img_vlad
            goal_img = unwrapped_env.experiment.goal_img_vlad
            path = nav.get_path(obs_img=obs_img, goal_img=goal_img)
        print(f"  [Replan] Final path: {len(path)} nodes (after {replan_attempts} attempts)")

        env.step([0.0, 0.0])

        while not done and episode_length < max_steps:
            # Track position
            curr_transform = unwrapped_env.core.hero.get_transform()
            hero_pos = np.array([curr_transform.location.x, curr_transform.location.y])
            traj_positions.append([curr_transform.location.x, curr_transform.location.y])

            # Replan until path has enough nodes or goal reached
            if current_path_idx >= len(path) and not info.get('is_success', False):
                replan_attempts = 0
                while len(path) < REPLAN_MIN_PATH_LEN and not info.get('is_success', False) and replan_attempts < 10:
                    replan_attempts += 1
                    print(f"  [Replan] Path exhausted at step {episode_length}, replanning with image (attempt {replan_attempts})")
                    if nav.localization_mode == 'image':
                        obs_img_now = unwrapped_env.experiment.obs_img_vlad
                        goal_img_now = unwrapped_env.experiment.goal_img_vlad
                        new_path = nav.get_path(obs_img=obs_img_now, goal_img=goal_img_now)
                    else:
                        from carla import Location
                        curr_loc = curr_transform.location
                        current_location = Location(x=hero_pos[0], y=hero_pos[1], z=curr_loc.z)
                        new_path = nav.get_path(current_location, goal_location)
                    if len(new_path) >= REPLAN_MIN_PATH_LEN:
                        path = new_path
                        current_path_idx = 1
                        print(f"  [Replan] New path: {len(path)} nodes")
                        break
                    # Drive one step to change observation before retrying
                    action = agent.eval_actions(obs)
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    obs, reward, done, truncated, info = env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    if done or truncated:
                        break

            # Advance subgoal if we're close enough to current target
            if current_path_idx < len(path):
                if nav.localization_mode == 'image':
                    # Image-based subgoal advancement: use VLAD cosine similarity
                    obs_img = unwrapped_env.experiment.obs_img_vlad
                    obs_feat = nav.extract_feature_from_image(obs_img)
                    obs_feat_norm = obs_feat / max(np.linalg.norm(obs_feat), 1e-8)
                    subgoal_feat = nav.get_node_feature_normalized(path[current_path_idx])
                    sim_to_subgoal = float(np.dot(obs_feat_norm, subgoal_feat))
                    while sim_to_subgoal >= SUBGOAL_REACH_SIM and current_path_idx < len(path) - 1:
                        current_path_idx += 1
                        subgoal_feat = nav.get_node_feature_normalized(path[current_path_idx])
                        sim_to_subgoal = float(np.dot(obs_feat_norm, subgoal_feat))
                else:
                    # GPS-based subgoal advancement: use spatial distance
                    subgoal_pos = nav.get_node_position(path[current_path_idx])
                    dist_to_subgoal = np.linalg.norm(hero_pos - subgoal_pos)
                    while dist_to_subgoal < SUBGOAL_REACH_DIST and current_path_idx < len(path) - 1:
                        current_path_idx += 1
                        subgoal_pos = nav.get_node_position(path[current_path_idx])
                        dist_to_subgoal = np.linalg.norm(hero_pos - subgoal_pos)

            # Override goal image with mapper subgoal
            if current_path_idx < len(path):
                # breakpoint()
                subgoal_img = nav.get_subgoal_image(path[current_path_idx])
                subgoal_features = encode_image(subgoal_img)
                obs["goal"] = subgoal_features
                # Set goal mask flag to 1.0 (goal is visible)
                vec = obs["vector"].copy()
                vec[3] = 1.0
                obs["vector"] = vec

            # Capture frame for video
            if save_video and isinstance(obs, dict) and "pixels" in obs:
                frame = obs["pixels"].copy()
                if frame.ndim == 3 and frame.shape[2] >= 3:
                    all_frames.append(frame[:, :, :3])

            # Take action
            action = agent.eval_actions(obs)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            if done or truncated:
                break

        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        agent_trajectory.extend(traj_positions)

        attempt_info = dict(info) if isinstance(info, dict) else {}
        if "is_success" in info:
            success_rate.append(float(info["is_success"]))
        else:
            success_rate.append(0.0)

        if "distance_completed" in info:
            agent_distance = float(info["distance_completed"])
            distance_completed.append(agent_distance)
            S = int(info.get("is_success", 0))
            spl = S * shortest_distance / max(shortest_distance, agent_distance)
            SPL.append(spl)

        if info.get("is_success", False):
            termination_budget = 0
        else:
            termination_budget -= 1
            unwrapped_env.set_start_transform(start_location)

    # Failure mode analysis
    final_pos = np.array(agent_trajectory[-1]) if agent_trajectory else np.array([0, 0])
    goal_pos_2d = np.array([goal_location.x, goal_location.y])
    final_dist_to_goal = float(np.linalg.norm(final_pos - goal_pos_2d))

    # Determine failure subtype from the last failed attempt
    failure_subtype = "none"
    if success_rate and np.max(success_rate) == 0:
        last_len = episode_lengths[-1] if episode_lengths else 0
        # Check if timeout (truncated at max_steps)
        if last_len >= max_steps - 1:
            failure_subtype = "timeout"
        else:
            failure_subtype = "deviation"

    # Mutually exclusive failure mode detection with priority:
    # 1. localization_failure, 2. planning_failure, 3. navigation_failure
    success = float(np.max(success_rate)) if success_rate else 0.0

    # Check for localization failure (image mode only)
    localization_failure = False
    if nav.localization_mode == 'image' and success == 0:
        localization_failure = len(path) <= 2

    # Check for planning failure (only if not localization)
    planning_failure = False
    if not localization_failure and success == 0:
        planning_failure = len(path) < REPLAN_MIN_PATH_LEN

    # Navigation failure = success == 0 and neither localization nor planning failed
    navigation_failure = (success == 0) and not localization_failure and not planning_failure

    results = {
        "trajectory_id": traj_id,
        "model": "drq_ig",
        "agent_trajectory": agent_trajectory,
        "start_location": aux_data["start"].tolist() if hasattr(aux_data["start"], 'tolist') else aux_data["start"],
        "goal_location": aux_data["goal"].tolist() if hasattr(aux_data["goal"], 'tolist') else aux_data["goal"],
        "success": float(np.max(success_rate)) if success_rate else 0.0,
        "success_rate": float(np.mean(success_rate)) if success_rate else 0.0,
        "mean_spl": float(np.mean(SPL)) if SPL else 0.0,
        "max_spl": float(np.max(SPL)) if SPL else 0.0,
        "mean_reward": float(np.mean(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "mean_distance": float(np.mean(distance_completed)) if distance_completed else 0.0,
        "shortest_distance": shortest_distance,
        "num_attempts": TERMINATION_BUDGET - termination_budget,
        "path_length": len(path),
        # Failure mode fields
        "planning_failure": bool(planning_failure),
        "localization_failure": bool(localization_failure),
        "navigation_failure": bool(np.max(success_rate) == 0 if success_rate else True),
        "failure_subtype": failure_subtype,
        "final_distance_to_goal": final_dist_to_goal,
        "episode_lengths": episode_lengths,
        "attempt_successes": [bool(s) for s in success_rate],
    }
    if save_video and all_frames:
        results["frames"] = all_frames
    return results


def evaluate_baseline_trajectory(env, model_type, nav, trajectory_path, traj_id,
                                   max_steps=MAX_EPISODE_STEPS, save_video=False):
    """Evaluate a baseline model using dense map subgoals as topomap."""
    unwrapped_env = env.unwrapped

    # Load trajectory data
    aux_path = os.path.join(trajectory_path, "aux.pkl")
    with open(aux_path, "rb") as f:
        aux_data = pickle.load(f)

    start_location = ndarray_to_location(aux_data["start"])
    goal_location = ndarray_to_location(aux_data["goal"])

    unwrapped_env.set_start_transform(start_location)
    unwrapped_env.set_destination_transform(goal_location)

    # Compute shortest path distance
    route_planner = GlobalRoutePlanner(unwrapped_env.core.map, 2.0)
    try:
        trace = route_planner.trace_route(start_location, goal_location)
        shortest_distance = 0.0
        prev_wp = None
        for wp, _ in trace:
            if prev_wp is not None:
                shortest_distance += prev_wp.transform.location.distance(wp.transform.location)
            prev_wp = wp
        if shortest_distance <= 2.0:
            shortest_distance = start_location.distance(goal_location)
    except Exception:
        shortest_distance = start_location.distance(goal_location)

    # Get path on dense map — use image-based localization if in image mode
    if nav.localization_mode == 'image':
        obs, info = env.reset()
        env.step([0.0, 0.0])
        obs_img = unwrapped_env.experiment.obs_img_vlad
        goal_img = unwrapped_env.experiment.goal_img_vlad
        path = nav.get_path(obs_img=obs_img, goal_img=goal_img)
    else:
        path = nav.get_path(start_location, goal_location)

    # Initialize baseline model with navigator + path (no disk I/O)
    MODEL = MODEL_CLASSES[model_type]
    checkpoint_path = MODEL_CHECKPOINTS[model_type]

    agent = MODEL(
        ckpt_path=checkpoint_path,
        nav=nav,
        path=path,
        search_radius=3 if model_type == "nomad" else 4,
        threshold=8.0,
        traj_id=traj_id,
    )

    sp_agent = SetPointAgent(unwrapped_env.core.hero)

    # Track subgoal advancement
    current_subgoal_idx = 1  # Start from second node

    # Evaluation
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    SPL = []
    distance_completed = []
    agent_trajectory = []
    all_frames = []
    termination_budget = TERMINATION_BUDGET

    while termination_budget > 0:
        observation, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        traj_positions = []
        current_subgoal_idx = 1  # Reset on retry

        # Replan until path has enough nodes or goal reached
        replan_attempts = 0
        while len(path) < REPLAN_MIN_PATH_LEN and replan_attempts < 10:
            replan_attempts += 1
            print(f"  [Replan] Path too short ({len(path)} nodes), replanning with image (attempt {replan_attempts})")
            obs_img = unwrapped_env.experiment.obs_img_vlad
            goal_img = unwrapped_env.experiment.goal_img_vlad
            path = nav.get_path(obs_img=obs_img, goal_img=goal_img)
        print(f"  [Replan] Final path: {len(path)} nodes (after {replan_attempts} attempts)")
        agent.path = path

        while not done and episode_length < max_steps:
            # Replan until path has enough nodes or goal reached
            if current_subgoal_idx >= len(path) and not info.get('is_success', False):
                curr_transform = unwrapped_env.core.hero.get_transform()
                hero_pos = np.array([curr_transform.location.x, curr_transform.location.y])
                replan_attempts = 0
                while len(path) < REPLAN_MIN_PATH_LEN and not info.get('is_success', False) and replan_attempts < 10:
                    replan_attempts += 1
                    print(f"  [Replan] Path exhausted at step {episode_length}, replanning with image (attempt {replan_attempts})")
                    if nav.localization_mode == 'image':
                        obs_img_now = unwrapped_env.experiment.obs_img_vlad
                        goal_img_now = unwrapped_env.experiment.goal_img_vlad
                        new_path = nav.get_path(obs_img=obs_img_now, goal_img=goal_img_now)
                    else:
                        from carla import Location
                        curr_loc = curr_transform.location
                        current_location = Location(x=hero_pos[0], y=hero_pos[1], z=curr_loc.z)
                        new_path = nav.get_path(current_location, goal_location)
                    if len(new_path) >= REPLAN_MIN_PATH_LEN:
                        path = new_path
                        agent.path = path
                        current_subgoal_idx = 1
                        print(f"  [Replan] New path: {len(path)} nodes")
                        break
                    # Drive one step to change observation before retrying
                    obs_bgr = unwrapped_env.experiment.obs_img
                    waypoints = np.array(agent.eval_action(obs_bgr[..., None]))
                    actions = sp_agent.run_step(waypoints, True)
                    observation, reward, done, truncated, info = env.step(actions)
                    episode_reward += reward
                    episode_length += 1
                    if done or truncated:
                        break

            # Advance subgoal based on localization mode
            if current_subgoal_idx < len(path):
                if nav.localization_mode == 'image':
                    # Image-based subgoal advancement: use VLAD cosine similarity
                    obs_img = unwrapped_env.experiment.obs_img_vlad
                    obs_feat = nav.extract_feature_from_image(obs_img)
                    obs_feat_norm = obs_feat / max(np.linalg.norm(obs_feat), 1e-8)
                    subgoal_feat = nav.get_node_feature_normalized(path[current_subgoal_idx])
                    sim_to_subgoal = float(np.dot(obs_feat_norm, subgoal_feat))
                    while sim_to_subgoal >= SUBGOAL_REACH_SIM and current_subgoal_idx < len(path) - 1:
                        current_subgoal_idx += 1
                        subgoal_feat = nav.get_node_feature_normalized(path[current_subgoal_idx])
                        sim_to_subgoal = float(np.dot(obs_feat_norm, subgoal_feat))
                else:
                    # GPS-based subgoal advancement: use spatial distance
                    curr_transform = unwrapped_env.core.hero.get_transform()
                    hero_pos = np.array([curr_transform.location.x, curr_transform.location.y])
                    subgoal_pos = nav.get_node_position(path[current_subgoal_idx])
                    dist = np.linalg.norm(hero_pos - subgoal_pos)
                    while dist < SUBGOAL_REACH_DIST and current_subgoal_idx < len(path) - 1:
                        current_subgoal_idx += 1
                        subgoal_pos = nav.get_node_position(path[current_subgoal_idx])
                        dist = np.linalg.norm(hero_pos - subgoal_pos)

            # Sync dense-map policy search window with external subgoal
            if hasattr(agent, 'set_subgoal_idx'):
                agent.set_subgoal_idx(current_subgoal_idx)

            # Baseline prediction — obs_img is RGB from post_process_image.
            # callback_obs applies cv2.COLOR_BGR2RGB (swaps R<->B), so pre-swap
            # to BGR so the model receives correct RGB after that conversion.
            obs_bgr = unwrapped_env.experiment.obs_img

            # Pass the subgoal image from the dense mapper path directly to
            # the model so it uses the navigator's subgoal as its goal.
            if current_subgoal_idx < len(path):
                subgoal_img = nav.get_subgoal_image(path[current_subgoal_idx])
                waypoints = np.array(agent.eval_action(obs_bgr[..., None], goal_image=subgoal_img))
            else:
                waypoints = np.array(agent.eval_action(obs_bgr[..., None]))
            # print((len(path),cur))
            actions = sp_agent.run_step(waypoints, True)
            loc_obj = unwrapped_env.core.hero.get_transform().location
            traj_positions.append((loc_obj.x, loc_obj.y, loc_obj.z))

            if save_video and isinstance(observation, dict) and "pixels" in observation:
                frame = observation["pixels"].copy()
                if frame.ndim == 3 and frame.shape[2] >= 3:
                    all_frames.append(frame[:, :, :3])

            _, reward, done, truncated, info = env.step(actions)
            episode_reward += reward
            episode_length += 1
            done = done or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        agent_trajectory.extend(traj_positions)

        if "is_success" in info:
            success_rate.append(float(info["is_success"]))
        else:
            success_rate.append(0.0)

        if "distance_travelled" in info:
            agent_distance = float(info["distance_travelled"])
            distance_completed.append(agent_distance)
            S = int(info.get("is_success", 0))
            spl = S * shortest_distance / max(shortest_distance, agent_distance)
            SPL.append(spl)

        if info.get("is_success", False):
            termination_budget = 0
        else:
            termination_budget -= 1
            unwrapped_env.set_start_transform(start_location)

    # Failure mode analysis
    final_pos = np.array(agent_trajectory[-1][:2]) if agent_trajectory else np.array([0, 0])
    goal_pos_2d = np.array([goal_location.x, goal_location.y])
    final_dist_to_goal = float(np.linalg.norm(final_pos - goal_pos_2d))

    # Determine failure subtype from the last failed attempt
    failure_subtype = "none"
    if success_rate and np.max(success_rate) == 0:
        last_len = episode_lengths[-1] if episode_lengths else 0
        if last_len >= max_steps - 1:
            failure_subtype = "timeout"
        else:
            failure_subtype = "deviation"

    # Mutually exclusive failure mode detection with priority:
    # 1. localization_failure, 2. planning_failure, 3. navigation_failure
    success = float(np.max(success_rate)) if success_rate else 0.0

    localization_failure = False
    if nav.localization_mode == 'image' and success == 0:
        localization_failure = len(path) <= 2

    planning_failure = False
    if not localization_failure and success == 0:
        planning_failure = len(path) < REPLAN_MIN_PATH_LEN

    navigation_failure = (success == 0) and not localization_failure and not planning_failure

    results = {
        "trajectory_id": traj_id,
        "model": model_type,
        "agent_trajectory": agent_trajectory,
        "start_location": aux_data["start"].tolist() if hasattr(aux_data["start"], 'tolist') else aux_data["start"],
        "goal_location": aux_data["goal"].tolist() if hasattr(aux_data["goal"], 'tolist') else aux_data["goal"],
        "success": float(np.max(success_rate)) if success_rate else 0.0,
        "success_rate": float(np.mean(success_rate)) if success_rate else 0.0,
        "mean_spl": float(np.mean(SPL)) if SPL else 0.0,
        "max_spl": float(np.max(SPL)) if SPL else 0.0,
        "mean_reward": float(np.mean(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "mean_distance": float(np.mean(distance_completed)) if distance_completed else 0.0,
        "shortest_distance": shortest_distance,
        "num_attempts": TERMINATION_BUDGET - termination_budget,
        "num_topomap_nodes": len(path),
        # Failure mode fields
        "planning_failure": bool(planning_failure),
        "localization_failure": bool(localization_failure) or len(path) < 3,
        "navigation_failure": bool(np.max(success_rate) == 0 if success_rate else True),
        "failure_subtype": failure_subtype,
        "final_distance_to_goal": final_dist_to_goal,
        "episode_lengths": episode_lengths,
        "attempt_successes": [bool(s) for s in success_rate],
    }
    if save_video and all_frames:
        results["frames"] = all_frames
    return results


# ============================================================================
# COMPARISON AND RESULTS
# ============================================================================

def print_comparison_table(all_results):
    print(f"\n{'='*70}")
    print("CoRL 2026 - COMPARISON TABLE (Image-Goal, Dense Map)")
    print(f"{'='*70}")
    header = f"{'Model':<12} {'Success%':<12} {'Mean SPL':<12} {'Mean Dist':<12} {'Trajs':<8}"
    print(header)
    print("-" * 56)
    for model_name, results in all_results.items():
        if "error" in results:
            print(f"{model_name:<12} ERROR: {results['error']}")
            continue
        traj_results = [r for r in results.get("per_trajectory_results", []) if "error" not in r]
        if not traj_results:
            print(f"{model_name:<12} No valid results")
            continue
        success_rate = np.mean([r["success"] for r in traj_results]) * 100
        mean_spl = np.mean([r["mean_spl"] for r in traj_results])
        mean_dist = np.mean([r["mean_distance"] for r in traj_results])
        num_trajs = len(traj_results)
        print(f"{model_name:<12} {success_rate:>8.1f}%   {mean_spl:>10.3f}   {mean_dist:>10.1f}m   {num_trajs:>4}")
    print("-" * 56)

    print(f"\n{'='*70}")
    print("Per-Trajectory Results")
    print(f"{'='*70}")
    models = list(all_results.keys())
    num_trajs = max(
        len(all_results[m].get("per_trajectory_results", []))
        for m in models if "per_trajectory_results" in all_results[m]
    ) if models else 0
    header = f"{'Traj':<6}"
    for m in models:
        header += f"{m:<12}"
    print(header)
    print("-" * (6 + 12 * len(models)))
    for traj_idx in range(num_trajs):
        row = f"{traj_idx:<6}"
        for m in models:
            per_traj = all_results[m].get("per_trajectory_results", [])
            if traj_idx < len(per_traj):
                r = per_traj[traj_idx]
                if "error" in r:
                    row += f"{'ERR':<12}"
                else:
                    success = "Y" if r["success"] > 0 else "N"
                    row += f"{success} SPL={r['mean_spl']:.2f}  "
            else:
                row += f"{'N/A':<12}"
        print(row)


def save_results(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for model_name, results in all_results.items():
        pkl_path = os.path.join(output_dir, f"{model_name}_ig_dense_results.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(results, f)
        summary = {
            "model": model_name,
            "timestamp": timestamp,
            "variant": "image_goal_dense_map",
            "overall_success_rate": results.get("overall_success_rate", 0),
            "overall_mean_spl": results.get("overall_mean_spl", 0),
            "overall_mean_distance": results.get("overall_mean_distance", 0),
            "num_trajectories": results.get("num_trajectories", 0),
        }
        json_path = os.path.join(output_dir, f"{model_name}_ig_dense_results.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

    comparison = {}
    for model_name, results in all_results.items():
        if "error" in results:
            comparison[model_name] = {"error": results["error"]}
        else:
            comparison[model_name] = {
                "success_rate": results.get("overall_success_rate", 0),
                "mean_spl": results.get("overall_mean_spl", 0),
                "mean_distance": results.get("overall_mean_distance", 0),
                "num_trajectories": results.get("num_trajectories", 0),
            }
    comparison_path = os.path.join(output_dir, "comparison_summary_ig_dense.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    table_path = os.path.join(output_dir, "comparison_table_ig_dense.txt")
    with open(table_path, 'w') as f:
        f.write("CoRL 2026 - Model Comparison (Image-Goal, Dense Map, No Compass)\n")
        f.write("=" * 56 + "\n")
        f.write(f"{'Model':<12} {'Success%':<12} {'Mean SPL':<12} {'Mean Dist':<12}\n")
        f.write("-" * 48 + "\n")
        for model_name, metrics in comparison.items():
            if "error" in metrics:
                f.write(f"{model_name:<12} ERROR: {metrics['error']}\n")
            else:
                f.write(f"{model_name:<12} {metrics['success_rate']*100:>8.1f}%   "
                        f"{metrics['mean_spl']:>10.3f}   {metrics['mean_distance']:>10.1f}m\n")
    print(f"\nResults saved to {output_dir}")


def save_trajectory_video(frames, output_path, fps=15):
    if not frames:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        if frame.shape[2] == 3:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            writer.write(frame)
    writer.release()


# ============================================================================
# MAIN
# ============================================================================

def main():
    global TERMINATION_BUDGET, MAX_EPISODE_STEPS

    parser = argparse.ArgumentParser(description="CoRL 2026: Evaluate all policies using dense map navigation")
    parser.add_argument('--model', type=str, default='all',
                        choices=['drq_ig', 'nomad', 'gnm', 'vint', 'all'])
    parser.add_argument('--town', type=str, default='Town02')
    parser.add_argument('--benchmark-dir', type=str, default='corl-2026-results/trajectories')
    parser.add_argument('--map-dir', type=str, default='corl-2026-results/town02_map')
    parser.add_argument('--output-dir', type=str, default='corl-2026-results')
    parser.add_argument('--checkpoint', type=str, default=DRQ_IG_CHECKPOINT)
    parser.add_argument('--max-steps', type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument('--termination-budget', type=int, default=TERMINATION_BUDGET)
    parser.add_argument('--save-videos', action='store_true')
    parser.add_argument('--video-dir', type=str, default=None)
    parser.add_argument('--localization-mode', type=str, default='image',
                        choices=['gps', 'image'],
                        help='Node localization mode: gps (position-based) or image (SIFT+VLAD FAISS)')
    args = parser.parse_args()

    TERMINATION_BUDGET = args.termination_budget
    MAX_EPISODE_STEPS = args.max_steps

    video_dir = args.video_dir or os.path.join(args.output_dir, "videos")
    if args.save_videos:
        os.makedirs(video_dir, exist_ok=True)

    models_to_evaluate = [ 'nomad', 'gnm', 'vint','drq_ig'] if args.model == 'all' else [args.model]

    # Connect to existing CARLA server
    ig_config["env_config"]["carla"]["start_server"] = False
    ig_config["env_config"]["carla"]["port"] = 2000

    # Load benchmark info
    with open(os.path.join(args.benchmark_dir, "benchmark_info.json"), 'r') as f:
        benchmark_info = json.load(f)

    # Load dense map navigator
    print(f"\nLoading dense topological map from {args.map_dir}...")
    nav = DenseMapNavigator(args.map_dir, localization_mode=args.localization_mode)

    print(f"\n{'#'*60}")
    print(f"CoRL 2026 EVALUATION (Image-Goal, Dense Map, No Compass)")
    print(f"{'#'*60}")
    print(f"Benchmark: {args.benchmark_dir}")
    print(f"Town: {benchmark_info.get('town', args.town)}")
    print(f"Trajectories: {len(benchmark_info['trajectories'])}")
    print(f"Models: {models_to_evaluate}")
    print(f"Map: {args.map_dir}")
    print(f"Localization: {args.localization_mode}")
    print(f"Save videos: {args.save_videos}")

    # Setup environment
    ig_config["env_config"]["carla"]["town"] = benchmark_info.get('town', args.town)
    env = CarlaEnv(config=ig_config)
    env.experiment.eval_mode = True
    env.experiment.map_mode = False
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    env = RecordEpisodeStatistics(env)

    all_results = {}

    for model_type in models_to_evaluate:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_type.upper()}")
        print(f"{'='*60}")

        if model_type == 'drq_ig':
            agent = load_drq_ig_agent(env, args.checkpoint, SEED)

            traj_results = []
            for traj_info in tqdm.tqdm(benchmark_info["trajectories"], desc=f"DrQ-IG eval"):
                traj_id = traj_info["trajectory_id"]
                traj_path = traj_info["path"]

                try:
                    result = evaluate_drq_ig_trajectory(
                        env, agent, nav, traj_path, traj_id, MAX_EPISODE_STEPS,
                        save_video=args.save_videos
                    )
                    traj_results.append(result)
                    print(f"  Traj {traj_id}: Success={result['success']:.0f}, "
                          f"SPL={result['mean_spl']:.3f}, Dist={result['mean_distance']:.1f}m")

                    if args.save_videos and 'frames' in result:
                        vpath = os.path.join(video_dir, f"drq_ig_traj_{traj_id:02d}.mp4")
                        save_trajectory_video(result['frames'], vpath)
                        result['video_path'] = vpath
                        del result['frames']
                except Exception as e:
                    print(f"  ERROR on trajectory {traj_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    traj_results.append({"trajectory_id": traj_id, "model": "drq_ig", "error": str(e), "success": 0.0})

            successful = [r for r in traj_results if "error" not in r]
            all_results["drq_ig"] = {
                "model": "drq_ig",
                "num_trajectories": len(benchmark_info["trajectories"]),
                "num_successful_evals": len(successful),
                "overall_success_rate": np.mean([r["success"] for r in successful]) if successful else 0.0,
                "overall_mean_spl": np.mean([r["mean_spl"] for r in successful]) if successful else 0.0,
                "overall_mean_distance": np.mean([r["mean_distance"] for r in successful]) if successful else 0.0,
                "per_trajectory_results": traj_results,
            }

        else:
            traj_results = []
            for traj_info in tqdm.tqdm(benchmark_info["trajectories"], desc=f"{model_type} eval"):
                traj_id = traj_info["trajectory_id"]
                traj_path = traj_info["path"]

                try:
                    result = evaluate_baseline_trajectory(
                        env, model_type, nav, traj_path, traj_id, MAX_EPISODE_STEPS,
                        save_video=args.save_videos
                    )
                    traj_results.append(result)
                    print(f"  Traj {traj_id}: Success={result['success']:.0f}, "
                          f"SPL={result['mean_spl']:.3f}, Dist={result['mean_distance']:.1f}m")

                    if args.save_videos and 'frames' in result:
                        vpath = os.path.join(video_dir, f"{model_type}_traj_{traj_id:02d}.mp4")
                        save_trajectory_video(result['frames'], vpath)
                        result['video_path'] = vpath
                        del result['frames']
                except Exception as e:
                    print(f"  ERROR on trajectory {traj_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    traj_results.append({"trajectory_id": traj_id, "model": model_type, "error": str(e), "success": 0.0})

            successful = [r for r in traj_results if "error" not in r]
            all_results[model_type] = {
                "model": model_type,
                "num_trajectories": len(benchmark_info["trajectories"]),
                "num_successful_evals": len(successful),
                "overall_success_rate": np.mean([r["success"] for r in successful]) if successful else 0.0,
                "overall_mean_spl": np.mean([r["mean_spl"] for r in successful]) if successful else 0.0,
                "overall_mean_distance": np.mean([r["mean_distance"] for r in successful]) if successful else 0.0,
                "per_trajectory_results": traj_results,
            }

    env.close()
    print_comparison_table(all_results)
    save_results(all_results, args.output_dir)
    return all_results


if __name__ == "__main__":
    main()