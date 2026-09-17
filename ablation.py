#!/usr/bin/env python
"""
Ablation: Effect of map sparsity on navigation success rate.

Creates sparse topological maps by subsampling the dense map at different
sparsity levels, then evaluates all agents on each sparsity level.

Two sparsity modes:
  - Spatial (default): keeps nodes at least min_distance meters apart
  - Feature-similarity: discards nodes whose VLAD feature cosine similarity
    exceeds a threshold (i.e., keeps only nodes that look sufficiently different)

Reuses benchmark trajectories from corl-2026-results-dense-image.

Usage:
    # Spatial ablation (default):
    python ablation.py

    # Feature-similarity ablation:
    python ablation.py --sparsity-mode similarity --similarity-thresholds 0.7 0.8 0.9 0.95 0.99

    # Combined (both spatial and similarity):
    python ablation.py --sparsity-mode both

    # Specific model and sparsity level:
    python ablation.py --model nomad --min-distances 5 10 20

    # Skip evaluation, just generate sparse maps:
    python ablation.py --maps-only

    # Plot results from a previous run:
    python ablation.py --plot-only --results-dir ablation-results
"""
import os
import pickle
import json
import argparse

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
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rlib_integration.carla_env import CarlaEnv
from rlib_integration.agent import GlobalRoutePlanner
from rlib_integration.helper import ndarray_to_location
from jaxrl2.agents import DrQLearner
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit
from src.configs.eval_env_config_image_goal import config as ig_config
from src.configs.drq_default import get_config
from src.agent_wrapper import SetPointAgent
from nomad_policy_dense import NoMaD_DenseMap
from gnm_policy_dense import GNM_Policy_DenseMap
from vint_policy_dense import ViNT_Policy_DenseMap
from eval_graph_builder_vlad_prebuilt import (
    SIFTVLADConfig, FAISSIndex
)

flax.config.update('flax_use_orbax_checkpointing', True)

# ============================================================================
# DEFAULTS
# ============================================================================
TERMINATION_BUDGET = 2
MAX_EPISODE_STEPS = 3000
SEED = 42
REPLAN_MIN_PATH_LEN = 3  # replan if path has fewer nodes than this
SUBGOAL_REACH_DIST = 5.0
SUBGOAL_REACH_SIM = 0.6

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

# Sparsity levels
DEFAULT_MIN_DISTANCES = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
DEFAULT_SIMILARITY_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6]

MODEL_COLORS = {
    'drq_ig': '#2196F3',
    'nomad': '#FF9800',
    'gnm': '#4CAF50',
    'vint': '#F44336',
}

MODEL_LABELS = {
    'drq_ig': 'DrQ-IG',
    'nomad': 'NoMaD',
    'gnm': 'GNM',
    'vint': 'ViNT',
}


# ============================================================================
# SPARSE MAP BUILDER
# ============================================================================

def build_sparse_map(map_dir, min_distance=None, similarity_threshold=None, output_dir=None):
    """Build a sparse topological map by subsampling nodes.

    Two modes:
      - Spatial (min_distance): keeps nodes at least min_distance meters apart
      - Feature-similarity (similarity_threshold): discards nodes whose VLAD
        cosine similarity exceeds the threshold (keeps visually distinct nodes)
      - Both: apply spatial first, then feature-similarity
    Builds sequential graph edges and FAISS index.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load full map data
    with open(os.path.join(map_dir, "map_data.pkl"), 'rb') as f:
        map_data = pickle.load(f)
    all_nodes = map_data['metadata']['images']

    # Load features
    with open(os.path.join(map_dir, "features.pkl"), 'rb') as f:
        features = pickle.load(f)
    all_features = features['all_features']
    feature_dim = features['feature_dim']

    # Load VLAD encoder
    vlad_config = SIFTVLADConfig()
    from eval_graph_builder_vlad_prebuilt import SIFTVLADFeatureExtractor
    feature_extractor = SIFTVLADFeatureExtractor(vlad_config)
    feature_extractor.load(os.path.join(map_dir, "vlad_encoder.pkl"))

    # Load aux for yaws
    with open(os.path.join(map_dir, "aux.pkl"), 'rb') as f:
        aux = pickle.load(f)
    yaws = aux['yaws']

    # Normalize features for cosine similarity
    all_features_norm = all_features.copy()
    norms = np.linalg.norm(all_features_norm, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    all_features_norm = all_features_norm / norms

    # Subsampling
    kept_indices = list(range(len(all_nodes)))

    if min_distance is not None and similarity_threshold is not None:
        # Both: spatial first, then similarity
        kept_indices = _spatial_subsample(all_nodes, kept_indices, min_distance)
        kept_indices = _similarity_subsample(all_features_norm, kept_indices, similarity_threshold)
    elif similarity_threshold is not None:
        # Feature-similarity only
        kept_indices = _similarity_subsample(all_features_norm, kept_indices, similarity_threshold)
    elif min_distance is not None:
        # Spatial only
        kept_indices = _spatial_subsample(all_nodes, kept_indices, min_distance)
    else:
        raise ValueError("Must specify min_distance, similarity_threshold, or both")

    kept_indices = sorted(kept_indices)

    # Build sparse node list
    sparse_nodes = []
    for idx in kept_indices:
        node = dict(all_nodes[idx])
        node['compass'] = np.radians(yaws[idx])
        sparse_nodes.append(node)

    # Build sequential directed graph
    G = nx.DiGraph()
    for i, node in enumerate(sparse_nodes):
        G.add_node(i, **node, original_idx=kept_indices[i])
    for i in range(len(sparse_nodes) - 1):
        G.add_edge(i, i + 1, weight=1, type='sequential')

    # Build FAISS index on sparse features
    sparse_features = all_features[kept_indices].copy()
    main_index = FAISSIndex(feature_dim, use_gpu=True, index_type='flat', n_samples=len(kept_indices))
    main_index.build(sparse_features, verbose=False)

    # Save
    with open(os.path.join(output_dir, "sparse_nodes.pkl"), 'wb') as f:
        pickle.dump(sparse_nodes, f)
    with open(os.path.join(output_dir, "sparse_graph.pkl"), 'wb') as f:
        pickle.dump(nx.node_link_data(G), f)
    with open(os.path.join(output_dir, "sparse_features.pkl"), 'wb') as f:
        pickle.dump({'features': sparse_features, 'feature_dim': feature_dim}, f)
    meta = {
        'min_distance': min_distance,
        'similarity_threshold': similarity_threshold,
        'total_nodes': len(all_nodes),
        'sparse_nodes': len(sparse_nodes),
        'reduction': 1 - len(sparse_nodes) / len(all_nodes),
    }
    with open(os.path.join(output_dir, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    mode_desc = []
    if min_distance is not None:
        mode_desc.append(f"min_dist={min_distance}m")
    if similarity_threshold is not None:
        mode_desc.append(f"sim_thresh={similarity_threshold}")
    desc = ", ".join(mode_desc)
    print(f"  {desc}: {len(all_nodes)} -> {len(sparse_nodes)} nodes "
          f"({meta['reduction']:.1%} reduction, {G.number_of_edges()} edges)")

    return sparse_nodes, G, sparse_features, feature_extractor, main_index, feature_dim


def _spatial_subsample(all_nodes, candidate_indices, min_distance):
    """Keep nodes at least min_distance meters apart."""
    positions = np.array([[all_nodes[i]['x'], all_nodes[i]['y']] for i in candidate_indices])
    kept = [0]
    for i in range(1, len(candidate_indices)):
        dists = np.linalg.norm(positions[kept] - positions[i], axis=1)
        if np.min(dists) >= min_distance:
            kept.append(i)
    return [candidate_indices[i] for i in kept]


def _similarity_subsample(all_features_norm, candidate_indices, similarity_threshold):
    """Keep nodes whose max cosine similarity to all kept nodes is below threshold.

    Nodes with high feature similarity (cosine sim >= threshold) are considered
    visually redundant and discarded. Lower threshold = fewer nodes kept.
    """
    kept = [candidate_indices[0]]
    kept_feats = all_features_norm[candidate_indices[0]:candidate_indices[0]+1].copy()

    for i in range(1, len(candidate_indices)):
        idx = candidate_indices[i]
        feat = all_features_norm[idx]
        sims = np.dot(kept_feats, feat)
        if np.max(sims) < similarity_threshold:
            kept.append(idx)
            kept_feats = np.vstack([kept_feats, feat[np.newaxis, :]])

    return kept


# ============================================================================
# SPARSE MAP NAVIGATOR
# ============================================================================

class SparseMapNavigator:
    """Navigator for a spatially or feature-subsampled topological map."""

    def __init__(self, map_dir, sparse_dir, localization_mode='gps',
                 image_confidence_threshold=0.4):
        self.map_dir = map_dir
        self.sparse_dir = sparse_dir
        self.localization_mode = localization_mode
        self.image_confidence_threshold = image_confidence_threshold

        # Load sparse artifacts (entropy maps use graph.pkl/features.pkl; ablation variants use sparse_*.pkl)
        with open(os.path.join(sparse_dir, "sparse_nodes.pkl"), 'rb') as f:
            self.nodes_data = pickle.load(f)

        graph_path = os.path.join(sparse_dir, "sparse_graph.pkl")
        if not os.path.exists(graph_path):
            graph_path = os.path.join(sparse_dir, "graph.pkl")
        with open(graph_path, 'rb') as f:
            graph_data = pickle.load(f)
        self.G = nx.node_link_graph(graph_data)

        features_path = os.path.join(sparse_dir, "sparse_features.pkl")
        if not os.path.exists(features_path):
            features_path = os.path.join(sparse_dir, "features.pkl")
        with open(features_path, 'rb') as f:
            feat_data = pickle.load(f)
        self.features = feat_data['features']
        self.feature_dim = feat_data['feature_dim']

        # Load VLAD encoder
        vlad_config = SIFTVLADConfig()
        from eval_graph_builder_vlad_prebuilt import SIFTVLADFeatureExtractor
        self.feature_extractor = SIFTVLADFeatureExtractor(vlad_config)
        self.feature_extractor.load(os.path.join(map_dir, "vlad_encoder.pkl"))

        # Build FAISS index
        self.main_index = FAISSIndex(self.feature_dim, use_gpu=True, index_type='flat',
                                      n_samples=len(self.nodes_data))
        # breakpoint()
        self.main_index.build(self.features.copy(), verbose=False)

        with open(os.path.join(sparse_dir, "meta.json"), 'r') as f:
            self.meta = json.load(f)

        # Load MobileNet for DrQ-IG goal encoding
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        self.mobilenet = self.mobilenet.features[:4].to(self.device).eval()
        self.img_transform = T.Compose([
            T.ToPILImage(), T.Resize((84, 84)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_path(self, start_location=None, goal_location=None, obs_img=None, goal_img=None):
        if self.localization_mode == 'image' and obs_img is not None and goal_img is not None:
            return self._get_path_from_images(obs_img, goal_img)

        # GPS mode
        start_pos = np.array([start_location.x, start_location.y])
        goal_pos = np.array([goal_location.x, goal_location.y])
        node_positions = np.array([[d['x'], d['y']] for d in self.nodes_data])
        start_idx = int(np.argmin(np.linalg.norm(node_positions - start_pos, axis=1)))
        goal_idx = int(np.argmin(np.linalg.norm(node_positions - goal_pos, axis=1)))

        try:
            path = nx.shortest_path(self.G, start_idx, goal_idx)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path = [start_idx, goal_idx]
        return path

    def _get_path_from_images(self, obs_img, goal_img):
        """Find path using SIFT+VLAD image features."""
        obs_feat = self.feature_extractor.extract(obs_img)
        goal_feat = self.feature_extractor.extract(goal_img)

        query_obs = obs_feat.reshape(1, -1).astype(np.float32)
        query_goal = goal_feat.reshape(1, -1).astype(np.float32)

        sims_obs, idxs_obs = self.main_index.search(query_obs, k=5)
        sims_goal, idxs_goal = self.main_index.search(query_goal, k=5)

        start_sim = float(np.max(sims_obs[0]))
        goal_sim = float(np.max(sims_goal[0]))
        start_idx = int(idxs_obs[0, np.argmax(sims_obs[0])])
        goal_idx = int(idxs_goal[0, np.argmax(sims_goal[0])])
        # breakpoint()
        print(f"  [Image loc] start_node={start_idx} (conf={start_sim:.3f}), "
              f"goal_node={goal_idx} (conf={goal_sim:.3f})")

        try:
            path = nx.shortest_path(self.G, start_idx, goal_idx)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path = [start_idx, goal_idx]
        return path

    def get_subgoal_image(self, node_idx):
        images_dir = os.path.join(self.map_dir, "images")
        filename = self.nodes_data[node_idx]['filename']
        img_path = os.path.join(images_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_node_position(self, node_idx):
        d = self.nodes_data[node_idx]
        return np.array([d['x'], d['y']])

    def encode_image(self, img):
        if img is None:
            return np.zeros(2904, dtype=np.float32)
        img_tensor = self.img_transform(img.copy()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent = self.mobilenet(img_tensor)
        return latent.squeeze(0).cpu().numpy().flatten()

    def extract_feature_normalized(self, img):
        """Extract SIFT+VLAD feature and normalize for cosine similarity."""
        feat = self.feature_extractor.extract(img)
        norm = np.linalg.norm(feat)
        if norm < 1e-8:
            return feat
        return feat / norm

    def extract_feature_from_image(self, img):
        """Extract SIFT+VLAD feature from image (uniform interface with DenseMapNavigator)."""
        return self.feature_extractor.extract(img)

    def get_node_feature_normalized(self, node_idx):
        """Get the normalized feature for a sparse node (for cosine similarity)."""
        feat = self.features[node_idx]
        norm = np.linalg.norm(feat)
        if norm < 1e-8:
            return feat
        return feat / norm


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def load_drq_ig_agent(env, checkpoint_path, seed=42):
    drq_config = get_config()
    agent = DrQLearner(seed, env.observation_space.sample(), env.action_space.sample(), **dict(drq_config))
    state_dict = {
        'actor_params': agent._actor, 'critic_params': agent._critic,
        'target_critic_params': agent._target_critic_params,
        'temp': agent._temp, 'rng': agent._rng,
    }
    state_dict = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=state_dict)
    agent._actor = state_dict['actor_params']
    agent._critic = state_dict['critic_params']
    agent._target_critic_params = state_dict['target_critic_params']
    agent._temp = state_dict['temp']
    agent._rng = state_dict['rng']
    return agent


def evaluate_drq_ig(env, agent, nav, trajectory_path, traj_id, max_steps=MAX_EPISODE_STEPS):
    unwrapped = env.unwrapped
    with open(os.path.join(trajectory_path, "aux.pkl"), "rb") as f:
        aux = pickle.load(f)
    start_location = ndarray_to_location(aux["start"])
    goal_location = ndarray_to_location(aux["goal"])
    unwrapped.set_start_transform(start_location)
    unwrapped.set_destination_transform(goal_location)

    route_planner = GlobalRoutePlanner(unwrapped.core.map, 2.0)
    trace = route_planner.trace_route(start_location, goal_location)
    shortest_distance = 0.0
    prev_wp = None
    for wp, _ in trace:
        if prev_wp is not None:
            shortest_distance += prev_wp.transform.location.distance(wp.transform.location)
        prev_wp = wp
    if shortest_distance <= 2.0:
        shortest_distance = start_location.distance(goal_location)

    path = nav.get_path(start_location=start_location, goal_location=goal_location)
    current_path_idx = 1
    success_rate, SPL, distance_completed, agent_trajectory = [], [], [], []
    termination_budget = TERMINATION_BUDGET

    while termination_budget > 0:
        obs, info = env.reset()
        done = False
        episode_length = 0
        traj_positions = []
        current_path_idx = 1

        # Replan until path has enough nodes or goal reached
        replan_attempts = 0
        while len(path) < REPLAN_MIN_PATH_LEN and replan_attempts < 10:
            replan_attempts += 1
            print(f"  [Replan] Path too short ({len(path)} nodes), replanning with image (attempt {replan_attempts})")
            obs_img = unwrapped.experiment.obs_img_vlad
            goal_img = unwrapped.experiment.goal_img_vlad
            path = nav.get_path(obs_img=obs_img, goal_img=goal_img)
        print(f"  [Replan] Final path: {len(path)} nodes (after {replan_attempts} attempts)")

        env.step([0.0, 0.0])

        while not done and episode_length < max_steps:
            curr_transform = unwrapped.core.hero.get_transform()
            hero_pos = np.array([curr_transform.location.x, curr_transform.location.y])
            traj_positions.append([curr_transform.location.x, curr_transform.location.y])

            # Replan until path has enough nodes or goal reached
            if current_path_idx >= len(path) and not info.get('is_success', False):
                replan_attempts = 0
                while len(path) < REPLAN_MIN_PATH_LEN and not info.get('is_success', False) and replan_attempts < 10:
                    replan_attempts += 1
                    print(f"  [Replan] Path exhausted at step {episode_length}, replanning with image (attempt {replan_attempts})")
                    if nav.localization_mode == 'image':
                        obs_img_now = unwrapped.experiment.obs_img_vlad
                        goal_img_now = unwrapped.experiment.goal_img_vlad
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

            if current_path_idx < len(path):
                if nav.localization_mode == 'image':
                    # Image-based subgoal advancement: use VLAD cosine similarity
                    obs_img = unwrapped.experiment.obs_img_vlad
                    obs_feat_norm = nav.extract_feature_normalized(obs_img)
                    subgoal_feat_norm = nav.get_node_feature_normalized(path[current_path_idx])
                    sim_to_sg = float(np.dot(obs_feat_norm, subgoal_feat_norm))
                    while sim_to_sg >= SUBGOAL_REACH_SIM and current_path_idx < len(path) - 1:
                        current_path_idx += 1
                        subgoal_feat_norm = nav.get_node_feature_normalized(path[current_path_idx])
                        sim_to_sg = float(np.dot(obs_feat_norm, subgoal_feat_norm))
                else:
                    # GPS-based subgoal advancement: use spatial distance
                    subgoal_pos = nav.get_node_position(path[current_path_idx])
                    dist_to_sg = np.linalg.norm(hero_pos - subgoal_pos)
                    while dist_to_sg < SUBGOAL_REACH_DIST and current_path_idx < len(path) - 1:
                        current_path_idx += 1
                        subgoal_pos = nav.get_node_position(path[current_path_idx])
                        dist_to_sg = np.linalg.norm(hero_pos - subgoal_pos)

            if current_path_idx < len(path):
                subgoal_img = nav.get_subgoal_image(path[current_path_idx])
                subgoal_features = nav.encode_image(subgoal_img)
                obs["goal"] = subgoal_features
                vec = obs["vector"].copy()
                vec[3] = 1.0
                obs["vector"] = vec

            action = agent.eval_actions(obs)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, _, done, truncated, info = env.step(action)
            episode_length += 1
            if done or truncated:
                break

        agent_trajectory.extend(traj_positions)
        if "is_success" in info:
            success_rate.append(float(info["is_success"]))
        else:
            success_rate.append(0.0)
        if "distance_completed" in info:
            agent_distance = float(info["distance_completed"])
            distance_completed.append(agent_distance)
            S = int(info.get("is_success", 0))
            SPL.append(S * shortest_distance / max(shortest_distance, agent_distance))

        if info.get("is_success", False):
            termination_budget = 0
        else:
            termination_budget -= 1
            unwrapped.set_start_transform(start_location)

    return {
        "trajectory_id": traj_id, "model": "drq_ig",
        "agent_trajectory": agent_trajectory,
        "success": float(np.max(success_rate)) if success_rate else 0.0,
        "success_rate": float(np.mean(success_rate)) if success_rate else 0.0,
        "mean_spl": float(np.mean(SPL)) if SPL else 0.0,
        "mean_distance": float(np.mean(distance_completed)) if distance_completed else 0.0,
        "path_length": len(path), "num_attempts": TERMINATION_BUDGET - termination_budget,
    }


def evaluate_baseline(env, model_type, nav, trajectory_path, traj_id, max_steps=MAX_EPISODE_STEPS):
    unwrapped = env.unwrapped
    with open(os.path.join(trajectory_path, "aux.pkl"), "rb") as f:
        aux = pickle.load(f)
    start_location = ndarray_to_location(aux["start"])
    goal_location = ndarray_to_location(aux["goal"])
    unwrapped.set_start_transform(start_location)
    unwrapped.set_destination_transform(goal_location)

    route_planner = GlobalRoutePlanner(unwrapped.core.map, 2.0)
    trace = route_planner.trace_route(start_location, goal_location)
    shortest_distance = 0.0
    prev_wp = None
    for wp, _ in trace:
        if prev_wp is not None:
            shortest_distance += prev_wp.transform.location.distance(wp.transform.location)
        prev_wp = wp
    if shortest_distance <= 2.0:
        shortest_distance = start_location.distance(goal_location)

    path = nav.get_path(start_location=start_location, goal_location=goal_location)

    MODEL = MODEL_CLASSES[model_type]
    ckpt = MODEL_CHECKPOINTS[model_type]
    class NavAdapter:
        def __init__(self, sparse_nav, path_nodes):
            self.nav = sparse_nav
            self.path = path_nodes
        def get_subgoal_image(self, idx):
            return self.nav.get_subgoal_image(idx)
        def get_node_position(self, idx):
            return self.nav.get_node_position(idx)

    adapter = NavAdapter(nav, path)
    agent = MODEL(ckpt_path=ckpt, nav=adapter, path=path,
                   search_radius=3 if model_type == "nomad" else 4,
                   threshold=8.0, traj_id=traj_id)

    sp_agent = SetPointAgent(unwrapped.core.hero)
    current_subgoal_idx = 1
    success_rate, SPL, distance_completed, agent_trajectory = [], [], [], []
    termination_budget = TERMINATION_BUDGET

    while termination_budget > 0:
        _, info = env.reset()
        done = False
        episode_length = 0
        traj_positions = []
        current_subgoal_idx = 1

        # Replan until path has enough nodes or goal reached
        replan_attempts = 0
        while len(path) < REPLAN_MIN_PATH_LEN and replan_attempts < 10:
            replan_attempts += 1
            print(f"  [Replan] Path too short ({len(path)} nodes), replanning with image (attempt {replan_attempts})")
            obs_img = unwrapped.experiment.obs_img_vlad
            goal_img = unwrapped.experiment.goal_img_vlad
            path = nav.get_path(obs_img=obs_img, goal_img=goal_img)
            adapter.path = path
            agent.path = path
        print(f"  [Replan] Final path: {len(path)} nodes (after {replan_attempts} attempts)")

        while not done and episode_length < max_steps:
            # Replan until path has enough nodes or goal reached
            if current_subgoal_idx >= len(path) and not info.get('is_success', False):
                curr_transform = unwrapped.core.hero.get_transform()
                hero_pos = np.array([curr_transform.location.x, curr_transform.location.y])
                replan_attempts = 0
                while len(path) < REPLAN_MIN_PATH_LEN and not info.get('is_success', False) and replan_attempts < 10:
                    replan_attempts += 1
                    print(f"  [Replan] Path exhausted at step {episode_length}, replanning with image (attempt {replan_attempts})")
                    if nav.localization_mode == 'image':
                        obs_img_now = unwrapped.experiment.obs_img_vlad
                        goal_img_now = unwrapped.experiment.goal_img_vlad
                        new_path = nav.get_path(obs_img=obs_img_now, goal_img=goal_img_now)
                    else:
                        from carla import Location
                        curr_loc = curr_transform.location
                        current_location = Location(x=hero_pos[0], y=hero_pos[1], z=curr_loc.z)
                        new_path = nav.get_path(current_location, goal_location)
                    if len(new_path) >= REPLAN_MIN_PATH_LEN:
                        path = new_path
                        adapter.path = path
                        agent.path = path
                        current_subgoal_idx = 1
                        print(f"  [Replan] New path: {len(path)} nodes")
                        break
                    # Drive one step to change observation before retrying
                    obs_bgr = unwrapped.experiment.obs_img
                    waypoints = np.array(agent.eval_action(obs_bgr[..., None]))
                    actions = sp_agent.run_step(waypoints, True)
                    _, _, done, truncated, info = env.step(actions)
                    episode_length += 1
                    if done or truncated:
                        break

            if current_subgoal_idx < len(path):
                if nav.localization_mode == 'image':
                    # Image-based subgoal advancement: use VLAD cosine similarity
                    obs_img = unwrapped.experiment.obs_img_vlad
                    obs_feat_norm = nav.extract_feature_normalized(obs_img)
                    subgoal_feat_norm = nav.get_node_feature_normalized(path[current_subgoal_idx])
                    sim_to_sg = float(np.dot(obs_feat_norm, subgoal_feat_norm))
                    while sim_to_sg >= SUBGOAL_REACH_SIM and current_subgoal_idx < len(path) - 1:
                        current_subgoal_idx += 1
                        subgoal_feat_norm = nav.get_node_feature_normalized(path[current_subgoal_idx])
                        sim_to_sg = float(np.dot(obs_feat_norm, subgoal_feat_norm))
                else:
                    # GPS-based subgoal advancement: use spatial distance
                    curr_transform = unwrapped.core.hero.get_transform()
                    hero_pos = np.array([curr_transform.location.x, curr_transform.location.y])
                    subgoal_pos = nav.get_node_position(path[current_subgoal_idx])
                    dist = np.linalg.norm(hero_pos - subgoal_pos)
                    while dist < SUBGOAL_REACH_DIST and current_subgoal_idx < len(path) - 1:
                        current_subgoal_idx += 1
                        subgoal_pos = nav.get_node_position(path[current_subgoal_idx])
                        dist = np.linalg.norm(hero_pos - subgoal_pos)

            if hasattr(agent, 'set_subgoal_idx'):
                agent.set_subgoal_idx(current_subgoal_idx)

            obs_bgr = unwrapped.experiment.obs_img
            if current_subgoal_idx < len(path):
                subgoal_img = nav.get_subgoal_image(path[current_subgoal_idx])
                waypoints = np.array(agent.eval_action(obs_bgr[..., None], goal_image=subgoal_img))
            else:
                waypoints = np.array(agent.eval_action(obs_bgr[..., None]))

            actions = sp_agent.run_step(waypoints, True)
            loc_obj = unwrapped.core.hero.get_transform().location
            traj_positions.append((loc_obj.x, loc_obj.y, loc_obj.z))
            _, _, done, truncated, info = env.step(actions)
            episode_length += 1
            done = done or truncated

        agent_trajectory.extend(traj_positions)
        if "is_success" in info:
            success_rate.append(float(info["is_success"]))
        else:
            success_rate.append(0.0)
        if "distance_travelled" in info:
            agent_distance = float(info["distance_travelled"])
            distance_completed.append(agent_distance)
            S = int(info.get("is_success", 0))
            SPL.append(S * shortest_distance / max(shortest_distance, agent_distance))

        if info.get("is_success", False):
            termination_budget = 0
        else:
            termination_budget -= 1
            unwrapped.set_start_transform(start_location)

    return {
        "trajectory_id": traj_id, "model": model_type,
        "agent_trajectory": agent_trajectory,
        "success": float(np.max(success_rate)) if success_rate else 0.0,
        "success_rate": float(np.mean(success_rate)) if success_rate else 0.0,
        "mean_spl": float(np.mean(SPL)) if SPL else 0.0,
        "mean_distance": float(np.mean(distance_completed)) if distance_completed else 0.0,
        "num_topomap_nodes": len(path),
        "num_attempts": TERMINATION_BUDGET - termination_budget,
    }


# ============================================================================
# PLOTTING
# ============================================================================

def plot_ablation_results(results_dir, output_dir):
    """Plot ablation results: success rate and SPL vs number of nodes."""
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "ablation_results.pkl")
    with open(results_path, 'rb') as f:
        ablation_data = pickle.load(f)

    # Extract data for plotting
    models = ['drq_ig', 'nomad', 'gnm', 'vint']
    node_counts = sorted(ablation_data.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Success rate vs nodes
    ax = axes[0]
    for model in models:
        successes = [ablation_data[n].get(model, {}).get('success_rate', 0) for n in node_counts]
        ax.plot(node_counts, successes, 'o-', color=MODEL_COLORS[model],
                label=MODEL_LABELS[model], linewidth=2, markersize=6)
    ax.set_xlabel('Number of Map Nodes', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Success Rate vs Map Density', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Plot 2: SPL vs nodes
    ax = axes[1]
    for model in models:
        spls = [ablation_data[n].get(model, {}).get('mean_spl', 0) for n in node_counts]
        ax.plot(node_counts, spls, 'o-', color=MODEL_COLORS[model],
                label=MODEL_LABELS[model], linewidth=2, markersize=6)
    ax.set_xlabel('Number of Map Nodes', fontsize=12)
    ax.set_ylabel('Mean SPL', fontsize=12)
    ax.set_title('SPL vs Map Density', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Plot 3: Mean distance vs nodes
    ax = axes[2]
    for model in models:
        dists = [ablation_data[n].get(model, {}).get('mean_distance', 0) for n in node_counts]
        ax.plot(node_counts, dists, 'o-', color=MODEL_COLORS[model],
                label=MODEL_LABELS[model], linewidth=2, markersize=6)
    ax.set_xlabel('Number of Map Nodes', fontsize=12)
    ax.set_ylabel('Mean Distance (m)', fontsize=12)
    ax.set_title('Distance Completed vs Map Density', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ablation_results.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Individual per-model plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for idx, model in enumerate(models):
        ax = axes[idx // 2, idx % 2]
        successes = [ablation_data[n].get(model, {}).get('success_rate', 0) for n in node_counts]
        spls = [ablation_data[n].get(model, {}).get('mean_spl', 0) for n in node_counts]

        color = MODEL_COLORS[model]
        ax.plot(node_counts, successes, 'o-', color=color, label='Success Rate', linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(node_counts, spls, 's--', color=color, alpha=0.6, label='SPL', linewidth=2)

        ax.set_xlabel('Number of Map Nodes', fontsize=11)
        ax.set_ylabel('Success Rate', fontsize=11, color=color)
        ax2.set_ylabel('SPL', fontsize=11, color=color, alpha=0.6)
        ax.set_title(MODEL_LABELS[model], fontsize=13)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        # Mark the best success rate
        best_idx = np.argmax(successes)
        best_nodes = node_counts[best_idx]
        ax.axvline(x=best_nodes, color='gray', linestyle=':', alpha=0.5)
        ax.annotate(f'Best: {best_nodes} nodes\n({successes[best_idx]:.0%})',
                     xy=(best_nodes, successes[best_idx]),
                     xytext=(best_nodes * 1.5, successes[best_idx] * 0.8),
                     fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ablation_per_model.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved plots to {output_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    global TERMINATION_BUDGET, MAX_EPISODE_STEPS

    parser = argparse.ArgumentParser(description="Ablation: Map sparsity vs navigation success")
    parser.add_argument('--map-dir', type=str,
                        default='corl-2026-results-dense-image/town02_map')
    parser.add_argument('--benchmark-dir', type=str,
                        default='corl-2026-results-dense-image/trajectories')
    parser.add_argument('--results-dir', type=str, default='ablation-results')
    parser.add_argument('--model', type=str, default='all',
                        choices=['drq_ig', 'nomad', 'gnm', 'vint', 'all'])
    parser.add_argument('--sparsity-mode', type=str, default='spatial',
                        choices=['spatial', 'similarity', 'both'],
                        help='Spatial: subsample by min distance. '
                             'Similarity: subsample by VLAD feature similarity. '
                             'Both: spatial then similarity.')
    parser.add_argument('--min-distances', type=float, nargs='+',
                        default=DEFAULT_MIN_DISTANCES,
                        help='Minimum inter-node distances (meters) for spatial mode')
    parser.add_argument('--similarity-thresholds', type=float, nargs='+',
                        default=DEFAULT_SIMILARITY_THRESHOLDS,
                        help='VLAD cosine similarity thresholds for feature mode. '
                             'Nodes with similarity >= threshold are discarded. '
                             'Typical range: 0.1-0.6 (adjacent nodes avg ~0.3)')
    parser.add_argument('--max-steps', type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument('--termination-budget', type=int, default=TERMINATION_BUDGET)
    parser.add_argument('--maps-only', action='store_true',
                        help='Only build sparse maps, skip evaluation')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only plot from existing results')
    parser.add_argument('--localization-mode', type=str, default='gps',
                        choices=['gps', 'image'])
    parser.add_argument('--image-confidence', type=float, default=0.4,
                        help='Minimum VLAD similarity for image localization (0-1)')
    args = parser.parse_args()

    TERMINATION_BUDGET = args.termination_budget
    MAX_EPISODE_STEPS = args.max_steps

    if args.plot_only:
        plot_ablation_results(args.results_dir, args.results_dir)
        return

    models_to_evaluate = (['drq_ig', 'nomad', 'gnm', 'vint']
                          if args.model == 'all' else [args.model])

    # Load benchmark
    with open(os.path.join(args.benchmark_dir, "benchmark_info.json"), 'r') as f:
        benchmark_info = json.load(f)

    # Determine sparsity levels
    if args.sparsity_mode == 'spatial':
        sparsity_configs = [(md, None) for md in args.min_distances]
    elif args.sparsity_mode == 'similarity':
        sparsity_configs = [(None, st) for st in args.similarity_thresholds]
    else:  # both
        sparsity_configs = [(md, st) for md in args.min_distances for st in args.similarity_thresholds]

    print(f"\n{'#'*60}")
    print("ABLATION: Map Sparsity vs Navigation Success")
    print(f"{'#'*60}")
    print(f"Map: {args.map_dir}")
    print(f"Benchmark: {args.benchmark_dir}")
    print(f"Trajectories: {len(benchmark_info['trajectories'])}")
    print(f"Models: {models_to_evaluate}")
    print(f"Sparsity mode: {args.sparsity_mode}")
    print(f"Configurations: {len(sparsity_configs)}")
    print(f"Localization: {args.localization_mode}")

    # Build sparse maps at each sparsity level
    print(f"\n{'='*60}")
    print("BUILDING SPARSE MAPS")
    print(f"{'='*60}")

    sparse_maps = {}
    for min_dist, sim_thresh in sparsity_configs:
        if min_dist is not None and sim_thresh is not None:
            sparse_dir = os.path.join(args.results_dir, f"sparse_{min_dist}m_sim{sim_thresh}")
        elif sim_thresh is not None:
            sparse_dir = os.path.join(args.results_dir, f"sparse_sim{sim_thresh}")
        else:
            sparse_dir = os.path.join(args.results_dir, f"sparse_{min_dist}m")

        sparse_data = build_sparse_map(args.map_dir, min_distance=min_dist,
                                       similarity_threshold=sim_thresh,
                                       output_dir=sparse_dir)
        key = (min_dist, sim_thresh)
        sparse_maps[key] = {
            'dir': sparse_dir,
            'data': sparse_data,
            'num_nodes': len(sparse_data[0]),
            'min_distance': min_dist,
            'similarity_threshold': sim_thresh,
        }

    if args.maps_only:
        print("\nSparse maps built. Done.")
        return

    # Setup environment
    ig_config["env_config"]["carla"]["start_server"] = False
    ig_config["env_config"]["carla"]["port"] = 2000
    ig_config["env_config"]["carla"]["town"] = benchmark_info.get('town', 'Town02')

    env = CarlaEnv(config=ig_config)
    env.experiment.eval_mode = True
    env.experiment.map_mode = False
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    env = RecordEpisodeStatistics(env)

    # Load DrQ agent once (shared across sparsity levels)
    drq_agent = None
    if 'drq_ig' in models_to_evaluate:
        print(f"\nLoading DrQ-IG agent...")
        drq_agent = load_drq_ig_agent(env, DRQ_IG_CHECKPOINT, SEED)

    # Run evaluation at each sparsity level
    ablation_results = {}

    for key, sparse_info in sparse_maps.items():
        num_nodes = sparse_info['num_nodes']
        min_dist = sparse_info['min_distance']
        sim_thresh = sparse_info['similarity_threshold']

        label_parts = []
        if min_dist is not None:
            label_parts.append(f"min_dist={min_dist}m")
        if sim_thresh is not None:
            label_parts.append(f"sim_thresh={sim_thresh}")
        label = ", ".join(label_parts)

        print(f"\n{'='*60}")
        print(f"EVALUATING: {label} ({num_nodes} nodes)")
        print(f"{'='*60}")

        nav = SparseMapNavigator(args.map_dir, sparse_info['dir'],
                                localization_mode=args.localization_mode,
                                image_confidence_threshold=args.image_confidence)

        level_results = {}

        for model_type in models_to_evaluate:
            print(f"\n--- {model_type.upper()} ({num_nodes} nodes) ---")
            traj_results = []

            for traj_info in tqdm.tqdm(benchmark_info["trajectories"],
                                        desc=f"{model_type} {num_nodes}n"):
                traj_id = traj_info["trajectory_id"]
                traj_path = os.path.join(args.benchmark_dir, f"trajectory_{traj_id:02d}")

                try:
                    if model_type == 'drq_ig':
                        result = evaluate_drq_ig(env, drq_agent, nav, traj_path, traj_id,
                                                  MAX_EPISODE_STEPS)
                    else:
                        result = evaluate_baseline(env, model_type, nav, traj_path, traj_id,
                                                   MAX_EPISODE_STEPS)
                    traj_results.append(result)
                    print(f"  Traj {traj_id}: Success={result['success']:.0f}, "
                          f"SPL={result['mean_spl']:.3f}")
                except Exception as e:
                    print(f"  ERROR on trajectory {traj_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    traj_results.append({"trajectory_id": traj_id, "model": model_type,
                                          "error": str(e), "success": 0.0})

            successful = [r for r in traj_results if "error" not in r]
            level_results[model_type] = {
                'success_rate': np.mean([r["success"] for r in successful]) if successful else 0.0,
                'mean_spl': np.mean([r["mean_spl"] for r in successful]) if successful else 0.0,
                'mean_distance': np.mean([r["mean_distance"] for r in successful]) if successful else 0.0,
                'num_nodes': num_nodes,
                'min_distance': min_dist,
                'similarity_threshold': sim_thresh,
                'per_trajectory': traj_results,
            }

            sr = level_results[model_type]['success_rate']
            spl = level_results[model_type]['mean_spl']
            print(f"  {model_type}: Success={sr:.0%}, SPL={spl:.3f}, Nodes={num_nodes}")

        ablation_results[num_nodes] = level_results

        # Save incremental results after each sparsity level
        os.makedirs(args.results_dir, exist_ok=True)
        with open(os.path.join(args.results_dir, "ablation_results.pkl"), 'wb') as f:
            pickle.dump(ablation_results, f)

    env.close()

    # Final save + plot
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "ablation_results.pkl"), 'wb') as f:
        pickle.dump(ablation_results, f)

    # Summary table
    print(f"\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")
    header = f"{'Nodes':<8}"
    for m in models_to_evaluate:
        header += f" {MODEL_LABELS.get(m, m):<12}"
    print(header)
    print("-" * (8 + 13 * len(models_to_evaluate)))

    for num_nodes in sorted(ablation_results.keys()):
        row = f"{num_nodes:<8}"
        for m in models_to_evaluate:
            sr = ablation_results[num_nodes].get(m, {}).get('success_rate', 0) * 100
            row += f" {sr:>8.0f}%    "
        print(row)

    # Plot
    plot_ablation_results(args.results_dir, args.results_dir)

    print(f"\nResults saved to {args.results_dir}/")


if __name__ == "__main__":
    main()