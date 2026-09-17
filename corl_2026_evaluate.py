#!/usr/bin/env python
"""
CoRL 2026: Unified evaluation of DrQ (FAISS graph nav) vs NoMaD/GNM/ViNT on benchmark trajectories.

Evaluates all models on the same 20 trajectories, computes success rate and SPL,
and saves reproducible results to corl-2026-results/.

Usage:
    # Evaluate a single model:
    python corl_2026_evaluate.py --model drq --town Town02 \\
        --benchmark-dir corl-2026-results/trajectories \\
        --map-dir corl-2026-results/town02_map \\
        --output-dir corl-2026-results

    # Evaluate all models:
    python corl_2026_evaluate.py --model all --town Town02 \\
        --benchmark-dir corl-2026-results/trajectories \\
        --map-dir corl-2026-results/town02_map \\
        --output-dir corl-2026-results
"""
import os
import signal
import pickle
import json
import time
import argparse
from datetime import datetime

import cv2
import numpy as np
import psutil
import tqdm
import flax
from flax.training import checkpoints
from scipy.spatial.transform import Rotation as R

os.environ['XLA_FLAGS'] = '--xla_gpu_enable_command_buffer='
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.20'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

from rlib_integration.carla_env import CarlaEnv
from rlib_integration.agent import GlobalRoutePlanner
from rlib_integration.helper import ndarray_to_location
from jaxrl2.agents import DrQLearner
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from jaxrl2.wrappers.timelimit import TimeLimit
from src.configs.eval_config_v2 import config
from src.configs.drq_default import get_config
from src.agent_wrapper import SetPointAgent
from navigation_policies.baseline_policies.nomad_policy import NoMaD
from navigation_policies.baseline_policies.gnm_policy import GNM_Policy
from navigation_policies.baseline_policies.vint_policy import ViNT_Policy
from eval_graph_builder_vlad import TopologicalGraphBuilder, GraphBuilderConfig, SIFTVLADConfig


def load_mapper(map_dir):
    """Load a pre-built TopologicalGraphBuilder from saved artifacts."""
    import networkx as nx
    from eval_graph_builder_vlad import SIFTVLADFeatureExtractor

    mapper = TopologicalGraphBuilder(config=GraphBuilderConfig(feature_type='sift_vlad', use_gpu=True))

    # Load filtered nodes data
    with open(os.path.join(map_dir, "map_filtered.pkl"), "rb") as f:
        mapper.nodes_data = pickle.load(f)

    # Add compass field from yaw (yaw is in degrees, compass is in radians)
    for node in mapper.nodes_data:
        if 'compass' not in node and 'yaw' in node:
            node['compass'] = np.radians(node['yaw'])

    # Load original nodes data — need full data with images
    # For compass lookup by original index, use the filtered nodes
    # The original_nodes_data is used for compass lookup by start_node index
    mapper.original_nodes_data = mapper.nodes_data

    # Load graph
    with open(os.path.join(map_dir, "graph.pkl"), "rb") as f:
        graph_data = pickle.load(f)
    mapper.G = nx.node_link_graph(graph_data)

    # Load features
    with open(os.path.join(map_dir, "features.pkl"), "rb") as f:
        features = pickle.load(f)
    mapper.all_features_normalized = features['all_features']
    mapper.filtered_features = features['filtered_features']
    mapper.feature_dim = features['feature_dim']

    # Load index mappings
    with open(os.path.join(map_dir, "index_mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)
    mapper.new_to_old = mappings['new_to_old']
    mapper.old_to_new = mappings['old_to_new']
    mapper.nodes_to_remove = set(mappings.get('nodes_to_remove', []))

    # Load VLAD encoder for feature extraction from new images
    vlad_config = SIFTVLADConfig()
    mapper.feature_extractor = SIFTVLADFeatureExtractor(vlad_config)
    mapper.feature_extractor.load(os.path.join(map_dir, "vlad_encoder.pkl"))

    # Build FAISS index from filtered features
    mapper.main_index = FAISSIndex(mapper.feature_dim, n_samples=len(mapper.filtered_features))
    mapper.main_index.build(mapper.filtered_features)

    print(f"Loaded mapper: {len(mapper.nodes_data)} nodes, {mapper.G.number_of_edges()} edges")
    return mapper

flax.config.update('flax_use_orbax_checkpointing', True)

# ============================================================================
# DEFAULTS
# ============================================================================
TERMINATION_BUDGET = 2
MAX_EPISODE_STEPS = 3000
SEED = 42

MODEL_CHECKPOINTS = {
    "gnm": "/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/gnm.pth",
    "nomad": "/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/nomad.pth",
    "vint": "/home/robotlab/scratch/carla-rl/dependencies/navigation_policies/navigation_policies/pretrained_models/vint.pth",
}

MODEL_CLASSES = {
    "gnm": GNM_Policy,
    "nomad": NoMaD,
    "vint": ViNT_Policy,
}

DRQ_CHECKPOINT = "/home/robotlab/scratch/carla-rl/checkpoints/final_drq/checkpoint_2"


# ============================================================================
# DRQ + FAISS NAVIGATION EVALUATION
# ============================================================================

def load_drq_agent(env, checkpoint_path, seed=42):
    """Load DrQ agent from checkpoint."""
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


def evaluate_drq_trajectory(env, agent, mapper, trajectory_path, traj_id, max_steps=MAX_EPISODE_STEPS, save_video=False):
    """Evaluate DrQ agent on a single benchmark trajectory using FAISS graph navigation."""
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
    trace = route_planner.trace_route(start_location, goal_location)
    shortest_distance = 0.0
    prev_wp = None
    for wp, _ in trace:
        if prev_wp is not None:
            shortest_distance += prev_wp.transform.location.distance(wp.transform.location)
        prev_wp = wp
    if shortest_distance <= 2.0:
        shortest_distance = start_location.distance(goal_location)

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

        # Initial step
        env.step([0.0, 0.0])

        nav_result = None
        is_localized = False
        get_next_heading = None

        while not done and episode_length < max_steps:
            # Navigate using images
            if not is_localized:
                nav_result = mapper.navigate_from_images(
                    unwrapped_env.experiment.obs_img,
                    unwrapped_env.experiment.goal_img,
                )
                is_localized = nav_result['is_localized']
                get_next_heading = nav_result['next_heading_func_image']

            # Track position
            curr_transform = unwrapped_env.core.hero.get_transform()
            traj_positions.append([curr_transform.location.x, curr_transform.location.y])

            # Capture frame for video
            if save_video and "pixels" in obs:
                frame = obs["pixels"].copy()
                if frame.ndim == 3 and frame.shape[2] >= 3:
                    all_frames.append(frame[:, :, :3])

            # Build observation vector
            vec = np.copy(obs["vector"])
            if is_localized:
                target_world_yaw = mapper.original_nodes_data[nav_result['start_node']]["compass"]
                car_yaw = np.radians(curr_transform.rotation.yaw)
                r_world_to_car = R.from_euler('z', car_yaw, degrees=False).as_matrix().T
                target_dir_world = np.array([np.cos(target_world_yaw), np.sin(target_world_yaw), 0.0])
                target_dir_car = r_world_to_car @ target_dir_world
                goal_yaw_car = np.arctan2(target_dir_car[1], target_dir_car[0])
                vec[3] = np.cos(goal_yaw_car)
                vec[4] = np.sin(goal_yaw_car)
                vec[5] = np.dot([1.0, 0.0], [vec[3], vec[4]])
                vec[6] = 1.0
            else:
                vec[3] = -1.0
                vec[4] = -1.0
                vec[5] = -1.0
                vec[6] = 0.0
            obs["vector"] = vec

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
            unwrapped_env.set_start_transform(unwrapped_env.experiment.wp.transform.location)

    results = {
        "trajectory_id": traj_id,
        "model": "drq",
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
        "path_length": len(nav_result['path']) if nav_result else 0,
    }
    if save_video and all_frames:
        results["frames"] = all_frames
    return results


def evaluate_baseline_trajectory(env, model_type, trajectory_path, traj_id, max_steps=MAX_EPISODE_STEPS, save_video=False):
    """Evaluate a baseline model (NoMaD/GNM/ViNT) on a single benchmark trajectory."""
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
    trace = route_planner.trace_route(start_location, goal_location)
    shortest_distance = 0.0
    prev_wp = None
    for wp, _ in trace:
        if prev_wp is not None:
            shortest_distance += prev_wp.transform.location.distance(wp.transform.location)
        prev_wp = wp
    if shortest_distance <= 2.0:
        shortest_distance = start_location.distance(goal_location)

    # Initialize model
    MODEL = MODEL_CLASSES[model_type]
    checkpoint_path = MODEL_CHECKPOINTS[model_type]
    similarity_matrix = f"cache/{model_type}/distance_matrix_{traj_id}.pth"

    agent = MODEL(
        ckpt_path=checkpoint_path,
        mode="navigate",
        skip_index=1,
        map_dir=trajectory_path,
        similarity_matrix=similarity_matrix,
        use_similarity=True,
        traj_id=traj_id,
        threshold=8.0,  # default threshold
    )

    sp_agent = SetPointAgent(unwrapped_env.core.hero)

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

        while not done and episode_length < max_steps:
            waypoints = np.array(agent.eval_action(unwrapped_env.experiment.obs_img[..., None]))
            actions = sp_agent.run_step(waypoints, False)
            loc_obj = unwrapped_env.core.hero.get_transform().location
            traj_positions.append((loc_obj.x, loc_obj.y, loc_obj.z))

            # Capture frame for video
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
            unwrapped_env.set_start_transform(unwrapped_env.experiment.wp.transform.location)

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
        "num_topomap_nodes": len(agent.topomap),
    }
    if save_video and all_frames:
        results["frames"] = all_frames
    return results


# ============================================================================
# COMPARISON AND RESULTS
# ============================================================================

def print_comparison_table(all_results):
    """Print formatted comparison table."""
    print(f"\n{'='*70}")
    print("CoRL 2026 - COMPARISON TABLE")
    print(f"{'='*70}")

    header = f"{'Model':<10} {'Success%':<12} {'Mean SPL':<12} {'Mean Dist':<12} {'Trajs':<8}"
    print(header)
    print("-" * 54)

    for model_name, results in all_results.items():
        if "error" in results:
            print(f"{model_name:<10} ERROR: {results['error']}")
            continue

        traj_results = [r for r in results.get("per_trajectory_results", []) if "error" not in r]
        if not traj_results:
            print(f"{model_name:<10} No valid results")
            continue

        success_rate = np.mean([r["success"] for r in traj_results]) * 100
        mean_spl = np.mean([r["mean_spl"] for r in traj_results])
        mean_dist = np.mean([r["mean_distance"] for r in traj_results])
        num_trajs = len(traj_results)

        print(f"{model_name:<10} {success_rate:>8.1f}%   {mean_spl:>10.3f}   {mean_dist:>10.1f}m   {num_trajs:>4}")

    print("-" * 54)

    # Per-trajectory breakdown
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
    """Save all results to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_name, results in all_results.items():
        # Per-model pkl
        pkl_path = os.path.join(output_dir, f"{model_name}_results.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(results, f)

        # Per-model JSON (summary only, no trajectories)
        summary = {
            "model": model_name,
            "timestamp": timestamp,
            "overall_success_rate": results.get("overall_success_rate", 0),
            "overall_mean_spl": results.get("overall_mean_spl", 0),
            "overall_mean_distance": results.get("overall_mean_distance", 0),
            "num_trajectories": results.get("num_trajectories", 0),
        }
        json_path = os.path.join(output_dir, f"{model_name}_results.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

    # Comparison summary
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

    comparison_path = os.path.join(output_dir, "comparison_summary.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    # Human-readable table
    table_path = os.path.join(output_dir, "comparison_table.txt")
    with open(table_path, 'w') as f:
        f.write("CoRL 2026 - Model Comparison\n")
        f.write("=" * 54 + "\n")
        f.write(f"{'Model':<10} {'Success%':<12} {'Mean SPL':<12} {'Mean Dist':<12}\n")
        f.write("-" * 50 + "\n")
        for model_name, metrics in comparison.items():
            if "error" in metrics:
                f.write(f"{model_name:<10} ERROR: {metrics['error']}\n")
            else:
                f.write(f"{model_name:<10} {metrics['success_rate']*100:>8.1f}%   "
                        f"{metrics['mean_spl']:>10.3f}   {metrics['mean_distance']:>10.1f}m\n")

    print(f"\nResults saved to {output_dir}")


# ============================================================================
# MAIN
# ============================================================================

def save_trajectory_video(frames, output_path, fps=15):
    """Save trajectory frames as video."""
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


def main():
    global TERMINATION_BUDGET, MAX_EPISODE_STEPS

    parser = argparse.ArgumentParser(description="CoRL 2026: Evaluate DrQ vs NoMaD/GNM/ViNT")
    parser.add_argument('--model', type=str, default='all',
                        choices=['drq', 'nomad', 'gnm', 'vint', 'all'],
                        help='Model(s) to evaluate')
    parser.add_argument('--town', type=str, default='Town02')
    parser.add_argument('--benchmark-dir', type=str, default='corl-2026-results/trajectories')
    parser.add_argument('--map-dir', type=str, default='corl-2026-results/town02_map')
    parser.add_argument('--output-dir', type=str, default='corl-2026-results')
    parser.add_argument('--checkpoint', type=str, default=DRQ_CHECKPOINT,
                        help='DrQ checkpoint path')
    parser.add_argument('--max-steps', type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument('--termination-budget', type=int, default=TERMINATION_BUDGET)
    parser.add_argument('--save-videos', action='store_true',
                        help='Save camera-ready videos of trajectories')
    parser.add_argument('--video-dir', type=str, default=None,
                        help='Directory for video output (default: output-dir/videos)')
    args = parser.parse_args()

    TERMINATION_BUDGET = args.termination_budget
    MAX_EPISODE_STEPS = args.max_steps

    video_dir = args.video_dir or os.path.join(args.output_dir, "videos")
    if args.save_videos:
        os.makedirs(video_dir, exist_ok=True)

    models_to_evaluate = ['drq', 'nomad', 'gnm', 'vint'] if args.model == 'all' else [args.model]

    # Connect to existing CARLA server — do NOT start/kill
    config["env_config"]["carla"]["start_server"] = False
    config["env_config"]["carla"]["port"] = 2000

    # Load benchmark info
    benchmark_info_path = os.path.join(args.benchmark_dir, "benchmark_info.json")
    with open(benchmark_info_path, 'r') as f:
        benchmark_info = json.load(f)

    print(f"\n{'#'*60}")
    print(f"CoRL 2026 EVALUATION")
    print(f"{'#'*60}")
    print(f"Benchmark: {args.benchmark_dir}")
    print(f"Town: {benchmark_info.get('town', args.town)}")
    print(f"Trajectories: {len(benchmark_info['trajectories'])}")
    print(f"Models: {models_to_evaluate}")
    print(f"Map: {args.map_dir}")
    print(f"Save videos: {args.save_videos}")

    # Setup environment
    config["env_config"]["carla"]["town"] = benchmark_info.get('town', args.town)
    env = CarlaEnv(config=config)
    env.experiment.eval_mode = True
    env.experiment.map_mode = False
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    env = RecordEpisodeStatistics(env)

    all_results = {}

    for model_type in models_to_evaluate:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_type.upper()}")
        print(f"{'='*60}")

        if model_type == 'drq':
            # Load graph navigator
            print("Loading topological graph...")
            graph_config = GraphBuilderConfig(feature_type='sift_vlad', use_gpu=True)
            mapper = TopologicalGraphBuilder(config=graph_config)
            mapper.build_from_file(os.path.join(args.map_dir, "map_filtered.pkl"))

            # Load DrQ agent
            print(f"Loading DrQ agent from {args.checkpoint}...")
            agent = load_drq_agent(env, args.checkpoint, SEED)

            # Evaluate
            traj_results = []
            for traj_info in tqdm.tqdm(benchmark_info["trajectories"], desc=f"DrQ eval"):
                traj_id = traj_info["trajectory_id"]
                traj_path = traj_info["path"]

                try:
                    result = evaluate_drq_trajectory(
                        env, agent, mapper, traj_path, traj_id, MAX_EPISODE_STEPS,
                        save_video=args.save_videos
                    )
                    traj_results.append(result)
                    print(f"  Traj {traj_id}: Success={result['success']:.0f}, "
                          f"SPL={result['mean_spl']:.3f}, Dist={result['mean_distance']:.1f}m")

                    # Save video
                    if args.save_videos and 'frames' in result:
                        vpath = os.path.join(video_dir, f"drq_traj_{traj_id:02d}.mp4")
                        save_trajectory_video(result['frames'], vpath)
                        result['video_path'] = vpath
                        del result['frames']
                except Exception as e:
                    print(f"  ERROR on trajectory {traj_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    traj_results.append({
                        "trajectory_id": traj_id, "model": "drq",
                        "error": str(e), "success": 0.0
                    })

            successful = [r for r in traj_results if "error" not in r]
            all_results["drq"] = {
                "model": "drq",
                "num_trajectories": len(benchmark_info["trajectories"]),
                "num_successful_evals": len(successful),
                "overall_success_rate": np.mean([r["success"] for r in successful]) if successful else 0.0,
                "overall_mean_spl": np.mean([r["mean_spl"] for r in successful]) if successful else 0.0,
                "overall_mean_distance": np.mean([r["mean_distance"] for r in successful]) if successful else 0.0,
                "per_trajectory_results": traj_results,
            }

        else:
            # Baseline model (NoMaD/GNM/ViNT)
            traj_results = []
            for traj_info in tqdm.tqdm(benchmark_info["trajectories"], desc=f"{model_type} eval"):
                traj_id = traj_info["trajectory_id"]
                traj_path = traj_info["path"]

                try:
                    result = evaluate_baseline_trajectory(
                        env, model_type, traj_path, traj_id, MAX_EPISODE_STEPS,
                        save_video=args.save_videos
                    )
                    traj_results.append(result)
                    print(f"  Traj {traj_id}: Success={result['success']:.0f}, "
                          f"SPL={result['mean_spl']:.3f}, Dist={result['mean_distance']:.1f}m")

                    # Save video
                    if args.save_videos and 'frames' in result:
                        vpath = os.path.join(video_dir, f"{model_type}_traj_{traj_id:02d}.mp4")
                        save_trajectory_video(result['frames'], vpath)
                        result['video_path'] = vpath
                        del result['frames']
                except Exception as e:
                    print(f"  ERROR on trajectory {traj_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    traj_results.append({
                        "trajectory_id": traj_id, "model": model_type,
                        "error": str(e), "success": 0.0
                    })

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

    # Print and save results
    print_comparison_table(all_results)
    save_results(all_results, args.output_dir)

    return all_results


if __name__ == "__main__":
    main()