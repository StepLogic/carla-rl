#!/usr/bin/env python
"""
SPTM Map Visualization and Evaluation

Generates map comparison plots overlaid on aerial images for Town02 and Town05:
  - SPTM pruned map (nodes + shortcut edges from learned graph)
  - Sparse-Image map (10m spacing, image-based localization)
  - Sparse-GPS map (10m spacing, GPS-based localization)
  - Dense map (GPS-based dense map, sequential edges)

Then evaluates SPTM navigation on dense-image trajectories.

Usage:
    python sptm_eval_with_map.py --no-eval
    python sptm_eval_with_map.py --edge-model sptm_output/edge_model_best.pt --carla-port 2000
"""

import os
import sys
import pickle
import json
import argparse

import cv2
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sptm_train import SPTMNavigator, CARLA_ACTIONS

# Map directories per town
TOWN_CONFIG = {
    'Town02': {
        'dense': 'corl-2026-results-dense-gps/town02_map',
        'sparse_image': 'ablation-results/sparse_10m',
        'sparse_gps': 'ablation-results-gps/sparse_10m',
        'aerial': 'rss_results/town02aerial.png',
        'sptm_graph': 'sptm_output/sptm_graph.pkl',
    },
    'Town05': {
        'dense': 'corl-2026-results-dense-gps/town05_map',
        'aerial': 'benchmark_data/Town05/trajectories_visualization.png',
    },
}

# Colors for each map type
MAP_COLORS = {
    'SPTM':         ('#FF5722', '#FF8A65'),   # red-orange
    'Sparse-Image': ('#9C27B0', '#CE93D8'),   # purple
    'Sparse-GPS':   ('#00BCD4', '#80DEEA'),   # cyan
    'Dense':        ('#4CAF50', '#81C784'),   # green
}

MAP_DISPLAY_ORDER = ['SPTM', 'Sparse-Image', 'Sparse-GPS', 'Dense']


def load_map_graph(map_dir):
    """Load node positions and edges from a map directory with topology_graph.pkl."""
    topo_path = os.path.join(map_dir, 'topology_graph.pkl')
    if not os.path.exists(topo_path):
        return None

    with open(topo_path, 'rb') as f:
        G = pickle.load(f)

    positions = []
    for node_idx in sorted(G.nodes()):
        if 'pos' in G.nodes[node_idx]:
            positions.append(G.nodes[node_idx]['pos'])
        else:
            pos_arr = np.load(os.path.join(map_dir, 'positions.npy'))
            positions = [(pos_arr[i, 0], pos_arr[i, 1]) for i in range(len(pos_arr))]
            break

    if not positions:
        pos_arr = np.load(os.path.join(map_dir, 'positions.npy'))
        positions = [(pos_arr[i, 0], pos_arr[i, 1]) for i in range(len(pos_arr))]

    edges = list(G.edges())
    edge_types = {}
    for u, v in edges:
        etype = G.edges[u, v].get('type', 'sequential')
        edge_types[(u, v)] = etype

    sequential = [(u, v) for u, v in edges
                  if edge_types.get((u, v)) == 'sequential'
                  or edge_types.get((v, u)) == 'sequential'
                  or abs(u - v) == 1]
    shortcuts = [(u, v) for u, v in edges
                 if (u, v) not in sequential and (v, u) not in sequential]

    return {
        'positions': positions,
        'sequential': sequential,
        'shortcuts': shortcuts,
        'graph': G,
    }


def load_sparse_map(sparse_dir):
    """Load a sparse map from ablation-results directory (node_link_graph format)."""
    graph_path = os.path.join(sparse_dir, 'sparse_graph.pkl')
    nodes_path = os.path.join(sparse_dir, 'sparse_nodes.pkl')

    if not os.path.exists(graph_path) or not os.path.exists(nodes_path):
        return None

    with open(nodes_path, 'rb') as f:
        nodes = pickle.load(f)

    positions = [(n['x'], n['y']) for n in nodes]

    with open(graph_path, 'rb') as f:
        G_data = pickle.load(f)

    G = nx.node_link_graph(G_data, directed=True)

    edges = list(G.edges())
    sequential = [(u, v) for u, v in edges if abs(u - v) == 1]
    shortcuts = [(u, v) for u, v in edges if abs(u - v) != 1]

    return {
        'positions': positions,
        'sequential': sequential,
        'shortcuts': shortcuts,
    }


def load_sptm_map(sptm_graph_path, edge_model_path=None, action_model_path=None, map_dir=None):
    """Load SPTM map from saved graph or build from models."""
    if os.path.exists(sptm_graph_path):
        with open(sptm_graph_path, 'rb') as f:
            data = pickle.load(f)

        G = data['G']
        coords = data['keyframe_coordinates']

        positions = [tuple(c) for c in coords]
        edges = list(G.edges())
        sequential = [(u, v) for u, v in edges if abs(u - v) == 1]
        shortcuts = [(u, v) for u, v in edges if abs(u - v) != 1]

        return {
            'positions': positions,
            'sequential': sequential,
            'shortcuts': shortcuts,
        }

    # Fall back to building from models
    if edge_model_path and os.path.exists(edge_model_path):
        nav = SPTMNavigator(
            edge_model_path=edge_model_path,
            action_model_path=action_model_path if os.path.exists(action_model_path) else None,
            map_dir=map_dir,
        )
        nav.build_graph()
        positions = nav.keyframe_coordinates
        edges = list(nav.G.edges())
        sequential = [(u, v) for u, v in edges if abs(u - v) == 1]
        shortcuts = [(u, v) for u, v in edges if abs(u - v) != 1]
        return {
            'positions': positions,
            'sequential': sequential,
            'shortcuts': shortcuts,
        }

    return None


def load_maps_for_town(town_name, edge_model=None, action_model=None):
    """Load all available map types for a given town."""
    cfg = TOWN_CONFIG.get(town_name, {})
    maps = {}

    # SPTM
    sptm_path = cfg.get('sptm_graph')
    if sptm_path:
        data = load_sptm_map(sptm_path, edge_model, action_model, cfg.get('dense'))
        if data:
            maps['SPTM'] = data

    # Sparse-Image (from image-based localization)
    si_dir = cfg.get('sparse_image')
    if si_dir:
        data = load_sparse_map(si_dir)
        if data:
            maps['Sparse-Image'] = data

    # Sparse-GPS (from GPS-based localization)
    sg_dir = cfg.get('sparse_gps')
    if sg_dir:
        data = load_sparse_map(sg_dir)
        if data:
            maps['Sparse-GPS'] = data

    # Dense (GPS-based map)
    d_dir = cfg.get('dense')
    if d_dir:
        data = load_map_graph(d_dir)
        if data:
            maps['Dense'] = data

    return maps


def plot_town_panel_comparison(maps, town_name, aerial_path, output_dir):
    """Create a multi-panel comparison of all map types overlaid on aerial image."""
    os.makedirs(output_dir, exist_ok=True)

    # Load aerial image
    aerial = None
    if os.path.exists(aerial_path):
        aerial = cv2.imread(aerial_path)
        if aerial is not None:
            aerial = cv2.cvtColor(aerial, cv2.COLOR_BGR2RGB)

    available = [k for k in MAP_DISPLAY_ORDER if k in maps]
    n_maps = len(available)
    if n_maps == 0:
        print(f"  No maps available for {town_name}, skipping panel comparison")
        return

    # Layout: up to 5 panels in one row, or 3+2 for 5
    if n_maps <= 3:
        ncols = n_maps
        nrows = 1
    else:
        ncols = 3
        nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 8 * nrows))
    if n_maps == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Compute common extent from all positions
    all_xs, all_ys = [], []
    for name in available:
        for p in maps[name]['positions']:
            all_xs.append(p[0])
            all_ys.append(p[1])

    if all_xs:
        x_min, x_max = min(all_xs) - 20, max(all_xs) + 20
        y_min, y_max = min(all_ys) - 20, max(all_ys) + 20
    else:
        x_min, x_max, y_min, y_max = 0, 1, 0, 1

    for idx, name in enumerate(available):
        ax = axes[idx]
        data = maps[name]
        positions = data['positions']
        seq_edges = data['sequential']
        shortcuts = data['shortcuts']
        nc, ec = MAP_COLORS[name]

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        # Aerial background
        if aerial is not None:
            ax.imshow(aerial, extent=[x_min, x_max, y_max, y_min],
                      alpha=0.35, aspect='auto')

        # Sequential edges
        for u, v in seq_edges:
            if u < len(positions) and v < len(positions):
                ax.plot([positions[u][0], positions[v][0]],
                        [positions[u][1], positions[v][1]],
                        color=ec, alpha=0.35, linewidth=0.5)

        # Shortcut edges
        for u, v in shortcuts:
            if u < len(positions) and v < len(positions):
                ax.plot([positions[u][0], positions[v][0]],
                        [positions[u][1], positions[v][1]],
                        color='#FF9800', alpha=0.15, linewidth=0.3,
                        linestyle='--')

        # Nodes
        ax.scatter(xs, ys, s=4, c=nc, alpha=0.7, zorder=5)

        n_seq = len(seq_edges)
        n_short = len(shortcuts)
        ax.set_title(f'{name}\n{len(positions)} nodes, {n_seq} seq, {n_short} shortcuts',
                     fontsize=11)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for idx in range(n_maps, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Topological Map Comparison: {town_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(output_dir, f'{town_name.lower()}_map_comparison_aerial.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Panel comparison saved to {out_path}")

    # Also save clean version (no aerial)
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(8 * ncols, 8 * nrows))
    if n_maps == 1:
        axes2 = np.array([axes2])
    axes2 = axes2.flatten()

    for idx, name in enumerate(available):
        ax = axes2[idx]
        data = maps[name]
        positions = data['positions']
        seq_edges = data['sequential']
        shortcuts = data['shortcuts']
        nc, ec = MAP_COLORS[name]

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        for u, v in seq_edges:
            if u < len(positions) and v < len(positions):
                ax.plot([positions[u][0], positions[v][0]],
                        [positions[u][1], positions[v][1]],
                        color=ec, alpha=0.4, linewidth=0.6)

        for u, v in shortcuts:
            if u < len(positions) and v < len(positions):
                ax.plot([positions[u][0], positions[v][0]],
                        [positions[u][1], positions[v][1]],
                        color='#FF9800', alpha=0.2, linewidth=0.3,
                        linestyle='--')

        ax.scatter(xs, ys, s=6, c=nc, alpha=0.8, zorder=5)

        n_seq = len(seq_edges)
        n_short = len(shortcuts)
        ax.set_title(f'{name}\n{len(positions)} nodes, {n_seq} seq, {n_short} shortcuts',
                     fontsize=11)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    for idx in range(n_maps, len(axes2)):
        axes2[idx].axis('off')

    plt.suptitle(f'Topological Map Comparison: {town_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path2 = os.path.join(output_dir, f'{town_name.lower()}_map_comparison_clean.png')
    fig2.savefig(out_path2, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"  Clean comparison saved to {out_path2}")


def plot_all_maps_overlay(maps, town_name, aerial_path, output_dir):
    """Create a single overlay plot with all map types on the aerial image."""
    if not os.path.exists(aerial_path):
        print(f"  No aerial image at {aerial_path}, skipping overlay")
        return

    aerial = cv2.imread(aerial_path)
    if aerial is None:
        print(f"  Failed to load aerial image from {aerial_path}")
        return
    aerial = cv2.cvtColor(aerial, cv2.COLOR_BGR2RGB)

    available = [k for k in MAP_DISPLAY_ORDER if k in maps]
    if not available:
        print(f"  No maps available for {town_name}, skipping overlay")
        return

    # Compute extent
    all_xs, all_ys = [], []
    for name in available:
        for p in maps[name]['positions']:
            all_xs.append(p[0])
            all_ys.append(p[1])

    x_min, x_max = min(all_xs) - 30, max(all_xs) + 30
    y_min, y_max = min(all_ys) - 30, max(all_ys) + 30

    fig, ax = plt.subplots(1, 1, figsize=(16, 13))
    ax.imshow(aerial, extent=[x_min, x_max, y_max, y_min],
              alpha=0.4, aspect='auto')

    for name in available:
        data = maps[name]
        positions = data['positions']
        seq_edges = data['sequential']
        nc, ec = MAP_COLORS[name]

        pos_x = [p[0] for p in positions]
        pos_y = [p[1] for p in positions]

        # Draw sequential edges lightly
        for u, v in seq_edges:
            if u < len(positions) and v < len(positions):
                ax.plot([positions[u][0], positions[v][0]],
                        [positions[u][1], positions[v][1]],
                        color=ec, alpha=0.12, linewidth=0.3)

        # Draw nodes
        n_nodes = len(positions)
        ax.scatter(pos_x, pos_y, s=5, c=nc, alpha=0.6, zorder=5,
                   label=f'{name} ({n_nodes} nodes)')

    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_title(f'{town_name}: All Maps Overlaid on Aerial Image', fontsize=13, fontweight='bold')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{town_name.lower()}_map_overlay_aerial.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Overlay map saved to {out_path}")


def evaluate_sptm_on_trajectories(edge_model_path, action_model_path, map_dir,
                                   trajectories_dir, carla_port, max_steps=3000):
    """Evaluate SPTM navigation on benchmark trajectories in CARLA."""
    from rlib_integration.carla_env import CarlaEnv
    from rlib_integration.helper import ndarray_to_location
    from src.configs.eval_env_config_image_goal import config as ig_config
    from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics
    import carla

    ig_config["env_config"]["carla"]["town"] = "Town02"
    ig_config["env_config"]["carla"]["start_server"] = False
    ig_config["env_config"]["carla"]["port"] = carla_port

    env = CarlaEnv(config=ig_config)
    env.experiment.eval_mode = True
    env.experiment.map_mode = False
    env = TimeLimit(env, max_episode_steps=max_steps)
    env = RecordEpisodeStatistics(env)

    # Load benchmark info
    benchmark_path = os.path.join(trajectories_dir, 'benchmark_info.json')
    with open(benchmark_path, 'r') as f:
        benchmark_info = json.load(f)

    # Build SPTM navigator
    nav = SPTMNavigator(
        edge_model_path=edge_model_path,
        action_model_path=action_model_path,
        map_dir=map_dir,
    )
    nav.build_graph()
    print(f"SPTM graph: {nav.G.number_of_nodes()} nodes, {nav.G.number_of_edges()} edges")

    # Evaluate each trajectory
    results = []
    for traj_info in benchmark_info['trajectories']:
        traj_id = traj_info['trajectory_id']
        traj_path = traj_info.get('path', os.path.join(trajectories_dir, f'trajectory_{traj_id:02d}'))

        aux_path = os.path.join(traj_path, 'aux.pkl')
        if not os.path.exists(aux_path):
            alt_path = traj_path.replace('corl-2026-results/', f'{trajectories_dir}/')
            aux_path = os.path.join(alt_path, 'aux.pkl')

        if not os.path.exists(aux_path):
            print(f"  Trajectory {traj_id}: aux.pkl not found, skipping")
            continue

        with open(aux_path, 'rb') as f:
            aux = pickle.load(f)

        start = aux.get('start')
        goal = aux.get('goal')
        if start is None or goal is None:
            continue

        start_location = ndarray_to_location(start)
        goal_location = ndarray_to_location(goal)

        print(f"  Trajectory {traj_id}: start=({start[0]:.1f},{start[1]:.1f}) "
              f"goal=({goal[0]:.1f},{goal[1]:.1f})")

        try:
            result = _evaluate_single_trajectory(
                env, nav, start_location, goal_location, traj_id, max_steps)
            results.append(result)
            status = "SUCCESS" if result['success'] else "FAIL"
            print(f"    {status} | SPL={result['spl']:.3f} | dist={result['distance_completed']:.1f}m")
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                'trajectory_id': traj_id,
                'success': False,
                'spl': 0.0,
                'distance_completed': 0.0,
                'error': str(e),
            })

    # Aggregate results
    success_rate = np.mean([r['success'] for r in results]) if results else 0
    mean_spl = np.mean([r['spl'] for r in results]) if results else 0
    mean_dist = np.mean([r['distance_completed'] for r in results]) if results else 0

    print(f"\n{'='*50}")
    print(f"SPTM Evaluation Results ({len(results)} trajectories)")
    print(f"{'='*50}")
    print(f"  Success Rate: {success_rate:.1%}")
    print(f"  Mean SPL:     {mean_spl:.3f}")
    print(f"  Mean Distance: {mean_dist:.1f}m")

    return results, {'success_rate': success_rate, 'mean_spl': mean_spl, 'mean_distance': mean_dist}


def _evaluate_single_trajectory(env, nav, start_location, goal_location,
                                 traj_id, max_steps):
    """Evaluate SPTM on a single trajectory."""
    env.unwrapped.set_start_transform(start_location)
    env.unwrapped.set_destination_transform(goal_location)

    obs, info = env.reset()
    env.step(np.array([0.0, 0.0]))

    hero = env.unwrapped.core.hero
    hero_pos = hero.get_transform().location
    _, _, goal_node_idx = nav.set_goal_by_location(goal_location)
    nav.compute_shortest_paths(goal_node_idx)

    from rlib_integration.agent import GlobalRoutePlanner
    carla_map = env.unwrapped.core.map
    planner = GlobalRoutePlanner(carla_map, 2.0)
    try:
        route = planner.trace_route(start_location, goal_location)
        shortest_distance = sum(
            wp1.transform.location.distance(wp2.transform.location)
            for (wp1, _), (wp2, _) in zip(route, route[1:]))
    except Exception:
        shortest_distance = start_location.distance(goal_location)

    done = False
    steps = 0
    agent_trajectory = []
    success = False

    while not done and steps < max_steps:
        hero = env.unwrapped.core.hero
        curr_transform = hero.get_transform()
        hero_pos = np.array([curr_transform.location.x, curr_transform.location.y])
        agent_trajectory.append(hero_pos.copy())

        obs_img = env.unwrapped.experiment.obs_img
        try:
            irg_idx = nav.find_intermediate_reachable_goal(obs_img)
            if irg_idx is not None:
                goal_frame = nav.keyframes[irg_idx % len(nav.keyframes)]
            else:
                goal_frame = nav.keyframes[-1]
        except Exception:
            goal_frame = nav.keyframes[-1]

        prev_frame = getattr(_evaluate_single_trajectory, '_prev_frame', obs_img)
        try:
            action_probs = nav.predict_action(prev_frame, obs_img, goal_frame)
            discrete_action = int(np.argmax(action_probs))
        except Exception:
            discrete_action = 0

        _evaluate_single_trajectory._prev_frame = obs_img

        ctrl = CARLA_ACTIONS[discrete_action]
        action = np.array([ctrl['steer'], ctrl['throttle']])

        obs, reward, done, truncated, info = env.step(action)
        steps += 1

        goal_pos = np.array([goal_location.x, goal_location.y])
        dist_to_goal = np.linalg.norm(hero_pos - goal_pos)
        if dist_to_goal < 2.0:
            success = True
            break

        if done or truncated:
            break

    if len(agent_trajectory) > 1:
        distance_completed = sum(
            np.linalg.norm(np.array(agent_trajectory[i]) - np.array(agent_trajectory[i-1]))
            for i in range(1, len(agent_trajectory)))
    else:
        distance_completed = 0.0

    if success and shortest_distance > 0:
        spl = shortest_distance / max(shortest_distance, distance_completed)
    else:
        spl = 0.0

    return {
        'trajectory_id': traj_id,
        'success': success,
        'spl': float(spl),
        'distance_completed': float(distance_completed),
        'shortest_distance': float(shortest_distance),
        'steps': steps,
        'agent_trajectory': [p.tolist() for p in agent_trajectory],
    }


def main():
    parser = argparse.ArgumentParser(
        description="SPTM Map Visualization and Evaluation")
    parser.add_argument('--edge-model', type=str,
                        default='sptm_output/edge_model_best.pt')
    parser.add_argument('--action-model', type=str,
                        default='sptm_output/action_model_best.pt')
    parser.add_argument('--output-dir', type=str, default='sptm_eval_output')
    parser.add_argument('--carla-port', type=int, default=2000)
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip map generation, only evaluate')
    parser.add_argument('--no-eval', action='store_true',
                        help='Skip CARLA evaluation, only generate map plots')
    parser.add_argument('--max-steps', type=int, default=3000)
    parser.add_argument('--towns', type=str, default='Town02,Town05',
                        help='Comma-separated towns to visualize')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    towns = [t.strip() for t in args.towns.split(',')]

    for town in towns:
        print(f"\n{'='*60}")
        print(f"  Processing {town}")
        print(f"{'='*60}")

        cfg = TOWN_CONFIG.get(town, {})
        aerial_path = cfg.get('aerial', '')

        # Load all available map types
        maps = load_maps_for_town(town, args.edge_model, args.action_model)

        if not maps:
            print(f"  No maps found for {town}, skipping")
            continue

        for name, data in maps.items():
            n = len(data['positions'])
            ns = len(data['sequential'])
            nsh = len(data['shortcuts'])
            print(f"  {name}: {n} nodes, {ns} seq edges, {nsh} shortcuts")

        # Generate plots
        if not args.eval_only:
            plot_town_panel_comparison(maps, town, aerial_path, args.output_dir)
            plot_all_maps_overlay(maps, town, aerial_path, args.output_dir)

    # Evaluate SPTM on dense-image trajectories (Town02 only, needs CARLA)
    if not args.no_eval and os.path.exists(args.edge_model):
        print(f"\nEvaluating SPTM on dense-image trajectories...")
        di_dir = 'corl-2026-results-dense-image/town02_map'
        if di_dir:
            traj_dir = os.path.join(os.path.dirname(di_dir), 'trajectories')
            results, summary = evaluate_sptm_on_trajectories(
                args.edge_model, args.action_model,
                di_dir, traj_dir,
                args.carla_port, args.max_steps,
            )

            results_path = os.path.join(args.output_dir, 'sptm_dense_image_results.pkl')
            with open(results_path, 'wb') as f:
                pickle.dump({'results': results, 'summary': summary}, f)

            summary_path = os.path.join(args.output_dir, 'sptm_dense_image_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Results saved to {results_path}")

    print(f"\nDone! Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()