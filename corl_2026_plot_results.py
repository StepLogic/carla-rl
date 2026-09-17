#!/usr/bin/env python
"""
CoRL 2026: Plot evaluation results on Town02 aerial map.

Generates:
  - map: Topological map overlaid on aerial image
  - individual: Per-trajectory plots for each model
  - combined: All models overlaid per trajectory
  - all: All of the above

Usage:
    python corl_2026_plot_results.py --plot-type all \
        --results-dir corl-2026-results \
        --map-dir corl-2026-results/town02_map \
        --aerial-image rss_results/town02aerial.png \
        --output-dir corl-2026-results/plots
"""
import os
import pickle
import json
import argparse

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from eval_graph_builder_vlad_prebuilt import TopologicalGraphBuilder, GraphBuilderConfig


# ============================================================================
# CALIBRATION PARAMETERS
# ============================================================================
# CARLA Town02 coordinate ranges (from map data)
_DEFAULT_X_RANGE = (-4, 194)
_DEFAULT_Y_RANGE = (105, 307)

# Runtime config updated from CLI
PLOT_CONFIG = {
    'x_range': _DEFAULT_X_RANGE,
    'y_range': _DEFAULT_Y_RANGE,
}

# Image transforms for CARLA aerial views
FLIP_HORIZONTAL = True
FLIP_VERTICAL = True

# Model color scheme
MODEL_COLORS = {
    'drq_ig': '#2196F3',    # Blue
    'nomad': '#FF9800',     # Orange
    'gnm': '#4CAF50',       # Green
    'vint': '#F44336',      # Red
}

MODEL_LABELS = {
    'drq_ig': 'DrQ-IG',
    'nomad': 'NoMaD',
    'gnm': 'GNM',
    'vint': 'ViNT',
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_map_data(map_dir):
    """Load topological map data from pre-built artifacts."""
    # Load graph builder (instant — no re-extraction of features)
    graph_config = GraphBuilderConfig(feature_type='sift_vlad', use_gpu=False)
    mapper = TopologicalGraphBuilder(config=graph_config)
    mapper.load_from_directory(map_dir)

    # nodes_data = full original image metadata (1492 entries with x, y, filename)
    nodes_data = mapper.original_nodes_data

    # Load positions
    positions_path = os.path.join(map_dir, "positions.npy")
    if os.path.exists(positions_path):
        positions = np.load(positions_path)
    else:
        positions = np.array([[d['x'], d['y'], d.get('z', 0)] for d in nodes_data])

    # Graph is already loaded by load_from_directory
    G = mapper.G

    return mapper, nodes_data, positions, G


def load_results(results_dir):
    """Load evaluation results from pkl files."""
    results = {}
    suffix = '_ig_dense_results.pkl'
    for model_name in MODEL_COLORS:
        pkl_path = os.path.join(results_dir, f"{model_name}{suffix}")
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                results[model_name] = pickle.load(f)
            print(f"Loaded {model_name}: {len(results[model_name].get('per_trajectory_results', []))} trajectories")
        else:
            # Try alternate naming
            alt_path = os.path.join(results_dir, f"{model_name}_results.pkl")
            if os.path.exists(alt_path):
                with open(alt_path, 'rb') as f:
                    results[model_name] = pickle.load(f)
                print(f"Loaded {model_name}: {len(results[model_name].get('per_trajectory_results', []))} trajectories")
    return results


def normalize_trajectory(traj):
    """Normalize trajectory data to (N, 2) numpy array of [x, y]."""
    traj = np.array(traj)
    if traj.ndim == 1:
        return np.zeros((0, 2))
    if traj.shape[1] == 3:
        return traj[:, :2]
    if traj.shape[1] == 2:
        return traj
    return traj[:, :2]


# ============================================================================
# PLOTTING
# ============================================================================

def load_aerial_image(path):
    """Load and transform aerial image for CARLA coordinate overlay."""
    img = Image.open(path)
    if FLIP_HORIZONTAL:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if FLIP_VERTICAL:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return np.array(img)


def plot_map_on_aerial(map_dir, aerial_image_path, output_dir):
    """Plot topological map nodes and edges overlaid on the aerial image."""
    os.makedirs(output_dir, exist_ok=True)

    mapper, nodes_data, positions, G = load_map_data(map_dir)
    aerial = load_aerial_image(aerial_image_path)

    fig, ax = plt.subplots(figsize=(14, 14))
    # Show aerial image with extent mapped to CARLA coordinates
    ax.imshow(aerial, extent=[PLOT_CONFIG['x_range'][0], PLOT_CONFIG['x_range'][1],
                               PLOT_CONFIG['y_range'][0], PLOT_CONFIG['y_range'][1]],
              aspect='auto', alpha=0.6)

    # Plot edges
    import networkx as nx
    for u, v in G.edges():
        u_data = nodes_data[u]
        v_data = nodes_data[v]
        ax.plot([u_data['x'], v_data['x']], [u_data['y'], v_data['y']],
                'gray', linewidth=0.5, alpha=0.3)

    # Plot all graph nodes
    all_x = [d['x'] for d in nodes_data]
    all_y = [d['y'] for d in nodes_data]
    ax.scatter(all_x, all_y, c='navy', s=5, alpha=0.6,
                label=f'Graph nodes ({len(nodes_data)})')

    ax.set_xlim(PLOT_CONFIG['x_range'])
    ax.set_ylim(PLOT_CONFIG['y_range'])
    ax.set_xlabel('X (CARLA)', fontsize=12)
    ax.set_ylabel('Y (CARLA)', fontsize=12)
    ax.set_title('Town02 Topological Map Overlay', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)

    output_path = os.path.join(output_dir, 'topological_map_overlay.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved map overlay to {output_path}")


def plot_individual_trajectories(results_dir, map_dir, aerial_image_path, output_dir):
    """Plot individual trajectories for each model on aerial image."""
    os.makedirs(output_dir, exist_ok=True)

    mapper, nodes_data, positions, G = load_map_data(map_dir)
    aerial = load_aerial_image(aerial_image_path)
    results = load_results(results_dir)

    for model_name, model_results in results.items():
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        color = MODEL_COLORS.get(model_name, 'black')
        label = MODEL_LABELS.get(model_name, model_name)

        for traj_result in model_results.get('per_trajectory_results', []):
            if 'error' in traj_result:
                continue

            traj_id = traj_result['trajectory_id']
            traj = normalize_trajectory(traj_result.get('agent_trajectory', []))
            start_loc = np.array(traj_result.get('start_location', [0, 0]))[:2]
            goal_loc = np.array(traj_result.get('goal_location', [0, 0]))[:2]

            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(aerial, extent=[PLOT_CONFIG['x_range'][0], PLOT_CONFIG['x_range'][1],
                                       PLOT_CONFIG['y_range'][0], PLOT_CONFIG['y_range'][1]],
                      aspect='auto', alpha=0.5)

            if len(traj) > 0:
                ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5,
                        alpha=0.8, label=label)

            ax.scatter(start_loc[0], start_loc[1], c='lime', s=200, marker='o',
                        edgecolors='black', linewidths=1.5, zorder=5, label='Start')
            ax.scatter(goal_loc[0], goal_loc[1], c='red', s=250, marker='*',
                        edgecolors='black', linewidths=1.5, zorder=5, label='Goal')

            success = traj_result.get('success', 0)
            spl = traj_result.get('mean_spl', 0)
            ax.set_title(f'{label} Trajectory {traj_id} (Success={success:.0f}, SPL={spl:.3f})',
                         fontsize=13)
            ax.set_xlabel('X (CARLA)', fontsize=11)
            ax.set_ylabel('Y (CARLA)', fontsize=11)
            ax.legend(loc='upper right', fontsize=10)
            ax.set_xlim(PLOT_CONFIG['x_range'])
            ax.set_ylim(PLOT_CONFIG['y_range'])

            fig.savefig(os.path.join(model_dir, f'{model_name}_traj_{traj_id:02d}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)

    print(f"Saved individual trajectory plots to {output_dir}")


def plot_combined_trajectories(results_dir, map_dir, aerial_image_path, output_dir):
    """Plot all models' trajectories for each trajectory_id on the same map."""
    os.makedirs(output_dir, exist_ok=True)

    mapper, nodes_data, positions, G = load_map_data(map_dir)
    aerial = load_aerial_image(aerial_image_path)
    results = load_results(results_dir)

    # Find all trajectory IDs
    all_traj_ids = set()
    for model_results in results.values():
        for traj_result in model_results.get('per_trajectory_results', []):
            all_traj_ids.add(traj_result['trajectory_id'])
    all_traj_ids = sorted(all_traj_ids)

    for traj_id in all_traj_ids:
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.imshow(aerial, extent=[PLOT_CONFIG['x_range'][0], PLOT_CONFIG['x_range'][1],
                                   PLOT_CONFIG['y_range'][0], PLOT_CONFIG['y_range'][1]],
                  aspect='auto', alpha=0.5)

        # Find start/goal from any model that has this trajectory
        start_loc = None
        goal_loc = None
        for model_name, model_results in results.items():
            for traj_result in model_results.get('per_trajectory_results', []):
                if traj_result['trajectory_id'] == traj_id and 'error' not in traj_result:
                    if start_loc is None:
                        start_loc = np.array(traj_result.get('start_location', [0, 0]))[:2]
                        goal_loc = np.array(traj_result.get('goal_location', [0, 0]))[:2]

                    traj = normalize_trajectory(traj_result.get('agent_trajectory', []))
                    color = MODEL_COLORS.get(model_name, 'black')
                    label = MODEL_LABELS.get(model_name, model_name)

                    if len(traj) > 0:
                        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2.0,
                                alpha=0.85, label=f'{label} (SPL={traj_result.get("mean_spl", 0):.2f})')

        if start_loc is not None:
            ax.scatter(start_loc[0], start_loc[1], c='lime', s=250, marker='o',
                        edgecolors='black', linewidths=2, zorder=5, label='Start')
            ax.scatter(goal_loc[0], goal_loc[1], c='red', s=300, marker='*',
                        edgecolors='black', linewidths=2, zorder=5, label='Goal')

        ax.set_title(f'All Models - Trajectory {traj_id}', fontsize=14)
        ax.set_xlabel('X (CARLA)', fontsize=11)
        ax.set_ylabel('Y (CARLA)', fontsize=11)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlim(PLOT_CONFIG['x_range'])
        ax.set_ylim(PLOT_CONFIG['y_range'])

        fig.savefig(os.path.join(output_dir, f'combined_traj_{traj_id:02d}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved combined trajectory plots to {output_dir}")


def plot_all_trajectories_overview(results_dir, map_dir, aerial_image_path, output_dir):
    """Plot all trajectories for all models on a single overview figure."""
    os.makedirs(output_dir, exist_ok=True)

    aerial = load_aerial_image(aerial_image_path)
    results = load_results(results_dir)

    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 12))
    if len(results) == 1:
        axes = [axes]

    for idx, (model_name, model_results) in enumerate(results.items()):
        ax = axes[idx]
        ax.imshow(aerial, extent=[PLOT_CONFIG['x_range'][0], PLOT_CONFIG['x_range'][1],
                                   PLOT_CONFIG['y_range'][0], PLOT_CONFIG['y_range'][1]],
                  aspect='auto', alpha=0.5)

        color = MODEL_COLORS.get(model_name, 'black')
        label = MODEL_LABELS.get(model_name, model_name)

        for traj_result in model_results.get('per_trajectory_results', []):
            if 'error' in traj_result:
                continue
            traj = normalize_trajectory(traj_result.get('agent_trajectory', []))
            success = traj_result.get('success', 0)
            alpha = 0.9 if success else 0.4
            if len(traj) > 0:
                ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.0, alpha=alpha)

        # Mark starts and goals
        for traj_result in model_results.get('per_trajectory_results', []):
            if 'error' in traj_result:
                continue
            start_loc = np.array(traj_result.get('start_location', [0, 0]))[:2]
            goal_loc = np.array(traj_result.get('goal_location', [0, 0]))[:2]
            ax.scatter(start_loc[0], start_loc[1], c='lime', s=40, marker='o',
                        edgecolors='black', linewidths=0.5, zorder=5)
            ax.scatter(goal_loc[0], goal_loc[1], c='red', s=50, marker='*',
                        edgecolors='black', linewidths=0.5, zorder=5)

        success_rate = model_results.get('overall_success_rate', 0) * 100
        mean_spl = model_results.get('overall_mean_spl', 0)
        ax.set_title(f'{label}\n(Success={success_rate:.0f}%, SPL={mean_spl:.3f})', fontsize=12)
        ax.set_xlim(PLOT_CONFIG['x_range'])
        ax.set_ylim(PLOT_CONFIG['y_range'])
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'all_trajectories_overview.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved overview plot to {output_dir}/all_trajectories_overview.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CoRL 2026: Plot evaluation results on Town02 aerial map")
    parser.add_argument('--plot-type', type=str, default='all',
                        choices=['map', 'individual', 'combined', 'all'],
                        help='Type of plot to generate')
    parser.add_argument('--results-dir', type=str, default='corl-2026-results')
    parser.add_argument('--map-dir', type=str, default='corl-2026-results/town02_map')
    parser.add_argument('--aerial-image', type=str, default='rss_results/town02aerial.png')
    parser.add_argument('--output-dir', type=str, default='corl-2026-results/plots')
    parser.add_argument('--x-range', type=float, nargs=2, default=list(PLOT_CONFIG['x_range']),
                        help='CARLA X coordinate range for aerial overlay')
    parser.add_argument('--y-range', type=float, nargs=2, default=list(PLOT_CONFIG['y_range']),
                        help='CARLA Y coordinate range for aerial overlay')
    args = parser.parse_args()

    PLOT_CONFIG['x_range'] = tuple(args.x_range)
    PLOT_CONFIG['y_range'] = tuple(args.y_range)

    print(f"\n{'#'*60}")
    print(f"CoRL 2026 PLOTTING")
    print(f"{'#'*60}")
    print(f"Results: {args.results_dir}")
    print(f"Map: {args.map_dir}")
    print(f"Aerial: {args.aerial_image}")
    print(f"Output: {args.output_dir}")
    print(f"Plot type: {args.plot_type}")
    print(f"X range: {PLOT_CONFIG['x_range']}")
    print(f"Y range: {PLOT_CONFIG['y_range']}")

    if args.plot_type in ('map', 'all'):
        print("\nPlotting topological map overlay...")
        plot_map_on_aerial(args.map_dir, args.aerial_image, args.output_dir)

    if args.plot_type in ('individual', 'all'):
        print("\nPlotting individual trajectories...")
        plot_individual_trajectories(args.results_dir, args.map_dir, args.aerial_image,
                                     os.path.join(args.output_dir, 'individual'))

    if args.plot_type in ('combined', 'all'):
        print("\nPlotting combined trajectories...")
        plot_combined_trajectories(args.results_dir, args.map_dir, args.aerial_image,
                                    os.path.join(args.output_dir, 'combined'))

    if args.plot_type == 'all':
        print("\nPlotting all-trajectories overview...")
        plot_all_trajectories_overview(args.results_dir, args.map_dir, args.aerial_image,
                                        args.output_dir)

    print(f"\nDone! Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()