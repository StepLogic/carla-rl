#!/usr/bin/env python
"""Plot VLAD feature similarity distribution for the topological map.

Shows how visually similar map nodes are to each other, which informs
the similarity thresholds used in the ablation study.

Usage:
    python plot_similarity_distribution.py --map-dir corl-2026-results-dense-image/town02_map
"""

import os
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot VLAD similarity distribution")
    parser.add_argument('--map-dir', type=str,
                        default='corl-2026-results-dense-image/town02_map')
    parser.add_argument('--output-dir', type=str, default='ablation-results')
    parser.add_argument('--sample-pairs', type=int, default=50000,
                        help='Number of random pairs to sample for all-pairs histogram')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load features
    with open(os.path.join(args.map_dir, 'features.pkl'), 'rb') as f:
        feat = pickle.load(f)
    features = feat['all_features']

    # Load positions for nearby analysis
    with open(os.path.join(args.map_dir, 'map_data.pkl'), 'rb') as f:
        map_data = pickle.load(f)
    nodes = map_data['metadata']['images']

    positions = np.array([[n['x'], n['y']] for n in nodes])
    n = len(nodes)

    # Normalize for cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    features_norm = features / norms

    print(f"Total nodes: {n}")
    print(f"Feature dim: {features.shape[1]}")

    # ===== 1. Adjacent (sequential) node similarities =====
    adjacent_sims = []
    for i in range(n - 1):
        sim = np.dot(features_norm[i], features_norm[i + 1])
        adjacent_sims.append(float(sim))
    adjacent_sims = np.array(adjacent_sims)

    # ===== 2. Nearby nodes (within 5m) =====
    nearby_sims = []
    nearby_dists = []
    for i in range(n):
        for j in range(i + 1, min(i + 50, n)):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < 5.0:
                sim = np.dot(features_norm[i], features_norm[j])
                nearby_sims.append(float(sim))
                nearby_dists.append(float(d))
    nearby_sims = np.array(nearby_sims)
    nearby_dists = np.array(nearby_dists)

    # ===== 3. Random pairs =====
    np.random.seed(42)
    idx_i = np.random.randint(0, n, size=args.sample_pairs)
    idx_j = np.random.randint(0, n, size=args.sample_pairs)
    # Filter out same-node pairs
    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    random_sims = np.array([np.dot(features_norm[i], features_norm[j])
                            for i, j in zip(idx_i, idx_j)])

    # ===== Print statistics =====
    print(f"\nAdjacent node similarities (sequential):")
    print(f"  Mean: {np.mean(adjacent_sims):.4f}, Median: {np.median(adjacent_sims):.4f}")
    print(f"  P90: {np.percentile(adjacent_sims, 90):.4f}, P95: {np.percentile(adjacent_sims, 95):.4f}")
    print(f"  Max: {np.max(adjacent_sims):.4f}")

    print(f"\nNearby (<5m) node similarities:")
    print(f"  Count: {len(nearby_sims)}")
    print(f"  Mean: {np.mean(nearby_sims):.4f}, Median: {np.median(nearby_sims):.4f}")
    print(f"  P90: {np.percentile(nearby_sims, 90):.4f}, P95: {np.percentile(nearby_sims, 95):.4f}")
    print(f"  Max: {np.max(nearby_sims):.4f}")

    print(f"\nRandom pair similarities:")
    print(f"  Mean: {np.mean(random_sims):.4f}, Median: {np.median(random_sims):.4f}")
    print(f"  P90: {np.percentile(random_sims, 90):.4f}, P95: {np.percentile(random_sims, 95):.4f}")

    # ===== Plot =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram of all three distributions
    ax = axes[0, 0]
    bins = np.linspace(-0.15, 0.8, 100)
    ax.hist(random_sims, bins=bins, alpha=0.5, density=True, label='Random pairs', color='#2196F3')
    ax.hist(nearby_sims, bins=bins, alpha=0.5, density=True, label='Nearby (<5m)', color='#FF9800')
    ax.hist(adjacent_sims, bins=bins, alpha=0.5, density=True, label='Sequential', color='#4CAF50')
    ax.set_xlabel('Cosine Similarity (VLAD)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('SIFT+VLAD Feature Similarity Distribution', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. Similarity vs spatial distance
    ax = axes[0, 1]
    if len(nearby_dists) > 0:
        ax.scatter(nearby_dists, nearby_sims, alpha=0.05, s=1, color='#2196F3')
        # Binned mean
        dist_bins = np.linspace(0, 5, 20)
        bin_centers = [(dist_bins[i] + dist_bins[i+1])/2 for i in range(len(dist_bins)-1)]
        bin_means = []
        for i in range(len(dist_bins)-1):
            mask_bin = (nearby_dists >= dist_bins[i]) & (nearby_dists < dist_bins[i+1])
            if mask_bin.sum() > 0:
                bin_means.append(np.mean(nearby_sims[mask_bin]))
            else:
                bin_means.append(np.nan)
        ax.plot(bin_centers, bin_means, 'r-', linewidth=2, label='Binned mean')
    ax.set_xlabel('Spatial Distance (m)', fontsize=11)
    ax.set_ylabel('Cosine Similarity (VLAD)', fontsize=11)
    ax.set_title('Feature Similarity vs Spatial Distance', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. CDF of adjacent similarities
    ax = axes[1, 0]
    sorted_adj = np.sort(adjacent_sims)
    cdf = np.arange(1, len(sorted_adj) + 1) / len(sorted_adj)
    ax.plot(sorted_adj, cdf, linewidth=2, color='#4CAF50', label='Sequential pairs')
    sorted_nearby = np.sort(nearby_sims)
    cdf_nearby = np.arange(1, len(sorted_nearby) + 1) / len(sorted_nearby)
    ax.plot(sorted_nearby, cdf_nearby, linewidth=2, color='#FF9800', label='Nearby (<5m)')

    # Mark typical thresholds
    for thresh in [0.2, 0.3, 0.4, 0.5, 0.6]:
        ax.axvline(x=thresh, color='gray', linestyle=':', alpha=0.4)
        ax.annotate(f'{thresh}', xy=(thresh, 0.02), fontsize=8, color='gray',
                    ha='center')
    ax.set_xlabel('Cosine Similarity (VLAD)', fontsize=11)
    ax.set_ylabel('CDF', fontsize=11)
    ax.set_title('Cumulative Distribution of Feature Similarity', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. Nodes retained vs similarity threshold
    ax = axes[1, 1]
    thresholds = np.linspace(0.1, 0.7, 30)
    nodes_kept = []
    for thresh in thresholds:
        # Count how many nodes survive greedy similarity pruning
        kept = [0]
        kept_feats = features_norm[0:1].copy()
        for i in range(1, n):
            sims = np.dot(kept_feats, features_norm[i])
            if np.max(sims) < thresh:
                kept.append(i)
                kept_feats = np.vstack([kept_feats, features_norm[i:i+1]])
        nodes_kept.append(len(kept))
    ax.plot(thresholds, nodes_kept, 'b-', linewidth=2)
    ax.set_xlabel('Similarity Threshold', fontsize=11)
    ax.set_ylabel('Nodes Retained', fontsize=11)
    ax.set_title('Nodes Retained vs Similarity Threshold', fontsize=13)
    ax.axhline(y=755, color='gray', linestyle='--', alpha=0.5, label=f'Total nodes ({n})')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Annotate key thresholds
    for thresh in [0.2, 0.3, 0.4, 0.5, 0.6]:
        idx = np.argmin(np.abs(thresholds - thresh))
        ax.annotate(f'{nodes_kept[idx]}', xy=(thresh, nodes_kept[idx]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'similarity_distribution.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"\nSaved plots to {args.output_dir}/similarity_distribution.png")


if __name__ == "__main__":
    main()