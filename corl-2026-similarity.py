#!/usr/bin/env python
"""
CoRL 2026: Cross-map VLAD similarity distribution analysis.

Compares SIFT+VLAD feature similarity distributions across three CARLA Town02
topological maps to understand how map density and collection strategy affect
the visual redundancy of nodes.

The three maps:
  1. dense-gps:  1492 nodes, collected with GPS waypoint following
  2. dense-image: 755 nodes, collected with image-goal policy
  3. graph_output: 2050 nodes, earlier full-coverage collection

Generates:
  - Cross-map similarity histograms
  - Per-map similarity CDFs
  - Similarity vs spatial distance per map
  - Nodes retained vs similarity threshold per map
  - Cross-map nearest-neighbor similarity (how well maps cover each other)
  - Ablation readiness: nodes remaining at each similarity threshold

Usage:
    python corl-2026-similarity.py
    python corl-2026-similarity.py --output-dir ablation-results
"""

import os
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


MAP_CONFIGS = {
    'dense-gps': {
        'dir': 'corl-2026-results-dense-gps/town02_map',
        'label': 'Dense-GPS',
        'color': '#2196F3',
        'features_file': 'features.pkl',
        'map_data_file': 'map_data.pkl',
        'feature_key': 'all_features',
    },
    'dense-image': {
        'dir': 'corl-2026-results-dense-image/town02_map',
        'label': 'Dense-Image',
        'color': '#FF9800',
        'features_file': 'features.pkl',
        'map_data_file': 'map_data.pkl',
        'feature_key': 'all_features',
    },
    'sparse-gps-5m': {
        'dir': 'corl-2026-results-dense-gps/town02_map',
        'label': 'Sparse-GPS 5m',
        'color': '#E91E63',
        'features_file': 'features.pkl',
        'map_data_file': 'map_data.pkl',
        'feature_key': 'all_features',
        'spatial_subsample': 5.0,
    },
    'sparse-image-sim03': {
        'dir': 'corl-2026-results-dense-image/town02_map',
        'label': 'Sparse-Image sim≤0.3',
        'color': '#9C27B0',
        'features_file': 'features.pkl',
        'map_data_file': 'map_data.pkl',
        'feature_key': 'all_features',
        'similarity_subsample': 0.3,
    },
    'town05-gps': {
        'dir': 'corl-2026-results-dense-gps/town05_map',
        'label': 'Town05 Dense-GPS',
        'color': '#00BCD4',
        'features_file': 'features.pkl',
        'map_data_file': 'map_data.pkl',
        'feature_key': 'all_features',
    },
    'town05-sparse-gps-5m': {
        'dir': 'corl-2026-results-dense-gps/town05_map',
        'label': 'Town05 Sparse-GPS 5m',
        'color': '#009688',
        'features_file': 'features.pkl',
        'map_data_file': 'map_data.pkl',
        'feature_key': 'all_features',
        'spatial_subsample': 5.0,
    },
    'town05-sparse-image-sim03': {
        'dir': 'corl-2026-results-dense-gps/town05_map',
        'label': 'Town05 Sparse sim≤0.3',
        'color': '#795548',
        'features_file': 'features.pkl',
        'map_data_file': 'map_data.pkl',
        'feature_key': 'all_features',
        'similarity_subsample': 0.3,
    },
}


def _spatial_subsample(positions, min_distance):
    """Greedy spatial subsampling: keep nodes at least min_distance apart."""
    kept = [0]
    for i in range(1, len(positions)):
        dists = np.linalg.norm(positions[kept] - positions[i], axis=1)
        if np.min(dists) >= min_distance:
            kept.append(i)
    return kept


def _similarity_subsample(features_norm, threshold):
    """Greedy similarity subsampling: keep nodes with max cosine sim < threshold."""
    kept = [0]
    kept_feats = features_norm[0:1].copy()
    for i in range(1, len(features_norm)):
        sims = np.dot(kept_feats, features_norm[i])
        if np.max(sims) < threshold:
            kept.append(i)
            kept_feats = np.vstack([kept_feats, features_norm[i:i+1]])
    return kept


def load_map(config):
    """Load features and positions for a map."""
    features_path = os.path.join(config['dir'], config['features_file'])
    with open(features_path, 'rb') as f:
        feat_data = pickle.load(f)
    features = feat_data[config['feature_key']]
    feature_dim = feat_data.get('feature_dim', features.shape[1])

    # Normalize for cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    features_norm = features / norms

    # Load positions
    positions = None
    if config['map_data_file'] is not None:
        map_path = os.path.join(config['dir'], config['map_data_file'])
        with open(map_path, 'rb') as f:
            map_data = pickle.load(f)
        nodes = map_data['metadata']['images']
        positions = np.array([[n['x'], n['y']] for n in nodes])
    else:
        alt_path = os.path.join(config['dir'], 'map_filtered.pkl')
        if os.path.exists(alt_path):
            with open(alt_path, 'rb') as f:
                nodes = pickle.load(f)
            loc = nodes[0]['location']
            if isinstance(loc, (tuple, list)):
                positions = np.array([[n['location'][0], n['location'][1]] for n in nodes])
            else:
                positions = np.array([[n['location']['x'], n['location']['y']] for n in nodes])
        else:
            positions = None

    if positions is not None and len(positions) != len(features):
        print(f"  WARNING: positions ({len(positions)}) != features ({len(features)}), dropping positions")
        positions = None

    # Apply subsampling if configured
    if 'spatial_subsample' in config and positions is not None:
        min_dist = config['spatial_subsample']
        kept = _spatial_subsample(positions, min_dist)
        features = features[kept]
        features_norm = features_norm[kept]
        positions = positions[kept]
        config = dict(config)
        config['label'] = f"{config['label']} ({len(kept)} nodes)"
        print(f"  Spatial subsample {min_dist}m: {len(kept)} nodes kept")

    if 'similarity_subsample' in config:
        threshold = config['similarity_subsample']
        kept = _similarity_subsample(features_norm, threshold)
        features = features[kept]
        features_norm = features_norm[kept]
        if positions is not None:
            positions = positions[kept]
        config = dict(config)
        config['label'] = f"{config['label']} ({len(kept)} nodes)"
        print(f"  Similarity subsample ≤{threshold}: {len(kept)} nodes kept")

    return features, features_norm, positions, feature_dim


def compute_adjacent_sims(features_norm):
    """Compute cosine similarities between sequentially adjacent nodes."""
    sims = []
    for i in range(len(features_norm) - 1):
        sim = np.dot(features_norm[i], features_norm[i + 1])
        sims.append(float(sim))
    return np.array(sims)


def compute_nearby_sims(features_norm, positions, max_dist=5.0, max_offset=50):
    """Compute similarities for nodes within max_dist meters (vectorized per-node)."""
    n = len(features_norm)
    if positions is not None:
        sims = []
        dists = []
        for i in range(n):
            j_start = i + 1
            j_end = min(i + max_offset, n)
            if j_start >= j_end:
                continue
            d = np.linalg.norm(positions[j_start:j_end] - positions[i], axis=1)
            mask = d < max_dist
            if mask.any():
                sim_vals = features_norm[j_start:j_end][mask] @ features_norm[i]
                sims.append(sim_vals)
                dists.append(d[mask])
        if sims:
            return np.concatenate(sims), np.concatenate(dists)
        return np.array([]), np.array([])
    else:
        all_sims = []
        for i in range(n):
            j_start = i + 1
            j_end = min(i + max_offset, n)
            if j_start >= j_end:
                continue
            sim_vals = features_norm[j_start:j_end] @ features_norm[i]
            all_sims.append(sim_vals)
        if all_sims:
            return np.concatenate(all_sims), np.zeros(len(np.concatenate(all_sims)))
        return np.array([]), np.array([])


def compute_random_pair_sims(features_norm, n_pairs=50000, seed=42):
    """Compute similarities for random node pairs."""
    np.random.seed(seed)
    n = len(features_norm)
    idx_i = np.random.randint(0, n, size=n_pairs)
    idx_j = np.random.randint(0, n, size=n_pairs)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    # Batch compute
    batch_size = 5000
    sims = []
    for k in range(0, len(idx_i), batch_size):
        bi, bj = idx_i[k:k+batch_size], idx_j[k:k+batch_size]
        sims.extend(np.sum(features_norm[bi] * features_norm[bj], axis=1).tolist())
    return np.array(sims)


def compute_cross_map_sims(features_norm_a, features_norm_b, n_queries=200, k=5):
    """For nodes in map A, find nearest k nodes in map B and return similarities."""
    np.random.seed(42)
    query_idx = np.random.choice(len(features_norm_a), size=min(n_queries, len(features_norm_a)), replace=False)
    queries = features_norm_a[query_idx]
    # Compute all pairwise to map B
    all_sims = []
    for q in queries:
        sims = np.dot(features_norm_b, q)
        top_k = np.sort(sims)[-k:][::-1]
        all_sims.append(top_k)
    return np.array(all_sims)


def compute_nodes_retained(features_norm, thresholds):
    """For each similarity threshold, count nodes retained after greedy pruning."""
    n = len(features_norm)
    results = {}
    for thresh in thresholds:
        kept = [0]
        kept_feats = features_norm[0:1].copy()
        for i in range(1, n):
            sims = np.dot(kept_feats, features_norm[i])
            if np.max(sims) < thresh:
                kept.append(i)
                kept_feats = np.vstack([kept_feats, features_norm[i:i+1]])
        results[thresh] = len(kept)
    return results


def main():
    parser = argparse.ArgumentParser(description="CoRL 2026: Cross-map similarity analysis")
    parser.add_argument('--output-dir', type=str, default='ablation-results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load all maps
    maps = {}
    for name, config in MAP_CONFIGS.items():
        if not os.path.exists(os.path.join(config['dir'], config['features_file'])):
            print(f"Skipping {name}: features not found at {config['dir']}")
            continue
        print(f"Loading {name} from {config['dir']}...")
        features, features_norm, positions, feature_dim = load_map(config)
        maps[name] = {
            'features': features,
            'features_norm': features_norm,
            'positions': positions,
            'feature_dim': feature_dim,
            'config': config,
        }
        print(f"  {len(features)} nodes, {feature_dim}-dim features")

    map_names = list(maps.keys())
    print(f"\nLoaded {len(map_names)} maps: {map_names}")

    # Compute similarity distributions
    for name, m in maps.items():
        m['adj_sims'] = compute_adjacent_sims(m['features_norm'])
        m['nearby_sims'], m['nearby_dists'] = compute_nearby_sims(
            m['features_norm'], m['positions'])
        m['random_sims'] = compute_random_pair_sims(m['features_norm'])

    # Compute cross-map similarities
    if len(map_names) >= 2:
        for i, name_a in enumerate(map_names):
            for j, name_b in enumerate(map_names):
                if i != j:
                    cross = compute_cross_map_sims(
                        maps[name_a]['features_norm'],
                        maps[name_b]['features_norm'])
                    maps[name_a][f'cross_{name_b}'] = cross

    # Compute nodes retained at each threshold (fewer thresholds for speed)
    thresholds = np.linspace(0.05, 0.7, 14)
    for name, m in maps.items():
        m['nodes_retained'] = compute_nodes_retained(m['features_norm'], thresholds)

    # ===== PLOTTING =====

    # --- Figure 1: Per-map similarity distributions (3x2 grid) ---
    fig, axes = plt.subplots(len(map_names), 2, figsize=(16, 5 * len(map_names)))
    if len(map_names) == 1:
        axes = axes[np.newaxis, :]

    for row, name in enumerate(map_names):
        m = maps[name]
        cfg = m['config']

        # Left: histogram of all three similarity types
        ax = axes[row, 0]
        bins = np.linspace(-0.15, 0.8, 100)
        ax.hist(m['random_sims'], bins=bins, alpha=0.5, density=True,
                label='Random pairs', color='#9E9E9E')
        ax.hist(m['nearby_sims'], bins=bins, alpha=0.5, density=True,
                label='Nearby (<5m)', color='#FF9800')
        ax.hist(m['adj_sims'], bins=bins, alpha=0.5, density=True,
                label='Sequential', color=cfg['color'])
        ax.set_xlabel('Cosine Similarity', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{cfg["label"]} — Similarity Distribution', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Right: CDF
        ax = axes[row, 1]
        sorted_adj = np.sort(m['adj_sims'])
        cdf = np.arange(1, len(sorted_adj) + 1) / len(sorted_adj)
        ax.plot(sorted_adj, cdf, linewidth=2, color=cfg['color'], label='Sequential')
        if len(m['nearby_sims']) > 0:
            sorted_near = np.sort(m['nearby_sims'])
            cdf_near = np.arange(1, len(sorted_near) + 1) / len(sorted_near)
            ax.plot(sorted_near, cdf_near, linewidth=2, color='#FF9800', label='Nearby (<5m)')
        for t in [0.2, 0.3, 0.4, 0.5, 0.6]:
            ax.axvline(x=t, color='gray', linestyle=':', alpha=0.3)
            ax.annotate(f'{t}', xy=(t, 0.02), fontsize=8, color='gray', ha='center')
        ax.set_xlabel('Cosine Similarity', fontsize=11)
        ax.set_ylabel('CDF', fontsize=11)
        ax.set_title(f'{cfg["label"]} — CDF', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'per_map_similarity.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved per_map_similarity.png")

    # --- Figure 2: Cross-map comparison (all maps on same axes) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Adjacent similarity distributions overlaid
    ax = axes[0, 0]
    for name, m in maps.items():
        cfg = m['config']
        ax.hist(m['adj_sims'], bins=np.linspace(-0.15, 0.8, 80), alpha=0.5,
                density=True, label=cfg['label'], color=cfg['color'])
    ax.set_xlabel('Cosine Similarity', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Sequential Node Similarity — All Maps', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Top-right: Nodes retained vs threshold
    ax = axes[0, 1]
    for name, m in maps.items():
        cfg = m['config']
        nodes_list = [m['nodes_retained'][t] for t in thresholds]
        ax.plot(thresholds, nodes_list, '-o', color=cfg['color'], label=cfg['label'],
                linewidth=2, markersize=3)
    ax.set_xlabel('Similarity Threshold', fontsize=11)
    ax.set_ylabel('Nodes Retained', fontsize=11)
    ax.set_title('Nodes Retained vs Similarity Threshold', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Similarity vs spatial distance
    ax = axes[1, 0]
    for name, m in maps.items():
        cfg = m['config']
        if m['positions'] is not None and len(m['nearby_dists']) > 0:
            mask = m['nearby_dists'] > 0
            ax.scatter(m['nearby_dists'][mask], m['nearby_sims'][mask],
                       alpha=0.03, s=1, color=cfg['color'])
            # Binned mean
            dist_bins = np.linspace(0.5, 5, 15)
            bin_centers = [(dist_bins[i] + dist_bins[i+1])/2 for i in range(len(dist_bins)-1)]
            bin_means = []
            for bi in range(len(dist_bins)-1):
                mask_bin = (m['nearby_dists'] >= dist_bins[bi]) & (m['nearby_dists'] < dist_bins[bi+1])
                bin_means.append(np.mean(m['nearby_sims'][mask_bin]) if mask_bin.sum() > 0 else np.nan)
            ax.plot(bin_centers, bin_means, '-', color=cfg['color'], linewidth=2.5,
                    label=cfg['label'])
    ax.set_xlabel('Spatial Distance (m)', fontsize=11)
    ax.set_ylabel('Mean Cosine Similarity', fontsize=11)
    ax.set_title('Feature Similarity vs Spatial Distance', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-right: CDF overlaid
    ax = axes[1, 1]
    for name, m in maps.items():
        cfg = m['config']
        sorted_adj = np.sort(m['adj_sims'])
        cdf = np.arange(1, len(sorted_adj) + 1) / len(sorted_adj)
        ax.plot(sorted_adj, cdf, linewidth=2, color=cfg['color'], label=cfg['label'])
    for t in [0.2, 0.3, 0.4, 0.5, 0.6]:
        ax.axvline(x=t, color='gray', linestyle=':', alpha=0.3)
        ax.annotate(f'{t}', xy=(t, 0.02), fontsize=8, color='gray', ha='center')
    ax.set_xlabel('Cosine Similarity', fontsize=11)
    ax.set_ylabel('CDF', fontsize=11)
    ax.set_title('Sequential Similarity CDF — All Maps', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'cross_map_similarity.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved cross_map_similarity.png")

    # --- Figure 3: Cross-map nearest-neighbor similarity ---
    if len(map_names) >= 2:
        n_cross = len(map_names)
        n_plots = n_cross * (n_cross - 1)
        ncols = min(n_plots, 4)
        nrows = (n_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        axes = np.atleast_2d(axes).flatten()
        plot_idx = 0
        for name_a in map_names:
            for name_b in map_names:
                if name_a == name_b:
                    continue
                ax = axes[plot_idx]
                key = f'cross_{name_b}'
                if key in maps[name_a]:
                    cross_sims = maps[name_a][key]
                    for k_idx in range(min(5, cross_sims.shape[1])):
                        col = cross_sims[:, k_idx]
                        n_unique = len(np.unique(col))
                        if n_unique > 1:
                            lo, hi = float(col.min()), float(col.max())
                            margin = max((hi - lo) * 0.05, 1e-6)
                            n_bins = min(20, max(5, n_unique))
                            bins = np.linspace(lo - margin, hi + margin, n_bins + 1)
                            ax.hist(col, bins=bins, alpha=0.5,
                                    label=f'NN-{k_idx+1}', density=True)
                    ax.set_xlabel('Cosine Similarity', fontsize=11)
                    ax.set_ylabel('Density', fontsize=11)
                    ax.set_title(f'{MAP_CONFIGS[name_a]["label"]}\n→ NN in {MAP_CONFIGS[name_b]["label"]}',
                                fontsize=10)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                plot_idx += 1

        plt.tight_layout()
        fig.savefig(os.path.join(args.output_dir, 'cross_map_nn_similarity.png'),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved cross_map_nn_similarity.png")

    # --- Figure 4: Ablation readiness table ---
    ablation_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, m in maps.items():
        cfg = m['config']
        nodes_at_thresh = compute_nodes_retained(m['features_norm'], ablation_thresholds)
        counts = [nodes_at_thresh[t] for t in ablation_thresholds]
        ax.plot(ablation_thresholds, counts, 'o-', color=cfg['color'],
                label=cfg['label'], linewidth=2, markersize=6)
        for t, c in zip(ablation_thresholds, counts):
            ax.annotate(f'{c}', xy=(t, c), xytext=(3, 5),
                        textcoords='offset points', fontsize=7)

    ax.set_xlabel('Similarity Threshold (keep nodes with max sim < threshold)', fontsize=11)
    ax.set_ylabel('Nodes Retained', fontsize=11)
    ax.set_title('Ablation Readiness: Nodes at Each Similarity Threshold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'ablation_readiness.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved ablation_readiness.png")

    # Print summary table
    print(f"\n{'='*80}")
    print("SIMILARITY DISTRIBUTION SUMMARY")
    print(f"{'='*80}")
    print(f"{'Map':<25} {'Nodes':>6} {'Adj Mean':>10} {'Adj P95':>10} {'Nearby Mean':>12} {'Max Sim':>10}")
    print("-" * 80)
    for name, m in maps.items():
        cfg = m['config']
        adj_mean = np.mean(m['adj_sims'])
        adj_p95 = np.percentile(m['adj_sims'], 95)
        nearby_mean = np.mean(m['nearby_sims']) if len(m['nearby_sims']) > 0 else 0
        max_sim = np.max(m['adj_sims'])
        print(f"{cfg['label']:<25} {len(m['features']):>6} {adj_mean:>10.4f} {adj_p95:>10.4f} {nearby_mean:>12.4f} {max_sim:>10.4f}")

    print(f"\n{'='*80}")
    print("ABLATION THRESHOLD TABLE (nodes retained)")
    print(f"{'='*80}")
    header = f"{'Threshold':<12}"
    for name, m in maps.items():
        header += f" {MAP_CONFIGS[name]['label']:>20}"
    print(header)
    print("-" * len(header))

    ablation_check_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for t in ablation_check_thresholds:
        row = f"{t:<12}"
        for name, m in maps.items():
            nodes = m['nodes_retained'].get(t, compute_nodes_retained(m['features_norm'], [t])[t])
            row += f" {nodes:>20}"
        print(row)

    print(f"\nPlots saved to {args.output_dir}/")
    print(f"  per_map_similarity.png")
    print(f"  cross_map_similarity.png")
    print(f"  cross_map_nn_similarity.png")
    print(f"  ablation_readiness.png")


if __name__ == "__main__":
    main()