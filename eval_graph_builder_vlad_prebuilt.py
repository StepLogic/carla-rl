#!/usr/bin/env python
"""
Topological Graph Builder with SIFT+VLAD Features — Prebuilt Loader

Copy of eval_graph_builder_vlad.py modified to load pre-built map artifacts
directly from a directory (e.g., corl-2026-results/town02_map/), skipping
the expensive feature extraction, similarity detection, and graph building
steps.

All classes (SIFTExtractor, VLADEncoder, SIFTVLADFeatureExtractor, FAISSIndex,
UnionFind, SimilarityResult, detect_similar_nodes) are kept intact for
feature extraction at runtime on new images. The TopologicalGraphBuilder gains
a new load_from_directory() method that restores from saved pkl files.
"""

import os
import pickle
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Any

import numpy as np
import cv2
import networkx as nx
import faiss
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.cluster import MiniBatchKMeans

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SIFTVLADConfig:
    """Configuration for SIFT+VLAD feature extraction."""
    n_keypoints: int = 500
    contrast_threshold: float = 0.04
    edge_threshold: int = 10
    sigma: float = 1.6
    n_clusters: int = 64
    vlad_normalize: bool = True
    intra_normalize: bool = True
    codebook_sample_size: int = 100000
    random_state: int = 42
    resize_height: int = 240
    resize_width: int = 320
    verbose: bool = True


@dataclass
class GraphBuilderConfig:
    """Configuration for graph building."""
    min_loop_closure_gap: int = 5
    min_similarity_score: float = 0.1
    max_duplicate_similarity: float = 0.2
    faiss_top_k: int = 50
    use_gpu: bool = True
    index_type: str = 'auto'
    max_bridge_distance: int = 50
    feature_type: str = 'sift_vlad'
    image_key: str = 'obs_img'
    latent_key: str = 'latent'
    sift_vlad_config: SIFTVLADConfig = field(default_factory=SIFTVLADConfig)
    verbose: bool = True
    save_diagnostics: bool = True


# =============================================================================
# SIFT EXTRACTOR
# =============================================================================

class SIFTExtractor:
    """Extracts SIFT descriptors from images."""

    def __init__(self, config: SIFTVLADConfig):
        self.config = config
        self.sift = cv2.SIFT_create(
            nfeatures=config.n_keypoints,
            contrastThreshold=config.contrast_threshold,
            edgeThreshold=config.edge_threshold,
            sigma=config.sigma
        )

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for SIFT."""
        if image is None:
            raise ValueError("Image is None")
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.config.resize_height and self.config.resize_width:
            image = cv2.resize(image, (self.config.resize_width, self.config.resize_height))
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        return image

    def extract(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract SIFT keypoints and descriptors."""
        gray = self.preprocess(image)
        kps, descs = self.sift.detectAndCompute(gray, None)
        if descs is None or len(kps) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 128), dtype=np.float32)
        keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
        return keypoints, descs.astype(np.float32)

    def extract_batch(self, images: List[np.ndarray], show_progress: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Extract SIFT from multiple images."""
        results = []
        if show_progress and self.config.verbose:
            try:
                from tqdm import tqdm
                images = tqdm(images, desc="Extracting SIFT")
            except ImportError:
                pass
        for img in images:
            results.append(self.extract(img))
        return results


# =============================================================================
# VLAD ENCODER
# =============================================================================

class VLADEncoder:
    """VLAD encoder for aggregating local descriptors."""

    def __init__(self, config: SIFTVLADConfig):
        self.config = config
        self.codebook: np.ndarray = None
        self.faiss_index: faiss.Index = None
        self.descriptor_dim: int = 128
        self.vlad_dim: int = None
        self.is_trained: bool = False

    def train(self, descriptors: np.ndarray):
        """Train codebook using k-means."""
        if self.config.verbose:
            print(f"\nTraining VLAD codebook...")
            print(f"  Descriptors: {descriptors.shape}")
            print(f"  Clusters: {self.config.n_clusters}")
        if len(descriptors) > self.config.codebook_sample_size:
            idx = np.random.choice(len(descriptors), self.config.codebook_sample_size, replace=False)
            descriptors = descriptors[idx]
        kmeans = MiniBatchKMeans(
            n_clusters=self.config.n_clusters,
            random_state=self.config.random_state,
            batch_size=1024, max_iter=100, n_init=3
        )
        kmeans.fit(descriptors)
        self.codebook = kmeans.cluster_centers_.astype(np.float32)
        self.descriptor_dim = self.codebook.shape[1]
        self.vlad_dim = self.config.n_clusters * self.descriptor_dim
        self.faiss_index = faiss.IndexFlatL2(self.descriptor_dim)
        self.faiss_index.add(self.codebook)
        self.is_trained = True

    def encode(self, descriptors: np.ndarray) -> np.ndarray:
        """Encode descriptors into VLAD vector."""
        if not self.is_trained:
            raise RuntimeError("VLAD not trained")
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.vlad_dim, dtype=np.float32)
        descriptors = descriptors.astype(np.float32)
        _, assignments = self.faiss_index.search(descriptors, 1)
        assignments = assignments.flatten()
        vlad = np.zeros((self.config.n_clusters, self.descriptor_dim), dtype=np.float32)
        for i, desc in enumerate(descriptors):
            cluster_idx = assignments[i]
            vlad[cluster_idx] += desc - self.codebook[cluster_idx]
        if self.config.intra_normalize:
            for i in range(self.config.n_clusters):
                norm = np.linalg.norm(vlad[i])
                if norm > 1e-8:
                    vlad[i] /= norm
        vlad = vlad.flatten()
        if self.config.vlad_normalize:
            norm = np.linalg.norm(vlad)
            if norm > 1e-8:
                vlad /= norm
        return vlad

    def encode_batch(self, descriptors_list: List[np.ndarray], show_progress: bool = True) -> np.ndarray:
        """Encode multiple descriptor sets."""
        vlad_vectors = []
        if show_progress and self.config.verbose:
            try:
                from tqdm import tqdm
                descriptors_list = tqdm(descriptors_list, desc="Encoding VLAD")
            except ImportError:
                pass
        for descs in descriptors_list:
            vlad_vectors.append(self.encode(descs))
        return np.array(vlad_vectors, dtype=np.float32)

    def save(self, path: str):
        """Save trained encoder."""
        with open(path, 'wb') as f:
            pickle.dump({
                'codebook': self.codebook,
                'config': self.config,
                'vlad_dim': self.vlad_dim,
                'descriptor_dim': self.descriptor_dim
            }, f)

    def load(self, path: str):
        """Load trained encoder."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.codebook = data['codebook']
        self.config = data.get('config', self.config)
        self.vlad_dim = data['vlad_dim']
        self.descriptor_dim = data['descriptor_dim']
        self.faiss_index = faiss.IndexFlatL2(self.descriptor_dim)
        self.faiss_index.add(self.codebook)
        self.is_trained = True


# =============================================================================
# SIFT+VLAD FEATURE EXTRACTOR
# =============================================================================

class SIFTVLADFeatureExtractor:
    """Combined SIFT + VLAD feature extractor."""

    def __init__(self, config: SIFTVLADConfig):
        self.config = config
        self.sift_extractor = SIFTExtractor(config)
        self.vlad_encoder = VLADEncoder(config)
        self.is_trained = False

    def train_from_images(self, images: List[np.ndarray]):
        """Train VLAD codebook from images."""
        all_descriptors = []
        sift_results = self.sift_extractor.extract_batch(images)
        for _, descs in sift_results:
            if len(descs) > 0:
                all_descriptors.append(descs)
        all_descriptors = np.vstack(all_descriptors)
        self.vlad_encoder.train(all_descriptors)
        self.is_trained = True

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract SIFT+VLAD feature from image."""
        if not self.is_trained:
            raise RuntimeError("Not trained")
        _, descriptors = self.sift_extractor.extract(image)
        return self.vlad_encoder.encode(descriptors)

    def extract_batch(self, images: List[np.ndarray], show_progress: bool = True) -> np.ndarray:
        """Extract features from multiple images."""
        if not self.is_trained:
            raise RuntimeError("Not trained")
        sift_results = self.sift_extractor.extract_batch(images, show_progress)
        descriptors_list = [descs for _, descs in sift_results]
        return self.vlad_encoder.encode_batch(descriptors_list, show_progress)

    def save(self, path: str):
        self.vlad_encoder.save(path)

    def load(self, path: str):
        self.vlad_encoder.load(path)
        self.is_trained = True

    @property
    def feature_dim(self) -> int:
        if not self.is_trained:
            return self.config.n_clusters * 128
        return self.vlad_encoder.vlad_dim


# =============================================================================
# FAISS INDEX
# =============================================================================

class FAISSIndex:
    """FAISS-based index for similarity search."""

    def __init__(self, feature_dim: int, use_gpu: bool = True, index_type: str = 'auto', n_samples: int = None):
        self.feature_dim = feature_dim
        self.use_gpu = use_gpu and self._check_gpu()
        self.index_type = self._select_type(index_type, n_samples)
        self.index = None
        self.is_built = False

    def _check_gpu(self) -> bool:
        try:
            return faiss.get_num_gpus() > 0
        except:
            return False

    def _select_type(self, index_type: str, n_samples: int) -> str:
        if index_type != 'auto':
            return index_type
        if n_samples is None or n_samples < 1000:
            return 'flat'
        elif n_samples < 100000:
            return 'IVF'
        return 'HNSW'

    def build(self, features: np.ndarray, verbose: bool = True):
        features = np.ascontiguousarray(features.astype(np.float32))
        n_samples, dim = features.shape
        faiss.normalize_L2(features)
        if verbose:
            print(f"Building FAISS index: {n_samples} samples, {dim} dims, type={self.index_type}")
        if self.index_type == 'flat':
            self.index = faiss.IndexFlatIP(dim)
        elif self.index_type == 'IVF':
            n_clusters = min(int(np.sqrt(n_samples) * 2), 256)
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
            self.index.train(features)
            self.index.nprobe = min(n_clusters // 4, 32)
        elif self.index_type == 'HNSW':
            self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 64
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.index.add(features)
        self.is_built = True

    def search(self, queries: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        queries = np.ascontiguousarray(queries.astype(np.float32))
        faiss.normalize_L2(queries)
        return self.index.search(queries, k)

    def search_single(self, query: np.ndarray) -> Tuple[int, float]:
        sims, idxs = self.search(query, k=1)
        return int(idxs[0, 0]), float(sims[0, 0])


# =============================================================================
# UNION-FIND
# =============================================================================

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

    def get_groups(self) -> Dict[int, List[int]]:
        groups = defaultdict(list)
        for i in range(len(self.parent)):
            groups[self.find(i)].append(i)
        return dict(groups)


# =============================================================================
# SIMILARITY RESULT
# =============================================================================

@dataclass
class SimilarityResult:
    duplicate_pairs: List[Dict] = field(default_factory=list)
    loop_closures: List[Dict] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


def detect_similar_nodes(features: np.ndarray, coords: np.ndarray, config: GraphBuilderConfig) -> SimilarityResult:
    """Find duplicates and loop closures using FAISS."""
    n_samples, feature_dim = features.shape
    index = FAISSIndex(feature_dim, config.use_gpu, config.index_type, n_samples)
    index.build(features.copy(), config.verbose)
    top_k = min(config.faiss_top_k, n_samples)
    similarities, indices = index.search(features, k=top_k)
    duplicate_pairs, loop_closures, all_sims = [], [], []
    for i in range(n_samples):
        for k_idx in range(top_k):
            j, sim = int(indices[i, k_idx]), float(similarities[i, k_idx])
            if j <= i:
                continue
            gap = j - i
            if gap < config.min_loop_closure_gap:
                continue
            all_sims.append(sim)
            spatial_dist = float(np.linalg.norm(coords[i] - coords[j]))
            entry = {'node1': i, 'node2': j, 'similarity': sim,
                      'spatial_dist': spatial_dist, 'sequence_gap': gap}
            if sim >= config.max_duplicate_similarity:
                duplicate_pairs.append(entry)
            elif sim >= config.min_similarity_score:
                loop_closures.append(entry)
    stats = {
        'n_samples': n_samples, 'feature_dim': feature_dim,
        'n_comparisons': len(all_sims),
        'sim_min': min(all_sims) if all_sims else 0,
        'sim_max': max(all_sims) if all_sims else 0,
        'sim_mean': float(np.mean(all_sims)) if all_sims else 0,
        'sim_std': float(np.std(all_sims)) if all_sims else 0,
        'n_duplicates': len(duplicate_pairs),
        'n_loop_closures': len(loop_closures),
        'feature_type': config.feature_type
    }
    return SimilarityResult(duplicate_pairs, loop_closures, stats)


# =============================================================================
# TOPOLOGICAL GRAPH BUILDER (with prebuilt loading)
# =============================================================================

class TopologicalGraphBuilder:
    """Builds topological navigation graph using SIFT+VLAD or latent features.

    Supports two initialization modes:
      - build_from_file(): Full pipeline — extract features, detect duplicates,
        build graph from scratch. Slow (seconds to minutes).
      - load_from_directory(): Load pre-built artifacts directly from a saved
        map directory. Fast (sub-second).
    """

    def __init__(self, config: GraphBuilderConfig = None):
        self.config = config or GraphBuilderConfig()

        self.original_nodes_data: List[Dict] = []
        self.nodes_data: List[Dict] = []
        self.G: nx.Graph = nx.Graph()

        self.new_to_old: Dict[int, int] = {}
        self.old_to_new: Dict[int, int] = {}
        self.nodes_to_remove: Set[int] = set()

        self.all_features_normalized: np.ndarray = None
        self.filtered_features: np.ndarray = None
        self.feature_dim: int = None

        self.feature_extractor: SIFTVLADFeatureExtractor = None
        self.main_index: FAISSIndex = None
        self.similarity_result: SimilarityResult = None
        self.build_stats: Dict[str, Any] = {}

    # =========================================================================
    # PREBUILT LOADING — instant initialization from saved map directory
    # =========================================================================

    def load_from_directory(self, map_dir: str) -> 'TopologicalGraphBuilder':
        """Load pre-built map artifacts from a directory.

        Uses ALL nodes (no filtering). The topology_graph.pkl provides the full
        1492-node graph with sequential edges, and all_features feeds the FAISS
        index so image-based localization searches over every captured frame.

        Expects the following files in map_dir:
          - topology_graph.pkl  : full graph with all nodes (sequential edges)
          - features.pkl        : dict with all_features, filtered_features, feature_dim
          - vlad_encoder.pkl     : trained VLAD encoder
          - map_data.pkl        : dict with metadata -> images (x, y, filename, etc.)
          - aux.pkl             : dict with positions, yaws, filenames
          - map_filtered.pkl    : filtered nodes (used if topology_graph missing)
          - graph.pkl           : filtered graph (used if topology_graph missing)
          - index_mappings.pkl  : dict with new_to_old, old_to_new (used if filtering)

        Returns self for chaining.
        """
        if self.config.verbose:
            print(f"\n{'='*60}")
            print("LOADING PREBUILT MAP")
            print(f"{'='*60}")
            print(f"Directory: {map_dir}")

        # 1. Load original nodes data with full metadata (x, y, filename, etc.)
        map_data_path = os.path.join(map_dir, "map_data.pkl")
        if os.path.exists(map_data_path):
            with open(map_data_path, "rb") as f:
                map_data = pickle.load(f)
            self.original_nodes_data = map_data.get('metadata', {}).get('images', [])
        else:
            raise FileNotFoundError(f"map_data.pkl not found in {map_dir}")

        # 2. Load features (all 1492)
        features_path = os.path.join(map_dir, "features.pkl")
        with open(features_path, "rb") as f:
            features = pickle.load(f)
        self.all_features_normalized = features['all_features']
        # Use all features (no filtering) — filtered_features is the same array
        self.filtered_features = self.all_features_normalized
        self.feature_dim = features['feature_dim']

        # 3. Load full topology graph (all nodes, sequential edges)
        topo_path = os.path.join(map_dir, "topology_graph.pkl")
        if os.path.exists(topo_path):
            with open(topo_path, "rb") as f:
                self.G = pickle.load(f)
        else:
            # Fall back to filtered graph
            graph_path = os.path.join(map_dir, "graph.pkl")
            with open(graph_path, "rb") as f:
                graph_data = pickle.load(f)
            self.G = nx.node_link_graph(graph_data)

        # 4. Use all original nodes as nodes_data (no filtering)
        self.nodes_data = self.original_nodes_data
        # Add compass from yaw if missing
        for node in self.nodes_data:
            if 'compass' not in node and 'yaw' in node:
                node['compass'] = np.radians(node['yaw'])

        # 5. Load auxiliary data (positions, yaws) for compass
        aux_path = os.path.join(map_dir, "aux.pkl")
        if os.path.exists(aux_path):
            with open(aux_path, "rb") as f:
                aux_data = pickle.load(f)
            if 'yaws' in aux_data and len(self.nodes_data) == len(aux_data['yaws']):
                for i, node in enumerate(self.nodes_data):
                    if 'compass' not in node:
                        node['compass'] = np.radians(aux_data['yaws'][i])

        # 6. Identity mapping (no filtering — original index == graph index)
        n_nodes = len(self.nodes_data)
        self.new_to_old = {i: i for i in range(n_nodes)}
        self.old_to_new = {i: i for i in range(n_nodes)}
        self.nodes_to_remove = set()

        # 7. Load VLAD encoder for runtime feature extraction from new images
        vlad_path = os.path.join(map_dir, "vlad_encoder.pkl")
        if os.path.exists(vlad_path):
            self.feature_extractor = SIFTVLADFeatureExtractor(self.config.sift_vlad_config)
            self.feature_extractor.load(vlad_path)

        # 8. Build FAISS navigation index from ALL features (not filtered)
        # Use flat index for exact search — 1492 vectors is small enough
        self.main_index = FAISSIndex(self.feature_dim, self.config.use_gpu, 'flat', n_nodes)
        self.main_index.build(self.all_features_normalized.copy(), self.config.verbose)

        # 9. Compile build stats
        self.build_stats = {
            'original_nodes': n_nodes,
            'filtered_nodes': n_nodes,
            'removed_nodes': 0,
            'total_edges': self.G.number_of_edges(),
            'feature_type': features.get('feature_type', 'sift_vlad'),
            'feature_dim': self.feature_dim,
            'loaded_from': map_dir,
        }

        if self.config.verbose:
            print(f"Loaded {n_nodes} nodes, {self.G.number_of_edges()} edges")
            print(f"Feature dim: {self.feature_dim}")
            print(f"FAISS index: {n_nodes} features (all, no filtering)")
            print(f"VLAD encoder: {'loaded' if self.feature_extractor and self.feature_extractor.is_trained else 'not available'}")

        return self

    # =========================================================================
    # FULL BUILD PIPELINE (kept from original for building new maps)
    # =========================================================================

    def build_from_file(self, map_path: str) -> nx.Graph:
        """Build graph from pickle file (full pipeline)."""
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"LOADING DATA")
            print(f"{'='*60}")
            print(f"File: {map_path}")
        with open(map_path, "rb") as f:
            self.original_nodes_data = pickle.load(f)
        if self.config.verbose:
            print(f"Loaded {len(self.original_nodes_data)} nodes")
            print(f"Keys: {list(self.original_nodes_data[0].keys())}")
        return self.build_from_data(self.original_nodes_data)

    def build_from_data(self, nodes_data: List[Dict]) -> nx.Graph:
        """Build graph from node data (full pipeline)."""
        self.original_nodes_data = nodes_data
        self._extract_features()
        coords = np.array([[d['location'][0], d['location'][1]] for d in self.original_nodes_data])
        self.similarity_result = detect_similar_nodes(self.all_features_normalized, coords, self.config)
        self._remove_duplicates()
        self._cluster_loop_closures()
        self._build_filtered_graph()
        self._add_sequential_edges()
        self._build_navigation_index()
        self._compile_stats()
        return self.G

    def _extract_features(self):
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"EXTRACTING FEATURES: {self.config.feature_type.upper()}")
            print(f"{'='*60}")
        if self.config.feature_type == 'sift_vlad':
            self._extract_sift_vlad_features()
        else:
            self._extract_latent_features()

    def _extract_sift_vlad_features(self):
        key = self.config.image_key
        images = [node[key] for node in self.original_nodes_data]
        if self.config.verbose:
            sample = np.array(images[0])
            print(f"Image key: '{key}'")
            print(f"Image shape: {sample.shape}, dtype: {sample.dtype}")
        self.feature_extractor = SIFTVLADFeatureExtractor(self.config.sift_vlad_config)
        self.feature_extractor.train_from_images(images)
        all_features = self.feature_extractor.extract_batch(images)
        self.feature_dim = all_features.shape[1]
        norms = np.linalg.norm(all_features, axis=1, keepdims=True)
        self.all_features_normalized = all_features / (norms + 1e-8)

    def _extract_latent_features(self):
        key = self.config.latent_key
        all_features = np.array([d[key].flatten() for d in self.original_nodes_data], dtype=np.float32)
        self.feature_dim = all_features.shape[1]
        norms = np.linalg.norm(all_features, axis=1, keepdims=True)
        self.all_features_normalized = all_features / (norms + 1e-8)

    def _remove_duplicates(self):
        uf = UnionFind(len(self.original_nodes_data))
        for dp in self.similarity_result.duplicate_pairs:
            uf.union(dp['node1'], dp['node2'])
        groups = {k: v for k, v in uf.get_groups().items() if len(v) > 1}
        for nodes in groups.values():
            for node in sorted(nodes)[1:]:
                self.nodes_to_remove.add(node)

    def _cluster_loop_closures(self):
        uf = UnionFind(len(self.original_nodes_data))
        for lc in self.similarity_result.loop_closures:
            if lc['node1'] not in self.nodes_to_remove and lc['node2'] not in self.nodes_to_remove:
                uf.union(lc['node1'], lc['node2'])
        groups = defaultdict(list)
        for i in range(len(self.original_nodes_data)):
            if i not in self.nodes_to_remove:
                groups[uf.find(i)].append(i)
        loop_groups = {k: v for k, v in groups.items() if len(v) > 1}
        for nodes in loop_groups.values():
            counts = defaultdict(int)
            for lc in self.similarity_result.loop_closures:
                if lc['node1'] in nodes:
                    counts[lc['node1']] += 1
                if lc['node2'] in nodes:
                    counts[lc['node2']] += 1
            rep = max(nodes, key=lambda n: counts.get(n, 0))
            for node in nodes:
                if node != rep:
                    self.nodes_to_remove.add(node)

    def _build_filtered_graph(self):
        filtered_indices = [i for i in range(len(self.original_nodes_data)) if i not in self.nodes_to_remove]
        self.nodes_data = [self.original_nodes_data[i] for i in filtered_indices]
        self.new_to_old = {new: old for new, old in enumerate(filtered_indices)}
        self.old_to_new = {old: new for new, old in enumerate(filtered_indices)}
        self.filtered_features = self.all_features_normalized[filtered_indices]
        self.G = nx.Graph()
        for i, data in enumerate(self.nodes_data):
            self.G.add_node(i, **data, original_idx=self.new_to_old[i])

    def _add_sequential_edges(self):
        seq_edges, bridge_edges = 0, 0
        for new_i in range(len(self.nodes_data)):
            old_i = self.new_to_old[new_i]
            for offset in range(1, self.config.max_bridge_distance + 1):
                old_j = old_i + offset
                if old_j >= len(self.original_nodes_data):
                    break
                if old_j in self.old_to_new:
                    new_j = self.old_to_new[old_j]
                    if self.G.has_edge(new_i, new_j):
                        break
                    di, dj = self.nodes_data[new_i], self.nodes_data[new_j]
                    if di.get("episode") != dj.get("episode"):
                        break
                    yaw_i, yaw_j = di.get('yaw', 0), dj.get('yaw', 0)
                    rel_yaw = np.arctan2(np.sin(yaw_j - yaw_i), np.cos(yaw_j - yaw_i))
                    edge_type = 'sequential' if offset == 1 else 'bridged'
                    self.G.add_edge(new_i, new_j, weight=offset, type=edge_type,
                                    relative_yaw=rel_yaw, skipped=offset - 1)
                    if offset == 1:
                        seq_edges += 1
                    else:
                        bridge_edges += 1
                    break
        self.build_stats['sequential_edges'] = seq_edges
        self.build_stats['bridged_edges'] = bridge_edges

    def _add_loop_closure_edges(self):
        loop_edges = 0
        for lc in self.similarity_result.loop_closures:
            old_i, old_j = lc['node1'], lc['node2']
            if old_i in self.old_to_new and old_j in self.old_to_new:
                new_i, new_j = self.old_to_new[old_i], self.old_to_new[old_j]
                if not self.G.has_edge(new_i, new_j):
                    di, dj = self.nodes_data[new_i], self.nodes_data[new_j]
                    yaw_i, yaw_j = di.get('yaw', 0), dj.get('yaw', 0)
                    rel_yaw = np.arctan2(np.sin(yaw_j - yaw_i), np.cos(yaw_j - yaw_i))
                    self.G.add_edge(new_i, new_j, weight=1.0 - lc['similarity'],
                                    type='loop_closure', relative_yaw=rel_yaw,
                                    similarity=lc['similarity'], sequence_gap=lc['sequence_gap'])
                    loop_edges += 1
        self.build_stats['loop_closure_edges'] = loop_edges

    def _build_navigation_index(self):
        self.main_index = FAISSIndex(self.feature_dim, self.config.use_gpu, 'flat', len(self.nodes_data))
        self.main_index.build(self.filtered_features.copy(), self.config.verbose)

    def _compile_stats(self):
        edge_types = defaultdict(int)
        for _, _, d in self.G.edges(data=True):
            edge_types[d.get('type', 'unknown')] += 1
        self.build_stats.update({
            'original_nodes': len(self.original_nodes_data),
            'filtered_nodes': len(self.nodes_data),
            'removed_nodes': len(self.nodes_to_remove),
            'reduction_percent': 100 * len(self.nodes_to_remove) / max(len(self.original_nodes_data), 1),
            'total_edges': self.G.number_of_edges(),
            'edge_types': dict(edge_types),
            'feature_type': self.config.feature_type,
            'feature_dim': self.feature_dim
        })
        if self.config.verbose:
            print(f"\n{'='*60}")
            print("BUILD COMPLETE")
            print(f"{'='*60}")
            print(f"Feature type: {self.config.feature_type}")
            print(f"Feature dim: {self.feature_dim}")
            print(f"Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    # =========================================================================
    # NAVIGATION METHODS
    # =========================================================================

    def extract_feature_from_image(self, image: np.ndarray) -> np.ndarray:
        """Extract SIFT+VLAD feature from image."""
        if self.feature_extractor is None:
            raise RuntimeError("Feature extractor not available (use sift_vlad mode)")
        return self.feature_extractor.extract(image)

    def find_nearest_node(self, feature: np.ndarray) -> int:
        """Find nearest node to feature vector."""
        idx, _ = self.main_index.search_single(np.array(feature).flatten())
        return idx

    def find_nearest_node_from_image(self, image: np.ndarray) -> int:
        """Find nearest node from image."""
        return self.find_nearest_node(self.extract_feature_from_image(image))

    def find_nearest_node_with_confidence(self, feature: np.ndarray, threshold: float = 0.4) -> Tuple[int, float]:
        """Find nearest node with confidence. Returns (node_index, similarity_score)."""
        query = np.array(feature).flatten().reshape(1, -1)
        sims, idxs = self.main_index.search(query, k=5)
        best_idx = int(idxs[0, np.argmax(sims[0])])
        best_sim = float(np.max(sims[0]))
        return best_idx, best_sim

    def find_nearest_node_from_image_with_confidence(self, image: np.ndarray, threshold: float = 0.8) -> Tuple[int, float]:
        """Find nearest node from image with confidence. Returns (node_index, similarity_score)."""
        return self.find_nearest_node_with_confidence(self.extract_feature_from_image(image), threshold)

    def find_path(self, start_node: int, goal_node: int) -> Optional[List[int]]:
        """Find shortest path."""
        try:
            return nx.shortest_path(self.G, start_node, goal_node, weight='weight')
        except nx.NetworkXNoPath:
            return None

    def navigate(self, current_feature: np.ndarray, goal_feature: np.ndarray,
                 confidence_threshold: float = 0.8) -> Dict[str, Any]:
        """Navigate from current to goal using features."""
        start_node, _ = self.find_nearest_node_with_confidence(current_feature, confidence_threshold)
        goal_node, is_localized = self.find_nearest_node_with_confidence(goal_feature, 0.4)

        path = self.find_path(start_node, goal_node) if is_localized else None
        if path is None:
            path = [start_node, goal_node]

        path_features = np.array([self.filtered_features[i] for i in path], dtype=np.float32)
        path_index = faiss.IndexFlatIP(self.feature_dim)
        faiss.normalize_L2(path_features)
        path_index.add(path_features)

        def next_heading(feat: np.ndarray) -> float:
            feat_arr = np.array(feat).flatten().reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(feat_arr)
            _, local_idxs = path_index.search(feat_arr, k=1)
            local_idx = int(local_idxs[0, 0])
            if local_idx >= len(path) - 1:
                return 0.0
            curr, nxt = path[local_idx], path[local_idx + 1]
            if self.G.has_edge(curr, nxt):
                return self.nodes_data[curr]['compass']
            return 0.0

        return {
            'path': path,
            'start_node': start_node,
            'goal_node': goal_node,
            'is_localized': is_localized,
            'next_heading_func': next_heading
        }

    def navigate_from_images(self, current_image: np.ndarray, goal_image: np.ndarray,
                             confidence_threshold: float = 0.8) -> Dict[str, Any]:
        """Navigate using images directly."""
        current_feat = self.extract_feature_from_image(current_image)
        goal_feat = self.extract_feature_from_image(goal_image)
        nav_result = self.navigate(current_feat, goal_feat, confidence_threshold)

        orig_func = nav_result['next_heading_func']

        def next_heading_from_image(img: np.ndarray) -> float:
            f = self.extract_feature_from_image(img)
            return orig_func(f)

        nav_result['next_heading_func_image'] = next_heading_from_image
        return nav_result

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def visualize_graph(self, save_path: str = "navigation_graph.png"):
        """Visualize the navigation graph."""
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        node_pos = np.array([[d['location'][0], d['location'][1]] for d in self.nodes_data])
        pos_dict = {i: node_pos[i] for i in range(len(self.nodes_data))}
        entropies = [d.get('entropy', 0.5) for d in self.nodes_data]
        for ax, pos, title in [(axes[0], pos_dict, "Geographic"),
                               (axes[1], nx.spring_layout(self.G, k=2, seed=42), "Topological")]:
            nx.draw_networkx_edges(self.G, pos, ax=ax, alpha=0.3)
            nx.draw_networkx_nodes(self.G, pos, node_size=50, node_color=entropies, cmap=plt.cm.viridis, ax=ax)
            ax.set_title(f"{title} Layout ({self.config.feature_type.upper()})")
            ax.axis('equal')
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def save(self, output_dir: str = "."):
        """Save all outputs."""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "map_filtered.pkl"), "wb") as f:
            pickle.dump(self.nodes_data, f)
        with open(os.path.join(output_dir, "graph.pkl"), "wb") as f:
            pickle.dump(nx.node_link_data(self.G), f)
        with open(os.path.join(output_dir, "build_stats.pkl"), "wb") as f:
            pickle.dump(self.build_stats, f)
        with open(os.path.join(output_dir, "index_mappings.pkl"), "wb") as f:
            pickle.dump({
                'new_to_old': self.new_to_old,
                'old_to_new': self.old_to_new,
                'nodes_to_remove': self.nodes_to_remove
            }, f)
        with open(os.path.join(output_dir, "features.pkl"), "wb") as f:
            pickle.dump({
                'all_features': self.all_features_normalized,
                'filtered_features': self.filtered_features,
                'feature_dim': self.feature_dim,
                'feature_type': self.config.feature_type
            }, f)
        if self.feature_extractor:
            self.feature_extractor.save(os.path.join(output_dir, "vlad_encoder.pkl"))

        # Save map_data.pkl for DenseMapNavigator compatibility
        images_for_map_data = []
        for node in self.nodes_data:
            loc = node.get('location', (0.0, 0.0, 0.0))
            if isinstance(loc, (tuple, list)) and len(loc) >= 2:
                x, y, z = float(loc[0]), float(loc[1]), float(loc[2]) if len(loc) > 2 else 0.0
            else:
                x = y = z = 0.0
            images_for_map_data.append({
                'index': node.get('index', 0),
                'filename': node.get('filename', ''),
                'x': x,
                'y': y,
                'z': z,
                'yaw': node.get('yaw', 0.0),
                'road_id': node.get('road_id', -1),
                'lane_id': node.get('lane_id', -1),
            })
        map_data = {
            'metadata': {'images': images_for_map_data},
            'graph': nx.node_link_data(self.G),
            'positions': np.array([[d['x'], d['y']] for d in images_for_map_data]),
        }
        with open(os.path.join(output_dir, "map_data.pkl"), "wb") as f:
            pickle.dump(map_data, f)

    def get_stats(self) -> Dict[str, Any]:
        """Get build statistics."""
        return self.build_stats.copy()