#!/usr/bin/env python
"""
SPTM (Semi-Parametric Topological Memory) — PyTorch Implementation

Faithful port of the SPTM baseline from https://github.com/nsavinov/SPTM
(ICLR 2018, Savinov et al.) from Keras/TF to PyTorch.

Two networks:
  - R-network (edge predictor): Siamese ResNet-18, 6-channel input → 2-class
    (reachable / not reachable). Uses two shared-weight ResNet-18 branches,
    concatenated embeddings → top network (BN+ReLU → 4×Dense(512) → Dense(2)).
  - L-network (action predictor): ResNet-18, 9-channel input → 7-class actions.
    Input is [prev_frame, current_frame, goal_frame] stacked along channels.

Training data generation mirrors the original:
  - Edge predictor: positive pairs (temporally close ≤5 steps),
    negative pairs (temporally distant ≥25 steps apart).
  - Action predictor: triplets (prev, current, goal) → action label.

For CARLA: uses collected trajectory images, downsamples to 84×84, and maps
7 discrete actions to continuous CARLA controls.

Usage:
    # Train edge predictor (R-network):
    python sptm_train.py --network edge --map-dir corl-2026-results/town02_map

    # Train action predictor (L-network):
    python sptm_train.py --network action --map-dir corl-2026-results/town02_map

    # Evaluate SPTM as baseline:
    python sptm_train.py --eval-only --map-dir corl-2026-results/town02_map
"""

import os
import sys
import argparse
import pickle
import json
import random
from collections import defaultdict

import cv2
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms as T

import networkx as nx

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# HYPERPARAMETERS (faithful to original SPTM constants)
# ============================================================================

NUM_EMBEDDING = 512
TOP_HIDDEN = 4
NORMALIZATION_ON = False

# MobileNetV3 output dimensions (features[:4] on 84x84 -> [24, 11, 11])
MOBILENET_FEATURE_DIM = 24 * 11 * 11  # 2904

# Image dimensions
NET_HEIGHT = 84
NET_WIDTH = 84
NET_CHANNELS = 3

# Action space — discrete CARLA actions
ACTION_STRAIGHT = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTIONS_LIST = [ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT]
ACTION_CLASSES = len(ACTIONS_LIST)
EDGE_CLASSES = 2

# Discrete action → [steer, throttle] mapping
CARLA_DISCRETE_ACTIONS = {
    ACTION_STRAIGHT: np.array([0.0, 0.5]),   # steer, throttle
    ACTION_LEFT:     np.array([-0.5, 0.5]),
    ACTION_RIGHT:    np.array([0.5, 0.5]),
}

# Training defaults
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EDGE_MAX_EPOCHS = 10000
ACTION_MAX_EPOCHS = 19000
DUMP_AFTER_BATCHES = 100
EDGE_EPISODES = 10
MAX_ACTION_DISTANCE = 5
NEGATIVE_SAMPLE_MULTIPLIER = 5
MAX_CONTINUOUS_PLAY = 10000
MODEL_CHECKPOINT_PERIOD = 100

# Early stopping
EARLY_STOPPING_PATIENCE = 10     # Stop if loss stays near zero for this many checks
EARLY_STOPPING_CHECK_INTERVAL = 10  # Check every N epochs
EARLY_STOPPING_LOSS_THRESHOLD = 1e-4   # Consider loss "zero" if below this

# Navigation defaults
MEMORY_SUBSAMPLING = 4
MIN_SHORTCUT_DISTANCE = 5
SMALL_SHORTCUTS_NUMBER = 2000
LARGE_SHORTCUTS_NUMBER = 100000
SHORTCUT_WINDOW = 10
NUMBER_OF_NEAREST_NEIGHBOURS = 5
WEAK_INTERMEDIATE_REACHABLE_GOAL_THRESHOLD = 0.7
INTERMEDIATE_REACHABLE_GOAL_THRESHOLD = 0.95
MIN_LOOK_AHEAD = 1
MAX_LOOK_AHEAD = 7
GOAL_DISTANCE_ALLOWANCE = 63
TESTING_BATCH_SIZE = 1024

# CARLA discrete action mapping (steer, throttle)
# ACTION_STRAIGHT = 0 → [0.0, 0.5]
# ACTION_LEFT     = 1 → [-0.5, 0.5]
# ACTION_RIGHT    = 2 → [0.5, 0.5]


class DomainRandomization:
    """Apply random augmentations to CARLA images for training robustness."""

    def __init__(self, brightness_range=(0.7, 1.3), contrast_range=(0.8, 1.2),
                 noise_std=0.02, random_flip=True):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.random_flip = random_flip

    def __call__(self, img):
        """Apply random augmentations to an (84, 84, 3) float32 image in [0,1]."""
        if random.random() < 0.5:
            factor = random.uniform(*self.brightness_range)
            img = img * factor
        if random.random() < 0.3:
            factor = random.uniform(*self.contrast_range)
            mean = img.mean()
            img = (img - mean) * factor + mean
        if random.random() < 0.3:
            noise = np.random.normal(0, self.noise_std, img.shape).astype(np.float32)
            img = img + noise
        if self.random_flip and random.random() < 0.5:
            img = img[:, ::-1, :]
        return np.clip(img, 0.0, 1.0)


# ============================================================================
# PYTORCH MODELS
# ============================================================================

class BasicBlock(nn.Module):
    """ResNet basic block (for ResNet-18/34)."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet18Backbone(nn.Module):
    """ResNet-18 feature extractor.

    Faithful to the original Keras implementation:
    - Conv1: 7×7, stride 2, 64 filters
    - MaxPool: 3×3, stride 2
    - 4 stages: [2, 2, 2, 2] basic blocks, channels [64, 128, 256, 512]
    - Ends with BN + ReLU + AdaptiveAvgPool + Flatten
    """

    def __init__(self, in_channels=3, embedding_dim=512):
        super().__init__()
        self.in_channels = 64
        self.embedding_dim = embedding_dim

        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.bn_final = nn.BatchNorm1d(512)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class TopNetwork(nn.Module):
    """Top network for Siamese edge predictor.

    Takes concatenated embeddings (2 × 512 = 1024 dim) and produces
    2-class reachable/not-reachable output.

    Architecture (faithful to original):
    BN → ReLU → Dense(512) → BN → ReLU → ... (×4) → Dense(2, softmax)
    """

    def __init__(self, in_features=2 * NUM_EMBEDDING, num_classes=EDGE_CLASSES):
        super().__init__()
        layers = []
        in_dim = in_features
        for _ in range(TOP_HIDDEN):
            layers.extend([
                nn.Linear(in_dim, NUM_EMBEDDING),
                nn.BatchNorm1d(NUM_EMBEDDING),
                nn.ReLU(inplace=True),
            ])
            in_dim = NUM_EMBEDDING
        layers.append(nn.Linear(NUM_EMBEDDING, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SiameseEdgePredictor(nn.Module):
    """R-network: Siamese ResNet-18 for edge prediction.

    Faithful to original:
    - Input: 6-channel image (2 frames concatenated: [:3] and [3:] channels)
    - Two branches share weights (single ResNet-18 backbone)
    - Embeddings concatenated → TopNetwork → 2-class softmax
    """

    def __init__(self, embedding_dim=NUM_EMBEDDING):
        super().__init__()
        self.branch = ResNet18Backbone(in_channels=NET_CHANNELS, embedding_dim=embedding_dim)
        self.top = TopNetwork(in_features=2 * embedding_dim, num_classes=EDGE_CLASSES)

    def forward(self, x):
        # Split 6-channel input into two 3-channel frames
        frame1 = x[:, :NET_CHANNELS, :, :]
        frame2 = x[:, NET_CHANNELS:, :, :]

        emb1 = self.branch(frame1)
        emb2 = self.branch(frame2)

        if NORMALIZATION_ON:
            emb1 = F.normalize(emb1, p=2, dim=1)
            emb2 = F.normalize(emb2, p=2, dim=1)

        combined = torch.cat([emb1, emb2], dim=1)
        return self.top(combined)

    def get_embedding(self, x):
        """Get 512-dim embedding from single 3-channel image."""
        return self.branch(x)


class ActionPredictor(nn.Module):
    """L-network: ResNet-18 for action prediction.

    Faithful to original:
    - Input: 9-channel image (prev + current + goal frames stacked)
    - Standard ResNet-18 with 9 input channels → 7-class softmax
    """

    def __init__(self, num_classes=ACTION_CLASSES):
        super().__init__()
        self.backbone = ResNet18Backbone(in_channels=3 * NET_CHANNELS,
                                          embedding_dim=512)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)


# ============================================================================
# MOBILENETV3-BASED MODELS (for CARLA)
# ============================================================================

class FrozenMobileNetV3Encoder(nn.Module):
    """Frozen pretrained MobileNetV3-Small features[:4] encoder."""

    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = mobilenet.features[:4]
        for p in self.features.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            feat = self.features(x)
        return feat.flatten(1)


class MobileNetV3EdgePredictor(nn.Module):
    """R-network: Siamese MobileNetV3 for edge prediction.

    - Takes two separate 3-channel images
    - Passes each through frozen MobileNetV3 → 2904-dim
    - Projects each to 512-dim via trainable Linear
    - Concatenates → TopNetwork → 2-class reachable/not-reachable
    """

    def __init__(self, embedding_dim=512):
        super().__init__()
        self.encoder = FrozenMobileNetV3Encoder()
        self.proj = nn.Sequential(
            nn.Linear(MOBILENET_FEATURE_DIM, embedding_dim),
            nn.ReLU(),
        )
        self.top = TopNetwork(in_features=2 * embedding_dim,
                              num_classes=EDGE_CLASSES)

    def forward(self, img1, img2):
        e1 = self.proj(self.encoder(img1))
        e2 = self.proj(self.encoder(img2))
        if NORMALIZATION_ON:
            e1 = F.normalize(e1, p=2, dim=1)
            e2 = F.normalize(e2, p=2, dim=1)
        combined = torch.cat([e1, e2], dim=1)
        return self.top(combined)

    def get_embedding(self, x):
        """Get projected embedding from single 3-channel image."""
        return self.proj(self.encoder(x))


class CARLAActionPredictor(nn.Module):
    """L-network: MobileNetV3 for discrete CARLA action prediction.

    - Takes current observation image and goal image as separate inputs
    - Passes each through frozen MobileNetV3 → 2904-dim each
    - Concatenates → MLP → 3-class logits (straight, left, right)
    """

    def __init__(self, hidden_dim=512, num_classes=ACTION_CLASSES):
        super().__init__()
        self.encoder = FrozenMobileNetV3Encoder()
        self.fc = nn.Sequential(
            nn.Linear(MOBILENET_FEATURE_DIM * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, obs_img, goal_img):
        obs_feat = self.encoder(obs_img)
        goal_feat = self.encoder(goal_img)
        return self.fc(torch.cat([obs_feat, goal_feat], dim=1))


# ============================================================================
# DATASETS
# ============================================================================

class EdgeDataset(Dataset):
    """Dataset for edge predictor training.

    Generates positive pairs (temporally close) and negative pairs
    (temporally distant) from sequential trajectory images.
    """

    def __init__(self, map_dir, max_action_distance=MAX_ACTION_DISTANCE,
                 neg_multiplier=NEGATIVE_SAMPLE_MULTIPLIER,
                 augment=True, augmentation_prob=0.5):
        self.max_action_distance = max_action_distance
        self.neg_multiplier = neg_multiplier
        self.augment = DomainRandomization() if augment else None
        self.augmentation_prob = augmentation_prob

        # ImageNet normalization for MobileNetV3
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        # Load map images
        with open(os.path.join(map_dir, 'map_data.pkl'), 'rb') as f:
            map_data = pickle.load(f)
        self.images_meta = map_data['metadata']['images']
        self.images_dir = os.path.join(map_dir, 'images')

        # Load adjacency info from topology graph for sequential pairs
        with open(os.path.join(map_dir, 'topology_graph.pkl'), 'rb') as f:
            G = pickle.load(f)
        self.num_nodes = G.number_of_nodes()

        # Build sequential edge list for positive pair sampling
        self.sequential_pairs = []
        for u, v in G.edges():
            self.sequential_pairs.append((u, v))
            self.sequential_pairs.append((v, u))

        self._cache_images()

    def _cache_images(self):
        """Pre-load all images."""
        self.image_cache = []
        for meta in self.images_meta:
            img_path = os.path.join(self.images_dir, meta['filename'])
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (NET_WIDTH, NET_HEIGHT))
                img = img.astype(np.float32) / 255.0
            else:
                img = np.zeros((NET_HEIGHT, NET_WIDTH, NET_CHANNELS), dtype=np.float32)
            self.image_cache.append(img)

    def _img_to_tensor(self, img):
        """Convert numpy image [H, W, C] to normalized tensor [C, H, W]."""
        if self.augment is not None and random.random() < self.augmentation_prob:
            img = self.augment(img)
        img_t = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img_t = self.normalize(img_t)
        return img_t

    def __len__(self):
        return len(self.sequential_pairs) * 2

    def __getitem__(self, idx):
        # Positive pair: temporally close (sequential edge)
        if random.random() < 0.5:
            pair_idx = random.randint(0, len(self.sequential_pairs) - 1)
            i, j = self.sequential_pairs[pair_idx]
            # Add small random offset for variety
            offset = random.randint(0, min(self.max_action_distance - 1, 3))
            j = min(j + offset, self.num_nodes - 1)
            label = 1.0
        else:
            # Negative pair: temporally distant
            i = random.randint(0, self.num_nodes - 1)
            min_dist = self.neg_multiplier * self.max_action_distance
            # Pick a far-away node
            j_candidates = list(range(max(0, i - self.num_nodes), i - min_dist)) + \
                           list(range(i + min_dist, self.num_nodes))
            if not j_candidates:
                j = random.randint(0, self.num_nodes - 1)
            else:
                j = random.choice(j_candidates) % self.num_nodes
            label = 0.0

        img_i = self.image_cache[i].copy()
        img_j = self.image_cache[j].copy()

        return (self._img_to_tensor(img_i),
                self._img_to_tensor(img_j),
                torch.tensor(label, dtype=torch.long))


class ActionDataset(Dataset):
    """Dataset for discrete CARLA action predictor training.

    Generates pairs (current_frame, goal_frame) → discrete action class
    (straight=0, left=1, right=2) from spatial relationship.
    """

    def __init__(self, map_dir, max_action_distance=MAX_ACTION_DISTANCE,
                 augment=True, augmentation_prob=0.5):
        self.max_action_distance = max_action_distance
        self.augment = DomainRandomization() if augment else None
        self.augmentation_prob = augmentation_prob

        # ImageNet normalization for MobileNetV3
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        with open(os.path.join(map_dir, 'map_data.pkl'), 'rb') as f:
            map_data = pickle.load(f)
        self.images_meta = map_data['metadata']['images']
        self.images_dir = os.path.join(map_dir, 'images')

        with open(os.path.join(map_dir, 'topology_graph.pkl'), 'rb') as f:
            G = pickle.load(f)
        self.num_nodes = G.number_of_nodes()

        # Build adjacency for action sampling
        self.edges = {}
        for u, v in G.edges():
            self.edges.setdefault(u, []).append(v)
            self.edges.setdefault(v, []).append(u)

        self._cache_images()

    def _cache_images(self):
        self.image_cache = []
        for meta in self.images_meta:
            img_path = os.path.join(self.images_dir, meta['filename'])
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (NET_WIDTH, NET_HEIGHT))
                img = img.astype(np.float32) / 255.0
            else:
                img = np.zeros((NET_HEIGHT, NET_WIDTH, NET_CHANNELS), dtype=np.float32)
            self.image_cache.append(img)

    def _img_to_tensor(self, img):
        """Convert numpy image [H, W, C] to normalized tensor [C, H, W]."""
        if self.augment is not None and random.random() < self.augmentation_prob:
            img = self.augment(img)
        img_t = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img_t = self.normalize(img_t)
        return img_t

    def __len__(self):
        return self.num_nodes * 4

    def __getitem__(self, idx):
        # Sample current node and a reachable neighbor
        current = random.randint(0, self.num_nodes - 1)

        neighbors = self.edges.get(current, [])
        if neighbors:
            goal = random.choice(neighbors)
        else:
            goal = min(current + 1, self.num_nodes - 1)

        # Derive discrete action from relative heading
        curr_meta = self.images_meta[current]
        goal_meta = self.images_meta[goal]
        dx = goal_meta['x'] - curr_meta['x']
        dy = goal_meta['y'] - curr_meta['y']
        curr_yaw = curr_meta.get('yaw', 0.0)

        heading_to_goal = np.arctan2(dy, dx)
        rel_heading = heading_to_goal - curr_yaw
        # Wrap to [-pi, pi]
        rel_heading = (rel_heading + np.pi) % (2 * np.pi) - np.pi

        # Discrete action based on heading angle
        if rel_heading < -np.pi / 6:
            action_label = ACTION_LEFT
        elif rel_heading > np.pi / 6:
            action_label = ACTION_RIGHT
        else:
            action_label = ACTION_STRAIGHT

        img_curr = self.image_cache[current].copy()
        img_goal = self.image_cache[goal].copy()

        return (self._img_to_tensor(img_curr),
                self._img_to_tensor(img_goal),
                torch.tensor(action_label, dtype=torch.long))


# ============================================================================
# TRAINING
# ============================================================================

def train_edge_predictor(map_dirs, output_dir, epochs=EDGE_MAX_EPOCHS,
                         batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                         early_stopping=True, es_patience=EARLY_STOPPING_PATIENCE,
                         es_check_interval=EARLY_STOPPING_CHECK_INTERVAL,
                         es_threshold=EARLY_STOPPING_LOSS_THRESHOLD,
                         augment=True, augmentation_prob=0.5):
    """Train the R-network (MobileNetV3 Siamese edge predictor)."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MobileNetV3EdgePredictor().to(device)
    optimizer = optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    # Support single dir or list of dirs
    if isinstance(map_dirs, str):
        map_dirs = [map_dirs]

    datasets = [EdgeDataset(d, augment=augment, augmentation_prob=augmentation_prob)
                for d in map_dirs]
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = torch.utils.data.ConcatDataset(datasets)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                       num_workers=0, pin_memory=True)

    print(f"\nTraining R-network (edge predictor)")
    print(f"  Map dirs: {map_dirs}")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Early stopping: {early_stopping} (patience={es_patience}, threshold={es_threshold})")
    print(f"  Augmentation: {augment} (prob={augmentation_prob})")

    losses = []
    accuracies = []
    best_loss = float('inf')
    zero_loss_count = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (img1, img2, labels) in enumerate(loader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        avg_loss = epoch_loss / total
        accuracy = correct / total
        losses.append(avg_loss)
        accuracies.append(accuracy)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': accuracy,
            }, os.path.join(output_dir, 'edge_model_best.pt'))

        # Early stopping check
        if early_stopping and (epoch + 1) % es_check_interval == 0:
            if avg_loss < es_threshold:
                zero_loss_count += 1
                if zero_loss_count >= es_patience:
                    print(f"  Early stopping at epoch {epoch+1}: loss={avg_loss:.6f} "
                          f"near zero for {zero_loss_count} checks")
                    break
            else:
                zero_loss_count = 0

        if (epoch + 1) % MODEL_CHECKPOINT_PERIOD == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'accuracy': accuracy,
            }, os.path.join(output_dir, f'edge_model_epoch_{epoch+1:06d}.pt'))

            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, acc={accuracy:.3f}")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': losses[-1] if losses else 0,
        'accuracy': accuracies[-1] if accuracies else 0,
    }, os.path.join(output_dir, 'edge_model_final.pt'))

    # Save training curves
    torch.save({'losses': losses, 'accuracies': accuracies},
               os.path.join(output_dir, 'edge_training_log.pt'))

    _plot_training_curves(losses, accuracies, output_dir, 'edge')
    return model


def train_action_predictor(map_dirs, output_dir, epochs=ACTION_MAX_EPOCHS,
                            batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                            early_stopping=True, es_patience=EARLY_STOPPING_PATIENCE,
                            es_check_interval=EARLY_STOPPING_CHECK_INTERVAL,
                            es_threshold=EARLY_STOPPING_LOSS_THRESHOLD,
                            augment=True, augmentation_prob=0.5):
    """Train the L-network (CARLA discrete action predictor)."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CARLAActionPredictor().to(device)
    optimizer = optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    # Support single dir or list of dirs
    if isinstance(map_dirs, str):
        map_dirs = [map_dirs]

    datasets = [ActionDataset(d, augment=augment, augmentation_prob=augmentation_prob)
                for d in map_dirs]
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = torch.utils.data.ConcatDataset(datasets)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                       num_workers=0, pin_memory=True)

    print(f"\nTraining L-network (CARLA discrete action predictor)")
    print(f"  Map dirs: {map_dirs}")
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Early stopping: {early_stopping} (patience={es_patience}, threshold={es_threshold})")
    print(f"  Augmentation: {augment} (prob={augmentation_prob})")

    losses = []
    accuracies = []
    best_loss = float('inf')
    zero_loss_count = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (img_curr, img_goal, labels) in enumerate(loader):
            img_curr = img_curr.to(device)
            img_goal = img_goal.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(img_curr, img_goal)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        avg_loss = epoch_loss / total
        accuracy = correct / total
        losses.append(avg_loss)
        accuracies.append(accuracy)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': accuracy,
            }, os.path.join(output_dir, 'action_model_best.pt'))

        # Early stopping check
        if early_stopping and (epoch + 1) % es_check_interval == 0:
            if avg_loss < es_threshold:
                zero_loss_count += 1
                if zero_loss_count >= es_patience:
                    print(f"  Early stopping at epoch {epoch+1}: loss={avg_loss:.6f} "
                          f"near zero for {zero_loss_count} checks")
                    break
            else:
                zero_loss_count = 0

        if (epoch + 1) % MODEL_CHECKPOINT_PERIOD == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'accuracy': accuracy,
            }, os.path.join(output_dir, f'action_model_epoch_{epoch+1:06d}.pt'))

            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, acc={accuracy:.3f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': losses[-1] if losses else 0,
        'accuracy': accuracies[-1] if accuracies else 0,
    }, os.path.join(output_dir, 'action_model_final.pt'))

    torch.save({'losses': losses, 'accuracies': accuracies},
               os.path.join(output_dir, 'action_training_log.pt'))

    _plot_training_curves(losses, accuracies, output_dir, 'action')
    return model


def _plot_training_curves(losses, accuracies, output_dir, prefix):
    """Plot and save training curves."""
    if accuracies:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax2.plot(accuracies)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'{prefix} Training Accuracy')
        ax2.grid(True, alpha=0.3)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))

    ax1.plot(losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{prefix} Training Loss')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{prefix}_training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# SPTM NAVIGATION (for evaluation)
# ============================================================================

class SPTMNavigator:
    """SPTM navigation module for CARLA evaluation using MobileNetV3."""

    def __init__(self, edge_model_path=None, action_model_path=None,
                 map_dir=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ImageNet normalization
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        # Load models
        if edge_model_path:
            self.edge_model = MobileNetV3EdgePredictor().to(self.device)
            ckpt = torch.load(edge_model_path, map_location=self.device)
            self.edge_model.load_state_dict(ckpt['model_state_dict'])
            self.edge_model.eval()

            # References to encoder, proj, and top for fast inference
            self.encoder = self.edge_model.encoder
            self.proj = self.edge_model.proj
            self.top_network = self.edge_model.top
        else:
            self.edge_model = None
            self.encoder = None
            self.proj = None
            self.top_network = None

        if action_model_path:
            self.action_model = CARLAActionPredictor().to(self.device)
            ckpt = torch.load(action_model_path, map_location=self.device)
            self.action_model.load_state_dict(ckpt['model_state_dict'])
            self.action_model.eval()
        else:
            self.action_model = None

        # Load map data
        if map_dir:
            self._load_map(map_dir)

        self.last_nn = None
        self.smoothed_memory = None
        self.shortest_paths = None

    def _load_map(self, map_dir):
        """Load map images and graph for navigation."""
        with open(os.path.join(map_dir, 'map_data.pkl'), 'rb') as f:
            map_data = pickle.load(f)
        self.nodes_data = map_data['metadata']['images']
        self.images_dir = os.path.join(map_dir, 'images')

        with open(os.path.join(map_dir, 'topology_graph.pkl'), 'rb') as f:
            self.G = pickle.load(f)

        # Pre-load and preprocess keyframes
        self.keyframes = []
        self.keyframe_coordinates = []
        self._preprocess_keyframes()

    def _preprocess_keyframes(self):
        """Load and preprocess all keyframe images with normalization."""
        self.keyframe_tensors = []
        for meta in self.nodes_data:
            img_path = os.path.join(self.images_dir, meta['filename'])
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (NET_WIDTH, NET_HEIGHT))
                img = img.astype(np.float32) / 255.0
            else:
                img = np.zeros((NET_HEIGHT, NET_WIDTH, NET_CHANNELS), dtype=np.float32)
            self.keyframes.append(img)
            self.keyframe_coordinates.append((meta['x'], meta['y']))

            img_t = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
            img_t = self.normalize(img_t)
            self.keyframe_tensors.append(img_t.unsqueeze(0).to(self.device))

    def _np_to_tensor(self, frame):
        """Convert numpy image [H, W, C] to normalized tensor on device."""
        if isinstance(frame, torch.Tensor):
            return frame
        t = torch.from_numpy(np.transpose(frame, (2, 0, 1))).float()
        if t.max() > 1.0:
            t = t / 255.0
        t = self.normalize(t)
        return t.unsqueeze(0).to(self.device)

    def predict_edge(self, frame1, frame2):
        """Predict reachability between two frames using Siamese network."""
        x1 = self._np_to_tensor(frame1)
        x2 = self._np_to_tensor(frame2)

        with torch.no_grad():
            output = self.edge_model(x1, x2)
            probs = torch.softmax(output, dim=1)
        return probs[0, 1].item()  # Probability of reachable

    def predict_edge_siamese(self, query_img, memory_codes):
        """Predict reachability using pre-computed memory embeddings."""
        with torch.no_grad():
            query_code = self.proj(self.encoder(query_img))

        if NORMALIZATION_ON:
            query_code = F.normalize(query_code, p=2, dim=1)

        # Batch top-network forward pass for speed
        query_repeated = query_code.repeat(memory_codes.shape[0], 1)
        combined = torch.cat([query_repeated, memory_codes], dim=1)
        with torch.no_grad():
            probs = torch.softmax(self.top_network(combined), dim=1)
        return probs[:, 1].cpu().numpy()

    def set_memory_buffer(self):
        """Pre-compute all keyframe embeddings for fast inference."""
        all_keyframes = torch.cat(self.keyframe_tensors, dim=0)
        with torch.no_grad():
            self.memory_codes = self.proj(self.encoder(all_keyframes))
        if NORMALIZATION_ON:
            self.memory_codes = F.normalize(self.memory_codes, p=2, dim=1)

    def build_graph(self):
        """Build topological graph with shortcuts (faithful to original SPTM)."""
        print("Building SPTM graph...")
        self.set_memory_buffer()

        # Compute all pairwise reachability probabilities
        num_keyframes = len(self.keyframes)
        shortcuts_matrix = np.zeros((num_keyframes, num_keyframes))

        for i in tqdm.tqdm(range(num_keyframes), desc="Computing shortcuts"):
            probs = self.predict_edge_siamese(
                self.keyframe_tensors[i], self.memory_codes)
            shortcuts_matrix[i] = probs

        # Smooth shortcuts (symmetric median filter)
        for i in range(num_keyframes):
            for j in range(i + 1, num_keyframes):
                shortcuts_matrix[i][j] = (shortcuts_matrix[i][j] +
                                           shortcuts_matrix[j][i]) / 2.0
                shortcuts_matrix[j][i] = shortcuts_matrix[i][j]

        # Collect shortcut candidates with temporal distance filtering
        shortcuts = []
        for i in range(num_keyframes):
            for j in range(i + 1 + MIN_SHORTCUT_DISTANCE, num_keyframes):
                values = []
                for shift in range(-SHORTCUT_WINDOW, SHORTCUT_WINDOW + 1):
                    i_shifted = i + shift
                    j_shifted = j + shift
                    if (0 <= i_shifted < num_keyframes and
                        0 <= j_shifted < num_keyframes):
                        values.append(shortcuts_matrix[i_shifted][j_shifted])
                if values:
                    quality = np.median(values)
                    dist = np.sqrt(
                        (self.keyframe_coordinates[i][0] -
                         self.keyframe_coordinates[j][0]) ** 2 +
                        (self.keyframe_coordinates[i][1] -
                         self.keyframe_coordinates[j][1]) ** 2)
                    shortcuts.append((quality, i, j, dist))

        shortcuts = np.array(shortcuts) if shortcuts else np.empty((0, 4))

        # Sieve: keep top LARGE_SHORTCUTS_NUMBER, then top SMALL_SHORTCUTS_NUMBER
        if len(shortcuts) > 0:
            if len(shortcuts) > LARGE_SHORTCUTS_NUMBER:
                quality_threshold = np.percentile(
                    shortcuts[:, 0],
                    (len(shortcuts) - LARGE_SHORTCUTS_NUMBER) * 100 / len(shortcuts))
                shortcuts = shortcuts[shortcuts[:, 0] >= quality_threshold]

            if len(shortcuts) > SMALL_SHORTCUTS_NUMBER:
                quality_threshold = np.percentile(
                    shortcuts[:, 0],
                    (len(shortcuts) - SMALL_SHORTCUTS_NUMBER) * 100 / len(shortcuts))
                shortcuts = shortcuts[shortcuts[:, 0] >= quality_threshold]

        # Build graph
        self.G = nx.Graph()
        self.G.add_nodes_from(range(num_keyframes))

        # Add sequential edges
        for i in range(num_keyframes - 1):
            self.G.add_edge(i, i + 1)
            self.G.add_edge(i + 1, i)

        # Add shortcut edges
        for s in shortcuts:
            _, i, j, _ = s
            self.G.add_edge(int(i), int(j))
            self.G.add_edge(int(j), int(i))

        print(f"  Graph: {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges, "
              f"{len(shortcuts)} shortcuts")

    def find_nn(self, input_frame):
        """Find nearest neighbor using edge predictor."""
        input_tensor = self._np_to_tensor(input_frame)

        probabilities = self.predict_edge_siamese(input_tensor, self.memory_codes)
        best_index = np.argmax(probabilities)
        best_probability = np.max(probabilities)
        return best_index, best_probability, probabilities

    def find_smoothed_nn(self, input_frame):
        """Find nearest neighbor with smoothed localization (faithful to original)."""
        nn = None
        if self.last_nn is not None and self.shortest_paths is not None:
            nn, probabilities = self.find_nn_on_last_shortest_path(input_frame)
        if nn is None:
            nn, probabilities, _ = self.find_knn_median_threshold(
                input_frame, NUMBER_OF_NEAREST_NEIGHBOURS,
                INTERMEDIATE_REACHABLE_GOAL_THRESHOLD)
        return nn, probabilities

    def find_nn_on_last_shortest_path(self, input_frame):
        """Check last shortest path for localization (faithful to original)."""
        if self.last_nn is None or self.shortest_paths is None:
            return None, None

        path = self.shortest_paths.get(self.last_nn)
        if path is None:
            return None, None

        path_prefix = path[:MAX_LOOK_AHEAD + 1]
        probabilities = self.predict_edge_siamese(
            self.keyframe_tensors[path_prefix[0]],
            self.memory_codes) if len(path_prefix) > 0 else np.array([])

        best_look_ahead = None
        best_probability = 0
        for look_ahead, idx in enumerate(path_prefix):
            prob = probabilities[idx] if idx < len(probabilities) else 0
            if prob > best_probability:
                best_probability = prob
                best_look_ahead = look_ahead

        if best_probability < WEAK_INTERMEDIATE_REACHABLE_GOAL_THRESHOLD:
            return None, None
        return path_prefix[best_look_ahead], probabilities

    def find_knn_median_threshold(self, input_frame, k, threshold):
        """k-NN with median thresholding (faithful to original)."""
        probabilities = self.predict_edge_siamese(
            self._np_to_tensor(input_frame),
            self.memory_codes)

        n = len(probabilities)
        knn_threshold = np.percentile(probabilities, (n - k) * 100 / n)
        final_threshold = max(threshold, knn_threshold)

        nns = [i for i in range(n) if probabilities[i] >= final_threshold]
        nns.sort()

        if nns:
            nn = nns[len(nns) // 2]
            return nn, probabilities, nns
        else:
            return None, probabilities, nns

    def set_goal(self, goal_frame):
        """Set navigation goal (faithful to original)."""
        best_index, probabilities, nns = self.find_knn_median_threshold(
            goal_frame, NUMBER_OF_NEAREST_NEIGHBOURS, 0.0)
        best_probability = 1.0
        if best_index is None:
            best_index, best_probability, _ = self.find_nn(goal_frame)

        goal_index = len(self.keyframes)
        self.G.add_node(goal_index)
        self.G.add_edge(best_index, goal_index)
        self.G.add_edge(goal_index, best_index)

        self.keyframes.append(goal_frame)
        self.keyframe_coordinates.append((0, 0))  # approximate
        self.last_nn = None
        self.smoothed_memory = None

        return best_index, best_probability

    def set_goal_by_location(self, goal_location):
        """Set navigation goal by CARLA location (find nearest keyframe)."""
        goal_pos = np.array([goal_location.x, goal_location.y])
        coords = np.array(self.keyframe_coordinates)
        dists = np.linalg.norm(coords - goal_pos, axis=1)
        best_index = int(np.argmin(dists))

        goal_node_idx = len(self.keyframes)
        self.G.add_node(goal_node_idx)
        self.G.add_edge(best_index, goal_node_idx)
        self.G.add_edge(goal_node_idx, best_index)

        self.keyframes.append(self.keyframes[best_index])
        self.keyframe_coordinates.append((goal_location.x, goal_location.y))
        self.last_nn = None
        self.smoothed_memory = None

        return best_index, 1.0, goal_node_idx

    def compute_shortest_paths(self, graph_goal):
        """Compute shortest paths to goal (faithful to original)."""
        self.shortest_paths = nx.shortest_path(self.G, target=graph_goal)

    def select_irg(self, nn, probabilities):
        """Select Intermediate Reachable Goal on shortest path (faithful to original)."""
        shortest_path = self.shortest_paths.get(nn, [nn])
        upper_limit = len(shortest_path) - 1
        valid_min = min(MIN_LOOK_AHEAD, upper_limit)
        valid_max = min(MAX_LOOK_AHEAD, upper_limit)

        best_look_ahead = valid_min
        for look_ahead in range(valid_min, valid_max + 1):
            idx = shortest_path[look_ahead]
            if idx < len(probabilities) and probabilities[idx] >= INTERMEDIATE_REACHABLE_GOAL_THRESHOLD:
                best_look_ahead = look_ahead

        return shortest_path[best_look_ahead]

    def find_intermediate_reachable_goal(self, input_frame):
        """Find IRG for navigation step (faithful to original)."""
        nn, probabilities = self.find_smoothed_nn(input_frame)
        self.last_nn = nn
        if nn is None:
            return None, None
        return self.select_irg(nn, probabilities), nn

    def predict_action(self, obs_frame, goal_frame):
        """Predict discrete action and map to [steer, throttle]."""
        obs_t = self._np_to_tensor(obs_frame)
        goal_t = self._np_to_tensor(goal_frame)

        with torch.no_grad():
            logits = self.action_model(obs_t, goal_t)
            action_idx = int(torch.argmax(logits, dim=1)[0])
        return CARLA_DISCRETE_ACTIONS[action_idx]


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_sptm(env, nav, trajectory_path, traj_id, max_steps=3000,
                  termination_budget=2):
    """Evaluate SPTM on a single trajectory in CARLA."""
    from rlib_integration.helper import ndarray_to_location
    from rlib_integration.agent import GlobalRoutePlanner

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

    # Set goal in SPTM graph
    goal_frame = nav.keyframes[-1]  # Use last keyframe as approximate goal
    goal_index, _ = nav.set_goal(goal_frame)
    nav.compute_shortest_paths(len(nav.keyframes) - 1)

    success_rate, SPL, distance_completed, agent_trajectory = [], [], [], []
    budget = termination_budget

    while budget > 0:
        obs, info = env.reset()
        done = False
        episode_length = 0
        traj_positions = []
        nav.last_nn = None

        env.step([0.0, 0.0])

        while not done and episode_length < max_steps:
            # Get current observation
            curr_transform = unwrapped.core.hero.get_transform()
            hero_pos = np.array([curr_transform.location.x, curr_transform.location.y])
            traj_positions.append([curr_transform.location.x, curr_transform.location.y])

            # Get current frame from observation
            curr_frame = unwrapped.experiment.obs_img
            if curr_frame is not None:
                curr_frame_resized = cv2.resize(curr_frame, (NET_WIDTH, NET_HEIGHT))
                curr_frame_float = curr_frame_resized.astype(np.float32) / 255.0
            else:
                curr_frame_float = np.zeros((NET_HEIGHT, NET_WIDTH, NET_CHANNELS), dtype=np.float32)

            # Find IRG
            irg, nn = nav.find_intermediate_reachable_goal(curr_frame_float)

            if irg is not None and irg < len(nav.keyframes):
                goal_frame = nav.keyframes[irg]
            else:
                goal_frame = nav.keyframes[-1]

            # Predict continuous [steer, throttle]
            if nav.action_model is not None:
                action = nav.predict_action(curr_frame_float, goal_frame)
                steer, throttle = float(action[0]), float(action[1])
            else:
                steer, throttle = 0.0, 0.0

            obs, reward, done, truncated, info = env.step(
                np.array([steer, throttle]))
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
            budget = 0
        else:
            budget -= 1
            unwrapped.set_start_transform(start_location)

    return {
        "trajectory_id": traj_id, "model": "sptm",
        "agent_trajectory": agent_trajectory,
        "success": float(np.max(success_rate)) if success_rate else 0.0,
        "success_rate": float(np.mean(success_rate)) if success_rate else 0.0,
        "mean_spl": float(np.mean(SPL)) if SPL else 0.0,
        "mean_distance": float(np.mean(distance_completed)) if distance_completed else 0.0,
        "num_attempts": termination_budget - budget,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SPTM (Semi-Parametric Topological Memory) — PyTorch Implementation")
    parser.add_argument('--network', type=str, default='edge',
                        choices=['edge', 'action', 'both'],
                        help='Which network to train')
    parser.add_argument('--map-dir', type=str, action='append',
                        help='Directory containing map images and topology (use multiple times)')
    parser.add_argument('--output-dir', type=str, default='sptm_output',
                        help='Directory for model checkpoints and logs')
    parser.add_argument('--epochs-edge', type=int, default=EDGE_MAX_EPOCHS,
                        help='Number of training epochs for edge predictor')
    parser.add_argument('--epochs-action', type=int, default=ACTION_MAX_EPOCHS,
                        help='Number of training epochs for action predictor')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training, only run evaluation')
    parser.add_argument('--edge-model', type=str, default=None,
                        help='Path to trained edge model for evaluation')
    parser.add_argument('--action-model', type=str, default=None,
                        help='Path to trained action model for evaluation')
    parser.add_argument('--benchmark-dir', type=str,
                        default='corl-2026-results/trajectories',
                        help='Directory with benchmark trajectories for evaluation')
    parser.add_argument('--no-early-stopping', action='store_true',
                        help='Disable early stopping when loss reaches near-zero')
    parser.add_argument('--early-stopping-patience', type=int,
                        default=EARLY_STOPPING_PATIENCE,
                        help='Number of consecutive near-zero loss checks before stopping')
    parser.add_argument('--early-stopping-threshold', type=float,
                        default=EARLY_STOPPING_LOSS_THRESHOLD,
                        help='Loss threshold below which is considered near-zero')
    parser.add_argument('--early-stopping-check-interval', type=int,
                        default=EARLY_STOPPING_CHECK_INTERVAL,
                        help='Check early stopping every N epochs')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Disable domain randomization augmentation')
    parser.add_argument('--augmentation-prob', type=float, default=0.5,
                        help='Probability of applying augmentation to each image')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    map_dirs = args.map_dir or ['corl-2026-results/town02_map']

    print(f"\n{'#'*60}")
    print("SPTM — Semi-Parametric Topological Memory (PyTorch)")
    print(f"{'#'*60}")
    print(f"Map: {map_dirs}")
    print(f"Output: {args.output_dir}")
    print(f"Network: {args.network}")

    if not args.eval_only:
        if args.network in ('edge', 'both'):
            train_edge_predictor(map_dirs, args.output_dir,
                                epochs=args.epochs_edge,
                                batch_size=args.batch_size, lr=args.lr,
                                early_stopping=not args.no_early_stopping,
                                es_patience=args.early_stopping_patience,
                                es_check_interval=args.early_stopping_check_interval,
                                es_threshold=args.early_stopping_threshold,
                                augment=not args.no_augmentation,
                                augmentation_prob=args.augmentation_prob)

        if args.network in ('action', 'both'):
            train_action_predictor(map_dirs, args.output_dir,
                                    epochs=args.epochs_action,
                                    batch_size=args.batch_size, lr=args.lr,
                                    early_stopping=not args.no_early_stopping,
                                    es_patience=args.early_stopping_patience,
                                    es_check_interval=args.early_stopping_check_interval,
                                    es_threshold=args.early_stopping_threshold,
                                    augment=not args.no_augmentation,
                                    augmentation_prob=args.augmentation_prob)

    if args.eval_only:
        edge_path = args.edge_model or os.path.join(args.output_dir, 'edge_model_final.pt')
        action_path = args.action_model or os.path.join(args.output_dir, 'action_model_final.pt')

        if not os.path.exists(edge_path):
            print(f"Warning: Edge model not found at {edge_path}")
            print("Building graph with random initialization (for testing)")
            edge_path = None

        nav = SPTMNavigator(edge_model_path=edge_path,
                           action_model_path=action_path if os.path.exists(action_path) else None,
                           map_dir=args.map_dir)

        nav.build_graph()
        print(f"SPTM navigator ready with {nav.G.number_of_nodes()} nodes "
              f"and {nav.G.number_of_edges()} edges")


if __name__ == "__main__":
    main()