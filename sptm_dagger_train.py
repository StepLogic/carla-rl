#!/usr/bin/env python
"""
SPTM DAgger Training — Real-time data collection with Dataset Aggregation.

Iteratively:
  1. Collect trajectories in CARLA using mixed expert/policy
  2. Label with expert (oracle) actions
  3. Aggregate with offline map data
  4. Fine-tune edge and action predictors

Usage:
    # From pretrained models (skip offline training):
    python sptm_dagger_train.py \
        --pretrained-edge-model sptm_output/edge_model_best.pt \
        --pretrained-action-model sptm_output/action_model_best.pt \
        --carla-port 2000

    # Full training from scratch:
    python sptm_dagger_train.py --carla-port 2000

Requires a CARLA server running on the specified port.
"""

import os
import sys
import argparse
import pickle
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add repo root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sptm_train import (
    SiameseEdgePredictor, ActionPredictor,
    EdgeDataset, ActionDataset, DomainRandomization,
    SPTMNavigator,
    NET_HEIGHT, NET_WIDTH, NET_CHANNELS,
    MOVE_FORWARD, MOVE_BACKWARD, MOVE_LEFT, MOVE_RIGHT,
    STAY_IDLE, TURN_LEFT, TURN_RIGHT,
    ACTIONS_LIST, ACTION_CLASSES, EDGE_CLASSES,
    CARLA_ACTIONS, LEARNING_RATE, BATCH_SIZE,
    NEGATIVE_SAMPLE_MULTIPLIER, MAX_ACTION_DISTANCE,
)


# ============================================================================
# ACTION MAPPING
# ============================================================================

class ContinuousToDiscreteActionMapper:
    """Quantize continuous [steer, throttle] to SPTM's 7 discrete actions."""

    STEER_THRESHOLD = 0.15
    THROTTLE_THRESHOLD = 0.15
    IDLE_THROTTLE = 0.05
    IDLE_STEER = 0.05
    REVERSE_THRESHOLD = -0.1

    def __call__(self, steer: float, throttle: float) -> int:
        """Map continuous action to discrete SPTM action index."""
        if throttle < self.REVERSE_THRESHOLD:
            return MOVE_BACKWARD
        if throttle < self.IDLE_THROTTLE and abs(steer) < self.IDLE_STEER:
            return STAY_IDLE
        if abs(steer) > self.STEER_THRESHOLD and throttle > self.THROTTLE_THRESHOLD:
            return MOVE_LEFT if steer < 0 else MOVE_RIGHT
        if abs(steer) > self.STEER_THRESHOLD:
            return TURN_LEFT if steer < 0 else TURN_RIGHT
        return MOVE_FORWARD

    def discrete_to_carla_control(self, discrete_action: int) -> np.ndarray:
        """Convert discrete action to [steer, throttle] for CARLA env."""
        ctrl = CARLA_ACTIONS[discrete_action]
        return np.array([ctrl['steer'], ctrl['throttle']], dtype=np.float32)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DaggerTransition:
    """A single transition collected during DAgger."""
    prev_image: np.ndarray   # (84, 84, 3) float32 [0, 1]
    curr_image: np.ndarray
    next_image: np.ndarray
    goal_image: np.ndarray
    expert_action: int       # discrete SPTM action index
    position: np.ndarray    # (x, y)


# ============================================================================
# COMBINED DATASETS
# ============================================================================

class CombinedEdgeDataset(Dataset):
    """Samples positive/negative pairs from offline map data + DAgger transitions."""

    def __init__(self, map_dir, dagger_transitions=None,
                 offline_sample_ratio=0.5,
                 augment=True, augmentation_prob=0.5):
        self.offline_dataset = EdgeDataset(map_dir, augment=augment,
                                           augmentation_prob=augmentation_prob)
        self.offline_sample_ratio = offline_sample_ratio
        self.augment = DomainRandomization() if augment else None
        self.augmentation_prob = augmentation_prob

        # Process DAgger transitions into edge pairs
        self.dagger_images = []     # list of (84,84,3) float32 arrays
        self.positive_pairs = []    # list of (i, j) index tuples
        self.negative_pairs = []    # list of (i, j) index tuples
        self.episode_boundaries = []  # indices where episodes start

        if dagger_transitions:
            self._process_dagger_transitions(dagger_transitions)

    def _process_dagger_transitions(self, transitions: List[DaggerTransition]):
        """Convert DAgger transitions into positive/negative edge pairs."""
        start_idx = len(self.dagger_images)

        # Add images from transitions
        for t in transitions:
            self.dagger_images.append(t.curr_image)
        # Add last next_image
        if transitions:
            self.dagger_images.append(transitions[-1].next_image)

        n_dagger = len(self.dagger_images) - start_idx

        # Track episode boundaries for negative pair generation
        # Episodes are separated by transitions where prev_image is None-like
        # For simplicity, treat all transitions from one collection round as one episode
        self.episode_boundaries.append((start_idx, start_idx + n_dagger))

        # Generate positive pairs: within MAX_ACTION_DISTANCE steps
        for i in range(start_idx, start_idx + n_dagger):
            for offset in range(1, min(MAX_ACTION_DISTANCE, n_dagger - (i - start_idx))):
                j = i + offset
                self.positive_pairs.append((i, j))

        # Generate negative pairs: distant frames or from different episodes
        for i in range(start_idx, start_idx + n_dagger):
            for _ in range(NEGATIVE_SAMPLE_MULTIPLIER):
                # Pick a frame far away or from a different episode
                j = random.randint(0, len(self.dagger_images) - 1)
                if abs(i - j) < NEGATIVE_SAMPLE_MULTIPLIER * MAX_ACTION_DISTANCE:
                    # Too close — pick from a different part or different episode
                    if len(self.episode_boundaries) > 1:
                        # Pick from a different episode
                        other_ep = random.choice(self.episode_boundaries)
                        j = random.randint(other_ep[0], other_ep[1] - 1)
                    else:
                        # Pick far away in same sequence
                        j = random.randint(0, len(self.dagger_images) - 1)
                self.negative_pairs.append((i, j))

    def __len__(self):
        offline_len = len(self.offline_dataset)
        dagger_len = len(self.positive_pairs) + len(self.negative_pairs)
        return offline_len + dagger_len

    def __getitem__(self, idx):
        if random.random() < self.offline_sample_ratio or len(self.dagger_images) == 0:
            return self.offline_dataset[idx % len(self.offline_dataset)]

        # Sample from DAgger data
        if random.random() < 0.5 and self.positive_pairs:
            i, j = random.choice(self.positive_pairs)
            label = 1.0
        elif self.negative_pairs:
            i, j = random.choice(self.negative_pairs)
            label = 0.0
        else:
            return self.offline_dataset[idx % len(self.offline_dataset)]

        img_i = self.dagger_images[i].copy()
        img_j = self.dagger_images[j].copy()

        # Apply domain randomization
        if self.augment is not None:
            if random.random() < self.augmentation_prob:
                img_i = self.augment(img_i)
            if random.random() < self.augmentation_prob:
                img_j = self.augment(img_j)

        combined = np.concatenate([img_i, img_j], axis=2)
        combined = np.transpose(combined, (2, 0, 1))

        return (torch.from_numpy(combined),
                torch.tensor(label, dtype=torch.long))


class CombinedActionDataset(Dataset):
    """Samples action triplets from offline map data + DAgger transitions."""

    def __init__(self, map_dir, dagger_transitions=None,
                 offline_sample_ratio=0.5,
                 augment=True, augmentation_prob=0.5):
        self.offline_dataset = ActionDataset(map_dir, augment=augment,
                                              augmentation_prob=augmentation_prob)
        self.dagger_transitions = dagger_transitions or []
        self.offline_sample_ratio = offline_sample_ratio
        self.augment = DomainRandomization() if augment else None
        self.augmentation_prob = augmentation_prob

    def __len__(self):
        return len(self.offline_dataset) + len(self.dagger_transitions)

    def __getitem__(self, idx):
        if random.random() < self.offline_sample_ratio or len(self.dagger_transitions) == 0:
            return self.offline_dataset[idx % len(self.offline_dataset)]

        # Sample from DAgger data
        t = random.choice(self.dagger_transitions)

        img_prev = t.prev_image.copy()
        img_curr = t.curr_image.copy()
        img_goal = t.goal_image.copy()

        # Apply domain randomization
        if self.augment is not None:
            if random.random() < self.augmentation_prob:
                img_prev = self.augment(img_prev)
            if random.random() < self.augmentation_prob:
                img_curr = self.augment(img_curr)
            if random.random() < self.augmentation_prob:
                img_goal = self.augment(img_goal)

        combined = np.concatenate([img_prev, img_curr, img_goal], axis=2)
        combined = np.transpose(combined, (2, 0, 1))

        return (torch.from_numpy(combined),
                torch.tensor(t.expert_action, dtype=torch.long))


# ============================================================================
# FINE-TUNING FUNCTIONS
# ============================================================================

def fine_tune_edge_predictor(model, map_dir, output_dir, dagger_transitions,
                             epochs=500, batch_size=BATCH_SIZE, lr=1e-5,
                             offline_sample_ratio=0.5,
                             early_stopping=True, es_patience=10,
                             es_check_interval=10, es_threshold=1e-4,
                             augment=True, augmentation_prob=0.5):
    """Fine-tune the edge predictor on combined offline + DAgger data."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    dataset = CombinedEdgeDataset(map_dir, dagger_transitions,
                                  offline_sample_ratio=offline_sample_ratio,
                                  augment=augment, augmentation_prob=augmentation_prob)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                       num_workers=4, pin_memory=True)

    print(f"\nFine-tuning R-network (edge predictor)")
    print(f"  Combined dataset size: {len(dataset)} (offline={len(dataset.offline_dataset)}, "
          f"dagger_pairs={len(dataset.positive_pairs)+len(dataset.negative_pairs)})")
    print(f"  Device: {device}, Epochs: {epochs}, LR: {lr}")

    best_loss = float('inf')
    zero_loss_count = 0
    losses, accuracies = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        avg_loss = epoch_loss / total
        accuracy = correct / total
        losses.append(avg_loss)
        accuracies.append(accuracy)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                        'loss': avg_loss, 'accuracy': accuracy},
                       os.path.join(output_dir, 'edge_model_best.pt'))

        if early_stopping and (epoch + 1) % es_check_interval == 0:
            if avg_loss < es_threshold:
                zero_loss_count += 1
                if zero_loss_count >= es_patience:
                    print(f"  Early stopping at epoch {epoch+1}: loss={avg_loss:.6f}")
                    break
            else:
                zero_loss_count = 0

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, acc={accuracy:.3f}")

    torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                'loss': losses[-1], 'accuracy': accuracies[-1]},
               os.path.join(output_dir, 'edge_model_dagger.pt'))
    return model


def fine_tune_action_predictor(model, map_dir, output_dir, dagger_transitions,
                               epochs=1000, batch_size=BATCH_SIZE, lr=1e-5,
                               offline_sample_ratio=0.5,
                               early_stopping=True, es_patience=10,
                               es_check_interval=10, es_threshold=1e-4,
                               augment=True, augmentation_prob=0.5):
    """Fine-tune the action predictor on combined offline + DAgger data."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    dataset = CombinedActionDataset(map_dir, dagger_transitions,
                                     offline_sample_ratio=offline_sample_ratio,
                                     augment=augment, augmentation_prob=augmentation_prob)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                       num_workers=4, pin_memory=True)

    print(f"\nFine-tuning L-network (action predictor)")
    print(f"  Combined dataset size: {len(dataset)} (offline={len(dataset.offline_dataset)}, "
          f"dagger={len(dataset.dagger_transitions)})")
    print(f"  Device: {device}, Epochs: {epochs}, LR: {lr}")

    best_loss = float('inf')
    zero_loss_count = 0
    losses, accuracies = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        avg_loss = epoch_loss / total
        accuracy = correct / total
        losses.append(avg_loss)
        accuracies.append(accuracy)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                        'loss': avg_loss, 'accuracy': accuracy},
                       os.path.join(output_dir, 'action_model_best.pt'))

        if early_stopping and (epoch + 1) % es_check_interval == 0:
            if avg_loss < es_threshold:
                zero_loss_count += 1
                if zero_loss_count >= es_patience:
                    print(f"  Early stopping at epoch {epoch+1}: loss={avg_loss:.6f}")
                    break
            else:
                zero_loss_count = 0

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, acc={accuracy:.3f}")

    torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                'loss': losses[-1], 'accuracy': accuracies[-1]},
               os.path.join(output_dir, 'action_model_dagger.pt'))
    return model


# ============================================================================
# DAGGER DATA COLLECTOR
# ============================================================================

class DaggerDataCollector:
    """Collects transitions by running mixed expert/policy in CARLA."""

    def __init__(self, env, oracle, action_mapper, nav, max_steps=500):
        self.env = env
        self.oracle = oracle
        self.action_mapper = action_mapper
        self.nav = nav
        self.max_steps = max_steps

    def collect_round(self, policy_fn, beta, num_transitions,
                      trajectory_dirs=None):
        """
        Collect transitions using DAgger mixing.

        Args:
            policy_fn: callable(obs_img_84x84) -> discrete action index, or None
            beta: probability of using expert action for execution (1.0 = all expert)
            num_transitions: number of transitions to collect
            trajectory_dirs: list of trajectory directories with aux.pkl for start/goal

        Returns:
            List of DaggerTransition objects
        """
        transitions = []
        episode = 0

        while len(transitions) < num_transitions:
            episode += 1
            print(f"  Episode {episode}, collected {len(transitions)}/{num_transitions} transitions")

            try:
                ep_transitions = self._run_episode(
                    policy_fn, beta, num_transitions - len(transitions),
                    trajectory_dirs)
                transitions.extend(ep_transitions)
            except Exception as e:
                print(f"  Episode {episode} failed: {e}")
                continue

            if episode > num_transitions * 2:
                print(f"  Warning: too many episodes, stopping collection")
                break

        return transitions[:num_transitions]

    def _run_episode(self, policy_fn, beta, max_remaining, trajectory_dirs):
        """Run a single episode and collect transitions."""
        from rlib_integration.helper import ndarray_to_location

        # Select a random trajectory for start/goal
        start, goal = None, None
        if trajectory_dirs and len(trajectory_dirs) > 0:
            traj_dir = random.choice(trajectory_dirs)
            aux_path = os.path.join(traj_dir, 'aux.pkl')
            if os.path.exists(aux_path):
                with open(aux_path, 'rb') as f:
                    aux = pickle.load(f)
                start = aux.get('start')
                goal = aux.get('goal')

        # Reset environment with start/goal
        if start is not None and goal is not None:
            start_location = ndarray_to_location(start)
            goal_location = ndarray_to_location(goal)
            self.env.unwrapped.set_start_transform(start_location)
            self.env.unwrapped.set_destination_transform(goal_location)

        obs, info = self.env.reset()

        # Take a zero step to initialize sensors
        obs, reward, done, truncated, info = self.env.step(np.array([0.0, 0.0]))

        # Set up navigator path for subgoal tracking
        self._current_path = None
        self._current_path_idx = 1
        if goal is not None and self.nav is not None and start is not None:
            try:
                obs_img = self._get_obs_image()
                goal_img = self._get_goal_image_from_env()
                path = self.nav.get_path(
                    start_location=start_location,
                    goal_location=goal_location,
                    obs_img=obs_img,
                    goal_img=goal_img)
                if path is not None and len(path) > 1:
                    self._current_path = path
                    self._current_path_idx = 1
                    print(f"  Navigator path: {len(path)} nodes")
            except Exception as e:
                print(f"  Navigator path setup failed: {e}")

        # Reset oracle
        if goal is not None:
            goal_location = ndarray_to_location(goal)
            self.oracle.set_env(self.env.unwrapped)
            self.oracle.reset()
            self.oracle.set_goal(goal_location)

        prev_img = None
        step_transitions = []
        subgoal_reach_dist = 5.0

        for step in range(self.max_steps):
            if len(step_transitions) >= max_remaining:
                break

            # Get current raw observation image
            curr_img = self._get_obs_image()

            # Get subgoal image from navigator
            subgoal_img = self._get_subgoal_image()

            # DAgger: decide expert vs policy for execution
            if policy_fn is not None and random.random() > beta:
                # Use learned policy
                try:
                    discrete_action = policy_fn(curr_img)
                    exec_control = self.action_mapper.discrete_to_carla_control(discrete_action)
                except Exception:
                    exec_control = np.array([0.0, 0.5])  # fallback: go forward
            else:
                # Use expert
                try:
                    expert_cont = self.oracle.select_action(obs, info)
                    exec_control = np.clip(expert_cont,
                                           self.env.action_space.low,
                                           self.env.action_space.high)
                except Exception:
                    exec_control = np.array([0.0, 0.5])

            # Always get expert label
            try:
                expert_cont = self.oracle.select_action(obs, info)
                expert_discrete = self.action_mapper(expert_cont[0], expert_cont[1])
            except Exception:
                expert_discrete = MOVE_FORWARD

            # Execute action
            obs, reward, done, truncated, info = self.env.step(exec_control)
            next_img = self._get_obs_image()

            # Get hero position and advance subgoal
            position = self._get_hero_position()
            self._advance_subgoal(position, subgoal_reach_dist)

            # Record transition (skip first step since no prev_image)
            if prev_img is not None:
                step_transitions.append(DaggerTransition(
                    prev_image=prev_img,
                    curr_image=curr_img,
                    next_image=next_img,
                    goal_image=subgoal_img,
                    expert_action=expert_discrete,
                    position=position,
                ))

            prev_img = curr_img

            if done or truncated:
                break

        return step_transitions

    def _get_goal_image_from_env(self):
        """Get goal image from the environment's goal sensor."""
        try:
            goal_img = self.env.unwrapped.experiment.goal_img_raw
            if goal_img is not None and goal_img.size > 0:
                if goal_img.dtype == np.uint8:
                    goal_img = goal_img.astype(np.float32) / 255.0
                else:
                    goal_img = goal_img.astype(np.float32)
                if goal_img.shape[:2] != (NET_HEIGHT, NET_WIDTH):
                    goal_img = cv2.resize(goal_img, (NET_WIDTH, NET_HEIGHT))
                return goal_img
        except Exception:
            pass
        return None

    def _get_obs_image(self):
        """Get 84x84 RGB float32 [0,1] image from environment.

        CarlaEnv with STBL3EvalExperiment stores raw image at
        env.unwrapped.experiment.obs_img (84x84x3 uint8 RGB).
        """
        try:
            img = self.env.unwrapped.experiment.obs_img
            if img is not None and img.size > 0:
                if img.dtype == np.uint8:
                    img = img.astype(np.float32) / 255.0
                else:
                    img = img.astype(np.float32)
                if img.shape[:2] != (NET_HEIGHT, NET_WIDTH):
                    img = cv2.resize(img, (NET_WIDTH, NET_HEIGHT))
                return img
        except Exception:
            pass
        return np.zeros((NET_HEIGHT, NET_WIDTH, NET_CHANNELS), dtype=np.float32)

    def _get_subgoal_image(self):
        """Get subgoal image from navigator or fallback to current obs.

        Uses DenseMapNavigator to get the subgoal image for the current path node.
        Falls back to the raw goal image from the environment if nav is unavailable.
        """
        try:
            if self.nav is not None and hasattr(self.nav, 'get_subgoal_image'):
                path = getattr(self, '_current_path', None)
                path_idx = getattr(self, '_current_path_idx', 1)
                if path is not None and path_idx < len(path):
                    raw_img = self.nav.get_subgoal_image(path[path_idx])
                    if raw_img is not None and raw_img.size > 0:
                        img = cv2.resize(raw_img, (NET_WIDTH, NET_HEIGHT))
                        if img.dtype == np.uint8:
                            img = img.astype(np.float32) / 255.0
                        return img
        except Exception:
            pass
        # Fallback: use goal image from environment
        try:
            goal_img = self.env.unwrapped.experiment.goal_img_raw
            if goal_img is not None and goal_img.size > 0:
                if goal_img.dtype == np.uint8:
                    goal_img = goal_img.astype(np.float32) / 255.0
                else:
                    goal_img = goal_img.astype(np.float32)
                if goal_img.shape[:2] != (NET_HEIGHT, NET_WIDTH):
                    goal_img = cv2.resize(goal_img, (NET_WIDTH, NET_HEIGHT))
                return goal_img
        except Exception:
            pass
        # Last fallback: use current obs as goal
        return self._get_obs_image()

    def _get_hero_position(self):
        """Get hero (x, y) position from environment."""
        try:
            hero = self.env.unwrapped.core.hero
            transform = hero.get_transform()
            return np.array([transform.location.x, transform.location.y], dtype=np.float32)
        except Exception:
            return np.array([0.0, 0.0], dtype=np.float32)

    def _advance_subgoal(self, hero_pos, reach_dist=5.0):
        """Advance subgoal index when hero is close enough to current subgoal."""
        if self._current_path is None:
            return
        if self._current_path_idx >= len(self._current_path) - 1:
            return
        try:
            node_pos = np.array(self.nav.get_node_position(
                self._current_path[self._current_path_idx]))
            dist = np.linalg.norm(hero_pos - node_pos)
            if dist < reach_dist:
                self._current_path_idx = min(
                    self._current_path_idx + 1, len(self._current_path) - 1)
        except Exception:
            pass


# ============================================================================
# DAGGER TRAINER
# ============================================================================

class DaggerTrainer:
    """Orchestrates the DAgger loop: collect -> aggregate -> retrain."""

    def __init__(self, args):
        self.args = args
        self.map_dir = args.map_dir
        self.output_dir = args.output_dir
        self.dagger_rounds = args.dagger_rounds
        self.transitions_per_round = args.dagger_transitions
        self.beta_decay = args.dagger_beta_decay
        self.beta_min = args.dagger_beta_min
        self.fine_tune_lr = args.fine_tune_lr
        self.fine_tune_epochs_edge = args.fine_tune_epochs_edge
        self.fine_tune_epochs_action = args.fine_tune_epochs_action
        self.offline_sample_ratio = args.offline_sample_ratio
        self.carla_port = args.carla_port
        self.town = args.town

        self.action_mapper = ContinuousToDiscreteActionMapper()
        self.all_dagger_transitions = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Models (loaded later)
        self.edge_model = None
        self.action_model = None

    def run(self):
        """Run the full DAgger training pipeline."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Phase 0: Load or train models
        self._load_or_train_models()

        # Phase 1: Create CARLA environment and collector
        env = self._create_carla_env()
        oracle = self._create_oracle(env)
        nav = self._create_navigator()
        collector = DaggerDataCollector(env, oracle, self.action_mapper, nav)

        # Load trajectory directories for episode management
        trajectory_dirs = self._load_trajectory_dirs()

        # DAgger rounds
        for round_idx in range(self.dagger_rounds):
            beta = max(self.beta_min, 1.0 * (self.beta_decay ** round_idx))

            print(f"\n{'='*60}")
            print(f"DAgger Round {round_idx + 1}/{self.dagger_rounds}")
            print(f"  Beta: {beta:.3f}")
            print(f"  Total DAgger transitions: {len(self.all_dagger_transitions)}")
            print(f"{'='*60}")

            # Step 1: Collect data
            policy_fn = self._make_policy_fn()
            new_transitions = collector.collect_round(
                policy_fn=policy_fn,
                beta=beta,
                num_transitions=self.transitions_per_round,
                trajectory_dirs=trajectory_dirs,
            )
            print(f"  Collected {len(new_transitions)} transitions")

            # Step 2: Aggregate
            self.all_dagger_transitions.extend(new_transitions)
            self._save_dagger_data(round_idx)

            # Step 3: Retrain
            self.edge_model = fine_tune_edge_predictor(
                self.edge_model, self.map_dir, self.output_dir,
                self.all_dagger_transitions,
                epochs=self.fine_tune_epochs_edge,
                lr=self.fine_tune_lr,
                offline_sample_ratio=self.offline_sample_ratio,
            )

            self.action_model = fine_tune_action_predictor(
                self.action_model, self.map_dir, self.output_dir,
                self.all_dagger_transitions,
                epochs=self.fine_tune_epochs_action,
                lr=self.fine_tune_lr,
                offline_sample_ratio=self.offline_sample_ratio,
            )

            # Step 4: Periodic evaluation
            if (round_idx + 1) % 2 == 0:
                self._evaluate(env, nav, trajectory_dirs)

        print(f"\nDAgger training complete! Models saved to {self.output_dir}")
        print(f"Total DAgger transitions collected: {len(self.all_dagger_transitions)}")

    def _load_or_train_models(self):
        """Load pretrained models or train from scratch."""
        if self.args.pretrained_edge_model:
            print(f"Loading pretrained edge model from {self.args.pretrained_edge_model}")
            self.edge_model = SiameseEdgePredictor().to(self.device)
            ckpt = torch.load(self.args.pretrained_edge_model, map_location=self.device)
            self.edge_model.load_state_dict(ckpt['model_state_dict'])
        else:
            print("Training edge predictor from scratch on offline data...")
            from sptm_train import train_edge_predictor
            self.edge_model = train_edge_predictor(
                self.map_dir, self.output_dir,
                epochs=self.args.epochs_edge if hasattr(self.args, 'epochs_edge') else 10000,
            )

        if self.args.pretrained_action_model:
            print(f"Loading pretrained action model from {self.args.pretrained_action_model}")
            self.action_model = ActionPredictor().to(self.device)
            ckpt = torch.load(self.args.pretrained_action_model, map_location=self.device)
            self.action_model.load_state_dict(ckpt['model_state_dict'])
        else:
            print("Training action predictor from scratch on offline data...")
            from sptm_train import train_action_predictor
            self.action_model = train_action_predictor(
                self.map_dir, self.output_dir,
                epochs=self.args.epochs_action if hasattr(self.args, 'epochs_action') else 19000,
            )

    def _create_carla_env(self):
        """Create CARLA environment for data collection.

        Uses CarlaEnv with STBL3EvalExperiment (same as evaluation script)
        which handles Waypoint-based start/destination correctly.
        Raw 84x84 images are accessible via env.unwrapped.experiment.obs_img.
        """
        from rlib_integration.carla_env import CarlaEnv
        from src.configs.eval_env_config_image_goal import config as ig_config
        from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics
        from corl_2026_evaluate_image_goal import MAX_EPISODE_STEPS

        ig_config["env_config"]["carla"]["town"] = self.town
        ig_config["env_config"]["carla"]["start_server"] = False
        ig_config["env_config"]["carla"]["port"] = self.carla_port

        env = CarlaEnv(config=ig_config)
        env.experiment.eval_mode = True
        env.experiment.map_mode = False
        env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
        env = RecordEpisodeStatistics(env)
        return env

    def _create_oracle(self, env):
        """Create oracle agent for expert actions."""
        from nomad_sac_finetune.oracle_agent import OracleAgentWrapper
        oracle = OracleAgentWrapper(env=env, target_speed=5.0, use_basic_agent=True)
        return oracle

    def _create_navigator(self):
        """Create navigator for subgoal images."""
        # Import DenseMapNavigator from evaluation script
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from corl_2026_evaluate_image_goal import DenseMapNavigator
            nav = DenseMapNavigator(self.map_dir, localization_mode='gps')
            return nav
        except ImportError as e:
            print(f"Warning: Could not import DenseMapNavigator: {e}")
            print("Subgoal images will fall back to current observation")
            return None

    def _load_trajectory_dirs(self):
        """Load trajectory directories for episode management."""
        traj_dir = self.args.benchmark_dir
        if not os.path.exists(traj_dir):
            print(f"Warning: trajectory dir {traj_dir} not found")
            return []

        dirs = []
        for name in sorted(os.listdir(traj_dir)):
            path = os.path.join(traj_dir, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, 'aux.pkl')):
                dirs.append(path)
        print(f"Loaded {len(dirs)} trajectory directories from {traj_dir}")
        return dirs

    def _make_policy_fn(self):
        """Create a function that takes obs_img -> discrete action using current models."""
        edge_path = os.path.join(self.output_dir, 'edge_model_best.pt')
        action_path = os.path.join(self.output_dir, 'action_model_best.pt')

        if not os.path.exists(edge_path) or not os.path.exists(action_path):
            return None  # No policy yet, use pure expert

        try:
            navigator = SPTMNavigator(
                edge_model_path=edge_path,
                action_model_path=action_path,
                map_dir=self.map_dir,
            )
            navigator.build_graph()
        except Exception as e:
            print(f"Warning: Could not build SPTM navigator: {e}")
            return None

        def policy_fn(obs_img):
            """obs_img is (84,84,3) float32 [0,1]. Returns discrete action index."""
            try:
                # Find nearest node in SPTM graph
                nn_idx = navigator.find_nn(obs_img)
                # Predict action
                prev_frame = getattr(policy_fn, '_prev_frame', obs_img)
                action_probs = navigator.predict_action(prev_frame, obs_img,
                                                         navigator.keyframes[nn_idx % len(navigator.keyframes)])
                policy_fn._prev_frame = obs_img
                return int(np.argmax(action_probs))
            except Exception:
                return MOVE_FORWARD  # fallback

        return policy_fn

    def _save_dagger_data(self, round_idx):
        """Persist collected DAgger data to disk."""
        data_path = os.path.join(self.output_dir, f'dagger_round_{round_idx}_data.npz')
        n = len(self.all_dagger_transitions)
        prev_imgs = np.stack([t.prev_image for t in self.all_dagger_transitions])
        curr_imgs = np.stack([t.curr_image for t in self.all_dagger_transitions])
        next_imgs = np.stack([t.next_image for t in self.all_dagger_transitions])
        goal_imgs = np.stack([t.goal_image for t in self.all_dagger_transitions])
        actions = np.array([t.expert_action for t in self.all_dagger_transitions], dtype=np.int64)
        positions = np.stack([t.position for t in self.all_dagger_transitions])

        np.savez_compressed(data_path,
                            prev_images=prev_imgs,
                            curr_images=curr_imgs,
                            next_images=next_imgs,
                            goal_images=goal_imgs,
                            expert_actions=actions,
                            positions=positions)
        print(f"  Saved DAgger data: {data_path} ({n} transitions)")

    def _evaluate(self, env, nav, trajectory_dirs):
        """Quick evaluation on a few trajectories."""
        if not trajectory_dirs:
            print("  No trajectories for evaluation, skipping")
            return

        print("  Running evaluation...")
        # Use SPTMNavigator for evaluation
        edge_path = os.path.join(self.output_dir, 'edge_model_best.pt')
        action_path = os.path.join(self.output_dir, 'action_model_best.pt')

        if not os.path.exists(edge_path) or not os.path.exists(action_path):
            print("  No models for evaluation, skipping")
            return

        # Just report model existence and loss for now
        edge_ckpt = torch.load(edge_path, map_location='cpu')
        action_ckpt = torch.load(action_path, map_location='cpu')
        print(f"  Edge model: epoch={edge_ckpt.get('epoch', '?')}, "
              f"loss={edge_ckpt.get('loss', '?'):.4f}")
        print(f"  Action model: epoch={action_ckpt.get('epoch', '?')}, "
              f"loss={action_ckpt.get('loss', '?'):.4f}")


# ============================================================================
# LOAD DAGGER DATA FROM DISK
# ============================================================================

def load_dagger_data(path):
    """Load previously saved DAgger data from .npz file."""
    data = np.load(path, allow_pickle=True)
    transitions = []
    n = len(data['prev_images'])
    for i in range(n):
        transitions.append(DaggerTransition(
            prev_image=data['prev_images'][i],
            curr_image=data['curr_images'][i],
            next_image=data['next_images'][i],
            goal_image=data['goal_images'][i],
            expert_action=int(data['expert_actions'][i]),
            position=data['positions'][i],
        ))
    return transitions


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SPTM DAgger Training — Real-time collection with Dataset Aggregation")

    # Data and model paths
    parser.add_argument('--map-dir', type=str,
                        default='corl-2026-results-dense-gps/town02_map',
                        help='Directory containing map data')
    parser.add_argument('--output-dir', type=str, default='sptm_dagger_output',
                        help='Directory for model checkpoints and DAgger data')
    parser.add_argument('--benchmark-dir', type=str,
                        default='corl-2026-results-dense-gps/trajectories',
                        help='Directory with benchmark trajectories for collection')
    parser.add_argument('--pretrained-edge-model', type=str, default=None,
                        help='Path to pretrained edge model (skip offline training)')
    parser.add_argument('--pretrained-action-model', type=str, default=None,
                        help='Path to pretrained action model (skip offline training)')

    # DAgger parameters
    parser.add_argument('--dagger-rounds', type=int, default=10,
                        help='Number of DAgger collection rounds')
    parser.add_argument('--dagger-transitions', type=int, default=500,
                        help='Number of transitions to collect per round')
    parser.add_argument('--dagger-beta-decay', type=float, default=0.85,
                        help='Exponential decay for beta (expert probability)')
    parser.add_argument('--dagger-beta-min', type=float, default=0.3,
                        help='Minimum beta value')

    # Fine-tuning parameters
    parser.add_argument('--fine-tune-lr', type=float, default=1e-5,
                        help='Learning rate for fine-tuning (10x lower than initial)')
    parser.add_argument('--fine-tune-epochs-edge', type=int, default=500,
                        help='Edge predictor fine-tune epochs per round')
    parser.add_argument('--fine-tune-epochs-action', type=int, default=1000,
                        help='Action predictor fine-tune epochs per round')
    parser.add_argument('--offline-sample-ratio', type=float, default=0.5,
                        help='Fraction of offline data in combined dataset')

    # Offline training parameters (if no pretrained models)
    parser.add_argument('--epochs-edge', type=int, default=10000,
                        help='Edge predictor training epochs (from scratch)')
    parser.add_argument('--epochs-action', type=int, default=19000,
                        help='Action predictor training epochs (from scratch)')

    # CARLA parameters
    parser.add_argument('--carla-port', type=int, default=2000,
                        help='CARLA server port')
    parser.add_argument('--town', type=str, default='Town02',
                        help='CARLA town for data collection')

    # Resume from saved DAgger data
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume from saved DAgger data (.npz file)')

    # Augmentation
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Disable domain randomization')
    parser.add_argument('--augmentation-prob', type=float, default=0.5,
                        help='Probability of applying augmentation to each image')

    args = parser.parse_args()

    trainer = DaggerTrainer(args)

    # Resume from saved data if specified
    if args.resume_from:
        print(f"Resuming from saved DAgger data: {args.resume_from}")
        trainer.all_dagger_transitions = load_dagger_data(args.resume_from)
        print(f"Loaded {len(trainer.all_dagger_transitions)} existing transitions")

    trainer.run()


if __name__ == "__main__":
    main()