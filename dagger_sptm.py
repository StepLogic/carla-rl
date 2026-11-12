#!/usr/bin/env python3
"""DAgger (Dataset Aggregation) training for SPTM action model in CARLA.

Iteratively collects live data, retrains, evaluates periodically.

Usage:
    # 200 iterations, evaluate every 10, 2 episodes per iter
    python dagger_sptm.py --town town02 --dagger-iterations 200 --eval-every 10 \
        --episodes-per-iter 2 --epochs-per-iter 20
"""

import argparse
import csv
import json
import os
import pickle
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms as T

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from sptm_train import (
    SPTMNavigator, CARLAActionPredictor,
    ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT,
    CARLA_DISCRETE_ACTIONS,
    ActionDataset, MOBILENET_FEATURE_DIM,
    NET_HEIGHT, NET_WIDTH, NET_CHANNELS,
    BATCH_SIZE, LEARNING_RATE, ACTION_MAX_EPOCHS,
    MODEL_CHECKPOINT_PERIOD,
)

TRAJ_DIR = os.path.join(PROJECT_DIR, "corl-2026-experiments", "trajectories")
SPTM_MAPS_DIR = os.path.join(PROJECT_DIR, "corl-2026-experiments", "sptm_sparse_maps")
DEFAULT_OUTPUT = os.path.join(PROJECT_DIR, "sptm_output_carla")
MAX_STEPS = 500
CARLA_PORT = 2000
TOWN_NAMES = {"town02": "Town02", "town05": "Town05"}


# ============================================================================
# EXPERT POLICY — Route Planner
# ============================================================================

class ExpertPolicy:
    """CARLA route-planner expert that computes the correct discrete action."""

    def __init__(self, carla_map, sampling_resolution=2.0):
        from rlib_integration.agent import GlobalRoutePlanner
        self.planner = GlobalRoutePlanner(carla_map, sampling_resolution)

    def get_expert_action(self, current_location, goal_location, current_yaw_deg):
        """Return discrete expert action (0=straight, 1=left, 2=right)."""
        try:
            route = self.planner.trace_route(current_location, goal_location)
            if len(route) > 1:
                next_wp = route[1][0]
            else:
                next_wp = route[0][0]
            target = next_wp.transform.location
        except Exception:
            target = goal_location

        dx = target.x - current_location.x
        dy = target.y - current_location.y
        heading_to_target = np.arctan2(dy, dx)

        current_yaw_rad = np.deg2rad(current_yaw_deg)
        rel_heading = heading_to_target - current_yaw_rad
        rel_heading = (rel_heading + np.pi) % (2 * np.pi) - np.pi

        if rel_heading < -np.pi / 6:
            return ACTION_LEFT
        elif rel_heading > np.pi / 6:
            return ACTION_RIGHT
        return ACTION_STRAIGHT


# ============================================================================
# LIVE DATASET
# ============================================================================

class LiveDAggerDataset(Dataset):
    def __init__(self, data_list, augment_prob=0.0):
        self.samples = data_list
        self.augment_prob = augment_prob
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        obs = torch.from_numpy(np.transpose(s['obs_img'], (2, 0, 1))).float()
        goal = torch.from_numpy(np.transpose(s['goal_img'], (2, 0, 1))).float()
        obs = self.normalize(obs)
        goal = self.normalize(goal)
        label = torch.tensor(s['action'], dtype=torch.long)
        return obs, goal, label


# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_dagger_episode(env, nav, expert, start_location, goal_location,
                           max_steps=MAX_STEPS, save_expert=True):
    """Roll out one episode. If save_expert=True, save expert labels."""
    unwrapped = env.unwrapped
    unwrapped.set_start_transform(start_location)
    unwrapped.set_destination_transform(goal_location)

    obs, info = env.reset()
    env.step(np.array([0.0, 0.0]))

    _, _, goal_node_idx = nav.set_goal_by_location(goal_location)
    nav.compute_shortest_paths(goal_node_idx)

    collected = []
    traj_positions = []
    done = False
    steps = 0

    while not done and steps < max_steps:
        hero = unwrapped.core.hero
        curr_transform = hero.get_transform()
        curr_loc = curr_transform.location
        curr_yaw = curr_transform.rotation.yaw
        traj_positions.append([curr_loc.x, curr_loc.y])

        # Get current observation image
        obs_img = unwrapped.experiment.obs_img
        if obs_img is not None:
            obs_img = cv2.resize(obs_img, (NET_WIDTH, NET_HEIGHT))
            if obs_img.dtype == np.uint8:
                obs_img = obs_img.astype(np.float32) / 255.0
        else:
            obs_img = np.zeros((NET_HEIGHT, NET_WIDTH, NET_CHANNELS), dtype=np.float32)

        # Find IRG
        try:
            irg_idx = nav.find_intermediate_reachable_goal(obs_img)
            if irg_idx is not None:
                goal_frame = nav.keyframes[irg_idx % len(nav.keyframes)]
            else:
                goal_frame = nav.keyframes[-1]
        except Exception:
            goal_frame = nav.keyframes[-1]

        if save_expert:
            expert_action = expert.get_expert_action(curr_loc, goal_location, curr_yaw)
            collected.append({
                'obs_img': obs_img.copy(),
                'goal_img': goal_frame.copy(),
                'action': int(expert_action),
            })

        # Execute current policy's action
        try:
            policy_action = nav.predict_action(obs_img, goal_frame)
        except Exception:
            policy_action = np.array([0.0, 0.0])

        obs, reward, done, truncated, info = env.step(policy_action)
        steps += 1

        goal_pos = np.array([goal_location.x, goal_location.y])
        hero_pos = np.array([curr_loc.x, curr_loc.y])
        if np.linalg.norm(hero_pos - goal_pos) < 2.0:
            info['is_success'] = True
            done = True

        if done or truncated:
            break

    # Compute distance covered
    distance = 0.0
    if len(traj_positions) > 1:
        for i in range(1, len(traj_positions)):
            dx = traj_positions[i][0] - traj_positions[i-1][0]
            dy = traj_positions[i][1] - traj_positions[i-1][1]
            distance += np.sqrt(dx*dx + dy*dy)

    return collected, steps, info.get('is_success', False), distance


def collect_dagger_data(env, nav, trajectories, expert, num_episodes=10,
                        max_steps=MAX_STEPS, dagger_dir='dagger_data'):
    """Collect live training data across multiple episodes."""
    from rlib_integration.helper import ndarray_to_location

    os.makedirs(dagger_dir, exist_ok=True)
    all_collected = []

    for ep in range(num_episodes):
        traj_path = random.choice(trajectories)
        aux_path = os.path.join(traj_path, 'aux.pkl')
        if not os.path.exists(aux_path):
            continue

        with open(aux_path, 'rb') as f:
            aux = pickle.load(f)

        start = ndarray_to_location(aux['start'])
        goal = ndarray_to_location(aux['goal'])

        collected, steps, success, distance = collect_dagger_episode(
            env, nav, expert, start, goal, max_steps)
        all_collected.extend(collected)

        status = "SUCCESS" if success else "FAIL"
        print(f"    Episode {ep+1}/{num_episodes}: {status} | {steps} steps | "
              f"dist={distance:.1f}m | {len(collected)} samples")

        time.sleep(1)

    iter_num = len([f for f in os.listdir(dagger_dir) if f.startswith('iter_')])
    save_path = os.path.join(dagger_dir, f'iter_{iter_num:03d}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(all_collected, f)
    print(f"  Saved {len(all_collected)} samples to {save_path}")

    return all_collected


# ============================================================================
# EVALUATION — Roll out current policy, record metrics
# ============================================================================

def evaluate_policy(env, nav, trajectories, num_episodes=5, max_steps=MAX_STEPS):
    """Evaluate current policy on random trajectories. Returns metrics dict."""
    from rlib_integration.helper import ndarray_to_location

    distances = []
    steps_list = []
    successes = []

    for ep in range(num_episodes):
        traj_path = random.choice(trajectories)
        aux_path = os.path.join(traj_path, 'aux.pkl')
        if not os.path.exists(aux_path):
            continue

        with open(aux_path, 'rb') as f:
            aux = pickle.load(f)

        start = ndarray_to_location(aux['start'])
        goal = ndarray_to_location(aux['goal'])

        _, steps, success, distance = collect_dagger_episode(
            env, nav, None, start, goal, max_steps, save_expert=False)

        distances.append(distance)
        steps_list.append(steps)
        successes.append(1.0 if success else 0.0)

        status = "SUCCESS" if success else "FAIL"
        print(f"      Eval {ep+1}/{num_episodes}: {status} | {steps} steps | dist={distance:.1f}m")

    metrics = {
        'mean_distance': float(np.mean(distances)) if distances else 0.0,
        'std_distance': float(np.std(distances)) if distances else 0.0,
        'mean_steps': float(np.mean(steps_list)) if steps_list else 0.0,
        'success_rate': float(np.mean(successes)) if successes else 0.0,
        'num_episodes': len(distances),
    }
    return metrics


def save_eval_log(log_path, iteration, metrics):
    """Append evaluation metrics to JSON log."""
    records = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            records = json.load(f)
    records.append({'iteration': iteration, **metrics})
    with open(log_path, 'w') as f:
        json.dump(records, f, indent=2)


def print_eval_summary(log_path):
    """Print evaluation progress from log."""
    if not os.path.exists(log_path):
        return
    with open(log_path, 'r') as f:
        records = json.load(f)
    if not records:
        return
    print(f"\n  {'Iter':>6} {'Dist':>8} {'Success':>8} {'Steps':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    for r in records:
        print(f"  {r['iteration']:>6} {r['mean_distance']:>8.1f} {r['success_rate']:>8.2f} {r['mean_steps']:>8.1f}")


# ============================================================================
# TRAINING
# ============================================================================

def load_live_dagger_data(dagger_dir):
    all_samples = []
    if not os.path.exists(dagger_dir):
        return all_samples
    for fname in sorted(os.listdir(dagger_dir)):
        if fname.startswith('iter_') and fname.endswith('.pkl'):
            with open(os.path.join(dagger_dir, fname), 'rb') as f:
                samples = pickle.load(f)
            all_samples.extend(samples)
    return all_samples


def train_action_model_with_dagger(offline_dirs, dagger_dir, output_dir,
                                    epochs=20, batch_size=64, lr=1e-4,
                                    augment=True, augmentation_prob=0.5):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CARLAActionPredictor().to(device)

    ckpt_path = os.path.join(output_dir, 'action_model_best.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Warm-start from {ckpt_path}")

    optimizer = optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    if isinstance(offline_dirs, str):
        offline_dirs = [offline_dirs]
    offline_datasets = [ActionDataset(d, augment=augment, augmentation_prob=augmentation_prob)
                        for d in offline_dirs]
    offline_dataset = offline_datasets[0] if len(offline_datasets) == 1 else ConcatDataset(offline_datasets)

    live_samples = load_live_dagger_data(dagger_dir)

    if live_samples:
        live_dataset = LiveDAggerDataset(live_samples)
        combined_dataset = ConcatDataset([offline_dataset, live_dataset])
        print(f"\nTraining: {len(offline_dataset)} offline + {len(live_dataset)} live = {len(combined_dataset)} total")
    else:
        combined_dataset = offline_dataset
        print(f"\nTraining: {len(combined_dataset)} offline samples")

    loader = DataLoader(combined_dataset, batch_size=batch_size,
                       shuffle=True, num_workers=0, pin_memory=True)

    losses = []
    accuracies = []
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for img_curr, img_goal, labels in loader:
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

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': accuracy,
            }, os.path.join(output_dir, 'action_model_best.pt'))

    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'loss': losses[-1],
        'accuracy': accuracies[-1],
    }, os.path.join(output_dir, 'action_model_final.pt'))

    print(f"  Final: loss={losses[-1]:.4f}, acc={accuracies[-1]:.3f}")
    return model


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DAgger training for SPTM in CARLA")
    parser.add_argument('--town', type=str, default='town01',
                        choices=['town01', 'town05', 'all'])
    parser.add_argument('--map-dir', type=str, action='append',
                        help='Offline map directories (use multiple times)')
    parser.add_argument('--sptm-map-dir', type=str, default=None,
                        help='SPTM sparse map dir for graph; defaults to --map-dir')
    parser.add_argument('--edge-model', type=str,
                        default=os.path.join(DEFAULT_OUTPUT, 'edge_model_best.pt'))
    parser.add_argument('--action-model', type=str,
                        default=os.path.join(DEFAULT_OUTPUT, 'action_model_best.pt'))
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT)
    parser.add_argument('--dagger-iterations', type=int, default=5)
    parser.add_argument('--episodes-per-iter', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=MAX_STEPS)
    parser.add_argument('--epochs-per-iter', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--carla-port', type=int, default=CARLA_PORT)
    parser.add_argument('--dagger-data-dir', type=str, default='dagger_data')
    parser.add_argument('--eval-every', type=int, default=10,
                        help='Evaluate policy every N iterations')
    parser.add_argument('--eval-episodes', type=int, default=5,
                        help='Number of evaluation episodes per eval round')
    args = parser.parse_args()

    map_dirs = args.map_dir or ['corl-2026-experiments/maps/town02_dense']
    sptm_map_dir = args.sptm_map_dir or map_dirs[0]

    towns = ['town01', 'town05'] if args.town == 'all' else [args.town]
    all_trajectories = []
    for town in towns:
        traj_dir = os.path.join(TRAJ_DIR, town)
        if os.path.exists(traj_dir):
            all_trajectories.extend([
                os.path.join(traj_dir, d)
                for d in os.listdir(traj_dir)
                if os.path.isdir(os.path.join(traj_dir, d)) and d.startswith('trajectory_')
            ])
    if not all_trajectories:
        print("ERROR: No trajectories found.")
        return

    print(f"Found {len(all_trajectories)} trajectories for DAgger rollouts")

    from rlib_integration.carla_env import CarlaEnv
    from src.configs.eval_env_config_image_goal import config as ig_config
    from gymnasium.wrappers import TimeLimit

    ig_config["env_config"]["carla"]["start_server"] = True
    ig_config["env_config"]["carla"]["port"] = args.carla_port

    env = None
    try:
        # Retry CarlaEnv creation until CARLA is ready
        for attempt in range(10):
            try:
                env = CarlaEnv(config=ig_config)
                break
            except RuntimeError as e:
                if "time-out" in str(e).lower() and attempt < 9:
                    print(f"  CARLA not ready, retrying in 5s... ({attempt+1}/10)")
                    time.sleep(5)
                else:
                    raise
        env.experiment.eval_mode = True
        env.experiment.map_mode = False
        env = TimeLimit(env, max_episode_steps=args.max_steps)

        nav = SPTMNavigator(
            edge_model_path=args.edge_model,
            action_model_path=args.action_model,
            map_dir=sptm_map_dir,
        )
        nav.build_graph()
        print(f"SPTM graph: {nav.G.number_of_nodes()} nodes, {nav.G.number_of_edges()} edges")

        carla_map = env.unwrapped.core.map
        expert = ExpertPolicy(carla_map)

        dagger_dir = os.path.join(args.output_dir, args.dagger_data_dir)
        os.makedirs(dagger_dir, exist_ok=True)

        eval_log_path = os.path.join(args.output_dir, 'dagger_eval_log.json')

        # Resume: detect existing iter files and skip completed iterations
        existing_iters = []
        if os.path.exists(dagger_dir):
            for f in os.listdir(dagger_dir):
                if f.startswith('iter_') and f.endswith('.pkl'):
                    try:
                        n = int(f.split('_')[1].split('.')[0])
                        existing_iters.append(n)
                    except ValueError:
                        pass
        resume_from = 0
        if existing_iters:
            resume_from = max(existing_iters) + 1
            print(f"\n*** Resuming from iteration {resume_from} (found {len(existing_iters)} existing iter files) ***")
            # Ensure model is loaded from best checkpoint
            action_ckpt = os.path.join(args.output_dir, 'action_model_best.pt')
            if os.path.exists(action_ckpt):
                nav.action_model = CARLAActionPredictor().to(nav.device)
                ckpt = torch.load(action_ckpt, map_location=nav.device)
                nav.action_model.load_state_dict(ckpt['model_state_dict'])
                nav.action_model.eval()
                print(f"  Loaded action model from {action_ckpt}")

        for iteration in range(resume_from, args.dagger_iterations):
            print(f"\n{'='*60}")
            print(f"DAgger Iteration {iteration + 1}/{args.dagger_iterations}")
            print(f"{'='*60}")

            # 1. Collect live data
            print("Collecting live rollouts...")
            collect_dagger_data(
                env, nav, all_trajectories, expert,
                num_episodes=args.episodes_per_iter,
                max_steps=args.max_steps,
                dagger_dir=dagger_dir)

            # 2. Retrain on aggregated data
            print("Retraining action model on aggregated data...")
            train_action_model_with_dagger(
                map_dirs, dagger_dir, args.output_dir,
                epochs=args.epochs_per_iter,
                batch_size=args.batch_size,
                lr=args.lr)

            # 3. Reload updated model
            action_ckpt = os.path.join(args.output_dir, 'action_model_best.pt')
            if os.path.exists(action_ckpt):
                nav.action_model = CARLAActionPredictor().to(nav.device)
                ckpt = torch.load(action_ckpt, map_location=nav.device)
                nav.action_model.load_state_dict(ckpt['model_state_dict'])
                nav.action_model.eval()
                print(f"  Reloaded action model")

            # 4. Periodic evaluation
            if (iteration + 1) % args.eval_every == 0:
                print(f"\n  === Evaluation after iteration {iteration + 1} ===")
                metrics = evaluate_policy(
                    env, nav, all_trajectories,
                    num_episodes=args.eval_episodes,
                    max_steps=args.max_steps)
                save_eval_log(eval_log_path, iteration + 1, metrics)
                print(f"  Mean distance: {metrics['mean_distance']:.1f}m")
                print(f"  Success rate:  {metrics['success_rate']:.2f}")
                print(f"  Mean steps:    {metrics['mean_steps']:.1f}")
                print_eval_summary(eval_log_path)

        print(f"\n{'='*60}")
        print("DAgger training complete!")
        print(f"Models saved to: {args.output_dir}")
        print(f"Live data saved to: {dagger_dir}")
        print(f"Eval log: {eval_log_path}")
        print(f"{'='*60}")

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
