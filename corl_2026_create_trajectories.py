#!/usr/bin/env python
"""
CoRL 2026: Generate 20 reproducible benchmark trajectories for Town02.

Creates start/goal pairs across Town02, computes shortest-path distances,
and copies corresponding images from the map for each trajectory.

Usage:
    python corl_2026_create_trajectories.py --town Town02 --map-dir corl-2026-results/town02_map --output-dir corl-2026-results/trajectories --num-trajectories 20 --seed 42
"""
import glob
import os
import sys
import json
import pickle
import argparse
import random

import carla
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

from rlib_integration.agent import GlobalRoutePlanner


def generate_trajectory_pairs(wmap, num_trajectories, seed=42):
    """Generate start/goal pairs across Town02 with varied distances and difficulty."""
    rng = random.Random(seed)
    topology = wmap.get_topology()
    waypoints = [wp for edge in topology for wp in edge if wp.lane_type == carla.LaneType.Driving]

    # Get spawn points for start locations (well-distributed)
    spawn_points = wmap.get_spawn_points()
    if not spawn_points:
        spawn_points = [wp.transform for wp in waypoints[:50]]

    # Filter to good driving waypoints
    valid_starts = []
    for sp in spawn_points:
        wp = wmap.get_waypoint(sp.location, project_to_road=True)
        if wp and wp.lane_type == carla.LaneType.Driving:
            valid_starts.append(sp.location)

    # Also add waypoints from topology
    for edge in topology:
        for wp in edge:
            if wp.lane_type == carla.LaneType.Driving:
                valid_starts.append(wp.transform.location)

    # Remove duplicates
    seen = set()
    unique_starts = []
    for loc in valid_starts:
        key = (round(loc.x, 1), round(loc.y, 1))
        if key not in seen:
            seen.add(key)
            unique_starts.append(loc)

    rng.shuffle(unique_starts)

    # Generate pairs with varied distances
    pairs = []
    min_distance = 30.0
    max_distance = 300.0
    target_distances = np.linspace(40, 250, num_trajectories)
    rng.shuffle(target_distances)

    planner = GlobalRoutePlanner(wmap, 2.0)

    for i in range(min(num_trajectories, len(unique_starts))):
        start_loc = unique_starts[i % len(unique_starts)]

        # Find a goal at approximately target distance
        best_goal = None
        best_dist_diff = float('inf')
        target_dist = target_distances[i]

        # Sample candidate goals
        candidate_goals = rng.sample(
            unique_starts, min(50, len(unique_starts))
        )
        for goal_loc in candidate_goals:
            eu_dist = start_loc.distance(goal_loc)
            if eu_dist < min_distance or eu_dist > max_distance * 1.5:
                continue
            dist_diff = abs(eu_dist - target_dist)
            if dist_diff < best_dist_diff:
                best_dist_diff = dist_diff
                best_goal = goal_loc

        if best_goal is None:
            continue

        # Compute shortest path distance
        try:
            route = planner.trace_route(start_loc, best_goal)
            route_dist = 0.0
            prev_wp = None
            for wp, _ in route:
                if prev_wp is not None:
                    route_dist += prev_wp.transform.location.distance(wp.transform.location)
                prev_wp = wp
            if route_dist <= 2.0:
                route_dist = start_loc.distance(best_goal)
        except Exception:
            route_dist = start_loc.distance(best_goal)

        pairs.append({
            'trajectory_id': i,
            'start': np.array([start_loc.x, start_loc.y, start_loc.z]),
            'goal': np.array([best_goal.x, best_goal.y, best_goal.z]),
            'route_length': float(route_dist),
        })

    return pairs


def create_trajectory_from_map(trajectory_info, map_data, map_dir, output_dir, wmap=None):
    """Extract trajectory images from the pre-built map using actual route waypoints."""
    traj_id = trajectory_info['trajectory_id']
    start = trajectory_info['start']
    goal = trajectory_info['goal']

    traj_dir = os.path.join(output_dir, f"trajectory_{traj_id:02d}")
    os.makedirs(traj_dir, exist_ok=True)

    # Load map positions
    positions = map_data['metadata']['images']  # list of dicts with x,y,z,yaw,filename
    map_positions = np.array([[p['x'], p['y']] for p in positions])

    # Find start and goal nodes (nearest by position)
    start_pos = np.array([start[0], start[1]])
    goal_pos = np.array([goal[0], goal[1]])
    start_idx = int(np.argmin(np.linalg.norm(map_positions - start_pos, axis=1)))
    goal_idx = int(np.argmin(np.linalg.norm(map_positions - goal_pos, axis=1)))

    # If we have CARLA map, use GlobalRoutePlanner to get actual route waypoints
    if wmap is not None:
        from rlib_integration.agent import GlobalRoutePlanner
        planner = GlobalRoutePlanner(wmap, 2.0)
        try:
            route = planner.trace_route(
                carla.Location(x=start[0], y=start[1], z=start[2]),
                carla.Location(x=goal[0], y=goal[1], z=goal[2])
            )
            # Extract route waypoints at 2m intervals
            route_positions = []
            last_pos = None
            for wp, _ in route:
                pos = np.array([wp.transform.location.x, wp.transform.location.y])
                if last_pos is None or np.linalg.norm(pos - last_pos) >= 2.0:
                    route_positions.append(pos)
                    last_pos = pos
            route_positions = np.array(route_positions)

            # Find nearest map node for each route waypoint
            if len(route_positions) > 0:
                sampled_indices = []
                seen = set()
                for rp in route_positions:
                    idx = int(np.argmin(np.linalg.norm(map_positions - rp, axis=1)))
                    if idx not in seen:
                        sampled_indices.append(idx)
                        seen.add(idx)
                # Ensure goal is included
                if goal_idx not in seen:
                    sampled_indices.append(goal_idx)
                path_indices = sampled_indices
            else:
                path_indices = [start_idx, goal_idx]
        except Exception:
            # Fallback to simple nearest-neighbor path
            path_indices = [start_idx, goal_idx]
    else:
        # Fallback: just start and goal nodes
        path_indices = [start_idx, goal_idx]

    # Copy images and build locations list
    import shutil
    locations = []
    images_src_dir = os.path.join(map_dir, "images")

    for new_idx, map_idx in enumerate(path_indices):
        src_filename = positions[map_idx]['filename']
        src_path = os.path.join(images_src_dir, src_filename)
        dst_path = os.path.join(traj_dir, f"{new_idx}.jpg")

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            locations.append([
                positions[map_idx]['x'],
                positions[map_idx]['y'],
                positions[map_idx]['z']
            ])
        else:
            print(f"  WARNING: Image {src_path} not found, skipping")

    # Save aux.pkl (NoMaD/GNM/ViNT compatibility)
    if len(locations) > 1:
        loc_array = np.array(locations)
        diffs = np.diff(loc_array[:, :2], axis=0)
        mean_dist = float(np.mean(np.linalg.norm(diffs, axis=1)))
    else:
        mean_dist = 2.0

    aux_data = {
        'start': start,
        'goal': goal,
        'locations': locations,
        'mean_distance_per_step': mean_dist,
    }
    with open(os.path.join(traj_dir, "aux.pkl"), 'wb') as f:
        pickle.dump(aux_data, f)

    # Save metadata
    metadata = {
        'trajectory_id': traj_id,
        'start': start.tolist(),
        'goal': goal.tolist(),
        'route_length': trajectory_info['route_length'],
        'num_images': len(locations),
        'path': traj_dir,
    }
    with open(os.path.join(traj_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="CoRL 2026: Create 20 benchmark trajectories")
    parser.add_argument('--town', type=str, default='Town02')
    parser.add_argument('--map-dir', type=str, default='corl-2026-results/town02_map')
    parser.add_argument('--output-dir', type=str, default='corl-2026-results/trajectories')
    parser.add_argument('--num-trajectories', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load map data
    map_data_path = os.path.join(args.map_dir, "map_data.pkl")
    if not os.path.exists(map_data_path):
        print(f"ERROR: Map data not found at {map_data_path}")
        print("Run corl_2026_map_town02.py first.")
        return

    with open(map_data_path, 'rb') as f:
        map_data = pickle.load(f)

    print(f"Loaded map data: {len(map_data['metadata']['images'])} images")

    # Connect to CARLA for route planning
    print("Connecting to CARLA for route planning...")
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.load_world(args.town)
    wmap = world.get_map()

    # Generate trajectory pairs
    print(f"\nGenerating {args.num_trajectories} trajectories...")
    pairs = generate_trajectory_pairs(wmap, args.num_trajectories, args.seed)
    print(f"  Generated {len(pairs)} trajectory pairs")

    # Create trajectory directories from map
    print("\nCreating trajectory directories from map...")
    trajectories = []
    for pair in pairs:
        metadata = create_trajectory_from_map(pair, map_data, args.map_dir, args.output_dir, wmap)
        trajectories.append(metadata)
        print(f"  Trajectory {pair['trajectory_id']:02d}: "
              f"{pair['route_length']:.1f}m, {metadata['num_images']} images")

    # Save benchmark info
    benchmark_info = {
        'town': args.town,
        'num_trajectories': len(trajectories),
        'seed': args.seed,
        'trajectories': trajectories,
    }

    with open(os.path.join(args.output_dir, "benchmark_info.json"), 'w') as f:
        json.dump(benchmark_info, f, indent=2)

    with open(os.path.join(args.output_dir, "benchmark_info.pkl"), 'wb') as f:
        pickle.dump(benchmark_info, f)

    print(f"\nBenchmark info saved to {args.output_dir}")
    print(f"  Trajectories: {len(trajectories)}")
    print(f"  Distances: {min(t['route_length'] for t in trajectories):.1f}m - "
          f"{max(t['route_length'] for t in trajectories):.1f}m")


if __name__ == "__main__":
    main()