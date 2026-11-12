#!/usr/bin/env python
"""
CoRL 2026: Map Town02 via waypoint teleportation + build FAISS/VLAD topological graph.

Pipeline:
1. Connect to CARLA, load Town02
2. Teleport vehicle to every road waypoint, capture RGB images
3. Build topological graph using SIFT+VLAD features + FAISS similarity + sequential edges
4. Save everything to corl-2026-results/town02_map/

Usage:
    python corl_2026_map_town02.py --town Town02 --output-dir corl-2026-results/town02_map
"""
import glob
import math
import os
import queue
import sys
import json
import pickle
import argparse
from collections import defaultdict
from datetime import datetime

import carla
import cv2
import numpy as np
import networkx as nx

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major, sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

from eval_graph_builder_vlad import TopologicalGraphBuilder, GraphBuilderConfig, SIFTVLADConfig

# ============================================================================
# CARLA SYNC MODE
# ============================================================================

class CarlaSyncMode:
    """Context manager for synchronous CARLA mode."""

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 30)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout=2.0):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


# ============================================================================
# WAYPOINT EXTRACTION
# ============================================================================

class TownMapBuilder:
    """Extracts ordered waypoints from CARLA town road network."""

    def __init__(self, wmap, sampling_resolution=2.0):
        self.wmap = wmap
        self.sampling_resolution = sampling_resolution

    def get_ordered_waypoints(self):
        """Get waypoints grouped by road segment, one lane per road."""
        all_waypoints = self.wmap.generate_waypoints(self.sampling_resolution)
        road_lanes = defaultdict(list)

        for wp in all_waypoints:
            if wp.lane_type != carla.LaneType.Driving:
                continue
            key = (wp.road_id, wp.section_id, wp.lane_id)
            road_lanes[key].append(wp)

        ordered_segments = []
        for key, waypoints in road_lanes.items():
            waypoints.sort(key=lambda w: w.s)
            road_id, section_id, lane_id = key
            if lane_id < 0:
                ordered_segments.append({
                    'road_id': road_id,
                    'section_id': section_id,
                    'lane_id': lane_id,
                    'waypoints': waypoints
                })

        unique_segments = {}
        for seg in ordered_segments:
            key = (seg['road_id'], seg['section_id'])
            if key not in unique_segments:
                unique_segments[key] = seg
            elif seg['lane_id'] > unique_segments[key]['lane_id']:
                unique_segments[key] = seg

        return list(unique_segments.values())


# ============================================================================
# MAP DATA COLLECTOR
# ============================================================================

class MapDataCollector:
    """Collects RGB images at each waypoint and builds a sequential graph."""

    def __init__(self, save_dir, image_size=(256, 256), capture_interval=0.5, town='Town02'):
        self.save_dir = save_dir
        self.town = town
        self.images_dir = os.path.join(save_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self.image_size = image_size
        self.capture_interval = capture_interval
        self.image_count = 0
        self.image_data = []
        self.graph = nx.DiGraph()
        self.nodes_data = []  # For graph builder input

    def save_image(self, image_raw, waypoint):
        """Save RGB image and record metadata. Returns True if saved."""
        array = np.frombuffer(image_raw.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image_raw.height, image_raw.width, 4))
        array = array[:, :, :3][:, :, ::-1]  # BGRA to RGB

        if self.image_size is not None:
            array = cv2.resize(array, self.image_size)

        filename = f"{self.image_count:06d}.jpg"
        save_path = os.path.join(self.images_dir, filename)
        cv2.imwrite(save_path, cv2.cvtColor(array, cv2.COLOR_RGB2BGR))

        loc = waypoint.transform.location
        rot = waypoint.transform.rotation

        self.image_data.append({
            'index': self.image_count,
            'filename': filename,
            'x': loc.x,
            'y': loc.y,
            'z': loc.z,
            'yaw': rot.yaw,
            'pitch': rot.pitch,
            'road_id': waypoint.road_id,
            'lane_id': waypoint.lane_id,
            'section_id': waypoint.section_id
        })

        # Node for sequential graph
        self.graph.add_node(
            self.image_count,
            pos=(loc.x, loc.y),
            filename=filename,
            road_id=waypoint.road_id
        )

        # Sequential edge to previous image
        if self.image_count > 0:
            prev = self.image_data[-2]
            dist = math.sqrt(
                (loc.x - prev['x'])**2 + (loc.y - prev['y'])**2
            )
            self.graph.add_edge(
                self.image_count - 1, self.image_count,
                distance=dist, type='sequential'
            )

        # Node data for graph builder (locations for debug only)
        self.nodes_data.append({
            'obs_img': array,
            'location': (loc.x, loc.y, loc.z),
            'yaw': rot.yaw,
            'road_id': waypoint.road_id,
            'lane_id': waypoint.lane_id,
            'filename': filename,
            'index': self.image_count,
        })

        self.image_count += 1
        return True

    def save_map_data(self):
        """Save all map data to disk."""
        metadata = {
            'town': self.town,
            'total_images': self.image_count,
            'image_size': list(self.image_size),
            'capture_interval': self.capture_interval,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'images': self.image_data
        }

        # JSON metadata
        json_path = os.path.join(self.save_dir, "map_metadata.json")
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Pickle with graph
        pkl_path = os.path.join(self.save_dir, "map_data.pkl")
        positions = np.array([[d['x'], d['y'], d['yaw']] for d in self.image_data])
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'metadata': metadata,
                'graph': self.graph,
                'positions': positions
            }, f)

        # Topology graph
        graph_path = os.path.join(self.save_dir, "topology_graph.pkl")
        with open(graph_path, 'wb') as f:
            pickle.dump(self.graph, f)

        # Positions numpy
        np.save(os.path.join(self.save_dir, "positions.npy"), positions)

        # aux.pkl for NoMaD/ViNT/GNM compatibility
        aux_data = {
            'positions': positions[:, :2],
            'yaws': positions[:, 2],
            'filenames': [d['filename'] for d in self.image_data],
            'road_ids': [d['road_id'] for d in self.image_data],
            'mean_distance_per_step': self.capture_interval,
        }
        with open(os.path.join(self.save_dir, "aux.pkl"), 'wb') as f:
            pickle.dump(aux_data, f)

        print(f"  Saved {self.image_count} images and metadata")
        return self.image_count


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CoRL 2026: Map Town02 with waypoint teleportation")
    parser.add_argument('--town', type=str, default='Town02')
    parser.add_argument('--output-dir', type=str, default='corl-2026-results/town02_map')
    parser.add_argument('--capture-interval', type=float, default=0.5,
                        help='Min distance (m) between captures')
    parser.add_argument('--sampling-resolution', type=float, default=2.0,
                        help='CARLA waypoint sampling resolution (m)')
    parser.add_argument('--image-size', type=int, nargs=2, default=[256, 256])
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip if map_metadata.json already exists')
    # Graph builder args
    parser.add_argument('--n-clusters', type=int, default=64)
    parser.add_argument('--min-gap', type=int, default=5)
    parser.add_argument('--min-sim', type=float, default=0.1)
    parser.add_argument('--max-dup-sim', type=float, default=0.2)
    parser.add_argument('--no-gpu', action='store_true')
    args = parser.parse_args()

    # Check if already done
    if args.skip_existing and os.path.exists(os.path.join(args.output_dir, "map_metadata.json")):
        print(f"Map already exists at {args.output_dir}, skipping collection.")
        print("Run without --skip-existing to overwrite.")
    else:
        # ==================== STEP 1: COLLECT MAP ====================
        print(f"\n{'='*60}")
        print(f"STEP 1: Collecting map of {args.town}")
        print(f"{'='*60}")

        os.makedirs(args.output_dir, exist_ok=True)
        image_size = tuple(args.image_size)

        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.load_world(args.town)
        wmap = world.get_map()

        print(f"  Town: {wmap.name}")
        map_builder = TownMapBuilder(wmap, args.sampling_resolution)
        road_segments = map_builder.get_ordered_waypoints()
        total_waypoints = sum(len(seg['waypoints']) for seg in road_segments)
        print(f"  Road segments: {len(road_segments)}")
        print(f"  Total waypoints: {total_waypoints}")

        # Spawn vehicle + camera
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.audi.a2')[0]
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(image_size[0]))
        camera_bp.set_attribute('image_size_y', str(image_size[1]))
        camera_bp.set_attribute('fov', '90')

        actor_list = []
        first_wp = road_segments[0]['waypoints'][0]
        spawn_transform = first_wp.transform
        spawn_transform.location.z += 1.0
        vehicle = world.spawn_actor(vehicle_bp, spawn_transform)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)

        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=-10)
        )
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)

        collector = MapDataCollector(
            args.output_dir,
            image_size=image_size,
            capture_interval=args.capture_interval,
            town=args.town
        )

        print("\nCollecting images...")
        with CarlaSyncMode(world, camera, fps=30) as sync_mode:
            last_capture_loc = None

            for seg_idx, segment in enumerate(road_segments):
                waypoints = segment['waypoints']
                for wp_idx, waypoint in enumerate(waypoints):
                    wp_transform = waypoint.transform
                    wp_transform.location.z += 0.5
                    vehicle.set_transform(wp_transform)

                    _, image_rgb = sync_mode.tick(timeout=2.0)

                    # Distance-based capture
                    current_loc = waypoint.transform.location
                    if last_capture_loc is not None:
                        dist = current_loc.distance(last_capture_loc)
                        if dist < args.capture_interval:
                            continue

                    collector.save_image(image_rgb, waypoint)
                    last_capture_loc = current_loc

                    if collector.image_count % 500 == 0:
                        print(f"    Collected {collector.image_count} images...")

        num_images = collector.save_map_data()

        # Cleanup
        for actor in actor_list:
            actor.destroy()

        print(f"\nMap collection complete: {num_images} images")

    # ==================== STEP 2: BUILD GRAPH ====================
    print(f"\n{'='*60}")
    print(f"STEP 2: Building topological graph with SIFT+VLAD + FAISS")
    print(f"{'='*60}")

    map_data_path = os.path.join(args.output_dir, "map_data.pkl")
    if not os.path.exists(map_data_path):
        print(f"ERROR: {map_data_path} not found. Run step 1 first.")
        return

    with open(map_data_path, 'rb') as f:
        map_data = pickle.load(f)

    # Build nodes_data from saved images
    nodes_data = []
    images_dir = os.path.join(args.output_dir, "images")
    for img_info in map_data['metadata']['images']:
        img_path = os.path.join(images_dir, img_info['filename'])
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        nodes_data.append({
            'obs_img': img,
            'location': (img_info['x'], img_info['y'], img_info['z']),
            'yaw': img_info['yaw'],
            'road_id': img_info['road_id'],
            'lane_id': img_info['lane_id'],
            'filename': img_info['filename'],
            'index': img_info['index'],
        })

    print(f"  Loaded {len(nodes_data)} nodes from map data")

    # Build graph
    graph_config = GraphBuilderConfig(
        min_loop_closure_gap=args.min_gap,
        min_similarity_score=args.min_sim,
        max_duplicate_similarity=args.max_dup_sim,
        feature_type='sift_vlad',
        use_gpu=not args.no_gpu,
        sift_vlad_config=SIFTVLADConfig(n_clusters=args.n_clusters),
    )

    builder = TopologicalGraphBuilder(config=graph_config)
    builder.build_from_data(nodes_data)

    # Save graph outputs
    builder.save(args.output_dir)

    stats = builder.get_stats()
    print(f"\n  Graph Statistics:")
    for key, value in stats.items():
        print(f"    {key}: {value}")

    # Save build stats as JSON for easy reading
    stats_path = os.path.join(args.output_dir, "build_stats.json")
    serializable_stats = {k: v for k, v in stats.items() if isinstance(v, (int, float, str, bool))}
    with open(stats_path, 'w') as f:
        json.dump(serializable_stats, f, indent=2)

    print(f"\nDone! Map saved to {args.output_dir}")


if __name__ == "__main__":
    main()