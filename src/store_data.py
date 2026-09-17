#!/usr/bin/env python

import glob
import os
import sys
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from enum import IntEnum
import carla
import cv2
import pygame
import queue
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

class RoadOption(IntEnum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6

def vector(location1, location2):
    """
    Returns the unit vector from location1 to location2
    """
    x = location2.x - location1.x
    y = location2.y - location1.y
    z = location2.z - location1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return [x / norm, y / norm, z / norm]

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors.
    """
    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
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

    def tick(self, timeout):
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

class SingleLaneTraverser:
    """
    Specialized class for traversing a single lane and capturing images
    """
    
    def __init__(self, grp):
        self.grp = grp
        self.graph = grp._graph
    
    def find_single_lane_path(self,start_lane_id, max_distance=1000):
        """
        Find a path that stays within a single lane across multiple road segments
        """
        # print(f"Finding single lane path for Road {start_road_id}, Lane {start_lane_id}")
        
        # Find starting edge for the specified road and lane
        start_edge = None
        breakpoint()
        for road_id in self.grp._road_id_to_edge:
                print(road_id)
                for section_id in self.grp._road_id_to_edge[road_id]:
                    for lane_id in self.grp._road_id_to_edge[road_id][section_id]:
                        if lane_id == start_lane_id:
                            start_edge = self.grp._road_id_to_edge[road_id][section_id][lane_id]
                #             break
                #     if start_edge:
                #         break
                # if start_edge:
                    # break
        
        # if not start_edge:
        #     print(f"No edge found for Road {start_road_id}, Lane {start_lane_id}")
        #     return []
        
        print(f"Starting edge: {start_edge}")
        
        # Traverse the lane
        path_edges = [start_edge]
        current_edge = start_edge
        traveled_distance = 0
        
        while traveled_distance < max_distance:
            next_edge = self._find_next_edge_same_lane(current_edge, start_lane_id)
            if not next_edge:
                break
            
            # Calculate distance of current edge
            edge_data = self.graph.edges[current_edge[0], current_edge[1]]
            edge_length = edge_data.get('length', 1) * self.grp._sampling_resolution
            traveled_distance += edge_length
            
            path_edges.append(next_edge)
            current_edge = next_edge
        
        print(f"Found path with {len(path_edges)} edges, total distance: {traveled_distance:.2f}m")
        return path_edges
    
    def _find_next_edge_same_lane(self, current_edge, target_lane_id):
        """
        Find the next edge that maintains the same lane ID
        """
        current_node = current_edge[1]  # Exit node of current edge
        
        # Get all outgoing edges from current node
        next_edges = []
        for successor in self.graph.successors(current_node):
            edge_data = self.graph.edges[current_node, successor]
            
            # Check if this edge maintains the same lane
            exit_wp = edge_data.get('exit_waypoint')
            if exit_wp and exit_wp.lane_id == target_lane_id:
                next_edges.append((current_node, successor))
        
        # If multiple options, choose the one that continues straightest
        if len(next_edges) == 1:
            return next_edges[0]
        elif len(next_edges) > 1:
            # Use vector similarity to choose the straightest continuation
            current_edge_data = self.graph.edges[current_edge[0], current_edge[1]]
            current_vector = current_edge_data.get('exit_vector')
            
            if current_vector is not None:
                best_similarity = -1
                best_edge = None
                
                for edge in next_edges:
                    edge_data = self.graph.edges[edge[0], edge[1]]
                    next_vector = edge_data.get('entry_vector')
                    
                    if next_vector is not None:
                        similarity = np.dot(current_vector, next_vector)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_edge = edge
                
                return best_edge
        
        return None
    
    def get_waypoints_from_path(self, path_edges):
        """
        Extract waypoints from a path of edges with proper ordering
        """
        waypoints = []
        
        for i, edge in enumerate(path_edges):
            edge_data = self.graph.edges[edge[0], edge[1]]
            
            # For first edge, add all waypoints
            if i == 0:
                if 'entry_waypoint' in edge_data:
                    waypoints.append(edge_data['entry_waypoint'])
                if 'path' in edge_data:
                    waypoints.extend(edge_data['path'])
                if 'exit_waypoint' in edge_data:
                    waypoints.append(edge_data['exit_waypoint'])
            else:
                # For subsequent edges, skip the entry waypoint (it's the same as previous exit)
                if 'path' in edge_data:
                    waypoints.extend(edge_data['path'])
                if 'exit_waypoint' in edge_data:
                    waypoints.append(edge_data['exit_waypoint'])
        
        return waypoints

class ImageCapture:
    """
    Class to handle image capture along the path
    """
    
    def __init__(self, save_dir="/home/robotlab/scratch/carla-rl/topomap"):
        self.save_dir = save_dir
        self.image_count = 0
        self.locations = []
        os.makedirs(save_dir, exist_ok=True)
        
        # Create subdirectories
        self.rgb_dir = os.path.join(save_dir, "rgb")
        self.semseg_dir = os.path.join(save_dir, "semseg")
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.semseg_dir, exist_ok=True)
    
    def save_image(self, image, image_type="rgb"):
        """
        Save image to disk
        """
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]  # Convert BGRA to RGB
        
        filename = f"{self.image_count:06d}.png"
        if image_type == "rgb":
            save_path = os.path.join(self.rgb_dir, filename)
            cv2.imwrite(save_path, array)
        else:
            save_path = os.path.join(self.semseg_dir, filename)
            cv2.imwrite(save_path, array)
        
        return filename
    
    def record_location(self, vehicle, waypoint, image_count):
        """
        Record vehicle location and waypoint information
        """
        loc = vehicle.get_transform().location
        waypoint_info = {
            'image_id': image_count,
            'location': [loc.x, loc.y, loc.z],
            'waypoint_info': {
                'road_id': waypoint.road_id,
                'lane_id': waypoint.lane_id,
                's': waypoint.s,
                'transform': {
                    'location': [waypoint.transform.location.x, waypoint.transform.location.y, waypoint.transform.location.z],
                    'rotation': [waypoint.transform.rotation.pitch, waypoint.transform.rotation.yaw, waypoint.transform.rotation.roll]
                }
            }
        }
        self.locations.append(waypoint_info)
        return waypoint_info
    
    def save_metadata(self):
        """
        Save metadata about the captured images
        """
        metadata = {
            "total_images": self.image_count,
            "locations": self.locations,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        import json
        with open(os.path.join(self.save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved with {len(self.locations)} location records")

class GlobalRoutePlanner(object):
    """
    This class provides a very high level route plan.
    """

    def __init__(self, wmap, sampling_resolution):
        self._sampling_resolution = sampling_resolution
        self._wmap = wmap
        self._topology = None
        self._graph = None
        self._id_map = None
        self._road_id_to_edge = None

        # Build the graph
        self._build_topology()
        self._build_graph()
        self._find_loose_ends()

    def _build_topology(self):
        """
        This function retrieves topology from the server
        """
        self._topology = []
        for segment in self._wmap.get_topology():
            wp1, wp2 = segment[0], segment[1]
            l1, l2 = wp1.transform.location, wp2.transform.location
            x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
            seg_dict = dict()
            seg_dict["entry"], seg_dict["exit"] = wp1, wp2
            seg_dict["entryxyz"], seg_dict["exitxyz"] = (x1, y1, z1), (x2, y2, z2)
            seg_dict["path"] = []
            endloc = wp2.transform.location
            if wp1.transform.location.distance(endloc) > self._sampling_resolution:
                w = wp1.next(self._sampling_resolution)[0]
                while w.transform.location.distance(endloc) > self._sampling_resolution:
                    seg_dict["path"].append(w)
                    next_ws = w.next(self._sampling_resolution)
                    if len(next_ws) == 0:
                        break
                    w = next_ws[0]
            else:
                next_wps = wp1.next(self._sampling_resolution)
                if len(next_wps) == 0:
                    continue
                seg_dict["path"].append(next_wps[0])
            self._topology.append(seg_dict)

    def _build_graph(self):
        """
        This function builds a networkx graph representation of topology
        """
        self._graph = nx.DiGraph()
        self._id_map = dict()
        self._road_id_to_edge = dict()

        for segment in self._topology:
            entry_xyz, exit_xyz = segment["entryxyz"], segment["exitxyz"]
            path = segment["path"]
            entry_wp, exit_wp = segment["entry"], segment["exit"]
            intersection = entry_wp.is_junction
            road_id, section_id, lane_id = (
                entry_wp.road_id,
                entry_wp.section_id,
                entry_wp.lane_id,
            )

            for vertex in entry_xyz, exit_xyz:
                if vertex not in self._id_map:
                    new_id = len(self._id_map)
                    self._id_map[vertex] = new_id
                    self._graph.add_node(new_id, vertex=vertex)
            n1 = self._id_map[entry_xyz]
            n2 = self._id_map[exit_xyz]
            if road_id not in self._road_id_to_edge:
                self._road_id_to_edge[road_id] = dict()
            if section_id not in self._road_id_to_edge[road_id]:
                self._road_id_to_edge[road_id][section_id] = dict()
            self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)

            entry_carla_vector = entry_wp.transform.rotation.get_forward_vector()
            exit_carla_vector = exit_wp.transform.rotation.get_forward_vector()

            self._graph.add_edge(
                n1,
                n2,
                length=len(path) + 1,
                path=path,
                entry_waypoint=entry_wp,
                exit_waypoint=exit_wp,
                entry_vector=np.array(
                    [entry_carla_vector.x, entry_carla_vector.y, entry_carla_vector.z]
                ),
                exit_vector=np.array(
                    [exit_carla_vector.x, exit_carla_vector.y, exit_carla_vector.z]
                ),
                net_vector=vector(
                    entry_wp.transform.location, exit_wp.transform.location
                ),
                intersection=intersection,
                type=RoadOption.LANEFOLLOW,
            )

    def _find_loose_ends(self):
        """
        This method finds road segments that have an unconnected end
        """
        count_loose_ends = 0
        hop_resolution = self._sampling_resolution
        for segment in self._topology:
            end_wp = segment["exit"]
            exit_xyz = segment["exitxyz"]
            road_id, section_id, lane_id = (
                end_wp.road_id,
                end_wp.section_id,
                end_wp.lane_id,
            )
            if (
                road_id in self._road_id_to_edge
                and section_id in self._road_id_to_edge[road_id]
                and lane_id in self._road_id_to_edge[road_id][section_id]
            ):
                pass
            else:
                count_loose_ends += 1
                if road_id not in self._road_id_to_edge:
                    self._road_id_to_edge[road_id] = dict()
                if section_id not in self._road_id_to_edge[road_id]:
                    self._road_id_to_edge[road_id][section_id] = dict()
                n1 = self._id_map[exit_xyz]
                n2 = -1 * count_loose_ends
                self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)
                next_wp = end_wp.next(hop_resolution)
                path = []
                while (
                    next_wp is not None
                    and next_wp
                    and next_wp[0].road_id == road_id
                    and next_wp[0].section_id == section_id
                    and next_wp[0].lane_id == lane_id
                ):
                    path.append(next_wp[0])
                    next_wp = next_wp[0].next(hop_resolution)
                if path:
                    n2_xyz = (
                        path[-1].transform.location.x,
                        path[-1].transform.location.y,
                        path[-1].transform.location.z,
                    )
                    self._graph.add_node(n2, vertex=n2_xyz)
                    self._graph.add_edge(
                        n1,
                        n2,
                        length=len(path) + 1,
                        path=path,
                        entry_waypoint=end_wp,
                        exit_waypoint=path[-1],
                        entry_vector=None,
                        exit_vector=None,
                        net_vector=None,
                        intersection=end_wp.is_junction,
                        type=RoadOption.LANEFOLLOW,
                    )

def main():
    # Setup directories
    save_dir = "/home/robotlab/scratch/carla-rl/topomap"
    os.makedirs(save_dir, exist_ok=True)
    
    # Connect to CARLA server
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.load_world("Town02")
        wmap = world.get_map()
        
        print(f"Connected to CARLA map: {wmap.name}")
        
        # Create GlobalRoutePlanner
        sampling_resolution = 2.0
        grp = GlobalRoutePlanner(wmap, sampling_resolution)
        
        print(f"Graph built with {grp._graph.number_of_nodes()} nodes and {grp._graph.number_of_edges()} edges")
        
        # Create single lane traverser
        traverser = SingleLaneTraverser(grp)
        
        # Choose a road and lane to traverse (Town02 specific)
        # target_road = 5  # Good road in Town02
        target_lane = 1   # Right lane
        
        # Find single lane path
        path_edges = traverser.find_single_lane_path(target_lane)
        
        # if not path_edges:
        #     print("No path found. Trying different road...")
        #     target_road = 7  # Alternative road
        #     path_edges = traverser.find_single_lane_path(target_road, target_lane, max_distance=500)
        
        # if not path_edges:
        #     print("Failed to find a suitable path. Exiting.")
        #     return
        
        # Get waypoints from the path
        waypoints = traverser.get_waypoints_from_path(path_edges)
        print(f"Extracted {len(waypoints)} waypoints from the path")
        
        # Set up image capture
        image_capture = ImageCapture(save_dir)
        
        # Setup vehicle and sensors
        actor_list = []
        pygame.init()
        
        display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
        font = pygame.font.Font(None, 36)
        clock = pygame.time.Clock()
        
        # Create vehicle
        start_pose = waypoints[0].transform
        start_pose.location.z += 2.0  # Raise slightly above ground
        blueprint_library = world.get_blueprint_library()
        
        # Choose a small vehicle
        vehicle_bp = blueprint_library.filter('vehicle.audi.a2')[0]
        vehicle = world.spawn_actor(vehicle_bp, start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)  # Disable physics for precise control
        
        # Setup cameras
        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-10)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)
        
        camera_semseg = world.spawn_actor(
            blueprint_library.find('sensor.camera.semantic_segmentation'),
            carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-10)),
            attach_to=vehicle)
        actor_list.append(camera_semseg)
        
        # Create synchronous mode context
        with CarlaSyncMode(world, camera_rgb, camera_semseg, fps=20) as sync_mode:
            print("Starting image capture along the path...")
            
            waypoint_index = 0
            total_waypoints = len(waypoints)
            capture_interval = 3  # Capture image every N waypoints
            
            for waypoint in waypoints:
                # Check for quit signal
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE):
                        print("User interrupted capture")
                        break
                
                # Advance the simulation
                snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=2.0)
                
                # Set vehicle position
                vehicle.set_transform(waypoint.transform)
                
                # Capture images at intervals
                if waypoint_index % capture_interval == 0:
                    # Convert semantic segmentation image
                    image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                    
                    # Save images
                    rgb_filename = image_capture.save_image(image_rgb, "rgb")
                    semseg_filename = image_capture.save_image(image_semseg, "semseg")
                    
                    # Record location and waypoint info
                    image_capture.record_location(vehicle, waypoint, image_capture.image_count)
                    
                    image_capture.image_count += 1
                    
                    # Display progress
                    display.fill((0, 0, 0))
                    progress_text = f"Progress: {waypoint_index}/{total_waypoints}"
                    road_text = f"Road: {waypoint.road_id}, Lane: {waypoint.lane_id}"
                    images_text = f"Images: {image_capture.image_count}"
                    
                    display.blit(font.render(progress_text, True, (255, 255, 255)), (20, 20))
                    display.blit(font.render(road_text, True, (255, 255, 255)), (20, 60))
                    display.blit(font.render(images_text, True, (255, 255, 255)), (20, 100))
                    pygame.display.flip()
                    
                    print(f"Captured image {image_capture.image_count} at waypoint {waypoint_index} "
                          f"(Road {waypoint.road_id}, Lane {waypoint.lane_id})")
                
                waypoint_index += 1
                clock.tick(20)
        
        # Save metadata
        image_capture.save_metadata()
        
        print(f"\n=== CAPTURE COMPLETE ===")
        print(f"Total images captured: {image_capture.image_count}")
        print(f"RGB images saved to: {image_capture.rgb_dir}")
        print(f"Semantic segmentation images saved to: {image_capture.semseg_dir}")
        print(f"Metadata saved to: {os.path.join(save_dir, 'metadata.json')}")
        
        # Create a summary file
        with open(os.path.join(save_dir, "capture_summary.txt"), "w") as f:
            f.write(f"Single Lane Image Capture Summary\n")
            f.write(f"=================================\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Map: {wmap.name}\n")
            f.write(f"Target Road: {target_road}\n")
            f.write(f"Target Lane: {target_lane}\n")
            f.write(f"Total waypoints: {len(waypoints)}\n")
            f.write(f"Total images: {image_capture.image_count}\n")
            f.write(f"Capture interval: Every {capture_interval} waypoints\n")
            f.write(f"RGB images: {image_capture.rgb_dir}\n")
            f.write(f"Semantic segmentation images: {image_capture.semseg_dir}\n")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print('\nCleaning up actors...')
        for actor in actor_list:
            try:
                actor.destroy()
            except:
                pass
        pygame.quit()
        print('Done!')

if __name__ == "__main__":
    main()