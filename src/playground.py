# from collections import defaultdict, deque
# from functools import partial
# import glob
# import os
# import pickle
# import random
# import carla
# import cv2
# import numpy as np
# import shutil
# from absl import app, flags
# from jaxrl2.wrappers.frame_stack import FrameStack
# from jaxrl2.wrappers.timelimit import TimeLimit
# from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
# from rlib_integration.agent import GlobalRoutePlanner
# from rlib_integration.helper import ndarray_to_location
# from carla_eval import CarlaEvalEnv
# import matplotlib.pyplot as plt
# # Set up environment variables
# os.environ['XLA_FLAGS'] = "--xla_gpu_enable_command_buffer="
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".20"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# # Initialize environment
# env = CarlaEvalEnv(start_server=False, town="Town02")
# env = TimeLimit(env, max_episode_steps=int(2e3))
# env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
# env = FrameStack(env=env, num_stack=1, stacking_key="goal")
# env = RecordEpisodeStatistics(env)

# # Set up paths and route planner
# route_planner = GlobalRoutePlanner(env.unwrapped.core.map, 2.0)
# base = "/home/robotlab/scratch/carla-rl/topomap"
# results_store = "/home/robotlab/scratch/carla-rl/evaluation_v2/uncertainty"

# # Load trajectory data
# with open(f"{base}/traj.pkl", "rb") as pipe:
#     traj_data = pickle.load(pipe)
# locations = traj_data["locations"]

# # Create output directory if it doesn't exist
# os.makedirs(results_store, exist_ok=True)

# def plot_and_save_route(trace, origin, destination, save_path=None):
#     """Plot and save the route visualization with all waypoints marked"""
#     # Extract waypoints and road options
#     xs = [wp.transform.location.x for wp, _ in trace]
#     ys = [wp.transform.location.y for wp, _ in trace]
#     road_options = [road_option for _, road_option in trace]
    
#     # Create figure
#     plt.figure(figsize=(12, 12))
    
#     # Plot route line
#     plt.plot(xs, ys, linewidth=3, color='royalblue', alpha=0.5, label="Route Path")
    
#     # Plot all waypoints with color coding by road option
#     for i, ((x, y), option) in enumerate(zip(zip(xs, ys), road_options)):
#         # color = {
#         #     carla.RoadOption.LEFT: 'red',
#         #     carla.RoadOption.RIGHT: 'green',
#         #     carla.RoadOption.STRAIGHT: 'blue',
#         #     carla.RoadOption.LANEFOLLOW: 'gray',
#         #     carla.RoadOption.CHANGELANELEFT: 'orange',
#         #     carla.RoadOption.CHANGELANERIGHT: 'purple',
#         #     carla.RoadOption.VOID: 'black'
#         # }.get(option, 'gray')
        
#         plt.scatter(x, y, s=80, edgecolor='white', linewidth=1,
#                    zorder=4, label=f"Waypoint {i+1}" if i < 5 else "")
    
#     # Mark start and end points prominently
#     plt.scatter(xs[0], ys[0], color='lime', s=300, zorder=5, 
#                marker='*', edgecolor='black', linewidth=1.5, label="Start")
#     plt.scatter(xs[-1], ys[-1], color='magenta', s=300, zorder=5,
#                marker='X', edgecolor='black', linewidth=1.5, label="End")
    
#     # Add labels and title
#     plt.xlabel("X (meters)", fontsize=14)
#     plt.ylabel("Y (meters)", fontsize=14)
#     plt.title(f"CARLA Route with Waypoints\nStart: {origin[:2]} → End: {destination[:2]}", 
#              fontsize=16, pad=20)
    
#     # Create custom legend
#     from matplotlib.lines import Line2D
#     legend_elements = [
#         Line2D([0], [0], color='royalblue', lw=3, label='Route Path'),
#         Line2D([0], [0], marker='*', color='lime', label='Start Point',
#               markersize=15, linestyle='None', markeredgecolor='black'),
#         Line2D([0], [0], marker='X', color='magenta', label='End Point',
#               markersize=15, linestyle='None', markeredgecolor='black'),
#         Line2D([0], [0], marker='o', color='blue', label='Straight',
#               markersize=10, linestyle='None', markeredgecolor='white'),
#         Line2D([0], [0], marker='o', color='red', label='Left Turn',
#               markersize=10, linestyle='None', markeredgecolor='white'),
#         Line2D([0], [0], marker='o', color='green', label='Right Turn',
#               markersize=10, linestyle='None', markeredgecolor='white')
#     ]
    
#     plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
#     # Add grid and adjust layout
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
    
#     # Add coordinate annotations for start/end
#     plt.annotate(f"({xs[0]:.1f}, {ys[0]:.1f})", (xs[0], ys[0]), 
#                 textcoords="offset points", xytext=(10,10), ha='left')
#     plt.annotate(f"({xs[-1]:.1f}, {ys[-1]:.1f})", (xs[-1], ys[-1]), 
#                 textcoords="offset points", xytext=(10,10), ha='left')
    
#     # Save or show
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
#         plt.close()
#         print(f"Saved route visualization to: {save_path}")
#     else:
#         plt.show()

# # Process each trajectory
# trajectories = iter([
#     [1362, 1354],
#     [627, 472],
#     [741, 1495],
#     [561, 244]
# ])
# gt_paths=dict()
# for i, (origin_idx, destination_idx) in enumerate(trajectories):
#     origin, destination = locations[origin_idx], locations[destination_idx]
#     trace = route_planner.trace_route(
#         ndarray_to_location(origin),
#         ndarray_to_location(destination)
#     )
#     gt_paths[i]={"x":[wp.transform.location.x for wp, _ in trace],"y":[wp.transform.location.y for wp, _ in trace]}
    
#     # Generate save path
#     save_path = os.path.join(
#         results_store,
#         f"route_{i+1}_origin_{origin_idx}_dest_{destination_idx}.png"
#     )
    
#     # Plot and save
#     plot_and_save_route(trace, origin, destination, save_path)
#     print(f"Saved route visualization to: {save_path}")
# with open("gt_path.pkl","wb") as pipe:
#     pickle.dump(gt_paths,pipe)


#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# import cv2 
# import glob
# import os
# import sys

# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

# import carla

# import random

# try:
#     import pygame
# except ImportError:
#     raise RuntimeError('cannot import pygame, make sure pygame package is installed')

# try:
#     import numpy as np
# except ImportError:
#     raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# try:
#     import queue
# except ImportError:
#     import Queue as queue

# import pickle
# import math

# class CarlaSyncMode(object):
#     """
#     Context manager to synchronize output from different sensors. Synchronous
#     mode is enabled as long as we are inside this context

#         with CarlaSyncMode(world, sensors) as sync_mode:
#             while True:
#                 data = sync_mode.tick(timeout=1.0)

#     """

#     def __init__(self, world, *sensors, **kwargs):
#         self.world = world
#         self.sensors = sensors
#         self.frame = None
#         self.delta_seconds = 1.0 / kwargs.get('fps', 20)
#         self._queues = []
#         self._settings = None

#     def __enter__(self):
#         self._settings = self.world.get_settings()
#         self.frame = self.world.apply_settings(carla.WorldSettings(
#             no_rendering_mode=False,
#             synchronous_mode=True,
#             fixed_delta_seconds=self.delta_seconds))

#         def make_queue(register_event):
#             q = queue.Queue()
#             register_event(q.put)
#             self._queues.append(q)

#         make_queue(self.world.on_tick)
#         for sensor in self.sensors:
#             make_queue(sensor.listen)
#         return self

#     def tick(self, timeout):
#         self.frame = self.world.tick()
#         data = [self._retrieve_data(q, timeout) for q in self._queues]
#         assert all(x.frame == self.frame for x in data)
#         return data

#     def __exit__(self, *args, **kwargs):
#         self.world.apply_settings(self._settings)

#     def _retrieve_data(self, sensor_queue, timeout):
#         while True:
#             data = sensor_queue.get(timeout=timeout)
#             if data.frame == self.frame:
#                 return data


# # Functional stuff
# image_count=0
# save_dir="/home/robotlab/scratch/carla-rl/topomap"
# locations=[]
# os.makedirs(save_dir, exist_ok=True)

# def draw_image(surface, image,vehicle=None ,blend=False):
#     global image_count
#     array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
#     array = np.reshape(array, (image.height, image.width, 4))
#     array = array[:, :, :3]
#     array = array[:, :, ::-1]
 
#     image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
#     if not vehicle is None:
#         cv2.imwrite(f"{save_dir}/{image_count}.jpg",cv2.cvtColor(array,cv2.COLOR_BGR2RGB))    
#         image_count+=1
#         loc=vehicle.get_transform().location
#         locations.append([loc.x,loc.y,loc.z])
#         with open(f"{save_dir}/traj.pkl", "wb") as pipe:
#             pickle.dump({"locations": locations}, pipe)
#     if blend:
#         image_surface.set_alpha(100)
#     surface.blit(image_surface, (0, 0))


# def get_font():
#     fonts = [x for x in pygame.font.get_fonts()]
#     default_font = 'ubuntumono'
#     font = default_font if default_font in fonts else fonts[0]
#     font = pygame.font.match_font(font)
#     return pygame.font.Font(font, 14)


# def should_quit():
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             return True
#         elif event.type == pygame.KEYUP:
#             if event.key == pygame.K_ESCAPE:
#                 return True
#     return False


# def filter_single_lane_waypoints(waypoints, target_lane_id=None, road_id=None):
#     """
#     Filter waypoints to only include those from a single lane.
    
#     Args:
#         waypoints: List of carla.Waypoint objects
#         target_lane_id: Specific lane ID to filter by (optional)
#         road_id: Specific road ID to filter by (optional)
    
#     Returns:
#         List of waypoints from a single lane
#     """
#     if not waypoints:
#         return []
    
#     # Method 1: Filter by specific lane and road ID
#     if target_lane_id is not None and road_id is not None:
#         filtered = [wp for wp in waypoints 
#                    if wp.lane_id == target_lane_id and wp.road_id == road_id]
#         return filtered
    
#     # Method 2: Group waypoints by lane and select the lane with most waypoints
#     lane_groups = {}
#     for wp in waypoints:
#         lane_key = (wp.road_id, wp.lane_id)
#         if lane_key not in lane_groups:
#             lane_groups[lane_key] = []
#         lane_groups[lane_key].append(wp)
    
#     # Select the lane with the most waypoints (usually the main driving lane)
#     if lane_groups:
#         best_lane = max(lane_groups.keys(), key=lambda k: len(lane_groups[k]))
#         print(f"Selected lane: Road ID {best_lane[0]}, Lane ID {best_lane[1]} with {len(lane_groups[best_lane])} waypoints")
#         return lane_groups[best_lane]
    
#     return []


# def filter_rightmost_driving_lane(waypoints):
#     """
#     Filter waypoints to only include those from the rightmost driving lane.
#     This is often the main lane for data collection.
#     """
#     if not waypoints:
#         return []
    
#     # Group by road_id first
#     road_groups = {}
#     for wp in waypoints:
#         if wp.road_id not in road_groups:
#             road_groups[wp.road_id] = []
#         road_groups[wp.road_id].append(wp)
    
#     filtered_waypoints = []
    
#     for road_id, road_waypoints in road_groups.items():
#         # Filter only driving lanes (positive lane_id indicates right side in CARLA)
#         driving_waypoints = [wp for wp in road_waypoints if wp.lane_type == carla.LaneType.Driving]
        
#         if driving_waypoints:
#             # Group by lane_id
#             lane_groups = {}
#             for wp in driving_waypoints:
#                 if wp.lane_id not in lane_groups:
#                     lane_groups[wp.lane_id] = []
#                 lane_groups[wp.lane_id].append(wp)
            
#             # Select rightmost lane (smallest positive lane_id for right-hand traffic)
#             if lane_groups:
#                 rightmost_lane_id = min([lid for lid in lane_groups.keys() if lid > 0], default=None)
#                 if rightmost_lane_id:
#                     filtered_waypoints.extend(lane_groups[rightmost_lane_id])
#                     print(f"Road {road_id}: Selected rightmost driving lane {rightmost_lane_id} with {len(lane_groups[rightmost_lane_id])} waypoints")
    
#     return filtered_waypoints


# def get_continuous_lane_path(start_waypoint, max_distance=1000):
#     """
#     Generate a continuous path along a single lane starting from a waypoint.
    
#     Args:
#         start_waypoint: Starting carla.Waypoint
#         max_distance: Maximum distance to travel along the lane
    
#     Returns:
#         List of waypoints forming a continuous path in the same lane
#     """
#     waypoints = [start_waypoint]
#     current_wp = start_waypoint
#     traveled_distance = 0
    
#     while traveled_distance < max_distance:
#         # Get next waypoint in the same lane
#         next_waypoints = current_wp.next(2.0)  # 2 meter spacing
        
#         if not next_waypoints:
#             break
            
#         # Select the waypoint that stays in the same lane
#         next_wp = None
#         for wp in next_waypoints:
#             if wp.lane_id == current_wp.lane_id and wp.road_id == current_wp.road_id:
#                 next_wp = wp
#                 break
        
#         if next_wp is None:
#             break
            
#         waypoints.append(next_wp)
#         traveled_distance += current_wp.transform.location.distance(next_wp.transform.location)
#         current_wp = next_wp
    
#     print(f"Generated continuous path with {len(waypoints)} waypoints, total distance: {traveled_distance:.2f}m")
#     return waypoints


# def main():
#     actor_list = []
#     pygame.init()
    
#     display = pygame.display.set_mode(
#         (800, 600),
#         pygame.HWSURFACE | pygame.DOUBLEBUF)
#     font = get_font()
#     clock = pygame.time.Clock()
    
#     client = carla.Client('localhost', 2000)
#     client.set_timeout(20.0)
#     world = client.load_world("Town02")
    
#     try:
#         m = world.get_map()
        
#         # OPTION 1: Filter waypoints from all generated waypoints
#         all_waypoints = m.generate_waypoints(2)
#         print(f"Generated {len(all_waypoints)} total waypoints")
        
#         # Choose one of these filtering methods:
        
#         # Method A: Filter rightmost driving lane
#         waypoints = filter_rightmost_driving_lane(all_waypoints)
        
#         # Method B: Filter by specific lane (uncomment to use)
#         # waypoints = filter_single_lane_waypoints(all_waypoints, target_lane_id=1, road_id=10)
        
#         # Method C: Use first waypoint and generate continuous path (uncomment to use)
#         # if all_waypoints:
#         #     waypoints = get_continuous_lane_path(all_waypoints[0], max_distance=500)
#         # else:
#         #     waypoints = []
        
#         print(f"Filtered to {len(waypoints)} waypoints for single lane collection")
        
#         if not waypoints:
#             print("No waypoints found for single lane. Exiting.")
#             return
        
#         start_pose = waypoints[0].transform
#         start_pose.location.z += 5
#         blueprint_library = world.get_blueprint_library()
#         vehicle = world.spawn_actor(
#             blueprint_library.filter('vehicle.*')[0],
#             start_pose)
#         actor_list.append(vehicle)
#         vehicle.set_simulate_physics(False)
        
#         camera_rgb = world.spawn_actor(
#             blueprint_library.find('sensor.camera.rgb'),
#             carla.Transform(carla.Location(x=1.0, z=2.8), carla.Rotation(pitch=-15)),
#             attach_to=vehicle)
#         actor_list.append(camera_rgb)
        
#         camera_semseg = world.spawn_actor(
#             blueprint_library.find('sensor.camera.semantic_segmentation'),
#             carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
#             attach_to=vehicle)
#         actor_list.append(camera_semseg)
        
#         # Create a synchronous mode context.
#         with CarlaSyncMode(world, camera_rgb, camera_semseg, fps=30) as sync_mode:
#             waypoint_index = 0
#             for waypoint in waypoints:
#                 if should_quit():
#                     return
#                 clock.tick()
                
#                 # Advance the simulation and wait for the data.
#                 snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=2.0)
                
#                 # Use the waypoint's rotation directly (maintains lane direction)
#                 vehicle.set_transform(waypoint.transform)
                
#                 waypoint_index += 1
                
#                 image_semseg.convert(carla.ColorConverter.CityScapesPalette)
#                 fps = round(1.0 / snapshot.timestamp.delta_seconds)
                
#                 # Draw the display.
#                 draw_image(display, image_rgb, vehicle)
#                 draw_image(display, image_semseg, None, blend=True)
#                 display.blit(
#                     font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
#                     (8, 10))
#                 display.blit(
#                     font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
#                     (8, 28))
#                 display.blit(
#                     font.render('Waypoint: %d/%d' % (waypoint_index, len(waypoints)), True, (255, 255, 255)),
#                     (8, 46))
#                 pygame.display.flip()
    
#     finally:
#         print('destroying actors.')
#         for actor in actor_list:
#             actor.destroy()
        
#         pygame.quit()
#         print('done.')


# if __name__ == '__main__':
#     try:
#         main()
#     except KeyboardInterrupt:
#         print('\nCancelled by user. Bye!')



#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import cv2 
import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

import pickle
import math

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

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


# Functional stuff
image_count=0
save_dir="/home/robotlab/scratch/carla-rl/topomap"
locations=[]
os.makedirs(save_dir, exist_ok=True)

def draw_image(surface, image,vehicle=None ,blend=False):
    global image_count
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
 
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if not vehicle is None:
        cv2.imwrite(f"{save_dir}/{image_count}.jpg",cv2.cvtColor(array,cv2.COLOR_BGR2RGB))    
        image_count+=1
        loc=vehicle.get_transform().location
        locations.append([loc.x,loc.y,loc.z])
        with open(f"{save_dir}/traj.pkl", "wb") as pipe:
            pickle.dump({"locations": locations}, pipe)
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def sort_waypoints_sequentially(waypoints):
    """
    Sort waypoints to create a sequential path by connecting nearest neighbors.
    
    Args:
        waypoints: List of carla.Waypoint objects
    
    Returns:
        List of waypoints in sequential order
    """
    if len(waypoints) <= 1:
        return waypoints
    
    # Start with the first waypoint
    sorted_waypoints = [waypoints[0]]
    remaining_waypoints = waypoints[1:]
    
    while remaining_waypoints:
        current_wp = sorted_waypoints[-1]
        current_loc = current_wp.transform.location
        
        # Find the nearest remaining waypoint
        min_distance = float('inf')
        nearest_wp = None
        nearest_idx = -1
        
        for i, wp in enumerate(remaining_waypoints):
            wp_loc = wp.transform.location
            distance = current_loc.distance(wp_loc)
            if distance < min_distance:
                min_distance = distance
                nearest_wp = wp
                nearest_idx = i
        
        if nearest_wp:
            sorted_waypoints.append(nearest_wp)
            remaining_waypoints.pop(nearest_idx)
        else:
            break
    
    return sorted_waypoints


def filter_single_lane_waypoints(waypoints, target_lane_id=None, road_id=None):
    """
    Filter waypoints to only include those from a single lane and sort them sequentially.
    
    Args:
        waypoints: List of carla.Waypoint objects
        target_lane_id: Specific lane ID to filter by (optional)
        road_id: Specific road ID to filter by (optional)
    
    Returns:
        List of waypoints from a single lane in sequential order
    """
    if not waypoints:
        return []
    
    # Method 1: Filter by specific lane and road ID
    if target_lane_id is not None and road_id is not None:
        filtered = [wp for wp in waypoints 
                   if wp.lane_id == target_lane_id and wp.road_id == road_id]
        return sort_waypoints_sequentially(filtered)
    
    # Method 2: Group waypoints by lane and select the lane with most waypoints
    lane_groups = {}
    for wp in waypoints:
        lane_key = (wp.road_id, wp.lane_id)
        if lane_key not in lane_groups:
            lane_groups[lane_key] = []
        lane_groups[lane_key].append(wp)
    
    # Select the lane with the most waypoints (usually the main driving lane)
    if lane_groups:
        best_lane = max(lane_groups.keys(), key=lambda k: len(lane_groups[k]))
        print(f"Selected lane: Road ID {best_lane[0]}, Lane ID {best_lane[1]} with {len(lane_groups[best_lane])} waypoints")
        filtered_waypoints = lane_groups[best_lane]
        return sort_waypoints_sequentially(filtered_waypoints)
    
    return []


def filter_rightmost_driving_lane(waypoints):
    """
    Filter waypoints to only include those from the rightmost driving lane and sort sequentially.
    This is often the main lane for data collection.
    """
    if not waypoints:
        return []
    
    # Group by road_id first
    road_groups = {}
    for wp in waypoints:
        if wp.road_id not in road_groups:
            road_groups[wp.road_id] = []
        road_groups[wp.road_id].append(wp)
    
    all_lane_waypoints = []
    
    for road_id, road_waypoints in road_groups.items():
        # Filter only driving lanes (positive lane_id indicates right side in CARLA)
        driving_waypoints = [wp for wp in road_waypoints if wp.lane_type == carla.LaneType.Driving]
        
        if driving_waypoints:
            # Group by lane_id
            lane_groups = {}
            for wp in driving_waypoints:
                if wp.lane_id not in lane_groups:
                    lane_groups[wp.lane_id] = []
                lane_groups[wp.lane_id].append(wp)
            
            # Select rightmost lane (smallest positive lane_id for right-hand traffic)
            if lane_groups:
                rightmost_lane_id = min([lid for lid in lane_groups.keys() if lid > 0], default=None)
                if rightmost_lane_id:
                    lane_waypoints = lane_groups[rightmost_lane_id]
                    # Sort this road segment's waypoints sequentially
                    sorted_lane_waypoints = sort_waypoints_sequentially(lane_waypoints)
                    all_lane_waypoints.extend(sorted_lane_waypoints)
                    print(f"Road {road_id}: Selected rightmost driving lane {rightmost_lane_id} with {len(sorted_lane_waypoints)} waypoints")
    
    # Sort all collected waypoints sequentially across roads
    return sort_waypoints_sequentially(all_lane_waypoints) if all_lane_waypoints else []


def get_continuous_lane_path(start_waypoint, max_distance=1000, spacing=2.0):
    """
    Generate a continuous path along a single lane starting from a waypoint.
    This method guarantees sequential order by following the lane topology.
    
    Args:
        start_waypoint: Starting carla.Waypoint
        max_distance: Maximum distance to travel along the lane
        spacing: Distance between waypoints in meters
    
    Returns:
        List of waypoints forming a continuous path in the same lane in sequential order
    """
    waypoints = [start_waypoint]
    current_wp = start_waypoint
    traveled_distance = 0
    
    while traveled_distance < max_distance:
        # Get next waypoint in the same lane
        next_waypoints = current_wp.next(spacing)
        
        if not next_waypoints:
            break
            
        # Select the waypoint that stays in the same lane
        next_wp = None
        for wp in next_waypoints:
            if wp.lane_id == current_wp.lane_id and wp.road_id == current_wp.road_id:
                next_wp = wp
                break
        
        if next_wp is None:
            break
            
        waypoints.append(next_wp)
        traveled_distance += current_wp.transform.location.distance(next_wp.transform.location)
        current_wp = next_wp
    
    print(f"Generated continuous path with {len(waypoints)} waypoints, total distance: {traveled_distance:.2f}m")
    return waypoints


def get_all_road_ids(world_map):
    """
    Get all unique road IDs from the map.
    
    Args:
        world_map: CARLA map object
    
    Returns:
        List of unique road IDs
    """
    waypoints = world_map.generate_waypoints(2.0)
    road_ids = list(set([wp.road_id for wp in waypoints if wp.lane_type == carla.LaneType.Driving]))
    road_ids.sort()
    return road_ids


def get_road_start_waypoint(world_map, road_id, lane_id=None):
    """
    Find the starting waypoint for a specific road.
    
    Args:
        world_map: CARLA map object
        road_id: Road ID to find start point for
        lane_id: Specific lane ID (optional, will use rightmost driving lane if None)
    
    Returns:
        Starting waypoint for the road
    """
    waypoints = world_map.generate_waypoints(2.0)
    road_waypoints = [wp for wp in waypoints if wp.road_id == road_id and wp.lane_type == carla.LaneType.Driving]
    
    if not road_waypoints:
        return None
    
    if lane_id is not None:
        # Filter by specific lane
        lane_waypoints = [wp for wp in road_waypoints if wp.lane_id == lane_id]
        if lane_waypoints:
            road_waypoints = lane_waypoints
    else:
        # Use rightmost driving lane (smallest positive lane_id)
        lane_ids = [wp.lane_id for wp in road_waypoints if wp.lane_id > 0]
        if lane_ids:
            rightmost_lane_id = min(lane_ids)
            road_waypoints = [wp for wp in road_waypoints if wp.lane_id == rightmost_lane_id]
    
    if not road_waypoints:
        return None
    
    # Find the waypoint with minimum s value (start of road segment)
    start_waypoint = min(road_waypoints, key=lambda wp: wp.s)
    return start_waypoint


def get_connected_roads_path(world_map, start_road_id, max_roads=5, max_distance_per_road=200, spacing=2.0):
    """
    Generate a path that follows connected roads in sequence.
    
    Args:
        world_map: CARLA map object
        start_road_id: Road ID to start from
        max_roads: Maximum number of roads to traverse
        max_distance_per_road: Maximum distance to travel on each road
        spacing: Distance between waypoints
    
    Returns:
        List of waypoints traversing multiple connected roads in order
    """
    all_waypoints = []
    current_road_id = start_road_id
    visited_roads = set()
    
    for road_count in range(max_roads):
        if current_road_id in visited_roads:
            break
            
        # Get starting waypoint for current road
        start_wp = get_road_start_waypoint(world_map, current_road_id)
        if not start_wp:
            print(f"Could not find start waypoint for road {current_road_id}")
            break
        
        print(f"Road {road_count + 1}: Collecting from road {current_road_id}, lane {start_wp.lane_id}")
        
        # Generate path along this road
        road_waypoints = get_continuous_lane_path(start_wp, max_distance_per_road, spacing)
        all_waypoints.extend(road_waypoints)
        visited_roads.add(current_road_id)
        
        if not road_waypoints:
            break
            
        # Find next connected road
        last_wp = road_waypoints[-1]
        next_waypoints = last_wp.next(spacing)
        
        next_road_id = None
        for next_wp in next_waypoints:
            if next_wp.road_id != current_road_id and next_wp.road_id not in visited_roads:
                if next_wp.lane_type == carla.LaneType.Driving and next_wp.lane_id > 0:
                    next_road_id = next_wp.road_id
                    break
        
        if next_road_id is None:
            print(f"No more connected roads found after road {current_road_id}")
            break
            
        current_road_id = next_road_id
    
    print(f"Collected {len(all_waypoints)} waypoints from {len(visited_roads)} roads: {list(visited_roads)}")
    return all_waypoints


def get_multiple_roads_from_spawn_points(world, num_spawn_points=3, max_distance_per_spawn=300, spacing=2.0):
    """
    Collect waypoints from multiple roads by using different spawn points.
    Each spawn point typically starts on a different road.
    
    Args:
        world: CARLA world object
        num_spawn_points: Number of different spawn points to use
        max_distance_per_spawn: Maximum distance to travel from each spawn point
        spacing: Distance between waypoints
    
    Returns:
        List of waypoints from multiple roads
    """
    spawn_points = world.get_map().get_spawn_points()
    world_map = world.get_map()
    all_waypoints = []
    used_roads = set()
    
    spawn_indices = list(range(min(num_spawn_points, len(spawn_points))))
    
    for i, spawn_idx in enumerate(spawn_indices):
        spawn_transform = spawn_points[spawn_idx]
        nearest_waypoint = world_map.get_waypoint(spawn_transform.location)
        
        # Skip if we've already covered this road
        if nearest_waypoint.road_id in used_roads:
            print(f"Spawn {spawn_idx}: Road {nearest_waypoint.road_id} already covered, skipping")
            continue
            
        print(f"Spawn {spawn_idx}: Starting from road {nearest_waypoint.road_id}, lane {nearest_waypoint.lane_id}")
        
        # Generate path from this spawn point
        spawn_waypoints = get_continuous_lane_path(nearest_waypoint, max_distance_per_spawn, spacing)
        
        if spawn_waypoints:
            all_waypoints.extend(spawn_waypoints)
            used_roads.add(nearest_waypoint.road_id)
            
            # Add separator waypoint for smooth transition (optional)
            if i < len(spawn_indices) - 1:  # Not the last spawn point
                all_waypoints.append(None)  # Separator marker
    
    # Remove separator markers
    all_waypoints = [wp for wp in all_waypoints if wp is not None]
    
    print(f"Collected {len(all_waypoints)} waypoints from {len(used_roads)} roads: {list(used_roads)}")
    return all_waypoints


def get_sequential_multiple_roads(world_map, target_road_ids=None, max_distance_per_road=200, spacing=2.0):
    """
    Generate waypoints from multiple specific roads in the order provided.
    
    Args:
        world_map: CARLA map object
        target_road_ids: List of road IDs to traverse in order (None for auto-selection)
        max_distance_per_road: Maximum distance per road
        spacing: Distance between waypoints
    
    Returns:
        List of waypoints from multiple roads in specified order
    """
    if target_road_ids is None:
        # Auto-select roads
        all_road_ids = get_all_road_ids(world_map)
        target_road_ids = all_road_ids[:5]  # Take first 5 roads
    
    all_waypoints = []
    
    for i, road_id in enumerate(target_road_ids):
        start_wp = get_road_start_waypoint(world_map, road_id)
        if not start_wp:
            print(f"Road {i+1}: Could not find start waypoint for road {road_id}, skipping")
            continue
            
        print(f"Road {i+1}: Collecting from road {road_id}, lane {start_wp.lane_id}")
        
        road_waypoints = get_continuous_lane_path(start_wp, max_distance_per_road, spacing)
        all_waypoints.extend(road_waypoints)
    
    print(f"Collected {len(all_waypoints)} waypoints from {len(target_road_ids)} roads: {target_road_ids}")
    return all_waypoints


def get_sequential_lane_path_from_spawn_point(world, spawn_index=0, max_distance=1000, spacing=2.0):
    """
    Generate a sequential path starting from a specific spawn point.
    This ensures a natural driving path from a realistic starting position.
    
    Args:
        world: CARLA world object
        spawn_index: Index of spawn point to start from (default: 0)
        max_distance: Maximum distance to travel
        spacing: Distance between waypoints
    
    Returns:
        List of waypoints in sequential order starting from spawn point
    """
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points or spawn_index >= len(spawn_points):
        print("No valid spawn points found")
        return []
    
    # Get the spawn point and find the nearest waypoint
    spawn_transform = spawn_points[spawn_index]
    world_map = world.get_map()
    nearest_waypoint = world_map.get_waypoint(spawn_transform.location)
    
    print(f"Starting from spawn point {spawn_index} at location ({spawn_transform.location.x:.2f}, {spawn_transform.location.y:.2f})")
    print(f"Nearest waypoint: Road {nearest_waypoint.road_id}, Lane {nearest_waypoint.lane_id}")
    
    # Generate continuous path from this waypoint
    return get_continuous_lane_path(nearest_waypoint, max_distance, spacing)


def main():
    actor_list = []
    pygame.init()
    
    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.load_world("Town02")
    
    try:
        m = world.get_map()
        
        # MULTI-ROAD DATA COLLECTION OPTIONS
        # Choose one of these methods for collecting from multiple roads:
        
        # Method A: Connected roads path (follows road topology)
        # waypoints = get_connected_roads_path(m, start_road_id=10, max_roads=5, max_distance_per_road=200, spacing=2.0)
        
        # Method B: Multiple spawn points (different roads via spawn points)  
        waypoints = get_multiple_roads_from_spawn_points(world, num_spawn_points=5, max_distance_per_spawn=300, spacing=2.0)
        
        # Method C: Specific roads in order (manual road selection)
        # target_roads = [10, 13, 15, 20, 25]  # Specify road IDs you want
        # waypoints = get_sequential_multiple_roads(m, target_road_ids=target_roads, max_distance_per_road=200, spacing=2.0)
        
        # Method D: Auto-select roads in ID order
        # waypoints = get_sequential_multiple_roads(m, target_road_ids=None, max_distance_per_road=200, spacing=2.0)
        
        # SINGLE ROAD OPTIONS (uncomment if you want single road instead):
        
        # Single road from spawn point
        # waypoints = get_sequential_lane_path_from_spawn_point(world, spawn_index=0, max_distance=500, spacing=2.0)
        
        # Single road filtered from all waypoints
        # all_waypoints = m.generate_waypoints(2)
        # waypoints = filter_rightmost_driving_lane(all_waypoints)
        
        print(f"Filtered to {len(waypoints)} waypoints for single lane collection")
        
        if not waypoints:
            print("No waypoints found for single lane. Exiting.")
            return
        
        start_pose = waypoints[0].transform
        start_pose.location.z += 5
        blueprint_library = world.get_blueprint_library()
        vehicle = world.spawn_actor(
            blueprint_library.filter('vehicle.*')[0],
            start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)
        
        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=1.0, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)
        
        camera_semseg = world.spawn_actor(
            blueprint_library.find('sensor.camera.semantic_segmentation'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_semseg)
        
        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_semseg, fps=30) as sync_mode:
            waypoint_index = 0
            for waypoint in waypoints:
                if should_quit():
                    return
                clock.tick()
                
                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=2.0)
                
                # Use the waypoint's rotation directly (maintains lane direction)
                vehicle.set_transform(waypoint.transform)
                
                waypoint_index += 1
                
                image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                
                # Draw the display.
                draw_image(display, image_rgb, vehicle)
                draw_image(display, image_semseg, None, blend=True)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                display.blit(
                    font.render('Waypoint: %d/%d' % (waypoint_index, len(waypoints)), True, (255, 255, 255)),
                    (8, 46))
                
                # Show current road info
                current_road_id = waypoint.road_id if waypoint else "N/A"
                display.blit(
                    font.render('Road ID: %s' % str(current_road_id), True, (255, 255, 255)),
                    (8, 64))
                pygame.display.flip()
    
    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()
        
        pygame.quit()
        print('done.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')