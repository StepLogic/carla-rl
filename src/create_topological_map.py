from datetime import datetime
import glob
import time
import cv2
import numpy as np
from tqdm import tqdm
import os
import pickle
from rlib_integration.agent import BasicAgent
# from train_online_pixels import CarlaGoalEnv,config,FrameStack,TimeLimit,RecordEpisodeStatistics,ReplayBuffer
from jaxrl2.wrappers.frame_stack import FrameStack
from jaxrl2.wrappers.timelimit import TimeLimit
from jaxrl2.wrappers.record_statistics import RecordEpisodeStatistics
from rlib_integration.helper import carla_location_to_np_array
from carla_eval import CarlaEvalEnv
from PIL import Image
from jaxrl2.noise import OrnsteinUhlenbeckActionNoise
# from jax_mapping_experiment import JAXMappingExperiments
from carla_eval import CarlaEvalEnv

#!/usr/bin/env python

import carla
import random
import numpy as np
import networkx as nx
from collections import deque
import time

# Import the BasicAgent class from the provided code
# This assumes the BasicAgent class is defined in a file named 'agents.py'
# from agents import BasicAgent, GlobalRoutePlanner, RoadOption
# Since we're extending the provided code, we'll reuse the classes directly
#!/usr/bin/env python

import carla
import random
import numpy as np
import networkx as nx
from collections import deque
import time

# Import the BasicAgent class from the provided code
# This assumes the BasicAgent class is defined in a file named 'agents.py'
# from agents import BasicAgent, GlobalRoutePlanner, RoadOption
# Since we're extending the provided code, we'll reuse the classes directly

class JunctionTrajectoryAgent(BasicAgent):
    """
    An agent that travels along a random trajectory with a minimum number of intersections
    but limited to a maximum distance.
    Extends BasicAgent with functionality to find and follow paths containing at least a certain
    number of junctions while staying within a maximum distance constraint.
    """

    def __init__(self, vehicle, target_speed=20, opt_dict={}, map_inst=None, grp_inst=None, 
                 min_junctions=3, max_distance=2000):
        """
        Initialize the agent with a specific minimum number of junctions and maximum distance.
        
        Args:
            vehicle: The vehicle actor to control
            target_speed: Speed in km/h
            opt_dict: Dictionary with options for the BasicAgent
            map_inst: CARLA map instance
            grp_inst: GlobalRoutePlanner instance
            min_junctions: Minimum number of junctions the random trajectory should contain
            max_distance: Maximum distance (in meters) the trajectory should cover
        """
        super(JunctionTrajectoryAgent, self).__init__(vehicle, target_speed, opt_dict, map_inst, grp_inst)
        
        self._min_junctions = min_junctions
        self._max_distance = max_distance
        self._junction_waypoints = []  # List of waypoints at junctions
        self._junction_ids = set()     # Set of unique junction IDs
        self._current_route = []       # Current route being followed
        self._debug = self._world.debug  # Debug helper for visualization
        
        # Initialize with default colors for visualization
        self._route_color = carla.Color(0, 255, 0)  # Green
        self._junction_color = carla.Color(255, 0, 0)  # Red
        
    def find_random_junction_trajectory(self, search_attempts=10, visualize=True):
        """
        Find a random trajectory that contains at least the specified number of junctions
        and stays within the maximum distance limit.
        
        Args:
            search_attempts: Maximum number of attempts to find a suitable path
            visualize: Whether to visualize the found path
            
        Returns:
            True if a valid path was found, False otherwise
        """
        print(f"Searching for a random trajectory with at least {self._min_junctions} junctions "
              f"and maximum distance of {self._max_distance}m...")
        
        # First, identify all junctions in the map if we haven't already
        if not self._junction_waypoints:
            self._find_all_junctions()
            
        if len(self._junction_ids) < self._min_junctions:
            print(f"Warning: Map only contains {len(self._junction_ids)} junctions, less than requested {self._min_junctions}")
            return False
            
        # Try to find a valid path with the required number of junctions
        for attempt in range(search_attempts):
            print(f"Attempt {attempt+1}/{search_attempts} to find trajectory")
            
            # Get current vehicle location as start point
            start_loc = self._vehicle.get_location()
            start_wp = self._map.get_waypoint(start_loc)
            
            # Find a valid path with the minimum number of junctions and within distance limit
            route = self._find_path_with_min_junctions(start_wp)
            
            if route and len(route) > 0:
                # Calculate the total route distance
                route_distance = self._calculate_route_distance(route)
                junction_count = self._count_junctions_in_route(route)
                
                if (junction_count >= self._min_junctions and junction_count<=self._min_junctions+2):
                    print(f"Found valid trajectory with {junction_count} junctions and distance of {route_distance:.2f}m")
                    
                    # Set the found route as the global plan
                    self.set_global_plan(route, clean_queue=True)
                    self._current_route = route
                    
                    # Visualize the route if requested
                    if visualize:
                        self._visualize_route(route)
                        
                    return True
                elif(junction_count >= self._min_junctions):
                    print(f"Found valid trajectory with {junction_count} junctions and distance of {route_distance:.2f}m")
                    
                    # Set the found route as the global plan
                    self.set_global_plan(route, clean_queue=True)
                    self._current_route = route
                    
                    # Visualize the route if requested
                    if visualize:
                        self._visualize_route(route)
                        
                    return True
                else:
                    reason = []
                    if junction_count < self._min_junctions:
                        reason.append(f"too few junctions ({junction_count} < {self._min_junctions})")
                    if route_distance > self._max_distance:
                        reason.append(f"too long ({route_distance:.2f}m > {self._max_distance}m)")
                    
                    print(f"Route rejected: {', '.join(reason)}")
        
        print(f"Failed to find a trajectory with the required constraints after {search_attempts} attempts")
        return False
    
    def _find_all_junctions(self):
        """Identify all junctions in the map"""
        print("Finding all junctions in the map...")
        
        # Get all waypoints in the map
        waypoint_list = self._map.generate_waypoints(2.0)
        
        # Filter for junction waypoints
        junction_waypoints = [wp for wp in waypoint_list if wp.is_junction]
        
        # Store unique junctions
        for wp in junction_waypoints:
            junction = wp.get_junction()
            if junction and junction.id not in self._junction_ids:
                self._junction_ids.add(junction.id)
                self._junction_waypoints.append(wp)
                
        print(f"Found {len(self._junction_ids)} unique junctions in the map")
    
    def _find_path_with_min_junctions(self, start_waypoint):
        """
        Find a path from a start waypoint that includes at least min_junctions intersections
        and stays within the maximum distance constraint.
        
        Args:
            start_waypoint: Starting waypoint for the path
            
        Returns:
            List of (waypoint, RoadOption) tuples representing the path
        """
        # Choose a random end point from junction waypoints
        if not self._junction_waypoints:
            print("No junction waypoints found. Cannot create path.")
            return []
        
        # Try to find a path with enough junctions
        max_attempts = 20
        for _ in range(max_attempts):
            # Randomly select a destination junction that's a reasonable distance away
            potential_destinations = []
            for wp in self._junction_waypoints:
                dist = wp.transform.location.distance(start_waypoint.transform.location)
                # Filter for junctions that are not too close and not too far
                # Use 0.7 * max_distance as a heuristic since route distance is typically
                # longer than direct distance
                if 50 < dist < self._max_distance:
                    potential_destinations.append(wp)
            
            if not potential_destinations:
                print("No suitable destination junctions found within distance constraints")
                # Use a fallback approach with closer destinations
                for wp in self._junction_waypoints:
                    dist = wp.transform.location.distance(start_waypoint.transform.location)
                    if 50 < dist:
                        potential_destinations.append(wp)
                
                if not potential_destinations:
                    print("No suitable destination junctions found at all")
                    continue
            
            # Select a random destination from the potential destinations
            end_waypoint = random.choice(potential_destinations)
            try:
            # Trace route between start and end waypoints
                route = self.trace_route(start_waypoint, end_waypoint)
                junction_count = self._count_junctions_in_route(route)
                route_distance = self._calculate_route_distance(route)
                print(f"Potential route found: {junction_count} junctions, {route_distance:.2f}m distance")
                # Return the route for further evaluation
                return route
            except:
                # return self._find_path_with_min_junctions(start_waypoint)
                return []
            # Check if the route has enough junctions
            
                
        print("Could not find a path with potential to meet requirements")
        return []
    
    def _count_junctions_in_route(self, route):
        """
        Count the number of unique junctions in a route
        
        Args:
            route: List of (waypoint, RoadOption) tuples
            
        Returns:
            Number of unique junctions in the route
        """
        junction_ids = set()
        
        for waypoint, _ in route:
            if waypoint.is_junction:
                junction = waypoint.get_junction()
                if junction:
                    junction_ids.add(junction.id)
                    
        return len(junction_ids)
    
    def _visualize_route(self, route, duration=10.0):
        """
        Visualize the route with debug helpers
        
        Args:
            route: List of (waypoint, RoadOption) tuples
            duration: How long the visualization should last (in seconds)
        """
        # Draw line along the route
        prev_loc = None
        junction_ids_shown = set()
        
        for i, (waypoint, road_option) in enumerate(route):
            loc = waypoint.transform.location
            
            # Draw point for each waypoint
            self._debug.draw_point(
                loc + carla.Location(z=0.5), 
                size=0.1, 
                color=self._route_color, 
                life_time=duration
            )
            
            # Draw line connecting waypoints
            if prev_loc:
                self._debug.draw_line(
                    prev_loc + carla.Location(z=0.5),
                    loc + carla.Location(z=0.5),
                    thickness=0.1,
                    color=self._route_color,
                    life_time=duration
                )
            
            # Highlight junctions
            if waypoint.is_junction:
                junction = waypoint.get_junction()
                junction_id = junction.id
                
                # Only visualize each junction once
                if junction_id not in junction_ids_shown:
                    junction_ids_shown.add(junction_id)
                    
                    # Draw box around junction
                    bbox = junction.bounding_box
                    self._debug.draw_box(
                        carla.BoundingBox(bbox.location, bbox.extent),
                        bbox.rotation, 
                        thickness=0.5,
                        color=self._junction_color,
                        life_time=duration
                    )
                    
                    # Label the junction
                    self._debug.draw_string(
                        bbox.location + carla.Location(z=bbox.extent.z + 1.0),
                        f"Junction {junction_id}",
                        color=self._junction_color,
                        life_time=duration
                    )
            
            prev_loc = loc
        
        # Show start and end points
        if route:
            # Start point
            start_loc = route[0][0].transform.location
            self._debug.draw_point(
                start_loc + carla.Location(z=2.0),
                size=0.2,
                color=carla.Color(0, 255, 0),  # Green
                life_time=duration
            )
            self._debug.draw_string(
                start_loc + carla.Location(z=2.5),
                "START",
                color=carla.Color(0, 255, 0),
                life_time=duration
            )
            
            # End point
            end_loc = route[-1][0].transform.location
            self._debug.draw_point(
                end_loc + carla.Location(z=2.0),
                size=0.2,
                color=carla.Color(255, 0, 0),  # Red
                life_time=duration
            )
            self._debug.draw_string(
                end_loc + carla.Location(z=2.5),
                "END",
                color=carla.Color(255, 0, 0),
                life_time=duration
            )
            
        print(f"Visualizing route with {len(junction_ids_shown)} junctions for {duration} seconds")
    
    def _calculate_route_distance(self, route):
        """
        Calculate the total distance of a route.
        
        Args:
            route: List of (waypoint, RoadOption) tuples
            
        Returns:
            Total distance in meters
        """
        total_distance = 0.0
        prev_loc = None
        
        for waypoint, _ in route:
            loc = waypoint.transform.location
            
            if prev_loc:
                total_distance += loc.distance(prev_loc)
                
            prev_loc = loc
            
        return total_distance
        
    def get_route_info(self):
        """
        Get information about the current route.
        
        Returns:
            Dictionary with route information (junction count, distance)
        """
        if not self._current_route:
            return {
                "junction_count": 0,
                "distance": 0,
                "has_route": False
            }
            
        return {
            "junction_count": self._count_junctions_in_route(self._current_route),
            "distance": self._calculate_route_distance(self._current_route),
            "has_route": True
        }


def collect_basic_agent_data(origin,destination,difficulty="easy"):
    
    env = CarlaEvalEnv(use_rgb=True,town="Town01",start_server=True,max_dist=100,image_size=96)
    env = FrameStack(env=env, num_stack=1, stacking_key="pixels")
    env = FrameStack(env=env, num_stack=1, stacking_key="goal")
    # env = TimeLimit(env, max_episode_steps=2500)
    env = RecordEpisodeStatistics(env)
    env.unwrapped.set_start_transform(origin)
    env.unwrapped.set_destination_transform(destination)
    locations=[]
#     env.unwrapped.core.world.debug.draw_point(
#     destination,
#     size=0.2,
#     color=carla.Color(255, 0, 0),
#     life_time=0
# )

#     env.unwrapped.core.world.debug.draw_point(
#     origin,
#     size=0.2,
#     color=carla.Color(0, 255, 0),
#     life_time=0
# )

    # Main collection loop
    observation, info, done = *env.reset(), False
    collection_start_time = time.time()
    start_location = env.unwrapped.core.hero.get_transform().location
    # destination = env.unwrapped.core.destination
    # Initialize BasicAgent
    agent = BasicAgent(env.unwrapped.core.hero, target_speed=5.0)
    # agent = JunctionTrajectoryAgent(
    #         env.unwrapped.core.hero, 
    #         target_speed=30,
    #         min_junctions=min_junction,
    #         max_distance=max_dist
    #         )
    agent.set_destination(destination)
    agent.ignore_traffic_lights(True)
    agent.ignore_stop_signs(True)
    # data=[]
    # dataset_folder = os.path.join("topomap")

    step=0

    path=f"evaluation_trajectory/{difficulty}/"
    os.makedirs(path, exist_ok=True)
    map_dir=path+str(len(os.listdir(path)))  
    os.makedirs(map_dir, exist_ok=True)  
    # for i in tqdm(range(1, replay_buffer_size + 10)):
    # while not agent.find_random_junction_trajectory(search_attempts=100, visualize=False):
    #     print("Planning")
    
    while not done:
            obs=(observation["pixels"][...,0]*255).astype(np.uint8)
            Image.fromarray(obs).save(f"{map_dir}/{step}.jpg")
            control = agent.run_step()
            action = np.array([control.steer, control.throttle])
            next_observation, reward, done, truncated, info = env.step(action)
            done=done or agent.done()
            observation = next_observation
            step+=1
            locations.append(carla_location_to_np_array(env.unwrapped.core.hero.get_transform().location))
    # Save final buffer
    with open(f"{map_dir}/aux.pkl", "wb") as f:
        pickle.dump(dict(start=carla_location_to_np_array(start_location),
                         goal=carla_location_to_np_array(env.unwrapped.core.hero.get_transform().location),
                         mean_distance_per_step=info.get("mean_distance_per_step",0.0),
                         locations=locations,
                         town="Town02"
                         ),f)
    collection_duration = time.time() - collection_start_time
    print(f"\nData collection completed in {collection_duration/3600:.2f} hours Distance Completed {info.get('distance_completed',0.0)}")
    print(f"Final dataset saved to:")

if __name__ == "__main__":
    user_defined_trajectories=None
    with open("src/user_defined_trajectories.pkl", 'rb') as handle:
            user_defined_trajectories=pickle.load(handle)
    

    if user_defined_trajectories:
       for difficult,trajectories in user_defined_trajectories.items():
           for trajectory in trajectories.values():
               location=trajectory[0]["location"]
               origin=carla.Location(x=location[0],y=location[1],z=location[2])
               location=trajectory[-1]["location"]
               destination=carla.Location(x=location[0],y=location[1],z=location[2])
               collect_basic_agent_data(origin,destination,difficulty=difficult)
