#!/usr/bin/env python

import carla
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

def count_and_visualize_intersections(world, visualize=True, save_map=True):
    """
    Count all intersections in the CARLA world and optionally visualize them.
    
    Args:
        world: CARLA world object
        visualize: Whether to visualize the intersections in the simulator
        save_map: Whether to save a matplotlib visualization of the map
        
    Returns:
        Number of intersections and a dictionary of intersections with details
    """
    # Get the map
    carla_map = world.get_map()
    
    # Get all waypoints
    waypoint_list = carla_map.generate_waypoints(2.0)
    
    # Find all waypoints that are in junctions
    junction_waypoints = [wp for wp in waypoint_list if wp.is_junction]
    
    # Create a set of unique junction IDs
    unique_junction_ids = set()
    intersections = {}
    
    for waypoint in junction_waypoints:
        junction = waypoint.get_junction()
        junction_id = junction.id
        
        if junction_id not in unique_junction_ids:
            unique_junction_ids.add(junction_id)
            
            # Store junction details
            bbox = junction.bounding_box
            intersections[junction_id] = {
                'id': junction_id,
                'location': (bbox.location.x, bbox.location.y, bbox.location.z),
                'bbox': {
                    'location': (bbox.location.x, bbox.location.y, bbox.location.z),
                    'extent': (bbox.extent.x, bbox.extent.y, bbox.extent.z),
                    'rotation': (bbox.rotation.pitch, bbox.rotation.yaw, bbox.rotation.roll)
                },
                'waypoint': waypoint
            }
            
            # Visualize the junction in the simulator
            if visualize:
                debug = world.debug
                
                # Choose a random color for this junction
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                color = carla.Color(r, g, b)
                
                # Draw junction bounding box
                debug.draw_box(
                    carla.BoundingBox(bbox.location, bbox.extent),
                    bbox.rotation,
                    thickness=0.5,
                    color=color,
                    life_time=0.0  # Persist until destroyed
                )
                
                # Label the junction
                debug.draw_string(
                    bbox.location + carla.Location(z=bbox.extent.z + 1.0),
                    f"Junction {junction_id}",
                    color=color,
                    life_time=0.0
                )
    
    intersection_count = len(unique_junction_ids)
    print(f"Found {intersection_count} unique intersections/junctions.")
    
    # Save a visualization of the map with intersections
    if save_map:
        plot_intersections_on_map(carla_map, intersections)
    
    return intersection_count, intersections

def plot_intersections_on_map(carla_map, intersections):
    """
    Plot a top-down view of the map with intersections highlighted.
    
    Args:
        carla_map: CARLA map object
        intersections: Dictionary of intersection information
    """
    # Get the map bounds by using topology or waypoints
    waypoints = carla_map.generate_waypoints(2.0)
    
    # Extract coordinates from waypoints
    x_coords = [wp.transform.location.x for wp in waypoints]
    y_coords = [wp.transform.location.y for wp in waypoints]
    
    # Calculate map bounds
    min_x = min(x_coords) if x_coords else -500
    max_x = max(x_coords) if x_coords else 500
    min_y = min(y_coords) if y_coords else -500
    max_y = max(y_coords) if y_coords else 500
    
    # Create figure
    plt.figure(figsize=(12, 12))
    
    # Plot all topology (road network)
    topology = carla_map.get_topology()
    for wp1, wp2 in topology:
        x1, y1 = wp1.transform.location.x, wp1.transform.location.y
        x2, y2 = wp2.transform.location.x, wp2.transform.location.y
        plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=0.5)
    
    # Plot all intersections
    for junction_id, junction_data in intersections.items():
        # Get bounding box info
        bbox_loc = junction_data['bbox']['location']
        bbox_extent = junction_data['bbox']['extent']
        
        # Create rectangle for junction
        x = bbox_loc[0] - bbox_extent[0]
        y = bbox_loc[1] - bbox_extent[1]
        width = bbox_extent[0] * 2
        height = bbox_extent[1] * 2
        
        # Rotation angle (yaw)
        angle = junction_data['bbox']['rotation'][1]
        
        # Create and add rectangle
        rect = Rectangle((x, y), width, height, angle=angle, 
                         facecolor='red', alpha=0.3, edgecolor='red')
        plt.gca().add_patch(rect)
        
        # Add junction ID
        plt.text(bbox_loc[0], bbox_loc[1], str(junction_id),
                 color='white', ha='center', va='center', fontsize=8)
    
    # Set limits based on map bounds
    plt.xlim(min_x - 50, max_x + 50)
    plt.ylim(min_y - 50, max_y + 50)
    
    # Add town name as title
    map_name = carla_map.name
    plt.title(f"Intersections in {map_name}")
    
    # Add legend and grid
    plt.grid(True, alpha=0.3)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    
    # Add count of intersections
    plt.figtext(0.5, 0.01, f"Total intersections: {len(intersections)}", 
                ha="center", fontsize=12)
    
    # Save the figure
    plt.savefig(f"intersections_test.png", dpi=300, bbox_inches='tight')
    print(f"Saved map visualization to intersections_{map_name}.png")
    
def get_intersection_analysis(world):
    """
    Perform a detailed analysis of intersections in the CARLA world.
    
    Args:
        world: CARLA world object
        
    Returns:
        Analysis statistics
    """
    # Get the map
    carla_map = world.get_map()
    
    # Get count and intersections data
    count, intersections = count_and_visualize_intersections(world, visualize=False, save_map=False)
    
    # Analyze size distribution
    sizes = []
    for junction_id, junction_data in intersections.items():
        bbox_extent = junction_data['bbox']['extent']
        # Calculate area
        area = bbox_extent[0] * 2 * bbox_extent[1] * 2
        sizes.append(area)
    
    # Calculate size statistics
    avg_size = np.mean(sizes)
    median_size = np.median(sizes)
    min_size = np.min(sizes)
    max_size = np.max(sizes)
    
    # Analyze spatial distribution
    x_coords = [data['location'][0] for data in intersections.values()]
    y_coords = [data['location'][1] for data in intersections.values()]
    
    # Calculate nearest neighbor distances
    distances = []
    for i, junction_id in enumerate(intersections.keys()):
        loc_i = np.array([x_coords[i], y_coords[i]])
        nearest_dist = float('inf')
        
        for j, other_id in enumerate(intersections.keys()):
            if i != j:
                loc_j = np.array([x_coords[j], y_coords[j]])
                dist = np.linalg.norm(loc_i - loc_j)
                nearest_dist = min(nearest_dist, dist)
        
        if nearest_dist < float('inf'):
            distances.append(nearest_dist)
    
    # Calculate distance statistics
    avg_distance = np.mean(distances) if distances else 0
    median_distance = np.median(distances) if distances else 0
    min_distance = np.min(distances) if distances else 0
    max_distance = np.max(distances) if distances else 0
    
    # Analyze connections per intersection
    connections = {}
    
    for junction_id, junction_data in intersections.items():
        wp = junction_data['waypoint']
        junction = wp.get_junction()
        
        # Get all waypoints in this junction
        junction_wps = junction.get_waypoints(carla.LaneType.Driving)
        connections[junction_id] = len(junction_wps)
    
    # Calculate connection statistics
    avg_connections = np.mean(list(connections.values())) if connections else 0
    max_connections = np.max(list(connections.values())) if connections else 0
    min_connections = np.min(list(connections.values())) if connections else 0
    
    # Compile results
    analysis = {
        'total_intersections': count,
        'map_name': carla_map.name,
        'size_statistics': {
            'average_area_m2': avg_size,
            'median_area_m2': median_size,
            'min_area_m2': min_size,
            'max_area_m2': max_size
        },
        'spatial_statistics': {
            'average_nearest_neighbor_distance_m': avg_distance,
            'median_nearest_neighbor_distance_m': median_distance,
            'min_nearest_neighbor_distance_m': min_distance,
            'max_nearest_neighbor_distance_m': max_distance
        },
        'connection_statistics': {
            'average_connections': avg_connections,
            'min_connections': min_connections,
            'max_connections': max_connections
        }
    }
    
    return analysis

def print_analysis(analysis):
    """Print intersection analysis in a readable format."""
    print("\n=== INTERSECTION ANALYSIS ===")
    print(f"Map: {analysis['map_name']}")
    print(f"Total Intersections: {analysis['total_intersections']}")
    
    print("\nSize Statistics:")
    print(f"  Average Area: {analysis['size_statistics']['average_area_m2']:.2f} m²")
    print(f"  Median Area: {analysis['size_statistics']['median_area_m2']:.2f} m²")
    print(f"  Size Range: {analysis['size_statistics']['min_area_m2']:.2f} - {analysis['size_statistics']['max_area_m2']:.2f} m²")
    
    print("\nSpatial Distribution:")
    print(f"  Average Distance Between Intersections: {analysis['spatial_statistics']['average_nearest_neighbor_distance_m']:.2f} m")
    print(f"  Median Distance: {analysis['spatial_statistics']['median_nearest_neighbor_distance_m']:.2f} m")
    print(f"  Distance Range: {analysis['spatial_statistics']['min_nearest_neighbor_distance_m']:.2f} - {analysis['spatial_statistics']['max_nearest_neighbor_distance_m']:.2f} m")
    
    print("\nConnection Statistics:")
    print(f"  Average Connections per Intersection: {analysis['connection_statistics']['average_connections']:.2f}")
    print(f"  Connection Range: {analysis['connection_statistics']['min_connections']} - {analysis['connection_statistics']['max_connections']}")
    print("============================\n")

def main():
    argparser = argparse.ArgumentParser(description='CARLA Intersection Counter')
    argparser.add_argument('--host', default='localhost', help='IP of the CARLA server')
    argparser.add_argument('--port', default=2000, type=int, help='Port of the CARLA server')
    argparser.add_argument('--timeout', default=2.0, type=float, help='Timeout for connection')
    argparser.add_argument('--no-visual', action='store_true', help='Disable visualization in CARLA')
    argparser.add_argument('--no-map', action='store_true', help='Disable map visualization')
    argparser.add_argument('--analysis', action='store_true', help='Perform detailed analysis of intersections')
    args = argparser.parse_args()

    try:
        # Connect to client
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        
        # Get world
        world = client.get_world()
        
        if args.analysis:
            # Perform detailed analysis
            analysis = get_intersection_analysis(world)
            print_analysis(analysis)
        else:
            # Basic count and visualization
            count_and_visualize_intersections(
                world, 
                visualize=not args.no_visual,
                save_map=not args.no_map
            )
            
        # Keep the script running to maintain visualization
        if not args.no_visual and not args.analysis:
            print("Press Ctrl+C to exit...")
            while True:
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print('\nCancelled by user.')
    except Exception as e:
        print(f'Error: {e}')
    finally:
        print('Done.')

if __name__ == '__main__':
    main()