#!/usr/bin/env python

import glob
import os
import queue
import random
import sys
import math
import time
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from enum import IntEnum
import carla
import pygame
import pickle
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

        self._intersection_end_node = -1
        self._previous_decision = RoadOption.VOID

        # Build the graph
        self._build_topology()
        self._build_graph()
        self._find_loose_ends()
        # self._lane_change_link()

    def trace_route(self):
        """
        This method returns list of (carla.Waypoint, RoadOption)
        from origin to destination
        """
        route_trace = []
        # path=[]
        route = self._path_search()
        # print(route)
        # current_waypoint = self._wmap.get_waypoint(origin)
        # destination_waypoint = self._wmap.get_waypoint(destination)

        for i in range(len(route) - 1):
            edge = self._graph.edges[route[i], route[i + 1]]
            # print(edge)
            # path =[[edge["entry_waypoint"]] + edge["path"] + [edge["exit_waypoint"]]]
            # closest_index = self._find_closest_in_list(path)
            route_trace.append(edge["entry_waypoint"])
            route_trace.append(edge["exit_waypoint"])
            for waypoint in edge["path"]:
                # current_waypoint = waypoint
                route_trace.append(waypoint)
            

        return route_trace

    def _build_topology(self):
        """
        This function retrieves topology from the server as a list of
        road segments as pairs of waypoint objects, and processes the
        topology into a list of dictionary objects with the following attributes

        - entry (carla.Waypoint): waypoint of entry point of road segment
        - entryxyz (tuple): (x,y,z) of entry point of road segment
        - exit (carla.Waypoint): waypoint of exit point of road segment
        - exitxyz (tuple): (x,y,z) of exit point of road segment
        - path (list of carla.Waypoint):  list of waypoints between entry to exit, separated by the resolution
        """
        self._topology = []
        # Retrieving waypoints to construct a detailed topology
        for segment in self._wmap.get_topology():
            wp1, wp2 = segment[0], segment[1]
            # if wp1.lane_id!=-1 or wp2.lane_id!=-1:
            #     continue

            l1, l2 = wp1.transform.location, wp2.transform.location
            # Rounding off to avoid floating point imprecision
            x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
            wp1.transform.location, wp2.transform.location = l1, l2
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
        # print(self._topology)
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
                road_id=road_id,
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

    def _lane_change_link(self):
        """
        This method places zero cost links in the topology graph
        representing availability of lane changes.
        """
        for segment in self._topology:
            left_found, right_found = False, False

            for waypoint in segment["path"]:
                if not segment["entry"].is_junction:
                    next_waypoint, next_road_option, next_segment = None, None, None

                    if (
                        waypoint.right_lane_marking
                        and waypoint.right_lane_marking.lane_change
                        & carla.LaneChange.Right
                        and not right_found
                    ):
                        next_waypoint = waypoint.get_right_lane()
                        if (
                            next_waypoint is not None
                            and next_waypoint.lane_type == carla.LaneType.Driving
                            and waypoint.road_id == next_waypoint.road_id
                        ):
                            next_road_option = RoadOption.CHANGELANERIGHT
                            next_segment = self._localize(
                                next_waypoint.transform.location
                            )
                            if next_segment is not None:
                                self._graph.add_edge(
                                    self._id_map[segment["entryxyz"]],
                                    next_segment[0],
                                    entry_waypoint=waypoint,
                                    exit_waypoint=next_waypoint,
                                    intersection=False,
                                    exit_vector=None,
                                    path=[],
                                    length=0,
                                    type=next_road_option,
                                    change_waypoint=next_waypoint,
                                )
                                right_found = True
                    if (
                        waypoint.left_lane_marking
                        and waypoint.left_lane_marking.lane_change
                        & carla.LaneChange.Left
                        and not left_found
                    ):
                        next_waypoint = waypoint.get_left_lane()
                        if (
                            next_waypoint is not None
                            and next_waypoint.lane_type == carla.LaneType.Driving
                            and waypoint.road_id == next_waypoint.road_id
                        ):
                            next_road_option = RoadOption.CHANGELANELEFT
                            next_segment = self._localize(
                                next_waypoint.transform.location
                            )
                            if next_segment is not None:
                                self._graph.add_edge(
                                    self._id_map[segment["entryxyz"]],
                                    next_segment[0],
                                    entry_waypoint=waypoint,
                                    exit_waypoint=next_waypoint,
                                    intersection=False,
                                    exit_vector=None,
                                    path=[],
                                    length=0,
                                    type=next_road_option,
                                    change_waypoint=next_waypoint,
                                )
                                left_found = True
                if left_found and right_found:
                    break

    def _localize(self, location):
        """
        This function finds the road segment that a given location
        is part of, returning the edge it belongs to
        """
        waypoint = self._wmap.get_waypoint(location)
        edge = None
        try:
            edge = self._road_id_to_edge[waypoint.road_id][waypoint.section_id][
                waypoint.lane_id
            ]
        except KeyError:
            pass
        return edge

    def _distance_heuristic(self, n1, n2):
        """
        Distance heuristic calculator for path searching
        """
        l1 = np.array(self._graph.nodes[n1]["vertex"])
        l2 = np.array(self._graph.nodes[n2]["vertex"])
        return 1/(np.linalg.norm(l1 - l2)+1e-8)

    def _path_search(self):
        """
        This function finds the shortest path connecting origin and destination
        """
        # start, end = self._localize(origin), self._localize(destination)
        # if start is None or end is None:
        #     return []
        start=1
        # route = nx.astar_path(
        #     self._graph,
        #     source=start[0],
        #     # target=end[0],
        #     heuristic=self._distance_heuristic,
        #     weight="length",
        # )
        
        # route = nx.find_(
        #     self._graph,
        #     source=start,
        #     orientation="original"
        #     # target=end[0],
        #     # heuristic=self._distance_heuristic,1
        #     # weight="length",
        # )
        # breakpoint()
        # route.append(end[1])
        nodes_from_edges = []

        # # Iterate through the edges and add each node to the list
        # for u, v,direc in route:
        #     if not u in nodes_from_edges:
        #         nodes_from_edges.append(u)
        #     if not  v in nodes_from_edges:
        #         nodes_from_edges.append(v)
        # # nodes_from_edges=set(nodes_from_edges)
        # # breakpoint()
        # return nodes_from_edges
        paths = list(nx.all_simple_paths(self._graph, 0, 43))

        # Find the longest path
        return max(paths, key=len)
        

    def _successive_last_intersection_edge(self, index, route):
        """
        This method returns the last successive intersection edge
        from a starting index on the route.
        This helps moving past tiny intersection edges to calculate
        proper turn decisions.
        """

        last_intersection_edge = None
        last_node = None
        for node1, node2 in [
            (route[i], route[i + 1]) for i in range(index, len(route) - 1)
        ]:
            candidate_edge = self._graph.edges[node1, node2]
            if node1 == route[index]:
                last_intersection_edge = candidate_edge
            if (
                candidate_edge["type"] == RoadOption.LANEFOLLOW
                and candidate_edge["intersection"]
            ):
                last_intersection_edge = candidate_edge
                last_node = node2
            else:
                break

        return last_node, last_intersection_edge

    def _turn_decision(self, index, route, threshold=math.radians(35)):
        """
        This method returns the turn decision (RoadOption) for pair of edges
        around current index of route list
        """

        decision = None
        previous_node = route[index - 1]
        current_node = route[index]
        next_node = route[index + 1]
        next_edge = self._graph.edges[current_node, next_node]
        if index > 0:
            if (
                self._previous_decision != RoadOption.VOID
                and self._intersection_end_node > 0
                and self._intersection_end_node != previous_node
                and next_edge["type"] == RoadOption.LANEFOLLOW
                and next_edge["intersection"]
            ):
                decision = self._previous_decision
            else:
                self._intersection_end_node = -1
                current_edge = self._graph.edges[previous_node, current_node]
                calculate_turn = (
                    current_edge["type"] == RoadOption.LANEFOLLOW
                    and not current_edge["intersection"]
                    and next_edge["type"] == RoadOption.LANEFOLLOW
                    and next_edge["intersection"]
                )
                if calculate_turn:
                    last_node, tail_edge = self._successive_last_intersection_edge(
                        index, route
                    )
                    self._intersection_end_node = last_node
                    if tail_edge is not None:
                        next_edge = tail_edge
                    cv, nv = current_edge["exit_vector"], next_edge["exit_vector"]
                    if cv is None or nv is None:
                        return next_edge["type"]
                    cross_list = []
                    for neighbor in self._graph.successors(current_node):
                        select_edge = self._graph.edges[current_node, neighbor]
                        if select_edge["type"] == RoadOption.LANEFOLLOW:
                            if neighbor != route[index + 1]:
                                sv = select_edge["net_vector"]
                                cross_list.append(np.cross(cv, sv)[2])
                    next_cross = np.cross(cv, nv)[2]
                    deviation = math.acos(
                        np.clip(
                            np.dot(cv, nv) / (np.linalg.norm(cv) * np.linalg.norm(nv)),
                            -1.0,
                            1.0,
                        )
                    )
                    if not cross_list:
                        cross_list.append(0)
                    if deviation < threshold:
                        decision = RoadOption.STRAIGHT
                    elif cross_list and next_cross < min(cross_list):
                        decision = RoadOption.LEFT
                    elif cross_list and next_cross > max(cross_list):
                        decision = RoadOption.RIGHT
                    elif next_cross < 0:
                        decision = RoadOption.LEFT
                    elif next_cross > 0:
                        decision = RoadOption.RIGHT
                else:
                    decision = next_edge["type"]

        else:
            decision = next_edge["type"]

        self._previous_decision = decision
        return decision
    def _find_closest_in_list(self, current_waypoint, waypoint_list):
        min_distance = float("inf")
        closest_index = -1
        for i, waypoint in enumerate(waypoint_list):
            distance = waypoint.transform.location.distance(
                current_waypoint.transform.location
            )
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        return closest_index

    def visualize_graph(self, paths=None, save_path="road_network_graph.png"):
        """
        Visualize the road network graph using matplotlib with node labels and optional paths
        """
        plt.figure(figsize=(20, 20))
        
        # Get node positions
        pos = {}
        for node in self._graph.nodes():
            vertex = self._graph.nodes[node]["vertex"]
            pos[node] = (vertex[0], vertex[1])
        
        # Draw nodes with different colors for intersections
        node_colors = []
        for node in self._graph.nodes():
            # Check if this node is part of an intersection
            is_intersection = False
            for edge in self._graph.edges(node, data=True):
                if edge[2].get("intersection", False):
                    is_intersection = True
                    break
            node_colors.append('red' if is_intersection else 'blue')
        
        # Draw the graph
        nx.draw_networkx_nodes(self._graph, pos, node_size=50, node_color=node_colors, alpha=0.7)
        
        # Draw edges with different styles based on type
        solid_edges = [(u, v) for u, v, data in self._graph.edges(data=True) 
                    if data.get("type", RoadOption.LANEFOLLOW) == RoadOption.LANEFOLLOW]
        dashed_edges = [(u, v) for u, v, data in self._graph.edges(data=True) 
                    if data.get("type") in [RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT]]
        
        nx.draw_networkx_edges(self._graph, pos, edgelist=solid_edges, edge_color='black', 
                            style='-', width=1, alpha=0.7)
        nx.draw_networkx_edges(self._graph, pos, edgelist=dashed_edges, edge_color='green', 
                            style='--', width=2, alpha=0.7)
        
        # Highlight paths if provided
        if paths:
            if not isinstance(paths, list):
                paths = [paths]  # Convert single path to list
            
            path_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            
            for i, path in enumerate(paths):
                if not path:
                    continue
                    
                path_color = path_colors[i % len(path_colors)]
                
                # Draw path nodes
                path_nodes = set(path)
                nx.draw_networkx_nodes(self._graph, pos, nodelist=list(path_nodes), 
                                    node_size=100, node_color=path_color, alpha=0.9)
                
                # Draw path edges
                path_edges = []
                for j in range(len(path) - 1):
                    if self._graph.has_edge(path[j], path[j + 1]):
                        path_edges.append((path[j], path[j + 1]))
                
                if path_edges:
                    nx.draw_networkx_edges(self._graph, pos, edgelist=path_edges, 
                                        edge_color=path_color, style='-', width=3, alpha=0.9)
                
                # Add path number labels
                for node in path:
                    if node in pos:
                        x, y = pos[node]
                        plt.text(x + 1.5, y + 1.5, f"P{i}", 
                                fontsize=6, fontweight='bold',
                                bbox=dict(facecolor=path_color, alpha=0.7, boxstyle='round,pad=0.2'),
                                ha='center', va='center',
                                color='white')
        
        # Add node labels (node numbers)
        node_labels = {node: str(node) for node in self._graph.nodes()}
        nx.draw_networkx_labels(self._graph, pos, node_labels, font_size=8, 
                            font_color='white', font_weight='bold',
                            bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
        
        # Create a dictionary to store road ID information for each node
        node_road_ids = {}
        for u, v, data in self._graph.edges(data=True):
            if 'entry_waypoint' in data:
                road_id = data['entry_waypoint'].road_id
                node_road_ids[u] = road_id
                node_road_ids[v] = road_id
        
        # Add road ID labels at node positions (slightly offset)
        for node, road_id in node_road_ids.items():
            if node in pos:
                x, y = pos[node]
                plt.text(x - 2.0, y - 2.0, f"R{road_id}", 
                        fontsize=6, fontweight='bold',
                        bbox=dict(facecolor='yellow', alpha=0.7, boxstyle='round,pad=0.2'),
                        ha='center', va='center',
                        color='black')
        
        # Build title with path information
        title = f"Road Network Graph - {self._wmap.name}\n"
        title += f"Nodes: {self._graph.number_of_nodes()}, Edges: {self._graph.number_of_edges()}, "
        title += f"Road Segments: {len(self._topology)}"
        
        if paths:
            title += f"\nShowing {len(paths)} path(s)"
            for i, path in enumerate(paths):
                if path:
                    title += f"\nPath {i}: {path[0]} → {path[-1]} ({len(path)} nodes)"
        
        plt.title(title)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Regular Node'),
            Patch(facecolor='red', label='Intersection Node'),
            Patch(facecolor='black', label='Lane Follow Edge'),
            Patch(facecolor='green', label='Lane Change Edge'),
            Patch(facecolor='yellow', label='Road ID Label'),
            Patch(facecolor='black', label='Node Number Label')
        ]
        
        # Add path colors to legend
        if paths:
            path_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            for i in range(min(len(paths), len(path_colors))):
                legend_elements.append(Patch(facecolor=path_colors[i], label=f'Path {i}'))
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graph visualization with node labels saved to {save_path}")
        
        # Print path information
        if paths:
            print(f"\n=== PATH INFORMATION ===")
            for i, path in enumerate(paths):
                if path:
                    print(f"Path {i}: {path}")
                    print(f"  Length: {len(path)} nodes")
                    print(f"  Start: {path[0]}, End: {path[-1]}")
                    
                    # Calculate path statistics
                    unique_nodes = len(set(path))
                    print(f"  Unique nodes: {unique_nodes}")
                    if len(path) > 1:
                        print(f"  Revisited nodes: {len(path) - unique_nodes}")
                    print()
        
        # Print detailed node information
        print(f"\n=== DETAILED NODE INFORMATION ===")
        print(f"{'Node':<6} {'Road ID':<8} {'Is Intersection':<15} {'Position'}")
        print("-" * 50)
        
        for node in sorted(self._graph.nodes()):
            vertex = self._graph.nodes[node]["vertex"]
            road_id = node_road_ids.get(node, "N/A")
            is_intersection = any(edge[2].get("intersection", False) 
                                for edge in self._graph.edges(node, data=True))
            
            print(f"{node:<6} {road_id:<8} {str(is_intersection):<15} ({vertex[0]:.1f}, {vertex[1]:.1f})")
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
        
        filename = f"{self.image_count}.jpg"
        # if image_type == "rgb":
        save_path = os.path.join(self.rgb_dir, filename)
        cv2.imwrite(save_path, cv2.cvtColor(array,cv2.COLOR_BGR2RGB))
        # else:
        #     save_path = os.path.join(self.semseg_dir, filename)
        #     cv2.imwrite(save_path, array)
        
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
        with open(os.path.join(self.save_dir, "traj_data.pkl"), "w") as f:
            json.dump(metadata, f, indent=2)
       
# def lengths(x):
#     if isinstance(x,list):
#         yield len(x)
#         for y in x:
#             yield from lengths(y)
def main():
    # Connect to CARLA server
    actor_list = []
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.load_world("Town02")
        wmap = world.get_map()
        
        print(f"Connected to CARLA map: {wmap.name}")
        
        # Create GlobalRoutePlanner
        sampling_resolution = 2.0
        grp = GlobalRoutePlanner(wmap, sampling_resolution)
        
  
        # grp.visualize_graph(paths=[longest_path])
        # breakpoint()
        # return
        # wps=wmap.generate_waypoints(sampling_resolution)
        # random_start=random.choice(wps)
        # p_wps=random_start.previous_until_lane_start(1.0)[-1]
        
        waypoints=grp.trace_route()
        locations=[]
        for wp in waypoints:
            loc=wp.transform.location
            locations.append([loc.x,loc.y,loc.z])
            # plt.scatter(loc.x,loc.y)
        # plt.show()
        # # print(locations)
        # breakpoint()
        # breakpoint()
        # # breakpoint()
        
        # Find single lane path
        # path_edges = traverser.find_single_lane_path(target_lane)
        
        # if not path_edges:
        #     print("No path found. Trying different road...")
        #     target_road = 7  # Alternative road
        #     path_edges = traverser.find_single_lane_path(target_road, target_lane, max_distance=500)
        
        # if not path_edges:
        #     print("Failed to find a suitable path. Exiting.")
        #     return
        
        # Get waypoints from the path
        # waypoints = traverser.get_waypoints_from_path(path_edges)
        # print(f"Extracted {len(waypoints)} waypoints from the path")
        
        # Set up image capture
        image_capture = ImageCapture()
        
        # Setup vehicle and sensors
   
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
            
            # waypoint_index = 0
            total_waypoints = len(waypoints)
            # capture_interval = 3  # Capture image every N waypoints
            
            for ix,waypoint in enumerate(waypoints):
                # Check for quit signal
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE):
                        print("User interrupted capture")
                        break
                
                # Advance the simulation
                snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=2.0)
                # print(waypoint.transform.location)
                # Set vehicle position
                vehicle.set_transform(waypoint.transform)
                
                # Capture images at intervals
                # if waypoint_index % capture_interval == 0:
                # Convert semantic segmentation image
                image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                
                # Save images
                image_capture.save_image(image_rgb, "rgb")
                # image_capture.save_image(image_semseg, "semseg")
                
                # Record location and waypoint info
                image_capture.record_location(vehicle, waypoint, image_capture.image_count)
                
                image_capture.image_count += 1
                
                # Display progress
                display.fill((0, 0, 0))
                progress_text = f"Progress: {ix}/{total_waypoints}"
                road_text = f"Road: {waypoint.road_id}, Lane: {waypoint.lane_id}"
                images_text = f"Images: {image_capture.image_count}"
                
                display.blit(font.render(progress_text, True, (255, 255, 255)), (20, 20))
                display.blit(font.render(road_text, True, (255, 255, 255)), (20, 60))
                display.blit(font.render(images_text, True, (255, 255, 255)), (20, 100))
                pygame.display.flip()
                
                print(f"Captured image {image_capture.image_count} at waypoint {ix} "
                        f"(Road {waypoint.road_id}, Lane {waypoint.lane_id})")
                
                # waypoint_index += 1
                clock.tick(20)
        image_capture.locations=locations
        # Save metadata
        image_capture.save_metadata()
        
      
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