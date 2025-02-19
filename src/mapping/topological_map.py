import cv2
import numpy as np
from pyflann import FLANN
from collections import defaultdict
import heapq
import matplotlib.pyplot as plt
import skimage
import skimage.feature
class TopologicalMap:
    def __init__(self,radius=1.0):
        """
        Initialize the path finder with a set of points and connection radius.
        
        Args:
            points: numpy array of points (n_points × dimensions)
            radius: maximum distance to consider points as connected
        """
        self.graph = None        
        # Build the graph connectivity
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.des_nodes=[]
        self.image_node=[]
        self.heading_nodes=[]
        self.flann = FLANN()
        self.radius = radius
        # self._build_graph()
    def update(self, image_obs,heading_obs):
        # breakpoint()
        # features = skimage.feature.hog(image_obs,channel_axis=-1)
        image_obs=np.ravel(image_obs)
        self.image_node.append(image_obs)
        self.des_nodes.append(image_obs)
        self.heading_nodes.append(heading_obs)
        # breakpoint()
        # print(np.shape(self.des_nodes))
        # print(image_obs.shape)
        self.flann.build_index(np.array(self.des_nodes), algorithm='kdtree', trees=4)
        
    # def _build_graph(self):
    #     """
    #     Builds an adjacency list representation of the graph using FLANN radius search.
    #     Each node is connected to all other nodes within the specified radius.
    #     """
    #     self.graph = defaultdict(list)  # Initialize adjacency list
    #     for i, point in enumerate(self.des_nodes):
    #         # Find all neighbors within radius using FLANN
    #         indices, distances = self.flann.nn_index(point)
    #         # print(distances)
    #         breakpoint()
    #         # Add edges to graph (excluding self-loops)
    #         for j, dist in zip(indices, distances):
    #             if i != j:  # Avoid self-loops
    #                 self.graph[i].append((j, np.sqrt(dist)))  # Store edge with Euclidean distance

    #     # Optional: Convert defaultdict to a regular dict for consistency
    #     self.graph = dict(self.graph)

    def create_navigation_guide(self,image_obs,goal_idx):
        # path,_= self.find_path_to_goal(image_obs,goal_idx)
        # print("Path",path,len(self.des_nodes))
        def subgoal(image_obs):
            features=image_obs
            indices,distances=self.flann.nn_index(features,num_neighbors=1)
            # print(distances,indices,goal_idx)
            # if path is None or len(path)==0:
            #     return None ,True
            # if len(indices)==0 or indices is None or path is None:
            #     print("None")
            #     return None,not indices is None and len(indices)==0
            # if indices[0] == path[0]:
            #     k=path.pop(0)
            #     print(f"passed {k}")
            #     if len(path)==0:
            #         return None ,True
            # return (self.des_nodes[path[0]],self.heading_nodes[path[0]]),False
            return (self.des_nodes[indices[0]],self.heading_nodes[indices[0]]),indices[0]==goal_idx
        return subgoal
            
            

    # def find_path_to_goal(self,features,goal_idx):
    #     print("obs",goal_idx<len(self.des_nodes)-1)
    #     if(goal_idx > len(self.des_nodes)-1):
    #         return [],[]
    #     # features = skimage.feature.hog(obs,channel_axis=-1)
    #     # print("obs",goal_idx,features.shape)
    #     indices,distances=self.flann.nn_index(features,num_neighbors=1)
    #     print("Found",distances,indices)
    #     if len(indices)==0:
    #         return [],[]
    #     return self.find_shortest_path(indices[0],goal_idx)
        

    # def find_shortest_path(self, start_idx, end_idx):
    #     print("No nodes in graph", start_idx, end_idx)
    #     if self.flann._FLANN__curindex is None:
    #         print("No nodes in graph")
    #         return [], []
    #     if self.graph is None:
    #         self._build_graph()
    #     if start_idx >= len(self.des_nodes) or end_idx >= len(self.des_nodes):
    #         raise ValueError("Start or end index out of range")

    #     # Initialize distances and predecessors
    #     distances = {i: float('infinity') for i in range(len(self.des_nodes))}
    #     distances[start_idx] = 0
    #     predecessors = {i: None for i in range(len(self.des_nodes))}

    #     # Priority queue for Dijkstra's algorithm
    #     pq = [(0, start_idx)]
    #     while pq:
    #         current_distance, current_node = heapq.heappop(pq)

    #         # If we've reached the target
    #         if current_node == end_idx:
    #             break

    #         # If we've found a longer path
    #         if current_distance > distances[current_node]:
    #             continue

    #         # Check all neighbors
    #         for neighbor, weight in self.graph.get(current_node, []):
    #             distance = current_distance + weight
    #             if distance < distances[neighbor]:
    #                 distances[neighbor] = distance
    #                 predecessors[neighbor] = current_node
    #                 heapq.heappush(pq, (distance, neighbor))

    #     # Reconstruct path
    #     if distances[end_idx] == float('infinity'):
    #         return None, float('infinity')  # No path exists

    #     path = []
    #     current_node = end_idx
    #     while current_node is not None:
    #         path.append(current_node)
    #         current_node = predecessors[current_node]
    #     path.reverse()

    #     return path, distances[end_idx]
    # def visualize_path(self, path, title="Shortest Path"):
    #     """
    #     Visualize the points and the found path.
        
    #     Args:
    #         path: list of indices representing the path
    #         title: title for the plot
    #     """
    #     if path is None:
    #         print("No path to visualize")
    #         return
    #     plt.figure(figsize=(10, 10))
    #     # Plot all points
    #     plt.scatter(self.points[:, 0], self.points[:, 1], c='blue', alpha=0.5, label='Points')
    #     # Highlight start and end points
    #     plt.scatter(self.points[path[0], 0], self.points[path[0], 1], 
    #                c='green', s=100, label='Start')
    #     plt.scatter(self.points[path[-1], 0], self.points[path[-1], 1], 
    #                c='red', s=100, label='End')
    #     # Plot the path
    #     path_points = self.points[path]
    #     plt.plot(path_points[:, 0], path_points[:, 1], 'r-', label='Path')
    #     # Plot connections within radius for start and end points (optional)
    #     for idx in [path[0], path[-1]]:
    #         indices, _ = self.flann.nn_radius(self.points[idx], self.radius**2)
    #         for neighbor_idx in indices:
    #             if neighbor_idx != idx:
    #                 plt.plot([self.points[idx][0], self.points[neighbor_idx][0]],
    #                         [self.points[idx][1], self.points[neighbor_idx][1]],
    #                         'gray', alpha=0.2)
    #     plt.legend()
    #     plt.title(title)
    #     plt.grid(True)
    #     plt.savefig("shortes_path.pdf")
    
    def __del__(self):
        """Clean up FLANN index"""
        if hasattr(self, 'flann'):
            self.flann.delete_index()

# def example_usage():
#     """
#     Demonstrate usage of the ShortestPathFLANN class
#     """
#     # Create sample points
#     np.random.seed(42)
#     n_points = 100
#     points = np.random.rand(n_points, 2)  # 2D points
#     # Initialize pathfinder
#     pathfinder = ShortestPathFLANN(points, radius=0.2)
#     # Find path between two points
#     start_idx = 0
#     end_idx = 99
#     path, distance = pathfinder.find_shortest_path(start_idx, end_idx)
    
#     if path is not None:
#         print(f"Found path with length {len(path)} and total distance {distance:.4f}")
#         print("Path indices:", path)
        
#         # Visualize the result
#         pathfinder.visualize_path(path)
#     else:
#         print("No path found")
#     return points, pathstart_idx

