import numpy as np
from pyflann import FLANN
from collections import defaultdict
import heapq
import matplotlib.pyplot as plt

class ShortestPathFLANN:
    def __init__(self, points, radius=1.0):
        """
        Initialize the path finder with a set of points and connection radius.
        
        Args:
            points: numpy array of points (n_points × dimensions)
            radius: maximum distance to consider points as connected
        """
        self.points = np.asarray(points, dtype=np.float32)
        self.radius = radius
        self.flann = FLANN()
        self.graph = None
        
        # Build the FLANN index
        self.flann.build_index(self.points, algorithm='kdtree', trees=4)
        
        # Build the graph connectivity
        self._build_graph()
    
    def _build_graph(self):
        """
        Builds an adjacency list representation of the graph using FLANN radius search.
        Each node is connected to all other nodes within the specified radius.
        """
        self.graph = defaultdict(list)
        
        for i, point in enumerate(self.points):
            # Find all neighbors within radius
            indices, distances = self.flann.nn_radius(point, self.radius**2)
            
            # Add edges to graph (excluding self-loops)
            for j, dist in zip(indices, distances):
                if i != j:
                    self.graph[i].append((j, np.sqrt(dist)))
    
    def find_shortest_path(self, start_idx, end_idx):
        """
        Find the shortest path between two nodes using Dijkstra's algorithm.
        
        Args:
            start_idx: index of starting node
            end_idx: index of target node
            
        Returns:
            path: list of node indices representing the shortest path
            distance: total distance of the path
        """
        if start_idx >= len(self.points) or end_idx >= len(self.points):
            raise ValueError("Start or end index out of range")
            
        # Initialize distances and predecessors
        distances = {i: float('infinity') for i in range(len(self.points))}
        distances[start_idx] = 0
        predecessors = {i: None for i in range(len(self.points))}
        
        # Priority queue for Dijkstra's algorithm
        pq = [(0, start_idx)]
        
        while pq:
            current_distance, current_node = heapq.heappop(pq)
            
            # If we've reached the target
            if current_node == end_idx:
                break
                
            # If we've found a longer path
            if current_distance > distances[current_node]:
                continue
            
            # Check all neighbors
            for neighbor, weight in self.graph[current_node]:
                distance = current_distance + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))
        
        # Reconstruct path
        if distances[end_idx] == float('infinity'):
            return None, float('infinity')  # No path exists
            
        path = []
        current_node = end_idx
        while current_node is not None:
            path.append(current_node)
            current_node = predecessors[current_node]
        path.reverse()
        
        return path, distances[end_idx]
    
    def visualize_path(self, path, title="Shortest Path"):
        """
        Visualize the points and the found path.
        
        Args:
            path: list of indices representing the path
            title: title for the plot
        """
        if path is None:
            print("No path to visualize")
            return
            
        plt.figure(figsize=(10, 10))
        
        # Plot all points
        plt.scatter(self.points[:, 0], self.points[:, 1], c='blue', alpha=0.5, label='Points')
        
        # Highlight start and end points
        plt.scatter(self.points[path[0], 0], self.points[path[0], 1], 
                   c='green', s=100, label='Start')
        plt.scatter(self.points[path[-1], 0], self.points[path[-1], 1], 
                   c='red', s=100, label='End')
        
        # Plot the path
        path_points = self.points[path]
        plt.plot(path_points[:, 0], path_points[:, 1], 'r-', label='Path')
        
        # Plot connections within radius for start and end points (optional)
        for idx in [path[0], path[-1]]:
            indices, _ = self.flann.nn_radius(self.points[idx], self.radius**2)
            for neighbor_idx in indices:
                if neighbor_idx != idx:
                    plt.plot([self.points[idx][0], self.points[neighbor_idx][0]],
                            [self.points[idx][1], self.points[neighbor_idx][1]],
                            'gray', alpha=0.2)
        
        plt.legend()
        plt.title(title)
        plt.grid(True)
        plt.savefig("shortes_path.pdf")
    
    def __del__(self):
        """Clean up FLANN index"""
        if hasattr(self, 'flann'):
            self.flann.delete_index()

def example_usage():
    """
    Demonstrate usage of the ShortestPathFLANN class
    """
    # Create sample points
    np.random.seed(42)
    n_points = 100
    points = np.random.rand(n_points, 2)  # 2D points
    # Initialize pathfinder
    pathfinder = ShortestPathFLANN(points, radius=0.2)
    # Find path between two points
    start_idx = 0
    end_idx = 99
    path, distance = pathfinder.find_shortest_path(start_idx, end_idx)
    
    if path is not None:
        print(f"Found path with length {len(path)} and total distance {distance:.4f}")
        print("Path indices:", path)
        
        # Visualize the result
        pathfinder.visualize_path(path)
    else:
        print("No path found")
    return points, path

if __name__ == "__main__":
    example_usage()