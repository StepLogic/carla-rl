"""Pure map-following logic for the NoMaD baseline (no torch, no CARLA).

Mirrors navigation_policies' NoMaD.eval_action (radius window around the closest node,
subgoal = closest node if its predicted distance < close_threshold else the next one,
stop within 2 nodes of the goal node), written as functions so the rules are testable.
"""
import numpy as np


def window(closest, radius, goal_node):
    """Topomap index range [start, end] searched each step (inclusive)."""
    start = max(closest - radius, 0)
    end = min(closest + radius + 1, goal_node)
    return start, end


def select_subgoal(dists, start, close_threshold=3.0):
    """-> (closest_node, subgoal_index_within_window). dists covers topomap[start:end+1]."""
    d = np.asarray(dists, np.float64).ravel()
    i = int(np.argmin(d))
    sg = i if d[i] < close_threshold else min(i + 1, len(d) - 1)
    return start + i, sg


def plan_finished(closest, goal_node):
    """NoMaD's own stop rule: within 2 nodes of the last topomap node."""
    return abs(closest - goal_node) < 2


def localise_all(dists):
    """Global localisation: the node with the smallest predicted temporal distance."""
    return int(np.argmin(np.asarray(dists, np.float64).ravel()))
