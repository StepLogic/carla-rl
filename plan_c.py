"""Pure logic for option C driving (user decisions 2026-09-13).

Re-localise at each node on arrival (and on a leg timeout), accepting only the
expected node or the next `hops` nodes along the current plan, so a look-alike
road elsewhere cannot make the plan jump. A match other than the expected node
replans the shortest hop path from the match to the goal node.
"""
from collections import deque

import numpy as np


def bfs_path(n, edges, s, g, undirected=True):
    """Shortest hop path s -> g as a node list, or None. Every edge weight 1."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        if undirected:
            adj[v].append(u)
    prev, q = {s: None}, deque([s])
    while q:
        u = q.popleft()
        if u == g:
            break
        for v in adj[u]:
            if v not in prev:
                prev[v] = u
                q.append(v)
    if g not in prev:
        return None
    path = [g]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return path[::-1]


def candidate_window(path, i, hops=3):
    """Nodes a re-localisation may accept: the expected node path[i] and the next `hops`."""
    return list(path[i:i + hops + 1])


def nearest_latent(z, Z, candidates):
    """Candidate whose stored latent is nearest the live latent (cosine)."""
    z = np.asarray(z, np.float64)
    z = z / max(np.linalg.norm(z), 1e-12)
    C = np.asarray(Z, np.float64)[list(candidates)]
    C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    return int(list(candidates)[int(np.argmax(C @ z))])


def on_node_event(path, i, matched, goal_node, n, edges, undirected=True):
    """-> (path, i, replanned). i indexes the NEXT target; i == len(path) means the plan is finished.

    matched == expected node: keep the plan and target the following node.
    Otherwise: replan from the matched node and target the node after it.
    """
    if matched == path[i]:
        return list(path), i + 1, False
    new = bfs_path(n, edges, matched, goal_node, undirected)
    if new is None:                                   # cannot happen on a connected two-way map; keep the old plan
        return list(path), i + 1, False
    return new, 1, True


def step_cap(path_len_m, m_per_step=0.24, slack=300):
    """Total step budget: twice the steps the planned path needs, plus slack (user-approved)."""
    return int(round(2.0 * float(path_len_m) / m_per_step)) + int(slack)
