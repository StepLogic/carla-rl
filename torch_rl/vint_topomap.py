"""ViNT-native topological map and navigator (user 2026-11-07: "start with default
trace as map and then drive" -> "ViNT-native topomap").

The expert-tour trace IS the map, recorded as ViNT's create_topomap.py records one:
an image every --dt 1.0 s (every 10 steps at our 10 Hz). Navigation is ViNT's
navigate.py (ViNT branch): each step ViNT scores the path nodes within --radius 4
of the current one; the closest node is the target unless it is nearer than
--close-threshold 3, in which case the next node is. Images decide everything.

What the benchmark adds to ViNT's single-route setting: arbitrary start and goal
points need ROUTES, so the trace's revisits of a place become graph links --
proposed by the trace's own odometry (<= radius_m, < 90 deg apart; allowed for map
building and disclosed) and KEPT only when ViNT predicts the second place is at
most `tau` ahead of the first (ViNG-style), tau calibrated on consecutive nodes.
Edge costs are ViNT's predicted distances.
"""
import heapq
import math

import numpy as np

EVERY = 10          # one node per second at 10 Hz (create_topomap.py --dt 1.0)
RADIUS = 4          # navigate.py --radius
CLOSE = 3.0         # navigate.py --close-threshold (ViNT distance units)


def node_steps(n_steps, every=EVERY):
    """Trace steps that become nodes."""
    return np.arange(0, int(n_steps), int(every))


def context_steps(t, context_size=5, stride=3):
    """ViNT's context for the frame at step t, oldest first; the drive start repeats."""
    return [max(0, int(t) - j * int(stride)) for j in range(int(context_size), -1, -1)]


def sequential_edges(n_nodes):
    """Drive-order links of one continuous trace."""
    return [(i, i + 1) for i in range(int(n_nodes) - 1)]


def loop_candidates(xy, yaw, radius_m=8.0, max_dyaw=math.pi / 2, min_gap=3):
    """Ordered node pairs (u, v) the trace's odometry proposes as revisits: within
    radius_m, headings less than max_dyaw apart, and more than min_gap nodes
    apart in drive order (closer ones are drive neighbours, linked already)."""
    from scipy.spatial import cKDTree
    xy = np.asarray(xy, np.float64)
    yaw = np.asarray(yaw, np.float64)
    out = []
    for u, v in cKDTree(xy).query_pairs(float(radius_m)):
        if abs(u - v) <= int(min_gap):
            continue
        dyaw = abs((yaw[u] - yaw[v] + math.pi) % (2 * math.pi) - math.pi)
        if dyaw < max_dyaw:
            out += [(u, v), (v, u)]
    return sorted(out)


def calibrate_close(seq_dists, q=0.95):
    """The revisit threshold: the q-quantile of ViNT's predicted distance from a
    node to the NEXT node of the drive -- 'no farther than a typical next node'."""
    return float(np.quantile(np.asarray(seq_dists, np.float64), float(q)))


def build_graph(seq_edges, seq_costs, cand_pairs, cand_dists, tau):
    """(u, v, cost) edges: every drive link at its predicted cost, and the revisit
    candidates ViNT predicts within tau."""
    edges = [(int(u), int(v), float(c)) for (u, v), c in zip(seq_edges, seq_costs)]
    edges += [(int(u), int(v), float(d)) for (u, v), d in zip(cand_pairs, cand_dists)
              if d <= tau]
    return edges


def shortest_path(n, edges, s, g):
    """Dijkstra on (u, v, cost) edges -> node list from s to g, or None."""
    if s == g:
        return [int(s)]
    adj = [[] for _ in range(int(n))]
    for u, v, c in edges:                      # a saved map's edges come back as floats
        adj[int(u)].append((int(v), max(float(c), 0.0)))
    s, g = int(s), int(g)
    dist = {s: 0.0}
    prev = {}
    heap = [(0.0, s)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == g:
            break
        if d > dist.get(u, math.inf):
            continue
        for v, c in adj[u]:
            nd = d + c
            if nd < dist.get(v, math.inf):
                dist[v], prev[v] = nd, u
                heapq.heappush(heap, (nd, v))
    if g not in dist:
        return None
    path = [g]
    while path[-1] != s:
        path.append(prev[path[-1]])
    return [int(x) for x in reversed(path)]


def scc_sizes(n, edges):
    """Strongly connected component sizes, largest first (Kosaraju)."""
    n = int(n)
    fwd, rev = [[] for _ in range(n)], [[] for _ in range(n)]
    for u, v, *_ in edges:
        fwd[int(u)].append(int(v))
        rev[int(v)].append(int(u))
    seen, order = [False] * n, []
    for s in range(n):
        if seen[s]:
            continue
        stack, seen[s] = [(s, 0)], True
        while stack:
            u, i = stack.pop()
            if i < len(fwd[u]):
                stack.append((u, i + 1))
                v = fwd[u][i]
                if not seen[v]:
                    seen[v] = True
                    stack.append((v, 0))
            else:
                order.append(u)
    comp, sizes = [-1] * n, []
    for s in reversed(order):
        if comp[s] >= 0:
            continue
        size, stack, comp[s] = 0, [s], s
        while stack:
            u = stack.pop()
            size += 1
            for v in rev[u]:
                if comp[v] < 0:
                    comp[v] = s
                    stack.append(v)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def build_topomap(frames, xy, yaw, policy, *, every=EVERY, context_size=5, stride=3,
                  radius_m=8.0, min_gap=3, q=0.95, to_frame=None, log=print):
    """One continuous trace -> the topological map.

    frames (T, H, W, 3) uint8, xy (T, 2) and yaw (T,) radians are the trace's own
    odometry; `policy` is a VintPolicy (crop85, pair_dists); `to_frame` maps a
    trace frame to the eval's frame size before ViNT's crop. -> dict with
    node_step, node_xy, node_yaw, ctx85 (N, context+1, H, W, 3) uint8 crops,
    edges (E, 3) [u, v, predicted distance] and a summary."""
    frames = np.asarray(frames)
    xy = np.asarray(xy, np.float64)
    yaw = np.asarray(yaw, np.float64)
    steps = node_steps(len(frames), every)
    ctx_idx = [context_steps(t, context_size, stride) for t in steps]
    need = sorted({s for c in ctx_idx for s in c})
    crops = {s: policy.crop85(to_frame(frames[s]) if to_frame else frames[s]) for s in need}
    ctx85 = np.stack([np.stack([crops[s] for s in c]) for c in ctx_idx])
    node_img = ctx85[:, -1]
    n = len(steps)
    seq = sequential_edges(n)
    seq_d = policy.pair_dists(ctx85[[u for u, _ in seq]], node_img[[v for _, v in seq]])
    tau = calibrate_close(seq_d, q)
    log(f"[Topomap] {len(frames)} steps -> {n} nodes; drive links {len(seq)}: ViNT distance "
        f"to the next node median {np.median(seq_d):.2f}, q{q:g} tau = {tau:.2f}")
    cand = loop_candidates(xy[steps], yaw[steps], radius_m=radius_m, min_gap=min_gap)
    cand_d = (policy.pair_dists(ctx85[[u for u, _ in cand]], node_img[[v for _, v in cand]])
              if cand else np.zeros(0))
    edges = build_graph(seq, seq_d, cand, cand_d, tau)
    kept = len(edges) - len(seq)
    sizes = scc_sizes(n, edges)
    summary = {"steps": int(len(frames)), "nodes": int(n), "every": int(every),
               "drive_links": len(seq), "seq_dist_median": float(np.median(seq_d)),
               "tau": float(tau), "q": float(q), "radius_m": float(radius_m),
               "min_gap": int(min_gap), "revisit_candidates": len(cand),
               "kept_revisits": int(kept), "edges": len(edges),
               "scc_largest": int(sizes[0]), "scc_largest_frac": float(sizes[0] / n),
               "scc_count": len(sizes)}
    log(f"[Topomap] revisit candidates {len(cand)} (odometry), kept by ViNT {kept}; "
        f"largest strongly connected component {sizes[0]}/{n} = {sizes[0] / n:.1%}")
    return {"node_step": steps, "node_xy": xy[steps], "node_yaw": yaw[steps], "ctx85": ctx85,
            "edges": np.asarray(edges, np.float32).reshape(-1, 3), "summary": summary}


class TopoNavigator:
    """navigate.py's ViNT rule along a planned path (indices into `path`)."""

    def __init__(self, path, radius=RADIUS, close=CLOSE):
        self.path = [int(x) for x in path]
        self.radius, self.close = int(radius), float(close)
        self.closest = 0

    @property
    def goal(self):
        return len(self.path) - 1

    @property
    def final(self):
        return self.closest >= self.goal

    def window(self):
        start = max(self.closest - self.radius, 0)
        end = min(self.closest + self.radius + 1, self.goal)
        return range(start, end + 1)

    def step(self, dists):
        """ViNT's distances to the window's nodes -> the target path index."""
        w = self.window()
        m = int(np.argmin(np.asarray(dists, np.float64)))
        if float(dists[m]) > self.close:
            target = self.closest = w.start + m
        else:
            target = w.start + min(m + 1, len(w) - 1)
            self.closest = min(w.start + m + 1, self.goal)
        return target
