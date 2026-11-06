"""Topological image maps from an exploration trace, for the image-goal arm.

User 2026-11-02: map arms for the v30 benchmark -- uniform spacing, real-time
entropy, random and SGM placement; localisation by TRUE POSE first; report all
episodes (unroutable = failure) and the routable subset.

NODES are trace samples: the true pose plus the frame the policy saw there,
which is handed to the policy as its goal image while that node is the subgoal.
Nodes are DIRECTIONAL. The two sides of a CARLA road are different lanes and a
frame faces one way, so every spatial test below also requires the headings to
agree within `heading_tol` -- without it a node 3.5 m away on the opposite
carriageway would merge with, or be visited from, the lane it cannot be
reached from without a U-turn.

EDGES are what the traced car demonstrably drove. Walking the trace in order,
a sample VISITS every node within `visit_m` of it whose heading agrees; each
pair of successive visits is a directed edge weighted by the metres driven
between them (the minimum over traversals). Continuity is carried across a
segment boundary only when the restart continues the drive (not a stuck/stall
JUMP, and the restart is close to and aligned with where the car stopped), so
no edge is a teleport. Revisits therefore stitch the trace into a graph with
the junction structure it actually drove, and a sparse placement simply yields
long, honestly-costed edges rather than a different kind of graph.

Placement (indices into the trace):
  place_uniform  a sample becomes a node when no node with its heading is
                 within `spacing` -- coverage at fixed density, revisits free.
  place_random   `count` samples drawn uniformly, seeded (the control).
  place_entropy  the peer's real-time online-MAD rule (online_maps.online_mad,
                 z=1.0, window 110 m); optional `max_gap_m` coverage fill.
  SGM            selected outside this module from critic distances
                 (sgm_placement.sgm_select); build_graph takes any indices.
"""
import heapq
import json

import numpy as np

from vae_nav import online_maps

HEADING_TOL = np.radians(60.0)
VISIT_M = 3.0          # a sample visits a node this close (and aligned)
CONTINUE_M = 8.0       # a non-jumped restart this close to the stop continues
NEAREST_M = 8.0        # default localisation radius for start nodes


def _wrap(a):
    return (np.asarray(a, np.float64) + np.pi) % (2 * np.pi) - np.pi


def _aligned(yaw_a, yaw_b, tol=HEADING_TOL):
    return np.abs(_wrap(np.asarray(yaw_a) - yaw_b)) <= tol


# ---- trace loading -------------------------------------------------------------

def load_trace(trace_dir):
    """-> dict(pos[N,2], yaw[N], entropy[N], seg[N], local[N], file[S], jumped
    set, meta) from a run_trace directory (seg_*.npz + index.json). Frames are
    not loaded; `node_frames` reads only the ones a map keeps."""
    import os
    with open(os.path.join(trace_dir, "index.json")) as fh:
        index = json.load(fh)
    pos, yaw, ent, seg, local, files, jumped = [], [], [], [], [], {}, set()
    for s in index["segments"]:
        if not s.get("kept"):
            continue
        z = np.load(os.path.join(trace_dir, s["file"]))
        n = len(z["t"])
        pos.append(np.asarray(z["pos"], np.float64)[:, :2])
        yaw.append(np.asarray(z["yaw"], np.float64))
        ent.append(np.asarray(z["entropy"], np.float64).reshape(n))
        seg.append(np.full(n, s["seg"]))
        local.append(np.arange(n))
        files[int(s["seg"])] = s["file"]
        if s.get("jumped"):
            jumped.add(int(s["seg"]))
    yaws = np.concatenate(yaw)
    if np.nanmax(np.abs(yaws)) > 2 * np.pi + 1e-6:   # CARLA reports degrees
        yaws = np.radians(yaws)
    return {"pos": np.concatenate(pos), "yaw": yaws, "entropy": np.concatenate(ent),
            "seg": np.concatenate(seg), "local": np.concatenate(local),
            "files": files, "jumped": jumped, "dir": trace_dir,
            "meta": {k: v for k, v in index.items() if k != "segments"}}


def node_frames(trace, indices):
    """uint8 frames of the given trace samples, in order."""
    import os
    cache, out = {}, []
    for i in indices:
        s = int(trace["seg"][i])
        if s not in cache:
            cache = {s: np.load(os.path.join(trace["dir"], trace["files"][s]))["frame"]}
        out.append(np.asarray(cache[s][int(trace["local"][i])], np.uint8))
    return np.stack(out) if out else np.zeros((0,), np.uint8)


# ---- placement -----------------------------------------------------------------

class _DirectionalCover:
    """Is any accepted point within r of xy with an aligned heading?"""

    def __init__(self, cell):
        self.cell, self.grid = float(cell), {}

    def _k(self, xy):
        return int(np.floor(xy[0] / self.cell)), int(np.floor(xy[1] / self.cell))

    def near(self, xy, yaw, r):
        kx, ky = self._k(xy)
        reach = int(np.ceil(r / self.cell))
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                for p, h in self.grid.get((kx + dx, ky + dy), ()):
                    if np.hypot(*(p - xy)) <= r + 1e-9 and _aligned(h, yaw):
                        return True
        return False

    def add(self, xy, yaw):
        self.grid.setdefault(self._k(xy), []).append((np.asarray(xy, np.float64), float(yaw)))


def place_uniform(trace, spacing, seed_nodes=()):
    """Trace order; keep a sample when no aligned node lies within `spacing`
    (strictly: a sample exactly `spacing` away is kept)."""
    pos, yaw = trace["pos"], trace["yaw"]
    cov = _DirectionalCover(max(spacing, 1.0))
    for i in seed_nodes:
        cov.add(pos[i], yaw[i])
    out = list(seed_nodes)
    for i in range(len(pos)):
        if not cov.near(pos[i], yaw[i], spacing - 1e-6):
            cov.add(pos[i], yaw[i])
            out.append(i)
    return np.asarray(sorted(set(out)), int)


def place_random(trace, count, seed=0, dedupe_m=2.0):
    """`count` samples uniformly at random (seeded), excluding aligned
    duplicates within `dedupe_m` so no node is an unreachable twin."""
    rng = np.random.default_rng(seed)
    pos, yaw = trace["pos"], trace["yaw"]
    cov = _DirectionalCover(max(dedupe_m, 1.0))
    out = []
    for i in rng.permutation(len(pos)):
        if len(out) >= count:
            break
        if not cov.near(pos[i], yaw[i], dedupe_m):
            cov.add(pos[i], yaw[i])
            out.append(int(i))
    return np.asarray(sorted(out), int)


def place_entropy(trace, z=online_maps.Z, window_m=online_maps.WINDOW_M,
                  min_gap_m=online_maps.DEDUPE_M, max_gap_m=None, return_stats=False):
    """The peer's real-time rule (online_maps.online_mad). `max_gap_m` (off by
    default -- it is not part of their recipe) adds coverage nodes wherever no
    aligned node is within that distance."""
    idx, stats = online_maps.online_mad(trace["entropy"], trace["pos"], trace["seg"],
                                        window_m=window_m, z=z, dedupe_m=min_gap_m)
    if max_gap_m:
        before = len(idx)
        idx = place_uniform(trace, max_gap_m, seed_nodes=[int(i) for i in idx])
        stats["gap_fill_nodes"] = int(len(idx) - before)
    return (idx, stats) if return_stats else idx


def match_uniform(trace, count, lo=1.0, hi=200.0, iters=40):
    """Spacing whose uniform placement has the node count closest to `count`."""
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        n = len(place_uniform(trace, mid))
        if best is None or abs(n - count) < abs(best[1] - count):
            best = (mid, n)
        if n > count:
            lo = mid
        elif n < count:
            hi = mid
        else:
            break
    return best[0], place_uniform(trace, best[0])


# ---- graph -----------------------------------------------------------------------

class Graph:
    def __init__(self, nodes, edges, meta=None):
        self.nodes = nodes            # [{"pos": (x, y), "yaw": rad, "idx": trace i}]
        self.edges = edges            # {(a, b): metres}
        self.meta = dict(meta or {})
        self._adj = {}
        for (a, b), w in edges.items():
            self._adj.setdefault(a, []).append((b, w))
        self._xy = np.array([n["pos"] for n in nodes], np.float64).reshape(-1, 2)
        self._yaw = np.array([n["yaw"] for n in nodes], np.float64)

    # -- localisation
    def nearest(self, pos, yaw, radius=NEAREST_M):
        """Nearest node within `radius` whose heading agrees with `yaw`
        (yaw=None ignores heading). None if there is none."""
        if not len(self._xy):
            return None
        d = np.hypot(*(self._xy - np.asarray(pos, np.float64)[:2]).T)
        ok = d <= radius
        if yaw is not None:
            ok &= _aligned(self._yaw, yaw)
        if not ok.any():
            return None
        return int(np.flatnonzero(ok)[np.argmin(d[ok])])

    def goal_node(self, start, goal_pos, radius=NEAREST_M):
        """Among nodes within `radius` of the goal (heading unknown), the one
        minimising route-from-start + distance-to-goal: the node just BEFORE
        the goal wins over one past it. None if none is reachable."""
        d = np.hypot(*(self._xy - np.asarray(goal_pos, np.float64)[:2]).T)
        cands = set(np.flatnonzero(d <= radius).tolist())
        if start is None or not cands:
            return None
        dist = self._dijkstra(start)
        best = min(((dist[c] + d[c], c) for c in cands if c in dist), default=None)
        return None if best is None else int(best[1])

    # -- planning
    def _dijkstra(self, src, dst=None):
        dist, prev, pq = {src: 0.0}, {}, [(0.0, src)]
        while pq:
            du, u = heapq.heappop(pq)
            if du > dist.get(u, np.inf):
                continue
            if u == dst:
                break
            for v, w in self._adj.get(u, ()):
                nd = du + w
                if nd < dist.get(v, np.inf):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(pq, (nd, v))
        self._prev = prev
        return dist

    def path(self, src, dst):
        """Shortest directed node path src..dst (inclusive), or None."""
        if src is None or dst is None:
            return None
        if src == dst:
            return [src]
        dist = self._dijkstra(src, dst)
        if dst not in dist:
            return None
        out = [dst]
        while out[-1] != src:
            out.append(self._prev[out[-1]])
        return out[::-1]

    def length(self, path):
        return float(sum(self.edges[(a, b)] for a, b in zip(path, path[1:])))

    # -- io
    def save(self, path):
        with open(path, "w") as fh:
            json.dump({"nodes": self.nodes, "meta": self.meta,
                       "edges": [[a, b, w] for (a, b), w in sorted(self.edges.items())]}, fh)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        nodes = [{"pos": tuple(n["pos"]), "yaw": float(n["yaw"]), "idx": int(n["idx"])}
                 for n in d["nodes"]]
        return cls(nodes, {(int(a), int(b)): float(w) for a, b, w in d["edges"]}, d.get("meta"))


def build_graph(trace, indices, jumped_segs=None, visit_m=VISIT_M,
                continue_m=CONTINUE_M, meta=None, max_turn_deg=None):
    """Nodes = trace[indices]; edges = successive node visits along the trace
    (see the module note). `jumped_segs` defaults to the trace's own record.
    `max_turn_deg` (None = off, every earlier map): no edge between two nodes
    whose headings differ by more than this -- the Habitat peer's opposite-lane
    rule (2026-11-06), so a revisit of the other lane never becomes a shortcut."""
    pos, yaw, seg = trace["pos"], trace["yaw"], np.asarray(trace["seg"])
    jumped = set(trace.get("jumped", ())) if jumped_segs is None else set(jumped_segs)
    indices = [int(i) for i in indices]
    nodes = [{"pos": (float(pos[i, 0]), float(pos[i, 1])), "yaw": float(yaw[i]), "idx": i}
             for i in indices]
    cov = {}
    cell = max(visit_m, 1.0)
    for k, i in enumerate(indices):
        cov.setdefault((int(np.floor(pos[i, 0] / cell)), int(np.floor(pos[i, 1] / cell))), []).append(k)
    own = {i: k for k, i in enumerate(indices)}

    def visits(i):
        """Nodes sample i visits, its own node (if it is one) first."""
        kx, ky = int(np.floor(pos[i, 0] / cell)), int(np.floor(pos[i, 1] / cell))
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for k in cov.get((kx + dx, ky + dy), ()):
                    j = indices[k]
                    if np.hypot(*(pos[j] - pos[i])) <= visit_m and _aligned(yaw[j], yaw[i]):
                        out.append(k)
        if i in own:
            out = [own[i]] + [k for k in out if k != own[i]]
        return out

    edges = {}
    last, since = None, 0.0           # last visited node, metres driven since
    for i in range(len(pos)):
        if i > 0:
            step = float(np.hypot(*(pos[i] - pos[i - 1])))
            new_seg = seg[i] != seg[i - 1]
            broken = step > continue_m or not _aligned(yaw[i], yaw[i - 1], np.radians(90))
            if new_seg and int(seg[i]) in jumped:
                broken = True
            if broken:
                last, since = None, 0.0
            else:
                since += step
        vs = visits(i)
        if not vs:
            continue
        # among several nodes visited at once, keep the nearest as "the" visit
        k = vs[0] if i in own else min(vs, key=lambda k: np.hypot(*(pos[indices[k]] - pos[i])))
        if (last is not None and k != last and max_turn_deg is not None
                and not _aligned(yaw[indices[last]], yaw[indices[k]],
                                 np.radians(max_turn_deg))):
            last, since = k, 0.0              # the opposite lane: no edge
            continue
        if last is not None and k != last:
            w = edges.get((last, k))
            edges[(last, k)] = since if w is None else min(w, since)
        if k != last:
            last, since = k, 0.0
    return Graph(nodes, edges, meta)


# ---- navigation --------------------------------------------------------------------

class Navigator:
    """Feeds subgoals along a planned node path; `final` = hand over the true
    goal image. path[0] is the start node (already here), so the first subgoal
    is path[1]. Reaching ANY later node on the path (within `switch_m`, heading
    aligned when given) advances past it -- a car that skips a node is not made
    to go back for it."""

    def __init__(self, graph, path, switch_m=4.0):
        self.g, self.path, self.switch_m = graph, list(path), float(switch_m)
        self.k = 1                    # index of the current subgoal in path
        self.switches = 0

    @property
    def final(self):
        return self.k >= len(self.path)

    def current(self):
        return None if self.final else self.path[self.k]

    def update(self, pos, yaw=None):
        """-> True if the subgoal changed."""
        if self.final:
            return False
        p = np.asarray(pos, np.float64)[:2]
        hit = None
        for j in range(self.k, len(self.path)):
            n = self.g.nodes[self.path[j]]
            if np.hypot(*(np.asarray(n["pos"]) - p)) <= self.switch_m and (
                    yaw is None or _aligned(n["yaw"], yaw)):
                hit = j
        if hit is None:
            return False
        self.k = hit + 1
        self.switches += 1
        return True


# ---- SGM over sparse critic values -----------------------------------------------

def sgm_place(cand, pairs, q, xy, yaw, gate_m=10.0, cutoff=5.0):
    """Sparse Graphical Memory selection (sgm_placement's two-way test) over a
    SPARSE critic read: d(i, j) = -Q_g(view i, goal view j) is known only for
    the aligned pairs within the read radius; `pairs` index into `cand`.
    Candidates are walked in trace order; candidate s merges into an existing
    member w within `gate_m` with an aligned heading (the spatial prefilter,
    nearest first) when

        C(s, w) = max( max_k |d(s,k) - d(w,k)|,  max_k |d(k,s) - d(k,w)| ) < cutoff

    over the members k for which BOTH distances are known (k = w included, as in
    the reference); a pair with no common k cannot merge. Returns (trace
    indices of the members, stats).

    gate_m=None is THE PEER'S RULE (habitat-map-spec, 2026-10-03): no spatial
    gate and no heading test, consistency alone. s is then compared with every
    member, in the order they were placed; on a sparse read only members within
    twice the read radius can share a known k with s, so only those are visited
    -- an index, not a gate: no other member can produce a C at all."""
    cand = np.asarray(cand, int)
    xy = np.asarray(xy, np.float64)
    yaw = np.asarray(yaw, np.float64)
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    out_d, in_d = {}, {}
    for (a, b), v in zip(pairs, np.asarray(q, np.float64)):
        out_d.setdefault(int(a), {})[int(b)] = -v
        in_d.setdefault(int(b), {})[int(a)] = -v
    if gate_m is None:
        span = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T) if len(pairs) else np.zeros(1)
        reach = 2.0 * float(span.max()) + 1e-6
        cell = max(reach, 1.0)
        grid = {}

        def _cell(i):
            return int(np.floor(xy[i, 0] / cell)), int(np.floor(xy[i, 1] / cell))
    members, mset, cs = [], set(), []
    for s in range(len(cand)):
        if gate_m is None:
            kx, ky = _cell(s)
            near = sorted(m for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                          for m in grid.get((kx + dx, ky + dy), ())
                          if np.hypot(*(xy[m] - xy[s])) <= reach)
        else:
            d = np.hypot(*(xy[members] - xy[s]).T) if members else np.zeros(0)
            near = [members[i] for i in np.argsort(d) if d[i] <= gate_m
                    and _aligned(yaw[members[i]], yaw[s])]
        merged = False
        for w in near:
            c = -np.inf
            for table in (out_d, in_d):
                ts, tw = table.get(s, {}), table.get(w, {})
                for k in mset.intersection(ts).intersection(tw):
                    c = max(c, abs(ts[k] - tw[k]))
            if c == -np.inf:
                continue
            cs.append(c)
            if c < cutoff:
                merged = True
                break
        if not merged:
            members.append(s)
            mset.add(s)
            if gate_m is None:
                grid.setdefault(_cell(s), []).append(s)
    stats = {"candidates": int(len(cand)), "members": len(members), "gate_m": gate_m,
             "cutoff": cutoff, "c_median": float(np.median(cs)) if cs else None,
             "c_p10": float(np.percentile(cs, 10)) if cs else None,
             "tested": len(cs)}
    return cand[np.asarray(members, int)], stats


def sgm_match(cand, pairs, q, xy, yaw, count, gate_m=None, lo=0.0, hi=None, iters=40):
    """The cutoff whose SGM selection keeps the node count closest to `count`,
    by bisection (the count falls as the cutoff rises): the matched-budget SGM
    arm. -> (cutoff, members, stats with target and cutoff)."""
    if hi is None:
        # every |d1 - d2| is below this, so at `hi` everything mergeable merges
        hi = 2.0 * float(np.max(np.abs(np.asarray(q, np.float64)))) + 1.0
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        keep, st = sgm_place(cand, pairs, q, xy, yaw, gate_m=gate_m, cutoff=mid)
        if best is None or abs(len(keep) - count) < abs(len(best[1]) - count):
            best = (mid, keep, st)
        if len(keep) > count:
            lo = mid
        elif len(keep) < count:
            hi = mid
        else:
            break
    cutoff, keep, st = best
    return cutoff, keep, {**st, "target": int(count), "cutoff": cutoff}
