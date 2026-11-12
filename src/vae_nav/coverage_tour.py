"""Closed coverage tours of a town's lane graph, for the single-trace evaluation maps.

User 2026-11-06: "CAN YOU CREATE MAP FROM A SINGLE TRACE?" -> "I am talking
about evaluations". The Gibson protocol builds each scene's map from ONE
continuous exploration trace. v30's map trace was many short policy segments
joined by respawns, so whole roads were never driven and edges could not cross
a respawn. Here the EXPERT drives one uninterrupted closed tour that traverses
every driving lane at least once in its legal direction, and the map is built
from that single trace.

Choosing the tour is the DIRECTED RURAL POSTMAN problem on the lane graph:
  * REQUIRED edges: the lanes outside junctions (GlobalRoutePlanner LANEFOLLOW
    edges with intersection=False). Junction connectors are optional links.
  * NO LANE CHANGES: only LANEFOLLOW edges exist in the graph. The clone was
    trained never to change lane (user 2026-09-28) and a map edge it cannot
    drive would be a broken edge; both towns' lane-following graphs are one
    strongly connected component (checked from the xodr, 2026-11-06), so no
    lane needs a lane change to be reached.

The heuristic is the standard one: take every required edge once, balance
in/out degree with a minimum-cost flow over shortest-path distances, join any
components left over with the cheapest round trips, and walk an Euler circuit
from the start. Pure: plain dicts in, a list of edges out; `lane_graph` is the
only function that reads CARLA objects, and it reads them duck-typed.
"""
import networkx as nx
import numpy as np


def restrict_to_largest_scc(edges, required):
    """For a town whose lane-following graph is NOT one strongly connected component
    (Town05, 2026-11-09: some lanes are reachable only by a lane change, which the
    tour never makes). Returns (required edges inside the component holding the most
    required metres, a start node there -- the same rule as the default start,
    `sorted(req)[0][0]` -- and the required edges left out). A strongly connected
    town loses nothing."""
    edges = {tuple(e): float(w) for e, w in dict(edges).items()}
    req = list(dict.fromkeys(tuple(e) for e in required))
    G = nx.DiGraph()
    G.add_edges_from(edges)
    best, best_m = None, -1.0
    for comp in nx.strongly_connected_components(G):
        m = sum(edges[e] for e in req if e[0] in comp and e[1] in comp)
        if m > best_m:
            best, best_m = comp, m
    keep = [e for e in req if e[0] in best and e[1] in best]
    dropped = [e for e in req if not (e[0] in best and e[1] in best)]
    if not keep:
        raise ValueError("no required lane lies inside any strongly connected component")
    return keep, sorted(keep)[0][0], dropped


def coverage_tour(edges, required, start):
    """Closed walk from `start` back to `start` traversing every `required`
    edge at least once. `edges` = {(u, v): metres} (directed). Returns the
    walk as a list of (u, v) edges."""
    edges = {tuple(e): float(w) for e, w in dict(edges).items()}
    req = list(dict.fromkeys(tuple(e) for e in required))
    for e in req:
        if e not in edges:
            raise ValueError(f"required {e} is not an edge of the graph")
    G = nx.DiGraph()
    G.add_node(start)
    for (u, v), w in edges.items():
        G.add_edge(u, v, weight=w)
    need = {start} | {n for e in req for n in e}
    fwd = nx.descendants(G, start) | {start}
    back = nx.ancestors(G, start) | {start}
    bad = sorted((n for n in need if n not in fwd or n not in back), key=str)
    if bad:
        raise ValueError(f"the tour cannot reach and return from {bad[:5]}"
                         f"{' ...' if len(bad) > 5 else ''}: the lanes are not "
                         f"one strongly connected component")
    if not req:
        return []

    M = nx.MultiDiGraph()
    M.add_node(start)
    for u, v in req:
        M.add_edge(u, v)
    dist, paths = {}, {}

    def sp(a):
        if a not in dist:
            dist[a], paths[a] = nx.single_source_dijkstra(G, a, weight="weight")
        return dist[a], paths[a]

    def add_path(a, b):
        p = sp(a)[1][b]
        for x, y in zip(p, p[1:]):
            M.add_edge(x, y)

    # 1. BALANCE. A node entered more often than it is left needs extra paths
    #    out of it, and the reverse; pair them at minimum total distance.
    imb = {n: M.out_degree(n) - M.in_degree(n) for n in M.nodes}
    srcs = {n: -d for n, d in imb.items() if d < 0}
    dsts = {n: d for n, d in imb.items() if d > 0}
    if srcs:
        F = nx.DiGraph()
        for s, k in srcs.items():
            F.add_node(("s", s), demand=-k)
        for t, k in dsts.items():
            F.add_node(("t", t), demand=k)
        for s in srcs:
            d = sp(s)[0]
            for t in dsts:
                F.add_edge(("s", s), ("t", t), weight=int(round(1000 * d[t])))
        flow = nx.min_cost_flow(F)
        for (_, s), out in ((k, v) for k, v in flow.items() if k[0] == "s"):
            for (_, t), f in out.items():
                for _ in range(int(f)):
                    add_path(s, t)

    # 2. CONNECT. A balanced graph with several components has an Euler circuit
    #    in each but no tour; join the start's component to the nearest other
    #    one with the cheapest round trip, until one is left.
    while True:
        comps = list(nx.weakly_connected_components(M))
        if len(comps) == 1:
            break
        home = next(c for c in comps if start in c)
        best = None
        for c in comps:
            if c is home:
                continue
            for x in home:
                dx = sp(x)[0]
                for y in c:
                    cost = dx[y] + sp(y)[0][x]
                    if best is None or cost < best[0]:
                        best = (cost, x, y)
        _, x, y = best
        add_path(x, y)
        add_path(y, x)

    return [(u, v) for u, v in nx.eulerian_circuit(M, source=start)]


def walk_length(walk, edges):
    """Metres of a walk under `edges` = {(u, v): metres}."""
    return float(sum(edges[tuple(e)] for e in walk))


def lane_graph(grp, follow_type="LANEFOLLOW", junctions=False):
    """(edges, required, waypoints) from a GlobalRoutePlanner's `_graph`.

    edges     {(u, v): metres along the edge's waypoints}, LANEFOLLOW only, so
              the tour never changes lane;
    required  the edges outside junctions (the lanes) -- and, with
              `junctions=True`, every junction connector too, i.e. every legal
              TURN (user 2026-11-07, "make sure paths can be planned with MAP":
              a lanes-only tour skips most turns, so the map built from it has
              no edge for them and plans 6-22x detours round the block);
    waypoints {(u, v): [entry, *path, exit]}, the polyline the expert follows."""
    edges, required, wps = {}, [], {}
    for u, v, d in grp._graph.edges(data=True):
        kind = getattr(d["type"], "name", str(d["type"]))
        if kind != follow_type:
            continue
        pts = [d["entry_waypoint"], *d["path"], d["exit_waypoint"]]
        xy = np.array([[w.transform.location.x, w.transform.location.y] for w in pts],
                      np.float64)
        edges[(u, v)] = float(np.hypot(*np.diff(xy, axis=0).T).sum()) if len(xy) > 1 else 0.0
        wps[(u, v)] = pts
        if junctions or not d["intersection"]:
            required.append((u, v))
    return edges, required, wps


def tour_waypoints(walk, waypoints, min_gap_m=0.5):
    """The walk's waypoints in order, dropping any within `min_gap_m` of the
    one kept before it (an edge's exit is the next edge's entry)."""
    out, last = [], None
    for e in walk:
        for w in waypoints[tuple(e)]:
            p = np.array([w.transform.location.x, w.transform.location.y], np.float64)
            if last is not None and float(np.hypot(*(p - last))) < min_gap_m:
                continue
            out.append(w)
            last = p
    return out


def closed_drive_plan(plan, start_k=2, overlap=10):
    """(drive plan, start index). The car spawns at plan[start_k], a few
    waypoints into the tour (a lane's first waypoint sits on the junction's
    edge), and a closed tour's last waypoint is its first. Ending there would
    stop the trace `start_k` waypoints short of where it began, so its first and
    last stretches would never be joined by a driven edge. The drive plan
    appends plan[1 : start_k + overlap + 1]: the car drives past its own spawn
    point by `overlap` waypoints, and the loop is closed in the TRACE."""
    k = int(start_k)
    return list(plan) + list(plan[1:k + int(overlap) + 1]), k
