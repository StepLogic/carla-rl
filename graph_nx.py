"""Load a driven map as an UNDIRECTED networkx graph, self-loops removed (user 2026-09-16).

The sequence maps store DIRECTED edges: sequence_link_map.py walks the trace and records each node change u -> v in
the driving direction, so a stretch driven both ways appears twice, once as (u, v) and once as (v, u). Collapsing to
an undirected graph merges those reciprocal pairs -- that, not self-loop removal, is why the edge count drops.

Self-loops cannot occur in these maps: sequence_edges only emits an edge when the node in force CHANGES, so u == v is
impossible by construction, and a scan of all 25 maps found none. remove_selfloop_edges is still applied and its count
reported, so the guarantee is checked rather than assumed -- a map built by some future rule may not hold it.

Node attributes carried through: x_m, y_m, trace_index, and kind where the map has it (decision / coverage / both).
Edge attributes: driven_length_m (the shorter of the two directions when reciprocal), crosses_episodes, reciprocal,
and straight_m, the straight-line distance between the endpoints.
"""
import argparse, json, os

import networkx as nx
import numpy as np


def build_graph(path, drop_selfloops=True):
    """-> (nx.Graph, info dict). Undirected, self-loops removed, reciprocal directed pairs merged."""
    d = json.load(open(path))
    G = nx.Graph()
    for n in d["nodes"]:
        attrs = {"x_m": float(n["x_m"]), "y_m": float(n["y_m"]), "trace_index": int(n["trace_index"])}
        if "kind" in n:
            attrs["kind"] = n["kind"]
        G.add_node(int(n["id"]), **attrs)

    directed, reciprocal = 0, 0
    for e in d["edges"]:
        u, v = int(e["from"]), int(e["to"])
        directed += 1
        L = float(e.get("driven_length_m", 0.0))
        if G.has_edge(u, v):
            reciprocal += 1
            G[u][v]["driven_length_m"] = min(G[u][v]["driven_length_m"], L)
            G[u][v]["reciprocal"] = True
            continue
        straight = float(np.hypot(G.nodes[u]["x_m"] - G.nodes[v]["x_m"], G.nodes[u]["y_m"] - G.nodes[v]["y_m"])) \
            if u in G.nodes and v in G.nodes else float("nan")
        G.add_edge(u, v, driven_length_m=L, crosses_episodes=int(e.get("crosses_episodes", 0)),
                   reciprocal=False, straight_m=straight)

    loops = list(nx.selfloop_edges(G))
    if drop_selfloops:
        G.remove_edges_from(loops)

    comps = nx.number_connected_components(G)
    info = {"map": os.path.basename(path), "nodes": G.number_of_nodes(), "directed_edges": directed,
            "undirected_edges": G.number_of_edges(), "reciprocal_pairs_merged": reciprocal,
            "self_loops_removed": len(loops), "components": comps,
            "cycles": G.number_of_edges() - G.number_of_nodes() + comps,
            "isolates": len(list(nx.isolates(G)))}
    deg = [k for _, k in G.degree()]
    info["degree_min"], info["degree_max"] = int(min(deg)), int(max(deg))
    info["degree_mean"] = round(float(np.mean(deg)), 2)
    if comps == 1:
        info["hops_mean"] = round(nx.average_shortest_path_length(G), 2)
        info["diameter_hops"] = nx.diameter(G)
    return G, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", required=True, help="map name (in --seq-dir) or a path to a sequence JSON")
    ap.add_argument("--seq-dir", default="results/maps/sequence")
    ap.add_argument("--keep-selfloops", action="store_true")
    ap.add_argument("--save-graphml", default="", help="directory to write <map>.graphml into")
    a = ap.parse_args()

    rows = []
    for m in a.maps:
        path = m if os.path.exists(m) else os.path.join(a.seq_dir, f"{m}_sequence.json")
        G, info = build_graph(path, drop_selfloops=not a.keep_selfloops)
        rows.append(info)
        if a.save_graphml:
            os.makedirs(a.save_graphml, exist_ok=True)
            dst = os.path.join(a.save_graphml, os.path.basename(path).replace("_sequence.json", ".graphml"))
            nx.write_graphml(G, dst)
            info["graphml"] = dst

    hdr = ["map", "nodes", "directed_edges", "undirected_edges", "reciprocal_pairs_merged", "self_loops_removed",
           "components", "cycles", "degree_mean", "degree_min", "degree_max", "hops_mean", "diameter_hops"]
    w = [max(len(h), max(len(str(r.get(h, "-"))) for r in rows)) for h in hdr]
    print("  ".join(h.ljust(x) for h, x in zip(hdr, w)))
    for r in rows:
        print("  ".join(str(r.get(h, "-")).ljust(x) for h, x in zip(hdr, w)))
    if any("graphml" in r for r in rows):
        print()
        for r in rows:
            if "graphml" in r:
                print("[graphml]", r["graphml"])


if __name__ == "__main__":
    main()
