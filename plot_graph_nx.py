"""Draw a map's undirected networkx graph two ways (user 2026-09-16: "show graph for DC").

LEFT  geographic: nodes at their real x/y on the town's lanes -- what the map looks like on the ground.
RIGHT topological: positions computed by networkx from the EDGES ALONE (Kamada-Kawai), ignoring where the car
      actually was. This is the view that shows what kind of graph it is: a path unfolds into a single thread with
      no branches, whereas a map with loop closures pulls into a connected shape with visible cycles.

Nodes are coloured by selection trigger where the map records one (decision / coverage / both), using the same
colours as plot_node_kinds.py so the two figures can be read together.
"""
import argparse, collections, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from graph_nx import build_graph

KIND_COLOR = {"coverage": "#7fb3d5", "both": "#e67e22", "decision": "#c0392b"}
KIND_SIZE = {"coverage": 16, "both": 55, "decision": 70}
KIND_LABEL = {"coverage": "Coverage only", "both": "Both triggers", "decision": "Decision only"}
DEFAULT_COLOR, DEFAULT_SIZE = "#444444", 22


def node_style(G):
    kinds = [G.nodes[n].get("kind") for n in G.nodes]
    if all(k is None for k in kinds):
        return [DEFAULT_COLOR] * len(kinds), [DEFAULT_SIZE] * len(kinds), collections.Counter()
    return ([KIND_COLOR.get(k, DEFAULT_COLOR) for k in kinds],
            [KIND_SIZE.get(k, DEFAULT_SIZE) for k in kinds],
            collections.Counter(k for k in kinds if k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="map name in --seq-dir, or a path to a sequence JSON")
    ap.add_argument("--seq-dir", default="results/maps/sequence")
    ap.add_argument("--lanes", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    path = a.map if os.path.exists(a.map) else os.path.join(a.seq_dir, f"{a.map}_sequence.json")
    G, info = build_graph(path)
    label = a.label or os.path.basename(path).replace("_sequence.json", "")
    colors, sizes, kinds = node_style(G)

    lanes = json.load(open(a.lanes))["lane_polylines_xy"]
    L = np.concatenate([np.asarray(l, float) for l in lanes if len(l)])
    pad = 15

    fig, (axg, axt) = plt.subplots(1, 2, figsize=(16.4, 8.4))

    geo = {n: (G.nodes[n]["x_m"], G.nodes[n]["y_m"]) for n in G.nodes}
    axg.add_collection(LineCollection(lanes, colors="#e6e6e6", linewidths=1.6, zorder=0))
    nx.draw_networkx_edges(G, geo, ax=axg, edge_color="#d62728", width=1.1, alpha=0.75)
    nx.draw_networkx_nodes(G, geo, ax=axg, node_color=colors, node_size=sizes,
                           edgecolors="white", linewidths=0.4)
    axg.set_xlim(L[:, 0].min() - pad, L[:, 0].max() + pad)
    axg.set_ylim(L[:, 1].max() + pad, L[:, 1].min() - pad)  # CARLA is left-handed: +y points down
    axg.set_aspect("equal")
    axg.set_xlabel("X (m)")
    axg.set_ylabel("Y (m)")
    axg.set_title(f"{label} — geographic layout\nNodes at their true positions", fontsize=11)

    # Rotate the layout onto its principal axis: a path is one-dimensional, so unrotated it renders as a narrow
    # strip wasting the panel. This is a rigid rotation, so it distorts nothing -- equal aspect is kept.
    pos = nx.kamada_kawai_layout(G)
    P = np.array([pos[n] for n in G.nodes], float)
    P -= P.mean(0)
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    P = P @ Vt.T
    pos = {n: tuple(P[i]) for i, n in enumerate(G.nodes)}
    nx.draw_networkx_edges(G, pos, ax=axt, edge_color="#d62728", width=1.1, alpha=0.75)
    nx.draw_networkx_nodes(G, pos, ax=axt, node_color=colors, node_size=sizes,
                           edgecolors="white", linewidths=0.4)
    axt.set_aspect("equal")
    axt.axis("off")
    axt.set_title(f"{label} — topological layout\nPositions from the edges alone (Kamada-Kawai)", fontsize=11)

    handles = []
    for k in ("coverage", "both", "decision"):
        if kinds.get(k):
            handles.append(Line2D([], [], marker="o", lw=0, markerfacecolor=KIND_COLOR[k], markeredgecolor="white",
                                  markersize=np.sqrt(KIND_SIZE[k]) * 1.15, label=f"{KIND_LABEL[k]} ({kinds[k]})"))
    handles.append(Line2D([], [], color="#d62728", lw=2, label=f"Undirected edges ({info['undirected_edges']})"))
    fig.legend(handles=handles, loc="lower center", ncol=max(2, len(handles)), fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.055))

    fig.text(0.5, 0.012,
             f"{info['nodes']} nodes, {info['undirected_edges']} undirected edges "
             f"({info['reciprocal_pairs_merged']} reciprocal pairs merged, {info['self_loops_removed']} self-loops "
             f"removed), {info['cycles']} cycles, max degree {info['degree_max']}, "
             f"diameter {info.get('diameter_hops', '-')} hops.", ha="center", fontsize=9.5)
    fig.tight_layout(rect=[0, 0.125, 1, 1])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=165, bbox_inches="tight")
    print("[wrote]", a.out)
    print(" ", {k: info[k] for k in ("nodes", "undirected_edges", "self_loops_removed", "cycles", "degree_max")})


if __name__ == "__main__":
    main()
