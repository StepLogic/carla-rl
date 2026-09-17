"""Link map nodes by trace order ACROSS episodes (one continuous drive), for every map.

Trace collection continues from the nearest waypoint after every episode end, so the
whole trace is one driven sequence (episode-boundary jumps <= ~2.6 m). Each trace sample
is assigned to the map node that holds it; walking the trace in order, every change of
node u -> v (u != v) is a directed edge in the driving direction, INCLUDING changes that
cross an episode reset. The edge's waypoints are the trace samples from the last sample
at u to the first sample at v; when the same (u, v) is driven several times, the shortest
driven span is kept. Nodes are unchanged from the input maps.

Sample -> node assignment per map kind:
  entropy (madent) maps: only the node samples themselves are assigned;
  uniform[g] maps:       the node that absorbed the sample in always_merge (itself if kept,
                         else the first node kept before it within the gate) -- the same
                         rule render_uniform_frame.py --edges trace uses;
  junction map:          the uniform6 assignment, mapped to the junction holding that
                         uniform6 node (samples at non-junction nodes are unassigned).
"""
import argparse, json, os, pickle, sys
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))

TRACE = "results/placement_v6n_town02_budget10k_laneoff_speed10/trace_s7.pkl"
UNI = "results/figures/uniform_frame_trace_edges/uniform{g}_placement_v6n_town02_budget10k_laneoff_speed10_s7.json"
MAPS = {
    "junction": "results/maps/town02_junctions_uniform6_trace_speed10.json",
    "uniform6_trace": UNI.format(g=6),
    "uniform2_trace": UNI.format(g=2),
    "entropy_smooth9_high": "results/maps/town02_speed10_s7_smooth9_high_driving_map.json",
    "entropy_smooth9_low": "results/maps/town02_speed10_s7_smooth9_low_driving_map.json",
    "entropy_nosmooth_high": "results/maps/town02_speed10_s7_nosmooth_high_driving_map.json",
    "entropy_nosmooth_low": "results/maps/town02_speed10_s7_nosmooth_low_driving_map.json",
    # user 2026-09-13: nosmooth-high with spatial thinning (4 m) after the MAD filter (thin_entropy_map.py)
    "entropy_nosmooth_high_thin4": "results/maps/town02_speed10_s7_nosmooth_high_thin4_driving_map.json",
    # user 2026-09-13: uniform[6 m] nodes between the thinned high-entropy nodes (hybrid_uniform_entropy_map.py)
    "hybrid_uniform6_entropy_thin4": "results/maps/town02_speed10_s7_hybrid_uniform6_entropy_thin4_driving_map.json",
}
# user 2026-09-14: "evaluate goal mode on raw trace": every trace sample is a node (build_raw_trace_map.py)
MAPS["raw_trace"] = "results/maps/town02_speed10_s7_raw_trace_driving_map.json"
# user 2026-09-14: explore-critic uncertainty maps (uncertainty_map.py), madent on each score; node samples only
MAPS["epistemic_high"] = "results/maps/town02_speed10_s7_epistemic_high_driving_map.json"
MAPS["aleatoric_high"] = "results/maps/town02_speed10_s7_aleatoric_high_driving_map.json"
# user 2026-09-14: SGM ablation maps (build_sgm_map.py), one per (gate, edge cutoff, consistency)
import glob as _glob
for _p in sorted(_glob.glob("results/maps/town02_speed10_s7_sgm_*_driving_map.json")):
    MAPS[os.path.basename(_p)[len("town02_speed10_s7_"):-len("_driving_map.json")]] = _p
OUT_MAPS = "results/maps/sequence"


def uniform_assign(P, kept, gate_m):
    """Sample -> uniform node, the always_merge absorption rule (render_uniform_frame.py)."""
    local = {int(k): n for n, k in enumerate(kept)}
    assign = np.full(len(P), -1, int)
    for i in range(len(P)):
        if i in local:
            assign[i] = local[i]
            continue
        prior = [n for n, k in enumerate(kept) if k < i and np.linalg.norm(P[k] - P[i]) < gate_m]
        if prior:
            assign[i] = prior[0]
    return assign


def load_nodes_and_assign(name, path, P):
    """-> nodes [{id, trace_index, x_m, y_m, yaw_rad, ...}], assign (len(P),), old edge count, old reach."""
    d = json.load(open(path))
    if "node_trace_index" in d:                                  # uniform[g] trace-edge graph
        kept = np.asarray(d["node_trace_index"], int)
        assign = uniform_assign(P, kept, float(d["gate_m"]))
        nodes = [{"id": n, "trace_index": int(t), "x_m": float(P[t, 0]), "y_m": float(P[t, 1])} for n, t in enumerate(kept)]
        return nodes, assign, len(d["edges"]), d.get("directed_reachability")
    nodes = [dict(x) for x in d["nodes"]]
    if name == "junction":
        u6 = json.load(open(d["source"]["uniform_graph"]))
        kept6 = np.asarray(u6["node_trace_index"], int)
        a6 = uniform_assign(P, kept6, float(u6["gate_m"]))
        member = {}
        for k, nd in enumerate(nodes):
            for t in nd["member_trace_indices"]:
                member[int(t)] = k
        node_of_u6 = np.asarray([member.get(int(t), -1) for t in kept6], int)
        assign = np.where(a6 >= 0, node_of_u6[np.maximum(a6, 0)], -1)
    elif any("member_trace_indices" in nd for nd in nodes):     # thinned entropy map: own + absorbed samples
        assign = np.full(len(P), -1, int)
        for k, nd in enumerate(nodes):
            for t in nd["member_trace_indices"]:
                assign[int(t)] = k
    else:                                                        # entropy maps: node samples only
        assign = np.full(len(P), -1, int)
        for k, nd in enumerate(nodes):
            assign[int(nd["trace_index"])] = k
    return nodes, assign, len(d["edges"]), d.get("directed_reachability")


def sequence_edges(assign, P, yaw, seg):
    best, last_node, last_i = {}, -1, -1
    for i in range(len(P)):
        v = int(assign[i])
        if v < 0:
            continue
        if last_node >= 0 and v != last_node:
            span = list(range(last_i, i + 1))
            L = float(np.sum(np.linalg.norm(np.diff(P[span], axis=0), axis=1)))
            key = (last_node, v)
            if key not in best or L < best[key]["driven_length_m"]:
                best[key] = {"from": last_node, "to": v, "directed": True, "driven_length_m": round(L, 2),
                             "crosses_episodes": int(seg[i] != seg[last_i]), "n_waypoints": len(span),
                             "waypoints": [{"x_m": float(P[t, 0]), "y_m": float(P[t, 1]), "trace_index": t,
                                            "yaw_rad": float(yaw[t])} for t in span]}
        last_node, last_i = v, i
    return [best[k] for k in sorted(best)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig-dir", default="results/localize_plan/sequence")
    ap.add_argument("--maps", nargs="*", default=None, help="build only these maps (default: all)")
    # user 2026-09-15 Town07 repeat: another town's trace and maps; defaults are the Town02 run
    ap.add_argument("--trace", default=TRACE)
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--out-maps", default=OUT_MAPS)
    ap.add_argument("--map", action="append", default=[], help="extra NAME=PATH map (repeatable)")
    a = ap.parse_args()
    extra = dict(m.split("=", 1) for m in a.map)
    todo = ({k: v for k, v in extra.items() if not a.maps or k in a.maps} if extra
            else {k: v for k, v in MAPS.items() if not a.maps or k in a.maps})

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from render_placement_maps import lane_polylines
    from vae_nav.sgm_frame import directed_reachability

    tr = pickle.load(open(a.trace, "rb"))
    P = np.asarray(tr["positions"])[:, :2]; yaw = np.asarray(tr["yaws"]); seg = np.asarray(tr["segments"])
    lines = lane_polylines(a.town); L = np.concatenate([np.asarray(l) for l in lines if len(l)]); pad = 20
    os.makedirs(a.out_maps, exist_ok=True); os.makedirs(a.fig_dir, exist_ok=True)
    summary = {}
    for name, path in todo.items():
        nodes, assign, old_edges, old_reach = load_nodes_and_assign(name, path, P)
        n = len(nodes)
        edges = sequence_edges(assign, P, yaw, seg)
        el = [(e["from"], e["to"], 1.0) for e in edges]
        reach_dir = directed_reachability(n, el)
        reach_und = directed_reachability(n, el + [(v, u, w) for u, v, w in el])
        lens = np.array([e["driven_length_m"] for e in edges])
        cross = sum(e["crosses_episodes"] for e in edges)
        for k, nd in enumerate(nodes):
            nd["source_id"] = nd.get("id"); nd["id"] = k
        out = {"schema": "carla-rl sequence map v1: nodes = the source map's nodes (id = position); edges = node changes "
                         "along the trace in order ACROSS episodes (trace continues from the nearest waypoint), directed, "
                         "shortest driven span per node pair",
               "town": a.town, "source_map": path, "trace": a.trace,
               "assigned_samples": int((assign >= 0).sum()), "trace_samples": len(P),
               "directed_reachability": reach_dir, "undirected_reachability": reach_und,
               "nodes": nodes, "edges": edges}
        dst = os.path.join(a.out_maps, f"{name}_sequence.json")
        json.dump(out, open(dst, "w"), indent=1)
        summary[name] = {"nodes": n, "old_edges": old_edges, "old_directed_reach": old_reach, "edges": len(edges),
                         "edges_crossing_episodes": cross, "directed_reach": reach_dir, "undirected_reach": reach_und,
                         "driven_length_median_m": float(np.median(lens)) if len(lens) else None,
                         "driven_length_max_m": float(lens.max()) if len(lens) else None, "map": dst}

        xy = np.asarray([[d["x_m"], d["y_m"]] for d in nodes])
        fig, ax = plt.subplots(figsize=(11, 10))
        ax.add_collection(LineCollection(lines, colors="#d6d6d6", linewidths=2.0, zorder=0))
        polys = [np.asarray([[w["x_m"], w["y_m"]] for w in e["waypoints"]]) for e in edges]
        ax.add_collection(LineCollection(polys, colors="#d62728", linewidths=1.4, alpha=0.7, zorder=2))
        for e, pl in zip(edges, polys):
            if len(pl) > 2 and e["driven_length_m"] > 8:
                mid = len(pl) // 2
                ax.annotate("", xy=pl[mid + 1], xytext=pl[mid - 1],
                            arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=1.0), zorder=3)
        ax.scatter(xy[:, 0], xy[:, 1], s=40 if n <= 160 else 12, c="white", edgecolors="black", linewidths=0.7, zorder=4)
        if n <= 160:
            for i in range(n):
                ax.annotate(str(i), xy[i], textcoords="offset points", xytext=(4, 4), fontsize=6, zorder=5)
        ax.scatter([], [], s=40, c="white", edgecolors="black", label=f"Nodes ({name}): {n}")
        ax.plot([], [], "-", color="#d62728", lw=2, label=f"Sequence edges (driven path): {len(edges)} (was {old_edges}), "
                                                          f"{cross} cross an episode reset")
        ax.plot([], [], " ", label=f"Driven edge length median {np.median(lens):.1f} m, max {lens.max():.1f} m; "
                                   f"reach directed {reach_dir:.2f} (was {('n/a' if old_reach is None else f'{old_reach:.2f}')}), undirected {reach_und:.2f}")
        ax.set_xlim(L[:, 0].min() - pad, L[:, 0].max() + pad); ax.set_ylim(L[:, 1].max() + pad, L[:, 1].min() - pad)
        ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), fontsize=8.5, framealpha=0.95)
        fig.savefig(os.path.join(a.fig_dir, f"map_{name}_sequence.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
        s = summary[name]
        print(f"[{name:>22}] {n:>4} nodes | edges {old_edges:>4} -> {len(edges):>4} ({cross} across resets) | "
              f"reach {('n/a' if old_reach is None else f'{old_reach:.3f}')} -> {reach_dir:.3f} (undirected {reach_und:.3f}) | driven median "
              f"{s['driven_length_median_m']:.1f} m max {s['driven_length_max_m']:.1f} m | assigned "
              f"{out['assigned_samples']}/{len(P)}", flush=True)
    summary_name = "sequence_maps_summary.json" if not a.maps else f"sequence_maps_summary_{'_'.join(a.maps)}.json"
    if len(summary_name) > 200:                                  # many maps (SGM ablation): name by count
        summary_name = f"sequence_maps_summary_{len(a.maps)}maps.json"
    json.dump(summary, open(os.path.join(a.fig_dir, summary_name), "w"), indent=1)


if __name__ == "__main__":
    main()
