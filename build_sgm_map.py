"""SGM driving map on the speed10 lane-off trace (user 2026-09-14: "ablate SGM and find best threshold").

Nodes come from the release-faithful SGM frame (vae_nav.sgm_frame.sgm_faithful: metric gate, two-way
consistency over the gated node's learned edges, learned directed edges under the edge cutoff), read from the
trace's cached critic matrix (trace_s7.pkl.dist.npy, steps). The merge callable is wrapped so each sample
records the node that absorbed it -- the first gated node passing the test, as the placer's loop breaks there.

Output: a driving-map JSON with nodes in trace order; each has `member_trace_indices` (own + absorbed samples),
which sequence_link_map.py uses to assign trace samples. SGM's learned edges are counted, not stored: the
driven maps use sequence edges (sequence_link_map.py), like every other row of the table.
"""
import argparse, json, os, pickle, sys
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))

TRACE = "results/placement_v6n_town02_budget10k_laneoff_speed10/trace_s7.pkl"


def map_name(gate_m, edge_cutoff, consistency):
    return f"sgm_g{gate_m:g}_e{edge_cutoff:g}_c{consistency:g}"


def sgm_with_members(dist, P, consistency, edge_cutoff, gate_m):
    """-> kept trace indices, absorber (trace index of the node holding each sample), learned edges (local)."""
    from vae_nav.sgm_frame import consistency_merge, sgm_frame_placer
    test = consistency_merge(consistency)
    absorber = np.arange(len(P))

    def merge(i, m, kept, preds, succs, ctx):
        ok = test(i, m, kept, preds, succs, ctx)
        if ok:
            absorber[i] = kept[m]
        return ok

    ctx = {"pair_distance": lambda i, j: dist[i, j]}
    kept = np.asarray(sgm_frame_placer(merge, edge_cutoff=edge_cutoff, gate_m=gate_m)(P, ctx), int)
    return kept, absorber, ctx["sgm_edges"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-m", type=float, required=True)
    ap.add_argument("--edge-cutoff", type=float, required=True)
    ap.add_argument("--consistency", type=float, required=True)
    ap.add_argument("--trace", default=TRACE)
    ap.add_argument("--out-dir", default="results/maps")
    ap.add_argument("--prefix", default="town02_speed10_s7")
    ap.add_argument("--town", default="Town02")
    a = ap.parse_args()
    from vae_nav.sgm_frame import directed_reachability

    tr = pickle.load(open(a.trace, "rb"))
    P = np.asarray(tr["positions"], np.float64)[:, :2]; yaw = np.asarray(tr["yaws"], np.float64)
    dist = np.load(a.trace + ".dist.npy")
    kept, absorber, edges = sgm_with_members(dist, P, a.consistency, a.edge_cutoff, a.gate_m)
    nodes = [{"id": k, "trace_index": int(t), "x_m": float(P[t, 0]), "y_m": float(P[t, 1]), "yaw_rad": float(yaw[t]),
              "member_trace_indices": np.nonzero(absorber == t)[0].astype(int).tolist()}
             for k, t in enumerate(kept)]
    name = map_name(a.gate_m, a.edge_cutoff, a.consistency)
    reach = directed_reachability(len(kept), edges)
    out = {"schema": "carla-rl SGM node map v1: nodes = release-faithful SGM frame on the trace's critic matrix; "
                     "member_trace_indices = own + absorbed samples",
           "town": a.town, "trace": a.trace, "name": name,
           "sgm": {"gate_m": a.gate_m, "edge_cutoff_steps": a.edge_cutoff, "consistency_steps": a.consistency,
                   "learned_edges": len(edges), "learned_directed_reachability": reach},
           "nodes": nodes, "edges": [],
           "note": "edges are built by sequence_link_map.py (trace order across episodes)"}
    dst = os.path.join(a.out_dir, f"{a.prefix}_{name}_driving_map.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(f"[sgm] {name}: {len(kept)}/{len(P)} nodes, {len(edges)} learned edges, learned reach {reach:.3f} -> {dst}")


if __name__ == "__main__":
    main()
