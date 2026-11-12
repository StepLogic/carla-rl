"""SGM nodes AROUND the thinned high-entropy (madent) seeds -- the SGM twin of
hybrid_uniform_entropy_map.py (user 2026-09-19: "we have madent + uniform spacing, how about
madent + sgm").

Same construction as the Entropy + Spacing hybrid, with SGM's rule in place of the uniform gate:
the madent seeds are PINNED (a seed sample is always kept, never merged) and every other trace
sample goes through the release-faithful SGM frame -- metric gate, two-way consistency over the
gated node's learned edges, learned directed edges under the edge cutoff. Nothing else changes,
so a difference against the plain SGM row can only come from the seeds.

Only the metric gate is varied to match a node budget, exactly as the SGM row does.
Output: driving-map JSON, nodes in trace order, `kind` = entropy | sgm, with `member_trace_indices`
(own + absorbed samples) for sequence_link_map.py.
"""
import argparse, json, os, pickle, sys
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))

TRACE = "results/placement_v6n_town02_budget10k_laneoff_speed10/trace_s7.pkl"
SEED_MAP = "results/maps/town02_speed10_s7_nosmooth_high_thin4_driving_map.json"


def seeded_sgm(dist, P, seeds, consistency, edge_cutoff, gate_m):
    """The sgm_frame_placer loop with pinned seeds -> kept, absorber, learned edges.

    Identical to vae_nav.sgm_frame.sgm_frame_placer except for the two `pinned` lines: a seed
    sample skips the merge test and is added as a node with the same learned edges as any other.
    """
    from vae_nav.sgm_frame import consistency_merge
    test = consistency_merge(consistency)
    ctx = {"pair_distance": lambda i, j: dist[i, j]}
    d = ctx["pair_distance"]
    plane = np.asarray(P, np.float64)[:, :2]
    seeds = set(int(s) for s in seeds)
    absorber = np.arange(len(plane))
    kept, kinds, preds, succs = [], [], [], []
    for i in range(len(plane)):
        pinned = i in seeds
        if kept and not pinned:
            K = np.asarray(kept)
            gate = np.nonzero(np.linalg.norm(plane[K] - plane[i], axis=1) < gate_m)[0]
            merged = False
            for m in gate:
                if test(i, int(m), kept, preds, succs, ctx):
                    absorber[i] = kept[int(m)]; merged = True; break
            if merged:
                continue
        new = len(kept)
        preds.append([]); succs.append([])
        for k in range(new):
            w_in = float(d(kept[k], i))
            if w_in < edge_cutoff:
                succs[k].append((new, w_in)); preds[new].append((k, w_in))
            w_out = float(d(i, kept[k]))
            if w_out < edge_cutoff:
                succs[new].append((k, w_out)); preds[k].append((new, w_out))
        kept.append(i); kinds.append("entropy" if pinned else "sgm")
    edges = [(a, b, float(w)) for a in range(len(kept)) for b, w in succs[a]]
    return kept, kinds, absorber, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-m", type=float, required=True)
    ap.add_argument("--edge-cutoff", type=float, default=55.0)
    ap.add_argument("--consistency", type=float, default=55.0)
    ap.add_argument("--seed-map", default=SEED_MAP)
    ap.add_argument("--trace", default=TRACE)
    ap.add_argument("--out-dir", default="results/maps")
    ap.add_argument("--prefix", default="town02_speed10_s7")
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--name", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the node count only, write nothing")
    a = ap.parse_args()
    from vae_nav.sgm_frame import directed_reachability

    tr = pickle.load(open(a.trace, "rb"))
    P = np.asarray(tr["positions"], np.float64)[:, :2]; yaw = np.asarray(tr["yaws"], np.float64)
    seg = np.asarray(tr["segments"])
    dist = np.load(a.trace + ".dist.npy")
    seed = json.load(open(a.seed_map))["nodes"]
    seeds = [d["trace_index"] for d in seed]
    kept, kinds, absorber, edges = seeded_sgm(dist, P, seeds, a.consistency, a.edge_cutoff, a.gate_m)
    name = a.name or f"hybrid_sgm_g{a.gate_m:g}_e{a.edge_cutoff:g}_c{a.consistency:g}_entropy_thin4"
    if a.dry_run:
        print(f"[hybrid-sgm] gate {a.gate_m:g}: {kinds.count('entropy')} entropy + {kinds.count('sgm')} sgm "
              f"= {len(kept)} nodes")
        return
    nodes = [{"id": k, "trace_index": int(t), "kind": kd,
              "x_m": float(P[t, 0]), "y_m": float(P[t, 1]), "yaw_rad": float(yaw[t]), "episode": int(seg[t]),
              "member_trace_indices": np.nonzero(absorber == t)[0].astype(int).tolist()}
             for k, (t, kd) in enumerate(zip(kept, kinds))]
    reach = directed_reachability(len(kept), edges)
    out = {"schema": "carla-rl hybrid SGM node map v1: pinned madent seeds + release-faithful SGM frame for "
                     "every other sample; member_trace_indices = own + absorbed samples",
           "town": a.town, "trace": a.trace, "seed_map": a.seed_map, "name": name,
           "sgm": {"gate_m": a.gate_m, "edge_cutoff_steps": a.edge_cutoff, "consistency_steps": a.consistency,
                   "learned_edges": len(edges), "learned_directed_reachability": reach,
                   "pinned_seeds": kinds.count("entropy")},
           "rule": "trace order; a madent seed is always kept; every other sample runs the SGM merge test",
           "nodes": nodes, "edges": [],
           "note": "edges are built by sequence_link_map.py (trace order across episodes)"}
    dst = os.path.join(a.out_dir, f"{a.prefix}_{name}_driving_map.json")
    json.dump(out, open(dst, "w"), indent=1)
    xy = P[kept]
    D = np.linalg.norm(xy[:, None] - xy[None], axis=-1); np.fill_diagonal(D, np.inf)
    print(f"[hybrid-sgm] {name}: {kinds.count('entropy')} entropy + {kinds.count('sgm')} sgm = {len(kept)} nodes; "
          f"{len(edges)} learned edges, learned reach {reach:.3f}; nearest-node gap median {np.median(D.min(1)):.1f} m; "
          f"assigned {sum(len(n['member_trace_indices']) for n in nodes)}/{len(P)} samples -> {dst}")


if __name__ == "__main__":
    main()
