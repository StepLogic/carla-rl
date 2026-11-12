"""Spatial thinning AFTER the MAD filter (user 2026-09-13: "entropy no smooth high with thinning after filter").

The madent filter's min separation is measured along the trace (arc length), so repeat passes over one
place all survive (median nearest-node distance 0.6 m on nosmooth-high). This pass thins in SPACE:
greedy over nodes by descending entropy; a node within --radius-m (Euclidean) of an already-kept node is
removed and absorbed by the FIRST kept node that suppressed it (the highest-entropy one within the radius).

Output: a driving-map JSON with the kept nodes in trace order; each keeps its own fields plus
`member_trace_indices` (its own sample and every absorbed node's sample), which sequence_link_map.py uses
to assign trace samples. Edges are left to sequence_link_map.py.
"""
import argparse, json
import numpy as np


def thin(xy, entropy, radius_m, max_nodes=0):
    """-> kept indices (in greedy order) and absorber[i] (index of the kept node absorbing node i).

    max_nodes caps the greedy at the N highest-scoring well-separated candidates (user 2026-09-20: the
    signals are only comparable as seeds at the SAME seed count, because damage tracks the seed share of
    the budget). Everything past the cap is absorbed by its nearest kept node, so no sample is orphaned.
    """
    order = np.argsort(-np.asarray(entropy, np.float64), kind="stable")
    kept, absorber = [], np.full(len(xy), -1, int)
    for i in order:
        hit = next((j for j in kept if np.linalg.norm(xy[i] - xy[j]) < radius_m), None)
        if hit is None and (not max_nodes or len(kept) < max_nodes):
            kept.append(int(i)); absorber[i] = i
        elif hit is not None:
            absorber[i] = hit
        else:
            absorber[i] = int(kept[int(np.argmin([np.linalg.norm(xy[i] - xy[j]) for j in kept]))])
    return kept, absorber


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="results/maps/town02_speed10_s7_nosmooth_high_driving_map.json")
    ap.add_argument("--radius-m", type=float, default=4.0)
    ap.add_argument("--out", default="results/maps/town02_speed10_s7_nosmooth_high_thin4_driving_map.json")
    ap.add_argument("--score-field", default="auto", help="auto = entropy_nats | epistemic_v | aleatoric_v, else z")
    ap.add_argument("--max-nodes", type=int, default=0, help="keep only the N highest-scoring separated nodes")
    a = ap.parse_args()
    m = json.load(open(a.map))
    N = m["nodes"]
    xy = np.asarray([[d["x_m"], d["y_m"]] for d in N], np.float64)
    field = a.score_field
    if field == "auto":
        field = next((f for f in ("entropy_nats", "epistemic_v", "aleatoric_v", "z") if f in N[0]), None)
        if field is None:
            raise SystemExit(f"no score field in {a.map}; pass --score-field")
    H = np.asarray([d[field] for d in N], np.float64)
    kept, absorber = thin(xy, H, a.radius_m, a.max_nodes)
    nodes = []
    for new_id, k in enumerate(sorted(kept, key=lambda k: N[k]["trace_index"])):
        d = dict(N[k])
        d["source_id"] = d.get("id"); d["id"] = new_id
        # UNION THE INPUT'S OWN MEMBERS when it has them (the decision+coverage map assigns EVERY trace sample to a
        # node, not just the node's own sample). Keeping only node samples would throw that assignment away and leave
        # sequence_link_map.py linking node samples alone. Falls back to the node sample for maps without members.
        members = set()
        for i in range(len(N)):
            if absorber[i] == k:
                members.update(int(t) for t in N[i].get("member_trace_indices", [N[i]["trace_index"]]))
        d["member_trace_indices"] = sorted(members)
        nodes.append(d)
    out = {"schema": "carla-rl entropy node map v1 (thinned): nodes = madent-selected candidates after SPATIAL greedy "
                     "thinning by descending entropy; member_trace_indices = own + absorbed node samples",
           "town": m.get("town"), "coordinates": m.get("coordinates"), "selection": m.get("selection"),
           "source": m.get("source"), "source_map": a.map, "thinning": {"radius_m": a.radius_m, "metric": "euclidean",
           "order": "descending entropy", "absorb": "first kept node within radius"},
           "nodes": nodes, "edges": [],
           "note": "edges are built by sequence_link_map.py (trace order across episodes)"}
    json.dump(out, open(a.out, "w"), indent=1)
    gaps = [min(np.linalg.norm(xy[i] - xy[j]) for j in kept if j != i) for i in kept]
    print(f"[thin] {len(N)} -> {len(nodes)} nodes at {a.radius_m:g} m; kept-node nearest gap median {np.median(gaps):.1f} m "
          f"min {np.min(gaps):.1f} m; largest cluster {max(len(d['member_trace_indices']) for d in nodes)} -> {a.out}")


if __name__ == "__main__":
    main()
