"""Uniform[6 m] nodes BETWEEN the thinned high-entropy nodes (user 2026-09-13, another baseline).

Seeds: the entropy_nosmooth_high_thin4 nodes (30). Walking the trace in order, a sample becomes a
uniform node when it is at least --gate-m (Euclidean) from every node placed so far -- the uniform[g]
always_merge rule, run with the entropy nodes already on the map.

Sample assignment (for sequence edges): an entropy node keeps its own and absorbed samples
(member_trace_indices from thin_entropy_map.py); every other sample goes to the NEAREST node within the gate.
Output: driving-map JSON, nodes in trace order with `kind` = entropy | uniform and `member_trace_indices`.
"""
import argparse, json, pickle
import numpy as np

TRACE = "results/placement_v6n_town02_budget10k_laneoff_speed10/trace_s7.pkl"


def place_hybrid(P, seed_idx, seed_members, gate_m):
    """-> (node trace indices in trace order, kinds, members per node).

    seed_idx: trace indices of the entropy nodes; seed_members: list of member-index lists (same order).
    """
    P = np.asarray(P, np.float64)
    kept, kind = [int(i) for i in seed_idx], ["entropy"] * len(seed_idx)
    claimed = {int(t) for m in seed_members for t in m}
    for i in range(len(P)):
        if i in claimed or i in kept:
            continue
        if np.min(np.linalg.norm(P[kept] - P[i], axis=1)) >= gate_m:
            kept.append(i); kind.append("uniform")
    members = [list(map(int, m)) for m in seed_members] + [[k] for k in kept[len(seed_idx):]]
    K = P[kept]
    for i in range(len(P)):
        if i in claimed or i in kept:
            continue
        d = np.linalg.norm(K - P[i], axis=1)
        j = int(np.argmin(d))
        if d[j] < gate_m:
            members[j].append(i)
    order = np.argsort(kept, kind="stable")
    return [kept[o] for o in order], [kind[o] for o in order], [sorted(members[o]) for o in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-map", default="results/maps/town02_speed10_s7_nosmooth_high_thin4_driving_map.json")
    ap.add_argument("--gate-m", type=float, default=6.0)
    ap.add_argument("--trace", default=TRACE)
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--out", default="results/maps/town02_speed10_s7_hybrid_uniform6_entropy_thin4_driving_map.json")
    a = ap.parse_args()
    tr = pickle.load(open(a.trace, "rb"))
    P = np.asarray(tr["positions"], np.float64)[:, :2]; yaw = np.asarray(tr["yaws"]); seg = np.asarray(tr["segments"])
    seed = json.load(open(a.seed_map))["nodes"]
    idx, kinds, members = place_hybrid(P, [d["trace_index"] for d in seed], [d["member_trace_indices"] for d in seed], a.gate_m)
    by_ti = {d["trace_index"]: d for d in seed}
    nodes = []
    for k, (ti, kd, mem) in enumerate(zip(idx, kinds, members)):
        base = {"x_m": float(P[ti, 0]), "y_m": float(P[ti, 1]), "yaw_rad": float(yaw[ti]), "episode": int(seg[ti])}
        if kd == "entropy":
            base.update({kk: v for kk, v in by_ti[ti].items() if kk in ("entropy_nats", "z")})
        nodes.append({"id": k, "trace_index": int(ti), "kind": kd, **base, "member_trace_indices": mem})
    json.dump({"schema": "carla-rl hybrid node map v1: thinned high-entropy nodes (seeds) + uniform[g] nodes between them",
               "town": a.town, "trace": a.trace, "seed_map": a.seed_map, "gate_m": a.gate_m,
               "rule": "trace order; a sample becomes a uniform node when >= gate_m from every node placed so far",
               "assignment": "entropy nodes keep own+absorbed samples; others go to the nearest node within gate_m",
               "nodes": nodes, "edges": [], "note": "edges are built by sequence_link_map.py"},
              open(a.out, "w"), indent=1)
    xy = P[idx]
    D = np.linalg.norm(xy[:, None] - xy[None], axis=-1); np.fill_diagonal(D, np.inf)
    n_un = sum(1 for d in members if not d)
    print(f"[hybrid] {kinds.count('entropy')} entropy + {kinds.count('uniform')} uniform = {len(idx)} nodes (gate {a.gate_m:g} m); "
          f"nearest-node gap median {np.median(D.min(1)):.1f} m min {D.min():.1f} m; assigned samples "
          f"{sum(len(m) for m in members)}/{len(P)}; empty nodes {n_un} -> {a.out}")


if __name__ == "__main__":
    main()
