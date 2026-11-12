"""Decision- and Coverage-Aware Topological Mapping (user 2026-09-16: new madent variant).

ONLINE AND CAUSAL, which is the point. The existing madent map scores the WHOLE trace, takes a global median/MAD and
then picks spikes; nothing about it could run while driving. This walks the trace once, in order, and decides at each
step using only what it has already seen:

  DECISION trigger  robust z of the policy entropy against a ROLLING window of the last W entropies:
                    r = (U_t - median(B)) / (1.4826 * MAD(B) + eps),  fires when r > tau_ent AND U_t > U_min.
                    U_min is an absolute floor so a spike inside a flat low-entropy stretch does not qualify;
                    it is the RUNNING median of every entropy seen so far (causal; converges to the trace median).
  COVERAGE trigger  the view stopped looking like the last node's: z_t . z_last < tau_loc (unit latents).
                    Always fires for the first sample, when there is no last node.
  COOLDOWN          a node is admitted only if t - t_last > T_cooldown, in trace samples (~2 m apart).

Each node records which trigger fired (decision / coverage / both), so the split can be reported. Sample -> node
assignment is "the node in force at that time" (the most recent admitted node), which is what sequence_link_map.py
needs to build edges; edges themselves are left to it, as for every other driving map.
"""
import argparse, json, pickle
import numpy as np


def select_nodes(entropies, latents_unit, w=50, tau_ent=2.0, tau_loc=0.80, cooldown=3, u_min=None, eps=1e-9):
    """The algorithm, as a pure function. -> [(index, type)], type in {decision, coverage, both}.

    u_min: None  -> the RUNNING median of entropies seen so far (causal).
           float -> that absolute floor.
    """
    U = np.asarray(entropies, np.float64)
    Z = np.asarray(latents_unit, np.float64)
    out, buf = [], []
    last_i, last_z = None, None
    seen = []
    for t in range(len(U)):
        seen.append(float(U[t]))
        floor = float(np.median(seen)) if u_min is None else float(u_min)
        decision = False
        if len(buf) >= w:
            m = float(np.median(buf))
            mad = float(np.median(np.abs(np.asarray(buf) - m)))
            r = (U[t] - m) / (1.4826 * mad + eps)
            decision = bool(r > tau_ent and U[t] > floor)
        coverage = True if last_i is None else bool(float(Z[t] @ last_z) < tau_loc)
        eligible = last_i is None or (t - last_i) > cooldown
        if (decision or coverage) and eligible:
            out.append((t, "both" if (decision and coverage) else ("decision" if decision else "coverage")))
            last_i, last_z = t, Z[t]
        buf.append(float(U[t]))
        if len(buf) > w:
            buf.pop(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--entropy-json", required=True, help="entropy_madent_hi_lo_*.json (per-sample entropy_nats)")
    ap.add_argument("--town", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--tau-ent", type=float, default=2.0)
    ap.add_argument("--tau-loc", type=float, default=0.80)
    ap.add_argument("--cooldown", type=int, default=3, help="in trace samples (~2 m apart)")
    ap.add_argument("--u-min", default="median", help="'median' (running) or an absolute float")
    a = ap.parse_args()

    tr = pickle.load(open(a.trace, "rb"))
    P = np.asarray(tr["positions"], np.float64)[:, :2]
    yaw = np.asarray(tr["yaws"], np.float64); seg = np.asarray(tr["segments"])
    U = np.asarray(json.load(open(a.entropy_json))["entropy_nats"], np.float64)
    Z = np.stack([np.asarray(o["latent"], np.float64) for o in tr["observations"]])
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    if not (len(U) == len(Z) == len(P)):
        raise SystemExit(f"length mismatch: entropy {len(U)}, latents {len(Z)}, positions {len(P)}")

    u_min = None if a.u_min == "median" else float(a.u_min)
    picked = select_nodes(U, Z, w=a.window, tau_ent=a.tau_ent, tau_loc=a.tau_loc, cooldown=a.cooldown, u_min=u_min)
    idx = [t for t, _ in picked]
    # sample -> the node in force at that time (leading samples go to the first node)
    assign = np.full(len(P), -1, int)
    for k, t in enumerate(idx):
        end = idx[k + 1] if k + 1 < len(idx) else len(P)
        assign[t:end] = k
    if idx:
        assign[:idx[0]] = 0

    nodes = []
    for k, (t, kind) in enumerate(picked):
        nodes.append({"id": k, "trace_index": int(t), "x_m": float(P[t, 0]), "y_m": float(P[t, 1]),
                      "yaw_rad": float(yaw[t]), "kind": kind, "entropy_nats": float(U[t]), "episode": int(seg[t]),
                      "member_trace_indices": np.nonzero(assign == k)[0].astype(int).tolist()})
    kinds = {k: sum(1 for _, kk in picked if kk == k) for k in ("decision", "coverage", "both")}
    out = {"schema": "carla-rl decision+coverage node map v1: online causal selection; rolling robust-z entropy "
                     "spike (decision) OR visual novelty against the last node (coverage), with a cooldown",
           "town": a.town, "trace": a.trace, "entropy_json": a.entropy_json,
           "params": {"window": a.window, "tau_ent": a.tau_ent, "tau_loc": a.tau_loc, "cooldown_samples": a.cooldown,
                      "u_min": a.u_min},
           "kind_counts": kinds, "nodes": nodes, "edges": [],
           "note": "edges are built by sequence_link_map.py (trace order across episodes)"}
    json.dump(out, open(a.out, "w"), indent=1)
    gaps = [np.linalg.norm(P[idx[i + 1]] - P[idx[i]]) for i in range(len(idx) - 1)]
    print(f"[dc] {len(nodes)} nodes of {len(P)} samples {kinds} | node spacing median "
          f"{np.median(gaps) if gaps else float('nan'):.1f} m max {max(gaps) if gaps else float('nan'):.1f} m -> {a.out}")


if __name__ == "__main__":
    main()
