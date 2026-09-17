"""Export per-arm node sets, edges and per-episode routing for a placement run.

The result JSONs hold summary fields only and the harness NEVER RECORDED DRIVEN
PATHS. Nodes and edges are REBUILT offline exactly as eval_placement.py builds
them (sgm_faithful at the row's calibration and gate; madent from analytic
actor entropy; uniform/random via place() at the matched count, seed 0; one
edge rule, learned_edges at the row's cutoff) and each arm is checked against
its logged node and edge counts -- a mismatch aborts.

Per episode: start/goal from the harness's seeded sampler; `routed` recomputed
with the harness's own plan() on the rebuilt graph (15 m localisation) and
checked against n_routed; `success` for ROUTED episodes from the row's legs
field (appended in routed-episode order). No-map (bypassed) outcomes are only
recorded in aggregate, so their per-episode success is null.
"""
import argparse, json, os, pickle, sys

import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))

ARMS = ["sgm", "madent", "uniform_at_sgm", "random_at_sgm", "uniform_at_madent", "random_at_madent"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run-env", default="results/run_env/carla_gated_her_v6n.env")
    ap.add_argument("--localize-radius", type=float, default=15.0)
    a = ap.parse_args()

    for line in open(a.run_env):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); os.environ[k] = v
    os.environ["CARLA_START_SERVER"] = "0"

    import networkx as nx
    from gymnasium import spaces
    import train_carla_tqc as T
    from vae_nav.policy_io import load_agent_state
    from vae_nav.node_placement import actor_entropy, arclength, place
    from vae_nav.sgm_frame import sgm_faithful, learned_edges
    from eval_placement import plan

    rows = {arm: json.load(open(os.path.join(a.run_dir, f"v6n__{arm}__20-50m__s{a.seed}.json"))) for arm in ARMS}
    r0 = rows["sgm"]
    tp = os.path.join(a.run_dir, f"trace_s{a.seed}.pkl")
    tr = pickle.load(open(tp, "rb")); dist = np.load(tp + ".dist.npy")
    P = np.asarray(tr["positions"], np.float64)[:, :2]
    s = arclength(tr["positions"], tr["segments"])
    edge_cut = float(r0["sgm_edge_cutoff"]); cons = float(r0["sgm_consistency"]); gate = float(r0["sgm_gate_m"])

    sgm_nodes, sgm_edges = sgm_faithful(dist, P, consistency_cutoff=cons, edge_cutoff=edge_cut, gate_m=gate)
    o0 = tr["observations"][0]
    obs = spaces.Dict({k: spaces.Box(-np.inf, np.inf, np.asarray(v).shape, np.float32) for k, v in o0.items()})
    agent = T.build_agent(obs, spaces.Box(-1.0, 1.0, (2,), np.float32), 0)
    st = pickle.load(open(r0["checkpoint"], "rb")); load_agent_state(agent, st.get("agent", st))
    H = actor_entropy(agent, tr["observations"], seed=0, analytic=True)
    madent_nodes, _ = place("madent", s=s, entropy=H, min_sep_m=float(rows["madent"]["min_sep_m"]),
                            smooth_w=int(rows["madent"]["smooth_w"]), z_thresh=float(rows["madent"]["mad_z"]), nms_w=0)

    built = {"sgm": (np.asarray(sgm_nodes, int), sgm_edges), "madent": (np.asarray(madent_nodes, int), None)}
    for arm in ("uniform_at_sgm", "random_at_sgm", "uniform_at_madent", "random_at_madent"):
        m, ref = arm.split("_at_")
        n = len(built[ref][0])
        nodes, _ = place(m, s=s, n=n, seed=0, min_sep_m=float(rows[arm]["min_sep_m"]), smooth_w=int(rows[arm]["smooth_w"]))
        built[arm] = (np.asarray(nodes, int), None)

    # episodes: the harness's seeded sampler, verbatim
    rng = np.random.default_rng(int(r0.get("episode_seed", 0)))
    eps, tries = [], 0
    while len(eps) < r0["episodes"] and tries < 200 * r0["episodes"]:
        tries += 1
        i = int(rng.integers(0, len(P)))
        d = np.linalg.norm(P - P[i], axis=1)
        ok = np.flatnonzero((d >= 20.0) & (d <= 50.0))
        if len(ok) == 0:
            continue
        j = int(rng.choice(ok)); rng.integers(0, 2 ** 31 - 1)
        eps.append((i, j))

    names = []
    for arm in ARMS:
        nodes, edges = built[arm]
        if edges is None:
            edges = learned_edges(dist, nodes, edge_cut)
        row = rows[arm]
        if len(nodes) != row["n_nodes"] or len(edges) != row["n_edges"]:
            raise SystemExit(f"{arm}: rebuilt {len(nodes)} nodes / {len(edges)} edges, row logged "
                             f"{row['n_nodes']} / {row['n_edges']} -- not exporting a different graph")
        G = nx.DiGraph()
        G.add_nodes_from(int(n) for n in nodes)
        for u, v, w in edges:
            G.add_edge(int(nodes[u]), int(nodes[v]), steps=max(float(w), 1.0))
        routed = [plan(G, nodes, P, P[i], P[j], a.localize_radius)[0] is not None for i, j in eps]
        # ROUTED IS RECOMPUTED FROM TRACE POSITIONS; the harness decided it from
        # the simulator's snapped spawn/goal after reset. They agree on arms that
        # route all 60, but an episode near the 15 m radius can flip (measured:
        # uniform_at_madent 38 recomputed vs 37 logged). When the count differs,
        # legs cannot be aligned to episodes, so per-episode success stays null.
        aligned = sum(routed) == row["n_routed"]
        if not aligned:
            print(f"[export] {arm}: recomputed {sum(routed)} routed vs logged {row['n_routed']} -- "
                  f"nodes/edges exported, per-episode success left null", flush=True)
        legs = iter(row["legs"])
        episodes = []
        for (i, j), r in zip(eps, routed):
            rec = {"start_xy": P[i].tolist(), "goal_xy": P[j].tolist(),
                   "routed": bool(r) if aligned else None, "routed_recomputed": bool(r),
                   "success": None, "driven_path_xy": None}
            if r and aligned:
                rec["success"] = bool(next(legs)[2])
            episodes.append(rec)
        out = {"arm": arm, "town": r0["town"], "seed": a.seed, "checkpoint": r0["checkpoint"],
               "sgm_gate_m": gate, "edge_cutoff_steps": edge_cut, "sgm_consistency": cons,
               "mad_z": row["mad_z"], "node_trace_index": nodes.tolist(), "node_xy": P[nodes].tolist(),
               "edges": [[int(nodes[u]), int(nodes[v]), float(w)] for u, v, w in edges],
               "edges_note": "[from_trace_index, to_trace_index, critic_steps]; directed",
               "episodes": episodes,
               "note": "driven paths were never recorded by the harness (driven_path_xy null); "
                       "per-episode success known for routed episodes only; no-map outcomes are aggregate "
                       f"(success_bypassed={row['success_bypassed']}, n_bypassed={row['n_bypassed']})"}
        name = f"nodes_{arm}_s{a.seed}.json"
        json.dump(out, open(os.path.join(a.run_dir, name), "w"))
        names.append(name)
        print(f"[export] {arm}: {len(nodes)} nodes, {len(edges)} edges, routed {sum(routed)}, "
              f"routed successes {sum(1 for e in episodes if e['success'])} -> {name}", flush=True)
    print("[export] files:", " ".join(names))


if __name__ == "__main__":
    main()
