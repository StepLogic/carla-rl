"""Top-down plot of the madent graph on a placement run, rebuilt offline.

madent's nodes were never saved, so they are REBUILT exactly as
eval_placement.py builds them -- analytic base-Gaussian actor entropy over the
trace's observations (seed 0), then mad_spikes(z, min_sep, smooth, nms=0) --
from the run's checkpoint, WITHOUT CARLA: the agent is built on spaces taken
from the saved observations (Dict of Boxes) and a 2-d action Box (the analytic
base entropy does not depend on the action bounds). The rebuilt node and edge
counts must equal the logged row or the script aborts.

Drawn: Town lanes, the trace coloured by actor entropy, the madent nodes with
their 15 m localisation radius, a sample of learned edges, and the 60 episode
starts/goals marked by whether the graph could attach them.
"""
import argparse, json, os, pickle, sys

import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))


def overlay_env(path):
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/placement_v6n_town02_budget10k")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--run-env", default="results/run_env/carla_gated_her_v6n.env")
    ap.add_argument("--localize-radius", type=float, default=15.0)
    ap.add_argument("--out", default="results/figures/placement_v6n_town02_budget10k")
    a = ap.parse_args()

    overlay_env(a.run_env)
    os.environ.setdefault("CARLA_START_SERVER", "0")
    row = json.load(open(os.path.join(a.run_dir, f"v6n__madent__20-50m__s{a.seed}.json")))
    calib = json.load(open(os.path.join(a.run_dir, "calibration.json")))["chosen"]
    tp = os.path.join(a.run_dir, f"trace_s{a.seed}.pkl")
    tr = pickle.load(open(tp, "rb"))
    dist = np.load(tp + ".dist.npy")
    P = np.asarray(tr["positions"], np.float64)[:, :2]
    seg = np.asarray(tr["segments"])

    from gymnasium import spaces
    import train_carla_tqc as T
    from vae_nav.policy_io import load_agent_state
    from vae_nav.node_placement import actor_entropy, arclength, place
    from vae_nav.sgm_frame import learned_edges
    o0 = tr["observations"][0]
    obs_space = spaces.Dict({k: spaces.Box(-np.inf, np.inf, np.asarray(v).shape, np.float32)
                             for k, v in o0.items()})
    act_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
    agent = T.build_agent(obs_space, act_space, row.get("seed", 0) if "seed" in row else 0)
    st = pickle.load(open(row["checkpoint"], "rb"))
    load_agent_state(agent, st.get("agent", st))

    H = actor_entropy(agent, tr["observations"], seed=0, analytic=True)
    s = arclength(tr["positions"], tr["segments"])
    nodes, _ = place("madent", s=s, entropy=H, min_sep_m=float(row["min_sep_m"]),
                     smooth_w=int(row["smooth_w"]), z_thresh=float(row["mad_z"]), nms_w=0)
    nodes = np.asarray(nodes, dtype=int)
    edges = learned_edges(dist, nodes, float(calib["edge_cutoff"]))
    print(f"[madent] rebuilt {len(nodes)} nodes / {len(edges)} edges; row logged "
          f"{row['n_nodes']} / {row['n_edges']}; entropy {H.mean():.3f} [{H.min():.3f}, {H.max():.3f}]")
    if len(nodes) != row["n_nodes"] or len(edges) != row["n_edges"]:
        raise SystemExit("rebuild does not match the logged row -- not drawing a different graph")

    # episodes: the harness's seeded sampler, verbatim
    rng = np.random.default_rng(int(row.get("episode_seed", 0)))
    eps, tries = [], 0
    while len(eps) < row["episodes"] and tries < 200 * row["episodes"]:
        tries += 1
        i = int(rng.integers(0, len(P)))
        d = np.linalg.norm(P - P[i], axis=1)
        ok = np.flatnonzero((d >= 20.0) & (d <= 50.0))
        if len(ok) == 0:
            continue
        j = int(rng.choice(ok)); rng.integers(0, 2 ** 31 - 1)
        eps.append((i, j))
    N = P[nodes]
    near = lambda x: np.linalg.norm(N - x, axis=1).min()
    start_ok = np.array([near(P[i]) <= a.localize_radius for i, _ in eps])
    goal_ok = np.array([near(P[j]) <= a.localize_radius for _, j in eps])
    print(f"[madent] starts with a node within {a.localize_radius:g} m: {start_ok.sum()}/60 "
          f"(row: {60 - row['end_reasons'].get('start_unlocalised', 0)}); goals: {goal_ok.sum()}/60; "
          f"both: {(start_ok & goal_ok).sum()} (row routed {row['n_routed']})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from render_placement_maps import lane_polylines, finish

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.add_collection(LineCollection(lane_polylines(a.town), colors="#d0d0d0", linewidths=2.2, zorder=0))
    ei = np.array([(u, v) for u, v, *_ in edges], dtype=int)
    ax.add_collection(LineCollection(np.stack([N[ei[:, 0]], N[ei[:, 1]]], axis=1), colors="#9467bd",
                                     linewidths=0.4, alpha=0.18, zorder=1))
    for k in np.unique(seg):
        idx = np.flatnonzero(seg == k)
        pts = P[idx]
        lc = LineCollection(np.stack([pts[:-1], pts[1:]], axis=1), cmap="viridis", linewidths=2.0, zorder=2)
        lc.set_array(H[idx][:-1]); lc.set_clim(H.min(), H.max())
        ax.add_collection(lc)
    cb = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Actor entropy (nats)")
    for x, y in N:
        ax.add_patch(plt.Circle((x, y), a.localize_radius, color="#d62728", fill=False, lw=0.6, alpha=0.5, zorder=3))
    ax.scatter(N[:, 0], N[:, 1], s=46, c="#d62728", edgecolors="black", linewidths=0.6, zorder=5,
               label=f"madent nodes ({len(nodes)}), {a.localize_radius:g} m attach radius")
    S = np.array([P[i] for i, _ in eps]); G = np.array([P[j] for _, j in eps])
    ax.scatter(S[start_ok, 0], S[start_ok, 1], marker="^", s=40, c="#2ca02c", zorder=6, label=f"Episode start attached ({start_ok.sum()})")
    ax.scatter(S[~start_ok, 0], S[~start_ok, 1], marker="^", s=40, facecolors="none", edgecolors="#2ca02c", zorder=6, label=f"Episode start not attached ({(~start_ok).sum()})")
    ax.scatter(G[goal_ok, 0], G[goal_ok, 1], marker="*", s=70, c="#ff7f0e", zorder=6, label=f"Episode goal attached ({goal_ok.sum()})")
    ax.scatter(G[~goal_ok, 0], G[~goal_ok, 1], marker="*", s=70, facecolors="none", edgecolors="#ff7f0e", zorder=6, label=f"Episode goal not attached ({(~goal_ok).sum()})")
    ax.plot([], [], color="#9467bd", label=f"Learned edges ({len(edges)})")
    ck = row["checkpoint"]; ckname = ck.split("/")[-2] + "@" + ck.split("_")[-1].split(".")[0]
    finish(ax, P, f"{ckname}  {a.town}  trace s{a.seed}  madent (MAD z>={row['mad_z']:g}): "
                  f"routed {row['n_routed']}/60, SR {row['success']:.3f}\n"
                  f"Trace coloured by actor entropy; nodes are entropy spikes")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"{a.town.lower()}_v6n_s{a.seed}_madent.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    json.dump({"node_trace_index": nodes.tolist(), "node_xy": N.tolist(), "entropy": H.tolist(),
               "edges_local": [[int(u), int(v), float(w)] for u, v, w in edges],
               "episodes": [{"start_xy": P[i].tolist(), "goal_xy": P[j].tolist(),
                             "start_attached": bool(so), "goal_attached": bool(go)}
                            for (i, j), so, go in zip(eps, start_ok, goal_ok)]},
              open(out.replace(".png", ".json"), "w"))
    print(f"[Render] {out} (+ .json)")


if __name__ == "__main__":
    main()
