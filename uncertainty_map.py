"""Epistemic and aleatoric uncertainty maps of a trace (user 2026-09-14: "create map from trace using epistemic and
aleatoric uncertainty"; design approved: two maps, one per type, driven like every Table III row).

Per trace sample, v6n's EXPLORE critic (goal-free; 5 TQC critics x 25 quantiles) at the policy's own action:
  epistemic = std across critics of each critic's mean value   (critics disagree: little data here)
  aleatoric = mean across critics of the std over its quantiles (the return itself is uncertain here)
Both in V-units. The saved trace observations carry mode = GOAL (1) -- they were stored for the goal critic's
distance matrix -- so the mode bit is set to EXPLORE (0) for the actor and the critic; the trace is not modified.

Nodes: the same madent detector as the entropy maps (vae_nav.node_placement.place("madent"), robust z >= 2,
2 m min separation along the trace, smooth_w = 1), run once per score. Offline, no CARLA.
Writes results/maps/town02_speed10_s7_{epistemic,aleatoric}_high_driving_map.json (edges are built by
sequence_link_map.py), the per-sample scores, and a figure per score.
"""
import argparse, json, os, pickle, sys
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))

TRACE = "results/placement_v6n_town02_budget10k_laneoff_speed10/trace_s7.pkl"
EXPLORE = 0          # vae_nav.gated_tqc_learner: GOAL, EXPLORE = 1, 0


def epistemic(q):
    """(num_qs, B, n_q) -> (B,): std across critics of the per-critic mean."""
    q = np.asarray(q, np.float64)
    return q.mean(axis=2).std(axis=0)


def aleatoric(q):
    """(num_qs, B, n_q) -> (B,): per-critic std over quantiles, averaged over critics."""
    q = np.asarray(q, np.float64)
    return q.std(axis=2).mean(axis=0)


def with_explore_mode(obs):
    """Shallow copy of a batched observation dict with the mode bit set to EXPLORE."""
    out = dict(obs)
    out["mode"] = np.full_like(np.asarray(obs["mode"], np.float32), EXPLORE)
    return out


def explore_quantiles(agent, obs):
    """Explore-critic quantiles at the explore-mode policy action. -> (num_qs, B, n_q)."""
    ob = with_explore_mode(obs)
    actions = agent.eval_actions(ob)
    critic = agent._critic_e
    return np.asarray(critic.apply_fn({"params": critic.params}, ob, actions), np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=TRACE)
    ap.add_argument("--run-env", default="results/run_env/carla_gated_her_v6n.env")
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--min-sep-m", type=float, default=2.0)
    ap.add_argument("--smooth-w", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--maps-dir", default="results/maps")
    ap.add_argument("--out", default="results/figures/uncertainty_maps")
    a = ap.parse_args()

    for line in open(a.run_env):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); os.environ[k] = v
    os.environ["CARLA_START_SERVER"] = "0"

    from gymnasium import spaces
    import train_carla_tqc as T
    from vae_nav.policy_io import load_agent_state
    from vae_nav.node_placement import arclength, place, smooth
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from render_placement_maps import lane_polylines

    tr = pickle.load(open(a.trace, "rb"))
    P = np.asarray(tr["positions"], np.float64)[:, :2]
    yaw = np.asarray(tr["yaws"], np.float64); seg = np.asarray(tr["segments"])
    O = tr["observations"]
    obs_space = spaces.Dict({k: spaces.Box(-np.inf, np.inf, np.asarray(v).shape, np.float32) for k, v in O[0].items()})
    agent = T.build_agent(obs_space, spaces.Box(-1.0, 1.0, (2,), np.float32), 0)
    ck = tr["meta"]["checkpoint"]
    st = pickle.load(open(ck, "rb")); load_agent_state(agent, st.get("agent", st))
    if not hasattr(agent, "_critic_e"):
        raise SystemExit(f"{type(agent).__name__} has no explore critic (_critic_e): not a gated agent")

    ep, al = [], []
    for i in range(0, len(O), a.batch):
        chunk = O[i:i + a.batch]
        batch = {k: np.stack([np.asarray(o[k], np.float32) for o in chunk]) for k in chunk[0]}
        q = explore_quantiles(agent, batch)
        ep.append(epistemic(q)); al.append(aleatoric(q))
    scores = {"epistemic": np.concatenate(ep), "aleatoric": np.concatenate(al)}
    s = arclength(tr["positions"], tr["segments"])

    lines = lane_polylines("Town02")
    L = np.concatenate([np.asarray(l) for l in lines if len(l)]); pad = 20
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.maps_dir, exist_ok=True)
    summary = {}
    for name, U in scores.items():
        nodes, _ = place("madent", s=s, entropy=U, min_sep_m=a.min_sep_m, smooth_w=a.smooth_w, z_thresh=a.z, nms_w=0)
        nodes = np.sort(np.asarray(nodes, int))
        us = smooth(U, a.smooth_w); c = float(np.median(us)); scale = 1.4826 * float(np.median(np.abs(us - c)))
        z = (us - c) / scale
        m = {"schema": "carla-rl uncertainty node map v1: nodes = madent-selected trace samples on an explore-critic "
                       "uncertainty score (edges built by sequence_link_map.py)",
             "town": "Town02", "coordinates": "CARLA world metres (left-handed; plot y inverted for top-down)",
             "selection": f"madent high-{name} (z>={a.z:g}, smooth_w={a.smooth_w})",
             "source": {"trace": a.trace, "checkpoint": ck, "trace_meta": tr["meta"], "critic": "explore (_critic_e)",
                        "mode_bit": "EXPLORE (0) for actor and critic",
                        "score": {"epistemic": "std over critics of per-critic mean quantile",
                                  "aleatoric": "mean over critics of std over quantiles"}[name],
                        "madent": {"z": a.z, "min_sep_m": a.min_sep_m, "smooth_w": a.smooth_w, "signal": name, "nms_w": 0}},
             "nodes": [{"id": k, "trace_index": int(t), "x_m": float(P[t, 0]), "y_m": float(P[t, 1]),
                        "yaw_rad": float(yaw[t]), f"{name}_v": float(U[t]), "z": float(z[t]), "episode": int(seg[t])}
                       for k, t in enumerate(nodes)],
             "edges": []}
        dst = os.path.join(a.maps_dir, f"town02_speed10_s7_{name}_high_driving_map.json")
        json.dump(m, open(dst, "w"), indent=1)
        summary[name] = {"nodes": len(nodes), "median": c, "mad_scale": scale, "min": float(U.min()), "max": float(U.max()),
                         "node_mean": float(U[nodes].mean()) if len(nodes) else None, "map": dst}
        print(f"[{name}] {len(U)} samples: median {c:.3f} robust scale {scale:.3f} range [{U.min():.3f}, {U.max():.3f}] V "
              f"| madent z>={a.z:g}: {len(nodes)} nodes -> {dst}", flush=True)

        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.add_collection(LineCollection(lines, colors="#d0d0d0", linewidths=2.0, zorder=0))
        lc = None
        for k in np.unique(seg):
            i = np.flatnonzero(seg == k)
            if len(i) < 2:
                continue
            lc = LineCollection(np.stack([P[i][:-1], P[i][1:]], 1), cmap="viridis", linewidths=2.2, zorder=1, alpha=0.55)
            lc.set_array(U[i][:-1]); lc.set_clim(U.min(), U.max()); ax.add_collection(lc)
        if len(nodes):
            ax.scatter(P[nodes, 0], P[nodes, 1], s=55, c="#d62728", edgecolors="black", linewidths=0.7, zorder=3,
                       label=f"High-{name} nodes ({len(nodes)})")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8, framealpha=0.95)
        ax.set_xlim(L[:, 0].min() - pad, L[:, 0].max() + pad); ax.set_ylim(L[:, 1].max() + pad, L[:, 1].min() - pad)
        ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.02); cb.set_label(f"{name.capitalize()} uncertainty (V)")
        fig.savefig(os.path.join(a.out, f"{name}_high_nodes.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

    corr = float(np.corrcoef(scores["epistemic"], scores["aleatoric"])[0, 1])
    summary["pearson_epistemic_aleatoric"] = corr
    np.savez(os.path.join(a.out, "uncertainty_scores.npz"), **scores)
    json.dump(summary, open(os.path.join(a.out, "uncertainty_summary.json"), "w"), indent=1)
    print(f"[uncertainty] correlation epistemic vs aleatoric {corr:.3f}; figures and scores -> {a.out}")


if __name__ == "__main__":
    main()
