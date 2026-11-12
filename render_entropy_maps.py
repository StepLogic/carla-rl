"""Entropy map of a trace, with HIGH- and LOW-entropy madent node sets.

Analytic base-Gaussian actor entropy per trace candidate (the protocol's madent
estimator, eval_placement.py's call), then:
  high: mad_spikes(entropy)   -- the madent arm as used in every placement row
  low:  mad_spikes(-entropy)  -- the same detector on negated entropy: states where
        the policy is unusually CONFIDENT (entropy dips z MADs below the median)
Same z threshold, smoothing and min separation for both. Offline: the agent is
built on spaces taken from the saved observations; no CARLA.
"""
import argparse, json, os, pickle, sys
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/placement_v6n_town02_budget10k_laneoff_speed10")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--run-env", default="results/run_env/carla_gated_her_v6n.env")
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--min-sep-m", type=float, default=2.0)
    ap.add_argument("--smooth-w", type=int, default=9)
    ap.add_argument("--out", default="results/figures/entropy_maps")
    a = ap.parse_args()

    for line in open(a.run_env):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); os.environ[k] = v
    os.environ["CARLA_START_SERVER"] = "0"

    from gymnasium import spaces
    import train_carla_tqc as T
    from vae_nav.policy_io import load_agent_state
    from vae_nav.node_placement import actor_entropy, arclength, place
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from render_placement_maps import lane_polylines

    tp = os.path.join(a.run_dir, f"trace_s{a.seed}.pkl")
    tr = pickle.load(open(tp, "rb"))
    P = np.asarray(tr["positions"], np.float64)[:, :2]
    seg = np.asarray(tr["segments"])
    o0 = tr["observations"][0]
    obs = spaces.Dict({k: spaces.Box(-np.inf, np.inf, np.asarray(v).shape, np.float32) for k, v in o0.items()})
    agent = T.build_agent(obs, spaces.Box(-1.0, 1.0, (2,), np.float32), 0)
    ck = tr["meta"]["checkpoint"]
    st = pickle.load(open(ck, "rb")); load_agent_state(agent, st.get("agent", st))

    H = np.asarray(actor_entropy(agent, tr["observations"], seed=0, analytic=True), np.float64)
    s = arclength(tr["positions"], tr["segments"])
    hi, _ = place("madent", s=s, entropy=H, min_sep_m=a.min_sep_m, smooth_w=a.smooth_w, z_thresh=a.z, nms_w=0)
    lo, _ = place("madent", s=s, entropy=-H, min_sep_m=a.min_sep_m, smooth_w=a.smooth_w, z_thresh=a.z, nms_w=0)
    hi, lo = np.asarray(hi, int), np.asarray(lo, int)
    med = float(np.median(H)); mad = float(np.median(np.abs(H - med)))
    print(f"[entropy] {len(H)} candidates: mean {H.mean():.3f} median {med:.3f} MAD {mad:.3f} "
          f"range [{H.min():.3f}, {H.max():.3f}] nats")
    print(f"[madent] z>={a.z:g}: HIGH-entropy nodes {len(hi)} (entropy {H[hi].mean() if len(hi) else float('nan'):.3f} mean) | "
          f"LOW-entropy nodes {len(lo)} (entropy {H[lo].mean() if len(lo) else float('nan'):.3f} mean)")

    lines = lane_polylines(a.town)
    L = np.concatenate([np.asarray(l) for l in lines if len(l)]); pad = 20
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.4))
    ckname = ck.split("/")[-2] + "@" + ck.split("_")[-1].split(".")[0]
    speed = tr["meta"].get("target_speed_kmh") or "20"
    panels = [("Actor entropy along the trace", None, None),
              (f"HIGH entropy: madent z>={a.z:g} ({len(hi)} nodes)", hi, "#d62728"),
              (f"LOW entropy: madent on -entropy, z>={a.z:g} ({len(lo)} nodes)", lo, "#1f77b4")]
    for ax, (title, nodes, colour) in zip(axes, panels):
        ax.add_collection(LineCollection(lines, colors="#d0d0d0", linewidths=2.0, zorder=0))
        for k in np.unique(seg):
            i = np.flatnonzero(seg == k)
            if len(i) < 2:
                continue
            lc = LineCollection(np.stack([P[i][:-1], P[i][1:]], 1), cmap="viridis", linewidths=2.2, zorder=1,
                                alpha=1.0 if nodes is None else 0.45)
            lc.set_array(H[i][:-1]); lc.set_clim(H.min(), H.max()); ax.add_collection(lc)
        if nodes is not None and len(nodes):
            ax.scatter(P[nodes, 0], P[nodes, 1], s=55, c=colour, edgecolors="black", linewidths=0.7, zorder=3,
                       label=f"{len(nodes)} nodes, mean entropy {H[nodes].mean():.3f}")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, framealpha=0.95)
        ax.set_xlim(L[:, 0].min() - pad, L[:, 0].max() + pad); ax.set_ylim(L[:, 1].max() + pad, L[:, 1].min() - pad)
        ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    cb = fig.colorbar(lc, ax=axes, fraction=0.015, pad=0.01); cb.set_label("Actor entropy (nats)")
    (lambda *_a, **_k: None)(  # NO TITLES (user 2026-09-13)
        f"{ckname}  {a.town}  trace s{a.seed} ({tr['meta'].get('collect_action_mode', 'eval')} actions, "
                 f"{speed} km/h)  -- median {med:.3f}, MAD {mad:.3f} nats", fontsize=11)
    os.makedirs(a.out, exist_ok=True)
    tag = os.path.basename(a.run_dir.rstrip("/"))
    out = os.path.join(a.out, f"entropy_madent_hi_lo_{tag}_s{a.seed}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    json.dump({"trace": tp, "checkpoint": ck, "z": a.z, "min_sep_m": a.min_sep_m, "smooth_w": a.smooth_w,
               "entropy_nats": H.tolist(), "median": med, "mad": mad,
               "high": {"node_trace_index": hi.tolist(), "node_xy": P[hi].tolist()},
               "low": {"node_trace_index": lo.tolist(), "node_xy": P[lo].tolist()}},
              open(out.replace(".png", ".json"), "w"))
    print(f"[Render] {out} (+ .json)")


if __name__ == "__main__":
    main()
