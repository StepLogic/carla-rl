"""Sample start/goal pairs from a trace and plot them on the town map (no title).

Sampler = eval_placement.py's --episodes-from-trace, verbatim: start = a random trace
candidate; goal = a random candidate whose STRAIGHT-LINE distance from the start is in
[goal_min, goal_max]; seeded numpy generator. Both ends are real trace candidates, so
their stored observations can be used as goals. Writes the pairs to JSON with the
per-episode env seed the harness would use, for a later driving run.
"""
import argparse, json, os, pickle, sys
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/placement_v6n_town02_budget10k_laneoff_speed10")
    ap.add_argument("--seed", type=int, default=7, help="trace collect seed (file name)")
    ap.add_argument("--episode-seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--goal-min", type=float, default=20.0)
    ap.add_argument("--goal-max", type=float, default=50.0)
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--out", default="results/figures/trace_episodes")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from render_placement_maps import lane_polylines

    tp = os.path.join(a.run_dir, f"trace_s{a.seed}.pkl")
    tr = pickle.load(open(tp, "rb"))
    P = np.asarray(tr["positions"], np.float64)[:, :2]
    yaw = np.asarray(tr["yaws"], np.float64); seg = np.asarray(tr["segments"])

    rng = np.random.default_rng(int(a.episode_seed))
    eps, tries = [], 0
    while len(eps) < a.n and tries < 200 * a.n:
        tries += 1
        i = int(rng.integers(0, len(P)))
        d = np.linalg.norm(P - P[i], axis=1)
        ok = np.flatnonzero((d >= a.goal_min) & (d <= a.goal_max))
        if len(ok) == 0:
            continue
        j = int(rng.choice(ok))
        env_seed = int(rng.integers(0, 2 ** 31 - 1))
        eps.append({"index": len(eps), "env_seed": env_seed,
                    "start_trace_index": i, "start_xy": P[i].tolist(), "start_yaw_rad": float(yaw[i]),
                    "goal_trace_index": j, "goal_xy": P[j].tolist(), "goal_yaw_rad": float(yaw[j]),
                    "straight_line_m": float(d[j])})

    # route distance along the road network (offline xodr lane graph, directed)
    try:
        import carla
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        from scipy.spatial import cKDTree
        m = carla.Map(a.town, open(f"/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive/{a.town}.xodr").read())
        xy, yw, rows, cols, first, last, ents, exs = [], [], [], [], [], [], [], []
        for en, ex in m.get_topology():
            ch = [en] + en.next_until_lane_end(1.0); b = len(xy)
            for w in ch:
                xy.append([w.transform.location.x, w.transform.location.y]); yw.append(np.deg2rad(w.transform.rotation.yaw))
            rows += list(range(b, b + len(ch) - 1)); cols += list(range(b + 1, b + len(ch)))
            first.append(b); last.append(b + len(ch) - 1)
            ents.append([en.transform.location.x, en.transform.location.y]); exs.append([ex.transform.location.x, ex.transform.location.y])
        xy, yw = np.asarray(xy), np.asarray(yw); et = cKDTree(np.asarray(ents))
        for s_, e_ in enumerate(exs):
            for t_ in et.query_ball_point(e_, 1.0):
                if t_ != s_:
                    rows.append(last[s_]); cols.append(first[t_])
        lens = [max(np.linalg.norm(xy[c] - xy[r]), 1e-3) for r, c in zip(rows, cols)]
        G = csr_matrix((lens, (rows, cols)), shape=(len(xy), len(xy))); tree = cKDTree(xy)
        for e in eps:
            sx, gx = np.asarray(e["start_xy"]), np.asarray(e["goal_xy"])
            cand = [c for c in tree.query_ball_point(sx, 3.0) if abs(np.angle(np.exp(1j * (yw[c] - e["start_yaw_rad"])))) < np.deg2rad(60)]
            tgt = tree.query_ball_point(gx, 2.0)
            if cand and tgt:
                src = min(cand, key=lambda c: np.linalg.norm(xy[c] - sx))
                dd = dijkstra(G, directed=True, indices=src)
                e["route_m"] = float(np.min(dd[tgt])) if np.isfinite(np.min(dd[tgt])) else None
            else:
                e["route_m"] = None
    except Exception as ex:                                            # noqa: BLE001
        print(f"[route] skipped: {type(ex).__name__}: {ex}")

    lines = lane_polylines(a.town)
    L = np.concatenate([np.asarray(l) for l in lines if len(l)]); pad = 20
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.add_collection(LineCollection(lines, colors="#d0d0d0", linewidths=2.0, zorder=0))
    ax.add_collection(LineCollection([P[seg == k] for k in np.unique(seg)], colors="#1f77b4", linewidths=0.8,
                                     alpha=0.35, zorder=1, label="Exploration trace"))
    cmap = plt.get_cmap("tab20")
    for e in eps:
        c = cmap(e["index"] % 20)
        s, g = np.asarray(e["start_xy"]), np.asarray(e["goal_xy"])
        ax.annotate("", xy=g, xytext=s, zorder=2, arrowprops=dict(arrowstyle="->", lw=1.3, color=c, alpha=0.9))
        ax.scatter(*s, marker="^", s=70, color=c, edgecolors="black", linewidths=0.6, zorder=3)
        ax.scatter(*g, marker="*", s=150, color=c, edgecolors="black", linewidths=0.6, zorder=3)
    ax.scatter([], [], marker="^", s=60, color="#999999", edgecolors="black", label="Start")
    ax.scatter([], [], marker="*", s=120, color="#999999", edgecolors="black", label="Goal")
    ax.set_xlim(L[:, 0].min() - pad, L[:, 0].max() + pad); ax.set_ylim(L[:, 1].max() + pad, L[:, 1].min() - pad)
    ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, framealpha=0.95)

    os.makedirs(a.out, exist_ok=True)
    tag = os.path.basename(a.run_dir.rstrip("/"))
    out = os.path.join(a.out, f"episodes{a.n}_seed{a.episode_seed}_{tag}_s{a.seed}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    json.dump({"trace": tp, "town": a.town, "sampler": "eval_placement --episodes-from-trace (verbatim)",
               "episode_seed": a.episode_seed, "goal_band_m_straight_line": [a.goal_min, a.goal_max],
               "episodes": eps}, open(out.replace(".png", ".json"), "w"), indent=1)
    sl = [e["straight_line_m"] for e in eps]; rt = [e.get("route_m") for e in eps if e.get("route_m") is not None]
    print(f"[episodes] {len(eps)} pairs; straight-line {min(sl):.1f}-{max(sl):.1f} m (median {np.median(sl):.1f}); "
          f"route {('%.1f-%.1f m (median %.1f)' % (min(rt), max(rt), np.median(rt))) if rt else 'n/a'} over {len(rt)} routable")
    for e in eps:
        print(f"  {e['index']:>2}: start {e['start_trace_index']:>4} ({e['start_xy'][0]:6.1f},{e['start_xy'][1]:6.1f}) -> "
              f"goal {e['goal_trace_index']:>4} ({e['goal_xy'][0]:6.1f},{e['goal_xy'][1]:6.1f})  "
              f"line {e['straight_line_m']:5.1f} m  route {e.get('route_m') if e.get('route_m') is None else round(e['route_m'], 1)}")
    print(f"[Render] {out} (+ .json)")


if __name__ == "__main__":
    main()
