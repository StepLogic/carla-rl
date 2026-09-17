"""Long-horizon start/goal episodes from a trace, mixing ON-graph and OFF-graph goals.

Starts: random trace candidates (their stored pose and observation), heading = the
recorded yaw. Horizon is ROUTE distance along the town's directed lane graph (built
offline from the OpenDRIVE .xodr: lanes driven in their own direction, junction
connectors included, no U-turns); straight-line distance is ignored.
  on-graph goal : a trace candidate whose route distance from the start is in [lo, hi]
  off-graph goal: a driving-lane waypoint (not in a junction) at least --off-min-m from
                  EVERY trace candidate (beyond the 15 m localisation radius, so no graph
                  node built from this trace can attach it), route distance in [lo, hi]
Off-graph goals have no stored observation; a driving run needs a position-only goal.
"""
import argparse, json, os, pickle, sys
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))
XODR = "/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive/{}.xodr"


def lane_graph(town):
    import carla
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree
    m = carla.Map(town, open(XODR.format(town)).read())
    xy, yw, junc, rows, cols, first, last, ents, exs = [], [], [], [], [], [], [], [], []
    for en, ex in m.get_topology():
        ch = [en] + en.next_until_lane_end(1.0); b = len(xy)
        for w in ch:
            xy.append([w.transform.location.x, w.transform.location.y])
            yw.append(np.deg2rad(w.transform.rotation.yaw)); junc.append(bool(w.is_junction))
        rows += list(range(b, b + len(ch) - 1)); cols += list(range(b + 1, b + len(ch)))
        first.append(b); last.append(b + len(ch) - 1)
        ents.append([en.transform.location.x, en.transform.location.y])
        exs.append([ex.transform.location.x, ex.transform.location.y])
    xy, yw, junc = np.asarray(xy), np.asarray(yw), np.asarray(junc)
    et = cKDTree(np.asarray(ents))
    for s, e in enumerate(exs):
        for t in et.query_ball_point(e, 1.0):
            if t != s:
                rows.append(last[s]); cols.append(first[t])
    lens = [max(np.linalg.norm(xy[c] - xy[r]), 1e-3) for r, c in zip(rows, cols)]
    return xy, yw, junc, csr_matrix((lens, (rows, cols)), shape=(len(xy), len(xy))), cKDTree(xy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/placement_v6n_town02_budget10k_laneoff_speed10")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--episode-seed", type=int, default=0)
    ap.add_argument("--n-on", type=int, default=15)
    ap.add_argument("--n-off", type=int, default=5)
    ap.add_argument("--route-min", type=float, default=150.0)
    ap.add_argument("--route-max", type=float, default=400.0)
    ap.add_argument("--off-min-m", type=float, default=20.0)
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--out", default="results/figures/trace_episodes")
    a = ap.parse_args()

    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from render_placement_maps import lane_polylines

    tp = os.path.join(a.run_dir, f"trace_s{a.seed}.pkl")
    tr = pickle.load(open(tp, "rb"))
    P = np.asarray(tr["positions"], np.float64)[:, :2]
    yaw = np.asarray(tr["yaws"], np.float64); seg = np.asarray(tr["segments"])
    lxy, lyw, ljunc, G, ltree = lane_graph(a.town)
    ttree = cKDTree(P)
    off_ok = np.array([ttree.query(p)[0] >= a.off_min_m for p in lxy]) & ~ljunc
    print(f"[graph] {len(lxy)} lane nodes; {int(off_ok.sum())} off-graph candidates (>= {a.off_min_m:g} m from every trace point, not junction)")

    # goal lane node for each trace candidate (any lane node within 2 m, nearest)
    cand_goal_node = [ltree.query(p)[1] for p in P]

    rng = np.random.default_rng(int(a.episode_seed))
    eps, tries = [], 0
    want = ["on"] * a.n_on + ["off"] * a.n_off
    rng.shuffle(want)
    for kind in want:
        for _ in range(4000):
            tries += 1
            i = int(rng.integers(0, len(P)))
            near = [c for c in ltree.query_ball_point(P[i], 3.0)
                    if abs(np.angle(np.exp(1j * (lyw[c] - yaw[i])))) < np.deg2rad(60)]
            if not near:
                continue
            src = min(near, key=lambda c: np.linalg.norm(lxy[c] - P[i]))
            dd = dijkstra(G, directed=True, indices=src)
            if kind == "on":
                route = np.array([dd[g] for g in cand_goal_node])
                ok = np.flatnonzero((route >= a.route_min) & (route <= a.route_max))
                if len(ok) == 0:
                    continue
                j = int(rng.choice(ok))
                goal = {"goal_kind": "on_graph", "goal_trace_index": j, "goal_xy": P[j].tolist(),
                        "goal_yaw_rad": float(yaw[j]), "route_m": float(route[j])}
            else:
                ok = np.flatnonzero(off_ok & (dd >= a.route_min) & (dd <= a.route_max))
                if len(ok) == 0:
                    continue
                g = int(rng.choice(ok))
                goal = {"goal_kind": "off_graph", "goal_trace_index": None, "goal_xy": lxy[g].tolist(),
                        "goal_yaw_rad": float(lyw[g]), "route_m": float(dd[g]),
                        "nearest_trace_point_m": float(ttree.query(lxy[g])[0])}
            eps.append({"index": len(eps), "env_seed": int(rng.integers(0, 2 ** 31 - 1)),
                        "start_trace_index": i, "start_xy": P[i].tolist(), "start_yaw_rad": float(yaw[i]),
                        **goal, "straight_line_m": float(np.linalg.norm(np.asarray(goal["goal_xy"]) - P[i]))})
            break
    n_on = sum(e["goal_kind"] == "on_graph" for e in eps); n_off = len(eps) - n_on
    print(f"[episodes] {len(eps)} ({n_on} on-graph, {n_off} off-graph) after {tries} draws")
    for e in eps:
        extra = f"  nearest trace pt {e['nearest_trace_point_m']:.0f} m" if e["goal_kind"] == "off_graph" else ""
        print(f"  {e['index']:>2} {e['goal_kind']:>9}: start ({e['start_xy'][0]:6.1f},{e['start_xy'][1]:6.1f}) -> "
              f"goal ({e['goal_xy'][0]:6.1f},{e['goal_xy'][1]:6.1f})  route {e['route_m']:5.1f} m  line {e['straight_line_m']:5.1f} m{extra}")

    lines = lane_polylines(a.town)
    L = np.concatenate([np.asarray(l) for l in lines if len(l)]); pad = 20
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.add_collection(LineCollection(lines, colors="#d0d0d0", linewidths=2.0, zorder=0))
    ax.add_collection(LineCollection([P[seg == k] for k in np.unique(seg)], colors="#1f77b4", linewidths=0.8,
                                     alpha=0.35, zorder=1, label="Exploration trace (graph support)"))
    cmap = plt.get_cmap("tab20")
    for e in eps:
        c = cmap(e["index"] % 20)
        s, g = np.asarray(e["start_xy"]), np.asarray(e["goal_xy"])
        ax.annotate("", xy=g, xytext=s, zorder=2, arrowprops=dict(arrowstyle="->", lw=1.2, color=c, alpha=0.85,
                                                                  linestyle="-" if e["goal_kind"] == "on_graph" else "--"))
        ax.scatter(*s, marker="^", s=70, color=c, edgecolors="black", linewidths=0.6, zorder=3)
        if e["goal_kind"] == "on_graph":
            ax.scatter(*g, marker="*", s=150, color=c, edgecolors="black", linewidths=0.6, zorder=3)
        else:
            ax.scatter(*g, marker="X", s=110, color=c, edgecolors="black", linewidths=0.8, zorder=3)
    ax.scatter([], [], marker="^", s=60, color="#999999", edgecolors="black", label="Start (trace point)")
    ax.scatter([], [], marker="*", s=120, color="#999999", edgecolors="black", label=f"On-graph goal ({n_on})")
    ax.scatter([], [], marker="X", s=90, color="#999999", edgecolors="black", label=f"Off-graph goal ({n_off}, >= {a.off_min_m:g} m from trace)")
    ax.set_xlim(L[:, 0].min() - pad, L[:, 0].max() + pad); ax.set_ylim(L[:, 1].max() + pad, L[:, 1].min() - pad)
    ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    # no legend on this figure (user 2026-09-13); markers are described in the README

    os.makedirs(a.out, exist_ok=True)
    tag = os.path.basename(a.run_dir.rstrip("/"))
    out = os.path.join(a.out, f"longhorizon_on{a.n_on}_off{a.n_off}_route{a.route_min:g}-{a.route_max:g}_seed{a.episode_seed}_{tag}_s{a.seed}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    json.dump({"trace": tp, "town": a.town, "episode_seed": a.episode_seed,
               "horizon": {"kind": "route distance along the directed OpenDRIVE lane graph (no U-turns)",
                           "route_m": [a.route_min, a.route_max]},
               "off_graph_rule": f"driving-lane waypoint, not in a junction, >= {a.off_min_m:g} m from every trace candidate",
               "note": "off_graph goals have no trace observation (goal_trace_index null): a driving run needs a position-only goal",
               "episodes": eps}, open(out.replace(".png", ".json"), "w"), indent=1)
    print(f"[Render] {out} (+ .json)")


if __name__ == "__main__":
    main()
