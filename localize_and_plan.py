"""Localise start and goal from live latents on each map, then plan the shortest path.

Stage B of the plan-only pipeline (offline; GPU for the critic, no CARLA).
User decisions 2026-09-13: localisation RESULT = IMAGE ONLY; every edge weight = 1
(hop count); report "no path"; summary table + one example figure per map.

For every map and every episode (short + long):
  1. LOCALISE start and goal separately:
       image (RESULT): nearest stored trace latent (cosine) among the map's nodes
       critic (reported, labelled): shortlist the top-k nearest latents, pick the node the
                      goal critic predicts is FEWEST STEPS away from the live observation.
                      USES TRUE POSE: the v6n goal block is built from true hero_pos/yaw and
                      the node position, so this is NOT image-based localisation.
     Both scored against the ground-truth pose (distance to the chosen node, and whether
     the chosen node is the GT-nearest node).
  2. PLAN: BFS/Dijkstra over the map's directed edges with weight 1 (hops) from the
     image-localised start node to the image-localised goal node. Also from the critic
     picks, and from the GT-nearest nodes as a reference.
"""
import argparse, csv, json, os, pickle, sys
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))

TRACE = "results/placement_v6n_town02_budget10k_laneoff_speed10/trace_s7.pkl"
MAPS = {
    "junction": "results/maps/town02_junctions_uniform6_trace_speed10.json",
    "uniform6_trace": "results/figures/uniform_frame_trace_edges/uniform6_placement_v6n_town02_budget10k_laneoff_speed10_s7.json",
    "uniform2_trace": "results/figures/uniform_frame_trace_edges/uniform2_placement_v6n_town02_budget10k_laneoff_speed10_s7.json",
    "entropy_smooth9_high": "results/maps/town02_speed10_s7_smooth9_high_driving_map.json",
    "entropy_smooth9_low": "results/maps/town02_speed10_s7_smooth9_low_driving_map.json",
    "entropy_nosmooth_high": "results/maps/town02_speed10_s7_nosmooth_high_driving_map.json",
    "entropy_nosmooth_low": "results/maps/town02_speed10_s7_nosmooth_low_driving_map.json",
    # user 2026-09-13: nosmooth-high thinned 4 m after the MAD filter (thin_entropy_map.py); no original-set edges
    "entropy_nosmooth_high_thin4": "results/maps/town02_speed10_s7_nosmooth_high_thin4_driving_map.json",
    # user 2026-09-13: uniform[6 m] nodes between the thinned high-entropy nodes (hybrid_uniform_entropy_map.py)
    "hybrid_uniform6_entropy_thin4": "results/maps/town02_speed10_s7_hybrid_uniform6_entropy_thin4_driving_map.json",
}
# user 2026-09-13: same nodes, edges = node changes along the trace ACROSS episodes (sequence_link_map.py)
# user 2026-09-14: SGM ablation maps (build_sgm_map.py), one per (gate, edge cutoff, consistency)
import glob as _glob
for _p in sorted(_glob.glob("results/maps/town02_speed10_s7_sgm_*_driving_map.json")):
    MAPS[os.path.basename(_p)[len("town02_speed10_s7_"):-len("_driving_map.json")]] = _p
MAPS_SEQUENCE = {k: f"results/maps/sequence/{k}_sequence.json" for k in MAPS}
EXAMPLES = [("short", 2), ("short", 10), ("long", 0), ("long", 1)]   # detour, direct, long on-graph, long off-graph


def load_map(path, P):
    """-> node trace indices, node xy (N,2), directed edges [(u, v)] over node ids."""
    d = json.load(open(path))
    if "node_trace_index" in d and "edges" in d and d["edges"] and isinstance(d["edges"][0], list):
        tidx = np.asarray(d["node_trace_index"], int)
        local = {int(t): n for n, t in enumerate(tidx)}
        return tidx, P[tidx], [(local[int(a)], local[int(b)]) for a, b, _s in d["edges"]]
    nodes = d["nodes"]
    pos = {int(n["id"]): k for k, n in enumerate(nodes)}
    tidx = np.asarray([int(n["trace_index"]) for n in nodes], int)
    xy = np.asarray([[n["x_m"], n["y_m"]] for n in nodes], np.float64)
    return tidx, xy, [(pos[int(e["from"])], pos[int(e["to"])]) for e in d["edges"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="results/episodes/town02_speed10_s7_episode_views_seed0.pkl")
    ap.add_argument("--ckpt", default="checkpoints/carla_gated_her_v6n/checkpoint_250000.pkl")
    ap.add_argument("--run-env", default="results/run_env/carla_gated_her_v6n.env")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--step-cost", type=float, default=1.0)
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--out", default="results/localize_plan")
    ap.add_argument("--map-set", choices=("original", "sequence"), default="original")
    ap.add_argument("--maps", nargs="*", default=None, help="run only these maps (default: all in the set)")
    ap.add_argument("--undirected", action="store_true",
                    help="plan over edges in both directions (user 2026-09-13: make edges undirected)")
    a = ap.parse_args()
    maps = MAPS if a.map_set == "original" else MAPS_SEQUENCE
    if a.maps:
        maps = {k: v for k, v in maps.items() if k in a.maps}

    for line in open(a.run_env):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            kk, vv = line.split("=", 1); os.environ[kk] = vv
    os.environ["CARLA_START_SERVER"] = "0"

    from gymnasium import spaces
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import train_carla_tqc as T
    from vae_nav.policy_io import load_agent_state
    from vae_nav.value_distance import edge_cost_steps, goal_relabelled_obs
    from vae_nav.vae_goal_wrapper import VAEGoalWrapper
    from render_placement_maps import lane_polylines

    tr = pickle.load(open(TRACE, "rb"))
    P = np.asarray(tr["positions"], np.float64)[:, :2]; seg = np.asarray(tr["segments"])
    Z = np.stack([np.asarray(o["latent"], np.float64) for o in tr["observations"]])
    Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    V = pickle.load(open(a.views, "rb"))
    views = {(v["set"], v["index"], v["role"]): v for v in V["views"]}
    bundle = pickle.load(open(V["meta"]["episodes"], "rb"))

    o0 = tr["observations"][0]
    obs_space = spaces.Dict({k: spaces.Box(-np.inf, np.inf, np.asarray(val).shape, np.float32) for k, val in o0.items()})
    agent = T.build_agent(obs_space, spaces.Box(-1.0, 1.0, (2,), np.float32), 0)
    st = pickle.load(open(a.ckpt, "rb")); load_agent_state(agent, st.get("agent", st))
    goal_block = VAEGoalWrapper.goal_block.__get__(object.__new__(VAEGoalWrapper))

    def localise(view, tidx, nxy):
        z = np.asarray(view["obs"]["latent"], np.float64); z = z / max(np.linalg.norm(z), 1e-12)
        cos = 1.0 - Zn[tidx] @ z
        order = np.argsort(cos)
        top = order[:min(a.k, len(order))]
        batch = [goal_relabelled_obs(view["obs"], view["fields"], nxy[n], goal_block) for n in top]
        stacked = {kk: np.stack([b[kk] for b in batch]) for kk in batch[0]}
        steps, _ = edge_cost_steps(agent, stacked, step_cost=a.step_cost)
        steps = np.asarray(steps, np.float64)
        crit = int(top[int(np.argmin(steps))])
        gt = np.asarray(view["gt_xy"]); gtn = int(np.argmin(np.linalg.norm(nxy - gt, axis=1)))
        img = int(order[0])
        return {"image_node": img, "image_err_m": float(np.linalg.norm(nxy[img] - gt)), "image_cos": float(cos[img]),
                "image_is_gt_node": bool(img == gtn),
                "critic_node": crit, "critic_err_m": float(np.linalg.norm(nxy[crit] - gt)),
                "critic_steps": float(steps.min()), "critic_is_gt_node": bool(crit == gtn),
                "shortlist": [int(n) for n in top],
                "gt_node": gtn, "gt_node_err_m": float(np.linalg.norm(nxy[gtn] - gt))}

    def hops(D, s, g):
        h = D[s][g]
        return int(h) if np.isfinite(h) else None

    os.makedirs(a.out, exist_ok=True)
    lines = lane_polylines(a.town)
    L = np.concatenate([np.asarray(l) for l in lines if len(l)]); pad = 20
    rows, summary = [], {}
    for mname, mpath in maps.items():
        tidx, nxy, edges = load_map(mpath, P)
        n = len(tidx)
        G = (csr_matrix((np.ones(len(edges)), ([u for u, _ in edges], [v for _, v in edges])), shape=(n, n))
             if edges else csr_matrix((n, n)))
        D, pred = dijkstra(G, directed=not a.undirected, unweighted=True, return_predecessors=True)
        mres = []
        for set_name in ("short", "long"):
            for e in bundle[set_name]["episodes"]:
                s = localise(views[(set_name, e["index"], "start")], tidx, nxy)
                g = localise(views[(set_name, e["index"], "goal")], tidx, nxy)
                r = {"map": mname, "set": set_name, "index": int(e["index"]),
                     "goal_kind": e.get("goal_kind", "on_graph"), "route_m": e.get("route_m"),
                     "start": s, "goal": g,
                     "image_hops": hops(D, s["image_node"], g["image_node"]),
                     "critic_hops": hops(D, s["critic_node"], g["critic_node"]),
                     "gt_hops": hops(D, s["gt_node"], g["gt_node"])}
                mres.append(r); rows.append(r)
        both = [(x, w) for x in mres for w in ("start", "goal")]
        f = lambda key: [x[w][key] for x, w in both]
        found = lambda key: [x[key] for x in mres if x[key] is not None]
        summary[mname] = {
            "nodes": n, "edges": len(edges), "episodes": len(mres),
            "image_err_median_m": float(np.median(f("image_err_m"))), "image_picks_gt_node": float(np.mean(f("image_is_gt_node"))),
            "critic_err_median_m": float(np.median(f("critic_err_m"))), "critic_picks_gt_node": float(np.mean(f("critic_is_gt_node"))),
            "gt_node_err_median_m": float(np.median(f("gt_node_err_m"))),
            "image_path_found": len(found("image_hops")), "critic_path_found": len(found("critic_hops")),
            "gt_path_found": len(found("gt_hops")),
            # a 0-hop "path" means start and goal localised to the SAME node; report multi-hop separately
            "image_path_multihop": sum(h > 0 for h in found("image_hops")),
            "critic_path_multihop": sum(h > 0 for h in found("critic_hops")),
            "gt_path_multihop": sum(h > 0 for h in found("gt_hops")),
            "image_hops_median": float(np.median(found("image_hops"))) if found("image_hops") else None,
            "gt_hops_median": float(np.median(found("gt_hops"))) if found("gt_hops") else None}
        s_ = summary[mname]
        print(f"[{mname:>22}] {n:>4} nodes {len(edges):>4} edges | IMAGE err median {s_['image_err_median_m']:6.1f} m "
              f"(GT node {s_['image_picks_gt_node']:.2f}) path {s_['image_path_found']}/{s_['episodes']} "
              f"(multi-hop {s_['image_path_multihop']}) | "
              f"critic(true pose) err {s_['critic_err_median_m']:6.1f} m (GT node {s_['critic_picks_gt_node']:.2f}) "
              f"path {s_['critic_path_found']}/{s_['episodes']} (multi-hop {s_['critic_path_multihop']}) | "
              f"GT-node path {s_['gt_path_found']}/{s_['episodes']} (multi-hop {s_['gt_path_multihop']})", flush=True)

        # one figure per map: the example episodes, image localisation + planned path
        fig, axes = plt.subplots(2, 2, figsize=(13, 13))
        ex = {(x["set"], x["index"]): x for x in mres}
        for ax, key in zip(axes.ravel(), EXAMPLES):
            x = ex[key]
            ax.add_collection(LineCollection(lines, colors="#d0d0d0", linewidths=1.8, zorder=0))
            ax.add_collection(LineCollection([P[seg == k] for k in np.unique(seg)], colors="#1f77b4",
                                             linewidths=0.6, alpha=0.3, zorder=1))
            if edges:
                ax.add_collection(LineCollection([[nxy[u], nxy[v]] for u, v in edges], colors="#9e9e9e",
                                                 linewidths=0.6, alpha=0.6, zorder=1))
            ax.scatter(nxy[:, 0], nxy[:, 1], s=8, c="#7f7f7f", zorder=2, label=f"Map nodes ({n})")
            si, gi = x["start"]["image_node"], x["goal"]["image_node"]
            if x["image_hops"] is not None:
                path, cur = [gi], gi
                while cur != si and pred[si][cur] >= 0:
                    cur = int(pred[si][cur]); path.append(cur)
                path = path[::-1]
                ax.plot(nxy[path, 0], nxy[path, 1], "-", color="#2ca02c", lw=2.4, zorder=3,
                        label=f"Planned path ({x['image_hops']} hops)")
            vs, vg = views[(key[0], key[1], "start")], views[(key[0], key[1], "goal")]
            ax.scatter(*vs["gt_xy"], marker="^", s=120, c="white", edgecolors="black", lw=1.2, zorder=4, label="True start")
            ax.scatter(*vg["gt_xy"], marker="*", s=220, c="white", edgecolors="black", lw=1.2, zorder=4, label="True goal")
            ax.scatter(*nxy[si], marker="^", s=90, c="#d62728", edgecolors="black", zorder=5,
                       label=f"Image-localised start ({x['start']['image_err_m']:.0f} m off)")
            ax.scatter(*nxy[gi], marker="*", s=170, c="#d62728", edgecolors="black", zorder=5,
                       label=f"Image-localised goal ({x['goal']['image_err_m']:.0f} m off)")
            ax.set_xlim(L[:, 0].min() - pad, L[:, 0].max() + pad); ax.set_ylim(L[:, 1].max() + pad, L[:, 1].min() - pad)
            ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
            txt = "no path" if x["image_hops"] is None else f"{x['image_hops']} hops"
            ax.text(0.01, 0.99, f"{key[0]} #{key[1]} ({x['goal_kind']}), route {x['route_m']:.0f} m: {txt}",
                    transform=ax.transAxes, va="top", fontsize=9, bbox=dict(fc="white", alpha=0.8, lw=0))
            ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
        fig.savefig(os.path.join(a.out, f"examples_{mname}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)

    json.dump({"meta": {"views": a.views, "k": a.k, "map_set": a.map_set, "undirected": a.undirected, "maps": maps, "trace": TRACE, "edge_weight": 1,
                        "result": "image-only localisation (nearest cosine latent)",
                        "note": "critic columns use the pose-based goal block (true pose): NOT image-based"},
               "summary": summary, "results": rows}, open(os.path.join(a.out, "localize_plan_results.json"), "w"), indent=1)
    with open(os.path.join(a.out, "localize_plan_summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["nodes", "edges", "episodes", "image_err_median_m", "image_picks_gt_node", "image_path_found",
                "image_path_multihop", "image_hops_median", "critic_err_median_m", "critic_picks_gt_node",
                "critic_path_found", "critic_path_multihop", "gt_node_err_median_m", "gt_path_found",
                "gt_path_multihop", "gt_hops_median"]
        w.writerow(["map"] + cols)
        for m, s_ in summary.items():
            w.writerow([m] + [s_[c] for c in cols])
    print(f"[done] {a.out}/localize_plan_results.json, localize_plan_summary.csv, examples_<map>.png")


if __name__ == "__main__":
    main()
