"""Draw planned node paths for chosen episodes, one map per column (user 2026-09-16: "show some plans").

For each episode: the town's lanes in grey, the map's nodes, the BFS plan as a polyline through the planned nodes
(plan_nodes in the drive JSONL), and the true start and goal. The title carries plan length, hop count, the true road
route and whether the episode succeeded, so a long plan is visible next to a short one for the same trajectory.
"""
import argparse, json, os
import numpy as np


def load_map(seq_dir, name):
    d = json.load(open(f"{seq_dir}/{name}_sequence.json"))
    return np.asarray([[n["x_m"], n["y_m"]] for n in d["nodes"]], float)


def load_drive(path, map_name):
    return {(r["set"], r["index"]): r
            for l in open(path) if l.strip() for r in [json.loads(l)] if r["map"] == map_name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", default="Town02")
    ap.add_argument("--seq-dir", default="results/maps/sequence")
    ap.add_argument("--maps", nargs="+", required=True, help="label=map_name=jsonl triples, '=' separated")
    ap.add_argument("--episodes", nargs="+", required=True, help="set:index, e.g. long:3")
    ap.add_argument("--episodes-pkl", default="results/episodes/town02_speed10_s7_episodes_short20_long20_seed0.pkl",
                    help="true start/goal come from the episode bundle; the drive JSONL stores only node ids")
    ap.add_argument("--out", default="results/figures/plans.png")
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from render_placement_maps import lane_polylines
    from matplotlib.collections import LineCollection

    import pickle
    _b = pickle.load(open(a.episodes_pkl, "rb"))
    EP = {(s_, int(e["index"])): e for s_ in ("short", "long") for e in _b[s_]["episodes"]}
    specs = [m.split("=") for m in a.maps]
    eps = [(e.split(":")[0], int(e.split(":")[1])) for e in a.episodes]
    lanes = lane_polylines(a.town)
    L = np.concatenate([np.asarray(l) for l in lanes if len(l)]); pad = 15

    fig, axes = plt.subplots(len(eps), len(specs), figsize=(5.2 * len(specs), 5.0 * len(eps)), squeeze=False)
    for r, key in enumerate(eps):
        for c, (label, map_name, jsonl) in enumerate(specs):
            ax = axes[r][c]
            xy = load_map(a.seq_dir, map_name)
            rec = load_drive(jsonl, map_name).get(key)
            ax.add_collection(LineCollection(lanes, colors="#dcdcdc", linewidths=1.6, zorder=0))
            ax.scatter(xy[:, 0], xy[:, 1], s=5, c="#b0b0b0", zorder=1, label=f"nodes ({len(xy)})")
            if rec and rec.get("plan_nodes"):
                p = np.asarray([xy[i] for i in rec["plan_nodes"]], float)
                ax.plot(p[:, 0], p[:, 1], "-", color="#d62728", lw=1.8, zorder=3, alpha=0.9)
                ax.scatter(p[:, 0], p[:, 1], s=14, c="#d62728", zorder=4)
            if rec:
                _e = EP[key]
                ax.scatter(*_e["start_xy"], s=90, marker="*", c="#1f77b4", zorder=5, label="start")
                ax.scatter(*_e["goal_xy"], s=90, marker="X", c="#2ca02c", zorder=5, label="goal")
                route = rec.get("route_m")
                ax.set_title(f"{label} — {key[0]}#{key[1]}: plan {rec['plan_m']:.0f} m, {rec['plan_hops']} hops"
                             + (f", route {route:.0f} m" if route else "")
                             + f"\n{'SUCCESS' if rec['success'] else rec['end'].upper()}", fontsize=9)
            ax.set_xlim(L[:, 0].min() - pad, L[:, 0].max() + pad)
            ax.set_ylim(L[:, 1].max() + pad, L[:, 1].min() - pad)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=170, bbox_inches="tight"); print("[wrote]", a.out)


if __name__ == "__main__":
    main()
