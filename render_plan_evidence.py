"""Individual, low-text plots of a plan-only run (localize_and_plan.py) for an evidence pack.

Offline, no agent. Writes:
  maps/<map>.png                  map nodes + edges drawn along the driven waypoints
  examples/<map>_<set><index>.png one example episode: planned path, true and image-localised start/goal
  summary_paths.png               image-localised paths found per map (multi-hop vs 0-hop), per results file
No titles; short legends; numbers go in the README.
"""
import argparse, json, os, pickle, sys
from collections import deque
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W); sys.path.insert(0, os.path.join(W, "src"))

EXAMPLES = [("short", 2), ("short", 10), ("long", 0), ("long", 1)]


# Empty interiors of Town02's blocks (world metres): legend anchors that normally cover nothing but tarmac-free space.
TOWN02_BLOCK_CENTRES = [(95.0, 148.0), (92.0, 272.0), (22.0, 248.0), (90.0, 214.0), (160.0, 214.0)]
LEGEND_CORNERS = ["upper left", "upper right", "lower left", "lower right"]


def place_legend(fig, ax, heavy_xy, light_xy=(), **kw):
    """Put the legend where it covers the least: path/markers/nodes (heavy) weigh 10, roads/edges (light) 1.

    Candidates are the empty block interiors of Town02 (anchored in data coordinates) and the four axes corners
    (user 2026-09-14: "place legend in a good place" -- not over the route, not below the plot).
    """
    heavy = np.asarray(heavy_xy, float).reshape(-1, 2); light = np.asarray(light_xy, float).reshape(-1, 2)
    fig.canvas.draw()
    hd = ax.transData.transform(heavy) if len(heavy) else np.zeros((0, 2))
    ld = ax.transData.transform(light) if len(light) else np.zeros((0, 2))
    cands = [dict(loc="center", bbox_to_anchor=c, bbox_transform=ax.transData) for c in TOWN02_BLOCK_CENTRES]
    cands += [dict(loc=c) for c in LEGEND_CORNERS]
    best = None
    for c in cands:
        leg = ax.legend(**c, **kw); fig.canvas.draw()
        bb = leg.get_window_extent(); ab = ax.get_window_extent()
        inside = lambda d: int(((d[:, 0] >= bb.x0) & (d[:, 0] <= bb.x1) & (d[:, 1] >= bb.y0) & (d[:, 1] <= bb.y1)).sum())
        outside_axes = bb.x0 < ab.x0 - 1 or bb.x1 > ab.x1 + 1 or bb.y0 < ab.y0 - 1 or bb.y1 > ab.y1 + 1
        score = 10 * inside(hd) + inside(ld) + (10 ** 6 if outside_axes else 0)
        leg.remove()
        if best is None or score < best[0]:
            best = (score, c)
    return ax.legend(**best[1], **kw)


def bfs_path(n, edges, s, g, undirected):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        if undirected:
            adj[v].append(u)
    prev, q = {s: None}, deque([s])
    while q:
        u = q.popleft()
        if u == g:
            break
        for v in adj[u]:
            if v not in prev:
                prev[v] = u; q.append(v)
    if g not in prev:
        return None
    path = [g]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return path[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/localize_plan/sequence_undirected/localize_plan_results.json")
    ap.add_argument("--compare", nargs="*", default=["results/localize_plan/localize_plan_results_7maps_backup.json",
                                                     "results/localize_plan/sequence/localize_plan_results.json"],
                    help="other results files for the summary plot (labelled by their map_set/undirected)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from render_placement_maps import lane_polylines
    from localize_and_plan import load_map, TRACE

    P = np.asarray(pickle.load(open(TRACE, "rb"))["positions"], np.float64)[:, :2]
    R = json.load(open(a.results)); meta = R["meta"]
    undirected = bool(meta.get("undirected", False))
    V = pickle.load(open(meta["views"], "rb"))
    views = {(v["set"], v["index"], v["role"]): v for v in V["views"]}
    lines = lane_polylines("Town02"); L = np.concatenate([np.asarray(l) for l in lines if len(l)]); pad = 15
    for sub in ("maps", "examples"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)

    def frame(ax):
        ax.add_collection(LineCollection(lines, colors="#d6d6d6", linewidths=2.0, zorder=0))
        ax.set_xlim(L[:, 0].min() - pad, L[:, 0].max() + pad); ax.set_ylim(L[:, 1].max() + pad, L[:, 1].min() - pad)
        ax.set_aspect("equal"); ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

    recs, all_paths = {}, []
    for r in R["results"]:
        recs[(r["map"], r["set"], r["index"])] = r
    for mname, mpath in meta["maps"].items():
        m = json.load(open(mpath))
        _tidx, xy, E = load_map(mpath, P)                       # node positions + edges over node positions
        n = len(xy)
        wp = {}
        for (u, v), e in zip(E, m["edges"]):
            if isinstance(e, dict) and e.get("waypoints"):
                # node -> driven waypoints -> node: sequence edges on merged maps (uniform, junction) store only the
                # samples from leaving u to reaching v, so add the node positions to get the full node-to-node leg
                wp[(u, v)] = np.vstack([xy[u], np.asarray([[w["x_m"], w["y_m"]] for w in e["waypoints"]]), xy[v]])
            else:                                               # uniform trace-edge graphs: nodes ~gate m apart
                wp[(u, v)] = np.asarray([xy[u], xy[v]])
        polys = list(wp.values())
        size = 36 if n <= 160 else 10

        fig, ax = plt.subplots(figsize=(7, 7)); frame(ax)
        ax.add_collection(LineCollection(polys, colors="#d62728", linewidths=1.3, alpha=0.75, zorder=2))
        ax.scatter(xy[:, 0], xy[:, 1], s=size, c="white", edgecolors="black", linewidths=0.7, zorder=3, label=f"Nodes ({n})")
        ax.plot([], [], color="#d62728", lw=1.8, label=f"Edges ({len(E)})")
        place_legend(fig, ax, xy, np.concatenate([L] + [p_ for p_ in polys]) if polys else L,
                     ncol=2, fontsize=9, framealpha=0.95)
        fig.savefig(os.path.join(a.out, "maps", f"{mname}.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

        eps = sorted({(s_, i_) for (mm, s_, i_) in recs if mm == mname}, key=lambda t: (t[0] != "short", t[1]))
        for set_name, idx in eps:
            r = recs[(mname, set_name, idx)]
            si, gi = r["start"]["image_node"], r["goal"]["image_node"]
            path = bfs_path(n, E, si, gi, undirected)
            rec = {"map": mname, "set": set_name, "index": idx, "goal_kind": r["goal_kind"], "route_m": r["route_m"],
                   "undirected": undirected}
            for kind in ("image", "critic", "gt"):
                p = bfs_path(n, E, r["start"][f"{kind}_node"], r["goal"][f"{kind}_node"], undirected)
                if p is None:
                    rec[kind] = {"start_node": r["start"][f"{kind}_node"], "goal_node": r["goal"][f"{kind}_node"], "found": False}
                    continue
                pts, against, dl = [xy[p[0]].tolist()], 0, 0.0
                for u, v in zip(p[:-1], p[1:]):
                    fwd = (u, v) in wp
                    seg_ = wp[(u, v)] if fwd else wp[(v, u)][::-1]
                    against += int(not fwd)
                    dl += float(np.sum(np.linalg.norm(np.diff(seg_, axis=0), axis=1)))
                    pts.extend(seg_[1:].tolist())
                rec[kind] = {"start_node": p[0], "goal_node": p[-1], "found": True, "hops": len(p) - 1, "node_path": p,
                             "node_xy": xy[p].tolist(), "edges_against_driven_direction": against,
                             "driven_length_m": round(dl, 2), "waypoints_xy": pts}
            all_paths.append(rec)
            fig, ax = plt.subplots(figsize=(7, 7)); frame(ax)
            ax.add_collection(LineCollection(polys, colors="#bdbdbd", linewidths=0.8, alpha=0.8, zorder=1))
            ax.scatter(xy[:, 0], xy[:, 1], s=size * 0.5, c="#8c8c8c", zorder=2)
            if path is not None and len(path) > 1:
                seglines = []
                for u, v in zip(path[:-1], path[1:]):
                    seglines.append(wp[(u, v)] if (u, v) in wp else wp[(v, u)][::-1])
                ax.add_collection(LineCollection(seglines, colors="#2ca02c", linewidths=2.6, zorder=3))
                ax.plot([], [], color="#2ca02c", lw=2.6, label=f"Path ({len(path) - 1} hops)")
            elif path is not None:
                ax.plot([], [], " ", label="Path (0 hops)")
            else:
                ax.plot([], [], " ", label="No path")
            vs, vg = views[(set_name, idx, "start")], views[(set_name, idx, "goal")]
            ax.scatter(*vs["gt_xy"], marker="^", s=130, c="white", edgecolors="black", lw=1.2, zorder=4, label="Start")
            ax.scatter(*vg["gt_xy"], marker="*", s=230, c="white", edgecolors="black", lw=1.2, zorder=4, label="Goal")
            ax.scatter(*xy[si], marker="^", s=100, c="#d62728", edgecolors="black", zorder=5, label="Localised start")
            ax.scatter(*xy[gi], marker="*", s=180, c="#d62728", edgecolors="black", zorder=5, label="Localised goal")
            heavy = [np.asarray(vs["gt_xy"])[None], np.asarray(vg["gt_xy"])[None], xy[[si]], xy[[gi]], xy]
            if path is not None and len(path) > 1:
                heavy.extend(seglines)
            place_legend(fig, ax, np.concatenate(heavy), np.concatenate([L] + list(polys)),
                         ncol=2, fontsize=8.5, framealpha=0.9)
            sub = "examples" if (set_name, idx) in EXAMPLES else "paths"
            os.makedirs(os.path.join(a.out, sub, mname), exist_ok=True)
            fig.savefig(os.path.join(a.out, sub, mname, f"{set_name}{idx:02d}.png"), dpi=120 if sub == "paths" else 150,
                        bbox_inches="tight")
            plt.close(fig)
    json.dump({"source_results": a.results, "undirected": undirected, "maps": meta["maps"],
               "note": "paths recomputed by BFS (hop count) over the map edges from the localised start/goal nodes of "
                       "the results file; image = result, critic = uses true pose, gt = GT-nearest nodes; "
                       "waypoints follow each edge's driven polyline (reversed when used against the driven direction)",
               "paths": all_paths}, open(os.path.join(a.out, "planned_paths.json"), "w"))

    # summary: multi-hop and 0-hop paths per map, one bar group per results file
    files = list(a.compare) + [a.results]
    runs = []
    for f in files:
        d = json.load(open(f)); mt = d["meta"]
        label = ("Original edges" if mt.get("map_set", "original") == "original" else "Sequence edges") + \
                (", two-way" if mt.get("undirected") else ", one-way")
        mh, zh = {}, {}
        for r in d["results"]:
            h = r["image_hops"]
            mh[r["map"]] = mh.get(r["map"], 0) + int(h is not None and h > 0)
            zh[r["map"]] = zh.get(r["map"], 0) + int(h == 0)
        runs.append((label, mh, zh))
    names = list(meta["maps"])
    x = np.arange(len(names)); w = 0.8 / len(runs)
    colors = ["#9e9e9e", "#1f77b4", "#2ca02c", "#ff7f0e"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for k, (label, mh, zh) in enumerate(runs):
        xs = x - 0.4 + w * (k + 0.5)
        ax.bar(xs, [mh.get(nm, 0) for nm in names], w, color=colors[k % 4], label=label)
        ax.bar(xs, [zh.get(nm, 0) for nm in names], w, bottom=[mh.get(nm, 0) for nm in names], color=colors[k % 4],
               alpha=0.35, hatch="//", edgecolor="white")
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#bdbdbd", alpha=0.5, hatch="//", edgecolor="white")); labels.append("0-hop (same node)")
    ax.set_xticks(x); ax.set_xticklabels([nm.replace("_trace", "").replace("entropy_", "entropy\n") for nm in names], fontsize=8.5)
    ax.set_ylabel("Episodes with a path (of 40)"); ax.set_ylim(0, 42)
    ax.legend(handles, labels, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.95)
    fig.savefig(os.path.join(a.out, "summary_paths.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[evidence plots] {a.out}: {len(meta['maps'])} maps, {len(meta['maps']) * len(EXAMPLES)} examples, summary_paths.png")


if __name__ == "__main__":
    main()
