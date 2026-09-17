"""Top-down figures of a placement run: exploration trace, SGM graph, and the
evaluation episodes coloured by outcome, on the town's lane network.

Offline and CPU-only. The placement JSONs do not store node positions, edges
or driven paths, so the SGM graph is REBUILT from the cached critic matrix
with the calibration the run used (sgm_faithful, same arguments as
eval_placement.py), and checked against the row's logged node and edge
counts -- a mismatch aborts rather than drawing a different graph. Episodes
are regenerated from the same seeded sampler (eval_placement.py
--episodes-from-trace) and coloured by the row's per-episode success, which
aligns only because SGM routed all 60. DRIVEN PATHS ARE NOT RECOVERABLE: the
harness never saved them, so episodes are drawn as start -> goal chords.

Also writes a compact JSON per trace (trace xy, nodes, edges, episodes) so the
figure can be redrawn in another style without this repo.
"""
import argparse, glob, json, os, pickle, sys
import numpy as np

sys.path.insert(0, "src")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import carla
from vae_nav.sgm_frame import sgm_faithful

XODR = "/home/robotlab/Apps/CARLA_0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive/{}.xodr"
EDGE_SAMPLE = 3000   # learned edges drawn per figure; the legend gives the true total


def lane_polylines(town):
    m = carla.Map(town, open(XODR.format(town)).read())
    lines = []
    for entry, _exit in m.get_topology():
        chain = [entry] + entry.next_until_lane_end(1.0)
        lines.append([[w.transform.location.x, w.transform.location.y] for w in chain])
    return lines


def trace_episodes(P, n, seed=0, lo=20.0, hi=50.0):
    """Verbatim copy of eval_placement.py's episodes-from-trace sampler."""
    rng = np.random.default_rng(int(seed))
    eps, tries = [], 0
    while len(eps) < n and tries < 200 * n:
        tries += 1
        i = int(rng.integers(0, len(P)))
        d = np.linalg.norm(P - P[i], axis=1)
        ok = np.flatnonzero((d >= lo) & (d <= hi))
        if len(ok) == 0:
            continue
        j = int(rng.choice(ok))
        rng.integers(0, 2 ** 31 - 1)  # the per-episode env seed draw, kept for stream alignment
        eps.append((i, j))
    return eps


def draw_roads(ax, lines):
    ax.add_collection(LineCollection(lines, colors="#c8c8c8", linewidths=2.2, zorder=0))


def finish(ax, P, title):
    pad = 15
    ax.set_xlim(P[:, 0].min() - pad, P[:, 0].max() + pad)
    ax.set_ylim(P[:, 1].min() - pad, P[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # CARLA is left-handed: +y points down on a top-down view
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    # NO TITLES on figures (user 2026-09-13); `title` is kept so callers need not change


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arm", default="v6n")
    ap.add_argument("--town", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-graph", action="store_true", help="trace only (no calibration yet)")
    ap.add_argument("--full-map", action="store_true", help="axes span the whole town, not the trace")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lines = lane_polylines(a.town)
    calib = None if a.no_graph else json.load(open(os.path.join(a.run_dir, "calibration.json")))
    if calib is not None and not calib.get("chosen"):
        raise SystemExit("calibration has no chosen setting; use --no-graph")

    for tp in sorted(glob.glob(os.path.join(a.run_dir, "trace_s*.pkl")),
                     key=lambda p: int(p.split("_s")[-1].split(".")[0])):
        s = int(tp.split("_s")[-1].split(".")[0])
        tr = pickle.load(open(tp, "rb"))
        P = np.asarray(tr["positions"], np.float64)[:, :2]
        seg = np.asarray(tr["segments"])
        ck = tr["meta"]["checkpoint"]
        ckname = ck.split("/")[-2] + "@" + ck.split("_")[-1].split(".")[0]
        # one polyline per exploration episode, so episodes are not joined
        tlines = [P[seg == k] for k in np.unique(seg)]
        # HOW THE TRACE WAS COLLECTED, measured rather than trusted from a flag:
        # gap from each episode's first position to the previous episode's last.
        ks = np.unique(seg)
        gaps = np.array([np.linalg.norm(P[seg == b][0] - P[seg == a][-1])
                         for a, b in zip(ks[:-1], ks[1:])]) if len(ks) > 1 else np.zeros(1)
        kind = ("continued trace" if np.median(gaps) < 15.0 else "random-respawn trace")
        kind += f", median restart gap {np.median(gaps):.0f} m"
        export = {"town": a.town, "seed": s, "checkpoint": ck, "trace_xy": P.tolist(),
                  "trace_segments": seg.tolist()}

        fig, ax = plt.subplots(figsize=(9, 9))
        draw_roads(ax, lines)
        ax.add_collection(LineCollection(tlines, colors="#1f77b4", linewidths=0.9,
                                         alpha=0.8, zorder=2, label="Exploration trace"))
        if a.no_graph:
            finish(ax, P, f"{ckname}  {a.town}  trace s{s}  ({len(P)} candidates, "
                          f"{tr['episodes_done']} episodes) -- No graph yet\n{kind.capitalize()}")
        else:
            row = json.load(open(os.path.join(a.run_dir, f"{a.arm}__sgm__20-50m__s{s}.json")))
            dist = np.load(tp + ".dist.npy")
            c, e = float(calib["chosen"]["consistency"]), float(calib["chosen"]["edge_cutoff"])
            kept, edges = sgm_faithful(dist, P, consistency_cutoff=c, edge_cutoff=e,
                                       gate_m=float(row.get("sgm_gate_m", 2.0)))
            if len(kept) != row["n_nodes"] or len(edges) != row["n_edges"]:
                raise SystemExit(f"s{s}: rebuilt {len(kept)} nodes / {len(edges)} edges, "
                                 f"row logged {row['n_nodes']} / {row['n_edges']} -- not drawing a different graph")
            N = P[kept]
            ei = np.array([(u, v) for u, v, *_ in edges], dtype=int)
            # ~400k edges at any alpha paint a solid block that hides the roads,
            # so a seeded random SAMPLE is drawn and the legend says so.
            n_draw = min(EDGE_SAMPLE, len(ei))
            pick = np.random.default_rng(0).choice(len(ei), size=n_draw, replace=False)
            ax.add_collection(LineCollection(np.stack([N[ei[pick, 0]], N[ei[pick, 1]]], axis=1),
                                             colors="#7f7f7f", linewidths=0.4, alpha=0.25,
                                             zorder=1, rasterized=True))
            ax.scatter(N[:, 0], N[:, 1], s=6, c="#ff7f0e", zorder=3,
                       label=f"SGM[{c:g}/{e:g}] nodes ({len(kept)})")
            eps = trace_episodes(P, row["episodes"], seed=row.get("episode_seed", 0))
            ok = [bool(l[2]) for l in row["legs"]]
            assert row["n_routed"] == len(eps) == len(ok), "episodes do not align"
            for (i, j), o in zip(eps, ok):
                ax.annotate("", xy=P[j], xytext=P[i], zorder=4,
                            arrowprops=dict(arrowstyle="->", lw=1.1,
                                            color="#2ca02c" if o else "#d62728", alpha=0.85))
            ax.plot([], [], color="#2ca02c", label=f"Episode start -> goal, success ({sum(ok)})")
            ax.plot([], [], color="#d62728", label=f"Episode start -> goal, failure ({len(ok)-sum(ok)})")
            ax.plot([], [], color="#7f7f7f", label=f"Learned edges (random {n_draw} of {len(edges)})")
            finish(ax, P, f"{ckname}  {a.town}  trace s{s}  SGM[{c:g}/{e:g}]: "
                          f"SR {row['success']:.3f} over {row['n_routed']} routed episodes\n"
                          f"{kind.capitalize()}; episodes drawn start -> goal (driven paths not saved)")
            export.update({"sgm": {"consistency": c, "edge_cutoff": e, "node_trace_index": kept.tolist(),
                                   "node_xy": N.tolist(),
                                   "edges_local": [[int(u), int(v), float(w)] for u, v, w in edges]},
                           "episodes": [{"start_xy": P[i].tolist(), "goal_xy": P[j].tolist(),
                                         "start_trace_index": i, "goal_trace_index": j,
                                         "success": o} for (i, j), o in zip(eps, ok)],
                           "note": "driven paths were not saved by the harness; episodes are start/goal only"})
        if a.full_map:
            # THE WHOLE TOWN, not the trace's bounding box, so coverage is visible.
            L = np.concatenate([np.asarray(l) for l in lines if len(l)])
            pad = 20
            ax.set_xlim(L[:, 0].min() - pad, L[:, 0].max() + pad)
            ax.set_ylim(L[:, 1].max() + pad, L[:, 1].min() - pad)   # y inverted, as finish() does
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, framealpha=0.95)
        base = os.path.join(a.out, f"{a.town.lower()}_{a.arm}_s{s}")
        fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        json.dump(export, open(base + ".json", "w"))
        print(f"[Render] {base}.png (+ .json)", flush=True)

    json.dump({"town": a.town, "lane_polylines_xy": lines,
               "source": XODR.format(a.town) + " via carla.Map.get_topology, 1 m spacing"},
              open(os.path.join(a.out, f"{a.town.lower()}_lanes.json"), "w"))


if __name__ == "__main__":
    main()
