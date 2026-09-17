"""Build a global node set over the map from per-episode entropy spikes.

    python map_nodes_entropy.py --in results/entropy_nogoal.json

Per-episode placement asks "given this trajectory, where do the nodes go". A
topological MAP is a different object: many episodes cross the same junction, so
their nodes must be MERGED, and what matters is whether the merged graph covers
the town's junctions -- not whether each episode covered its own.

THE MERGE IS THE WHOLE PROBLEM, and it is where a per-episode result can quietly
stop being true. Forty episodes each planting a node at the same junction is one
node, not forty; forty episodes each planting a node at a different point along
the same straight is forty. So a criterion that fires reliably at a repeatable
PLACE compresses under merging, while one that fires at arbitrary points does
not. That asymmetry is invisible per-episode and is exactly what a map exposes.

BUDGET MATCHING AFTER MERGING. The uniform arm cannot simply be given the same
per-episode budget: it merges differently. Instead its arc-length spacing is
BISECTED until its merged node count matches the entropy arm's, so both graphs
are compared at the same size. Without that the comparison silently becomes
"more nodes wins", which is the failure mode the whole node-count sweep exists
to avoid.

GROUND TRUTH is the same in_junction flag used throughout, pooled: each
contiguous in-junction run contributes its centroid, and those centroids are
merged at the same radius so one physical junction crossed by twelve episodes
counts once.
"""
import argparse
import json

import numpy as np

import node_placement_entropy as NP

PEAKS = "#0072B2"
UNIFORM = "#E69F00"
JUNC = "#009E73"
INK, INK2, MUTED, GRID = "#000000", "#3d3d3d", "#8a8a8a", "#dcdcdc"


def greedy_merge(pts, radius, weights=None):
    """Merge points within `radius`, keeping the highest-weighted representative."""
    if len(pts) == 0:
        return np.empty((0, 2))
    order = np.arange(len(pts)) if weights is None else np.argsort(-np.asarray(weights))
    keep = []
    for i in order:
        p = pts[i]
        if all(np.hypot(*(p - pts[j])) >= radius for j in keep):
            keep.append(i)
    return pts[keep]


def episode_arrays(e):
    xy = np.array([[s["x"], s["y"]] for s in e["steps"]], float)
    h = np.array([s["entropy"] for s in e["steps"]], float)
    inj = np.array([s["in_junction"] for s in e["steps"]], bool)
    return xy, h, inj, NP.arclength(xy)


def collect(episodes, detector, z, dens, min_sep, warmup=25):
    """Per-episode node positions, pooled but NOT yet merged.

    WARMUP EXCLUSION IS LOAD-BEARING, NOT TIDYING. Every episode opens with a
    large reset transient -- entropy ~1.8 nats decaying to ~0.8 within 20 steps.
    MEASURED: with warmup=0, 44 of 112 raw nodes (39.3%) landed in the first 20
    steps, i.e. at SPAWN POINTS rather than at anything the policy noticed. That
    is not a junction detector, it is a spawn detector wearing one's clothes.
    Excluding the first `warmup` steps halves the node count (66 -> 33 merged)
    and RAISES junction coverage from 87.0% to 91.3%, which is the signature of
    removing noise rather than removing signal.
    """
    pts, wts, traj, junc = [], [], [], []
    for e in episodes:
        if any(s.get("x") is None for s in e["steps"]) or len(e["steps"]) < 10:
            continue
        xy, h, inj, s = episode_arrays(e)
        if s[-1] < 2 * min_sep:
            continue
        if len(xy) < warmup + 30:
            continue
        traj.append(xy)
        junc.append(NP.junction_events(inj, xy))
        if detector == "mad":
            idx = [i + warmup for i in NP.mad_spikes(h[warmup:], s[warmup:],
                                                     min_sep, 9, z)]
        else:
            n = max(1, int(round(s[-1] / 100.0 * dens)))
            idx = [i + warmup for i in NP.place_peaks(h[warmup:], s[warmup:],
                                                      n, min_sep, 9)]
        if idx:
            pts.append(xy[idx]); wts.append(NP.smooth(h, 9)[idx])
    pts = np.vstack(pts) if pts else np.empty((0, 2))
    wts = np.concatenate(wts) if wts else np.empty(0)
    junc = np.vstack([j for j in junc if len(j)]) if junc else np.empty((0, 2))
    return pts, wts, traj, junc


def uniform_to_count(traj, target, merge_r, lo=1.0, hi=400.0, iters=40):
    """Uniform arc-length nodes, spacing bisected until the MERGED count matches."""
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pts = []
        for xy in traj:
            s = NP.arclength(xy)
            if s[-1] < mid:
                continue
            n = max(1, int(s[-1] // mid))
            pts.append(xy[NP.place_uniform(s, n)])
        pts = np.vstack(pts) if pts else np.empty((0, 2))
        merged = greedy_merge(pts, merge_r)
        if best is None or abs(len(merged) - target) < abs(len(best) - target):
            best = merged
        if len(merged) > target:
            lo = mid            # too many nodes -> increase spacing
        else:
            hi = mid
    return best


def coverage(nodes, junctions, radii):
    """RECALL and PRECISION, which answer different questions and disagree.

      recall     fraction of JUNCTIONS with a node within R -- "is the map
                 complete?" Distances measured junction -> nearest node.
      precision  fraction of NODES within R of a junction -- "is the map
                 clean?" Distances measured node -> nearest junction.

    Reporting recall alone flatters any criterion that simply places nodes
    everywhere, which is why both are returned. Precision must also be read
    against the BASE RATE -- the fraction of driven road within R of a junction
    -- since a node dropped at random already scores that much.
    """
    if len(junctions) == 0:
        return {}
    if len(nodes) == 0:
        dj = np.full(len(junctions), np.inf)
        dn = np.array([np.inf])
    else:
        pair = np.linalg.norm(junctions[:, None, :] - nodes[None, :, :], axis=2)
        dj = pair.min(axis=1)      # per junction  -> recall
        dn = pair.min(axis=0)      # per node      -> precision
    out = {"median": float(np.median(dj)),
           "median_node_to_junction": float(np.median(dn))}
    for r in radii:
        out[f"within_{r}m"] = float((dj <= r).mean())          # recall
        out[f"recall_{r}m"] = float((dj <= r).mean())
        out[f"precision_{r}m"] = float((dn <= r).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/entropy_nogoal.json")
    ap.add_argument("--out-png", default="results/map_nodes.png")
    ap.add_argument("--out-json", default="results/map_nodes.json")
    ap.add_argument("--detector", choices=("mad", "topn"), default="mad")
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("--nodes-per-100m", type=float, default=4.0)
    ap.add_argument("--min-sep-m", type=float, default=8.0)
    ap.add_argument("--merge-m", type=float, default=6.0)
    ap.add_argument("--warmup", type=int, default=25,
                    help="steps excluded at episode start; see collect()")
    args = ap.parse_args()

    data = json.load(open(args.inp))
    eps = data["episodes"]
    radii = (5, 10)

    raw, wts, traj, junc_raw = collect(eps, args.detector, args.z,
                                       args.nodes_per_100m, args.min_sep_m,
                                       args.warmup)
    nodes_e = greedy_merge(raw, args.merge_m, wts)
    junctions = greedy_merge(junc_raw, args.merge_m)
    nodes_u = uniform_to_count(traj, len(nodes_e), args.merge_m)

    print(f"[Map] {len(traj)} episodes, {len(junc_raw)} junction crossings "
          f"-> {len(junctions)} distinct junctions (merged at {args.merge_m} m)")
    print(f"[Map] entropy: {len(raw)} raw nodes -> {len(nodes_e)} merged "
          f"(compression {len(raw)/max(len(nodes_e),1):.1f}x)")
    print(f"[Map] uniform: matched to {len(nodes_u)} merged nodes")
    ce, cu = coverage(nodes_e, junctions, radii), coverage(nodes_u, junctions, radii)
    # BASE RATE: fraction of driven road within R of a junction, i.e. what a
    # randomly placed node scores. Precision is meaningless without it.
    allp = np.vstack(traj)
    dbase = np.linalg.norm(allp[:, None, :] - junctions[None, :, :],
                           axis=2).min(axis=1)
    print(f"\n{'arm':<10}{'nodes':>6}" +
          "".join(f"{'Recall<'+str(r)+'m':>13}{'Prec<'+str(r)+'m':>12}" for r in radii))
    for nm, n, c in (("entropy", len(nodes_e), ce), ("uniform", len(nodes_u), cu)):
        print(f"{nm:<10}{n:>6}" + "".join(
            f"{c[f'recall_{r}m']*100:>12.1f}%{c[f'precision_{r}m']*100:>11.1f}%"
            for r in radii))
    print(f"{'base rate':<10}{'--':>6}" + "".join(
        f"{'--':>13}{(dbase <= r).mean()*100:>11.1f}%" for r in radii))
    print(f"  median junction->node: entropy {ce['median']:.2f} m, "
          f"uniform {cu['median']:.2f} m")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.6), sharex=True, sharey=True)
    for ax, nodes, col, title in ((axes[0], nodes_e, PEAKS, "Entropy spikes"),
                                  (axes[1], nodes_u, UNIFORM, "Uniform spacing")):
        for xy in traj:
            ax.plot(xy[:, 0], xy[:, 1], color=MUTED, linewidth=0.35, alpha=0.35,
                    zorder=1, solid_capstyle="round")
        ax.scatter(junctions[:, 0], junctions[:, 1], s=95, facecolor="none",
                   edgecolor=JUNC, linewidth=1.0, zorder=2)
        ax.scatter(nodes[:, 0], nodes[:, 1], s=16, color=col, zorder=3,
                   edgecolor="white", linewidth=0.5)
        c = coverage(nodes, junctions, radii)
        ax.set_title(f"{title} — {len(nodes)} nodes   ·   "
                     f"Recall {c['recall_10m']*100:.0f}%   ·   "
                     f"Precision {c['precision_10m']*100:.0f}%",
                     fontsize=7.6, color=INK, pad=4)
        # 'box' not 'datalim': the two axes are shared, and datalim adjustment
        # is illegal on shared axes. Sharing is what makes the two panels
        # directly comparable, so it is the aspect mode that gives way.
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.grid(True, color=GRID, linewidth=0.4); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=INK2)
    axes[0].set_ylabel("Y (m)")
    from matplotlib.lines import Line2D
    # FIGURE-level legend BELOW the panels, not inside an axis. Town02's road
    # network reaches the top-right corner of both panels, so an in-axis legend
    # at "upper right" sat on top of real nodes -- and the panels share axes, so
    # there is no empty corner to retreat to in either. Placing it outside costs
    # a little height and cannot occlude data at any zoom.
    fig.legend(handles=[
        Line2D([0], [0], color=MUTED, lw=0.8, label="Driven trajectories"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=JUNC, markersize=7, label="Junction"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PEAKS,
               markersize=5, label="Graph node")],
        frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.055),
        handlelength=1.6, columnspacing=2.0)
    fig.tight_layout(pad=0.35, w_pad=0.8, rect=(0, 0.10, 1, 1))
    for ext, kw in ((".png", {"dpi": 600}), (".pdf", {})):
        fig.savefig(args.out_png.replace(".png", ext), bbox_inches="tight",
                    facecolor="white", **kw)
        print(f"[Map] -> {args.out_png.replace('.png', ext)}")

    json.dump({"detector": args.detector, "z": args.z, "merge_m": args.merge_m,
               "n_junctions": len(junctions), "entropy": {"n": len(nodes_e), **ce},
               "uniform": {"n": len(nodes_u), **cu},
               "nodes_entropy": nodes_e.tolist(), "nodes_uniform": nodes_u.tolist(),
               "junctions": junctions.tolist()}, open(args.out_json, "w"))
    print(f"[Map] -> {args.out_json}")


if __name__ == "__main__":
    main()
