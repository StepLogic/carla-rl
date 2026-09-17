"""Show what the MAD threshold does, on the entropy traces themselves.

    python figure_z_on_traces.py

The ablation table says raising z trades recall for precision. It does not show
WHICH spikes are being given up, and that is the part a reader can only judge by
looking. Here each threshold is drawn back onto the trace in ENTROPY UNITS --

    h_thresh(z) = median(h) + z * 1.4826 * MAD(h)

-- computed per episode from that episode's own smoothed trace, exactly as the
detector does. So the horizontal lines are not decoration: a candidate is
eligible at threshold z precisely when the curve is above that line.

Nodes are shaded by the HIGHEST z that still retains them. Dark markers survive
z=4.0; pale ones are dropped first. Raising z prunes from the bottom up, and the
figure shows whether what it prunes are junction spikes or something else.

SEQUENTIAL shading, not categorical: z is an ordered magnitude, so a single hue
running light to dark is the honest encoding. A categorical palette here would
imply the thresholds are unordered kinds.

The grey band is the first 25 steps, excluded from detection -- every episode
opens with a reset transient that would otherwise dominate the MAD baseline and
plant nodes at spawn points.
"""
import argparse
import json

import numpy as np

import map_nodes_entropy as M
import node_placement_entropy as NP

JUNC = "#E69F00"
LINE = "#4a4a4a"
MUTED, INK, INK2, GRID = "#8a8a8a", "#000000", "#3d3d3d", "#dcdcdc"
WARM, SEP, SMOOTH = 25, 8.0, 9
ZS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default="evidence/2026-09-09_entropy_at_junctions/data/"
                            "entropy_nogoal_500cap.json")
    ap.add_argument("--out", default="results/fig_z_on_traces")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--mode", choices=("spread", "worst"), default="spread",
                    help="spread: sample the distribution. worst: the episodes "
                         "where the detector performs WORST, chosen by measured "
                         "per-episode recall and precision, not by eye.")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.6,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    cmap = plt.get_cmap("Blues")
    zcol = {z: cmap(0.35 + 0.6 * i / (len(ZS) - 1)) for i, z in enumerate(ZS)}

    # SPREAD, not top-N. Ranking episodes by junction count and taking the head
    # shows the detector only where it has the most to work with, which is
    # exactly the selection a sceptical reader should distrust. This samples
    # across the distribution instead: the richest, the median, the sparsest
    # that still has a junction, and an episode that ended in FAILURE -- the
    # case where the precision analysis says spikes are not junctions at all.
    allep = [e for e in json.load(open(args.inp))["episodes"]
             if len(e["steps"]) >= WARM + 60]
    withj = sorted((e for e in allep if sum(s["in_junction"] for s in e["steps"])),
                   key=lambda e: -sum(s["in_junction"] for s in e["steps"]))
    failed = [e for e in withj if e.get("reasons")]
    pick, seen = [], set()
    order = []
    if withj:
        order += withj[:2]                                   # richest
        order += [withj[len(withj) // 2]]                     # median
        order += [withj[-1]]                                  # sparsest
    if failed:
        order += failed[:2]                                   # ended in failure
    order += withj                                            # top up if needed
    if args.mode == "worst":
        # SELECTED BY MEASUREMENT, not by eye. Per-episode recall (junctions
        # with a node within 10 m) then precision, both ascending, so the
        # panels are the detector's genuine failures rather than ones chosen to
        # look bad. Episodes where it placed NO nodes despite junctions being
        # present come first -- that is the most complete failure available.
        scored = []
        for e in allep:
            xy, h, inj, sa = M.episode_arrays(e)
            J = NP.junction_events(inj, xy)
            if len(J) == 0:
                continue
            idx = [i + WARM for i in NP.mad_spikes(h[WARM:], sa[WARM:], SEP,
                                                   SMOOTH, 3.0)]
            if not idx:
                scored.append((-1.0, -1.0, e))       # no nodes at all
                continue
            nd = xy[idx]
            pair = np.linalg.norm(J[:, None, :] - nd[None, :, :], axis=2)
            scored.append((float((pair.min(axis=1) <= 10).mean()),
                           float((pair.min(axis=0) <= 10).mean()), e))
        scored.sort(key=lambda r: (r[0], r[1]))
        order = [r[2] for r in scored]

    for e in order:
        if e["episode"] not in seen:
            pick.append(e); seen.add(e["episode"])
        if len(pick) >= args.n:
            break

    fig, axes = plt.subplots(len(pick), 1, figsize=(7.16, 1.9 * len(pick)))
    axes = np.atleast_1d(axes)

    for ax, e in zip(axes, pick):
        st = e["steps"]
        h = np.array([s["entropy"] for s in st], float)
        inj = np.array([s["in_junction"] for s in st], bool)
        xy = np.array([[s["x"], s["y"]] for s in st], float)
        s_arc = NP.arclength(xy)
        t = np.arange(len(h))

        # The detector's own view: smoothed, warmup excluded.
        hs_full = NP.smooth(h, SMOOTH)
        hs = NP.smooth(h[WARM:], SMOOTH)
        med = float(np.median(hs))
        mad = 1.4826 * float(np.median(np.abs(hs - med)))

        ax.axvspan(0, WARM, color=MUTED, alpha=0.18, linewidth=0, zorder=0)
        d = np.diff(np.concatenate([[0], inj.view(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            ax.axvspan(a, min(b, len(t) - 1), color=JUNC, alpha=0.30,
                       linewidth=0, zorder=1)
        ax.plot(t, h, color=LINE, linewidth=0.8, zorder=3)

        counts = {}
        surviving = {}
        for z in ZS:
            thr = med + z * mad
            ax.axhline(thr, color=zcol[z], linewidth=0.9, zorder=2,
                       linestyle=(0, (5, 2)))
            # Label ONLY the extremes. The six thresholds span ~0.15 nats on
            # these traces, so six right-margin labels overlap into an
            # unreadable stack. The lines themselves carry the ordering; the
            # two endpoints are enough to anchor it.
            # Threshold labels intentionally omitted: the six lines sit within
            # ~0.15 nats of each other and the shading already encodes their
            # order (pale = low z, dark = high). The counts are reported in
            # text alongside the figure rather than crowded onto it.
            idx = [i + WARM for i in NP.mad_spikes(h[WARM:], s_arc[WARM:],
                                                   SEP, SMOOTH, z)]
            counts[z] = len(idx)
            for i in idx:
                surviving[i] = z          # last (highest) z that keeps it

        for i, zmax in surviving.items():
            ax.plot(i, h[i], "o", color=zcol[zmax], markersize=5.0, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.7)

        if e.get("reasons"):
            ax.axvline(len(t) - 1, color="#D55E00", linewidth=1.1, zorder=6)
        ax.set_ylabel("Entropy\n(nats)")
        ax.set_xlim(0, len(t) - 1)
        ax.text(0.004, 0.985,
                f"Episode {e['episode']}"
                + (f" (ended: {','.join(e['reasons'])})" if e.get("reasons") else ""),
                transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
                color=INK2)
        ax.grid(True, color=GRID, linewidth=0.4)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=INK2)
    axes[-1].set_xlabel("Step")

    handles = [Patch(facecolor=JUNC, alpha=0.30, label="In junction"),
               Patch(facecolor=MUTED, alpha=0.18, label="Excluded (reset transient)"),
               Line2D([0], [0], color=zcol[ZS[0]], lw=0.9, linestyle=(0, (5, 2)),
                      label="MAD threshold"),
               Line2D([0], [0], marker="o", color="none",
                      markerfacecolor=cmap(0.35), markersize=5,
                      label="Node kept only at low $z$"),
               Line2D([0], [0], marker="o", color="none",
                      markerfacecolor=cmap(0.95), markersize=5,
                      label="Node surviving $z$=4.0")]
    axes[0].legend(handles=handles, frameon=False, ncol=5, loc="lower left",
                   bbox_to_anchor=(0, 1.02), handlelength=1.3,
                   columnspacing=1.0)

    fig.tight_layout(pad=0.3, h_pad=0.6)
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 600})):
        fig.savefig(args.out + ext, bbox_inches="tight", facecolor="white", **kw)
        print(f"[Fig] -> {args.out}{ext}")
    for e in pick:
        st = e["steps"]
        h = np.array([s["entropy"] for s in st], float)
        xy = np.array([[s["x"], s["y"]] for s in st], float)
        sa = NP.arclength(xy)
        cs = [len(NP.mad_spikes(h[WARM:], sa[WARM:], SEP, SMOOTH, z)) for z in ZS]
        print(f"[Fig] episode {e['episode']:>2}: nodes at z={ZS} -> {cs}")


if __name__ == "__main__":
    main()
