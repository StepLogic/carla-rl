"""Pareto plot: Entropy + Spacing[g] against plain Spacing[g] at the same intervals (user 2026-09-16).

One panel per town. X = map size (nodes, log scale), Y = success rate. Two series:
  Spacing[g] alone            g = 2, 6, 8, 15
  Entropy + Spacing[g]        g = 6, 8, 15   (30 / 25 thinned entropy nodes as seeds)
plus the other maps (Dense, SGM, NoMaD, Entropy alone) as reference points, and the Pareto staircase over ALL maps.
Error bars are the sd over runs where a map has more than one run; single-run points are drawn without bars and
marked in the legend. Values come from the drive JSONLs, never hard-coded.
"""
import argparse, glob, json, os
import numpy as np

TOWNS = {
    "Town02": {"files": ["results/drive_plan_c/sequence_undirected.jsonl", "results/drive_plan_c/raw_trace.jsonl",
                         "results/drive_plan_c/sgm_ablation.jsonl", "results/nomad/nomad_uniform2_trace_v6ncap_20kmh.jsonl",
                         "results/drive_plan_c/repeat/run2.jsonl", "results/drive_plan_c/repeat/run3.jsonl",
                         "results/nomad/repeat/nomad_uniform2_trace_v6ncap_20kmh_run2.jsonl",
                         "results/nomad/repeat/nomad_uniform2_trace_v6ncap_20kmh_run3.jsonl",
                         "results/drive_plan_c/hybrid8.jsonl", "results/drive_plan_c/hybrid15.jsonl",
                         "results/drive_plan_c/uniform_sweep.jsonl"],
               "seq": "results/maps/sequence"},
    "Town07": {"files": ["results/town07/drive_plan_c.jsonl", "results/town07/nomad/nomad_uniform2_trace_v6ncap_20kmh.jsonl",
                         "results/town07/repeat/run2.jsonl", "results/town07/repeat/run3.jsonl",
                         "results/town07/nomad/repeat/nomad_uniform2_trace_v6ncap_20kmh_run2.jsonl",
                         "results/town07/nomad/repeat/nomad_uniform2_trace_v6ncap_20kmh_run3.jsonl",
                         "results/town07/hybrid8.jsonl", "results/town07/hybrid15.jsonl",
                         "results/town07/uniform_sweep.jsonl"],
               "seq": "results/town07/maps/sequence"},
}
SPACING = [("uniform2_trace", 2), ("uniform6_trace", 6), ("uniform8_trace", 8), ("uniform15_trace", 15)]
HYBRID = [("hybrid_uniform6_entropy_thin4", 6), ("hybrid_uniform8_entropy_thin4", 8), ("hybrid_uniform15_entropy_thin4", 15)]
OTHER = [("raw_trace", "Dense"), ("entropy_nosmooth_high_thin4", "Entropy"), ("sgm_g2_e55_c55", "SGM"),
         ("nomad_uniform2_trace_v6ncap_20kmh", "NoMaD")]


def per_map(files):
    """map -> list of per-run success rates (a run = one file's 40 episodes of that map)."""
    out = {}
    for f in files:
        if not os.path.exists(f):
            continue
        rows = [json.loads(l) for l in open(f) if l.strip()]
        for m in dict.fromkeys(r["map"] for r in rows):
            L = [r for r in rows if r["map"] == m]
            if len(L) == 40:
                out.setdefault(m, []).append(float(np.mean([r["success"] for r in L])))
    return out


def nodes_of(seq_dir, m):
    p = f"{seq_dir}/{'uniform2_trace' if m.startswith('nomad') else m}_sequence.json"
    return len(json.load(open(p))["nodes"]) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures/pareto_spacing.png")
    # user 2026-09-16 "compare just uniform and entropy + uniform": drop Dense / SGM / NoMaD / Entropy-alone
    ap.add_argument("--pairs-only", action="store_true", help="plot only Spacing[g] and Entropy + Spacing[g]")
    a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(TOWNS), figsize=(11.5, 4.6), sharey=False)
    for ax, (town, cfg) in zip(np.atleast_1d(axes), TOWNS.items()):
        srs = per_map(cfg["files"])
        pts = []                                     # (nodes, mean sr, sd, label, single-run?)
        for series, marker, colour, name in ((SPACING, "s", "#7f7f7f", "Spacing[g]"),
                                             (HYBRID, "o", "#d62728", "Entropy + Spacing[g]")):
            xs, ys, es, singles = [], [], [], []
            for m, g in series:
                if m not in srs:
                    continue
                n, v = nodes_of(cfg["seq"], m), srs[m]
                xs.append(n); ys.append(float(np.mean(v)))
                es.append(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0); singles.append(len(v) == 1)
                pts.append((n, float(np.mean(v)), es[-1], f"{name}={g}", len(v) == 1))
                ax.annotate(f"{g} m", (n, ys[-1]), textcoords="offset points", xytext=(4, 5), fontsize=7.5, color=colour)
            order = np.argsort(xs)
            ax.errorbar(np.array(xs)[order], np.array(ys)[order], yerr=np.array(es)[order], fmt=marker + "-",
                        color=colour, ms=6, lw=1.6, capsize=3, label=name)
        for m, lab in ([] if a.pairs_only else OTHER):
            if m not in srs:
                continue
            n, v = nodes_of(cfg["seq"], m), srs[m]
            sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            ax.errorbar([n], [np.mean(v)], yerr=[sd], fmt="^", color="#1f77b4", ms=6, capsize=3)
            ax.annotate(lab, (n, float(np.mean(v))), textcoords="offset points", xytext=(5, -9), fontsize=7.5, color="#1f77b4")
            pts.append((n, float(np.mean(v)), sd, lab, len(v) == 1))
        # Pareto staircase over every map: fewer nodes and higher SR
        front = sorted((n, v) for n, v, _sd, _l, _s in pts
                       if not any(n2 <= n and v2 >= v and (n2 < n or v2 > v) for n2, v2, _s2, _l2, _q in pts if (n2, v2) != (n, v)))
        if front:
            fx, fy = zip(*front)
            ax.step(list(fx) + [max(p[0] for p in pts)], list(fy) + [fy[-1]], where="post", color="black", lw=1.0,
                    alpha=0.45, zorder=0,
                    label="Pareto front" + (" (spacing maps)" if a.pairs_only else " (all maps)"))
        ax.set_xscale("log"); ax.set_xlabel("Map size (nodes)"); ax.set_ylabel("Success rate")
        ax.set_title(town); ax.grid(alpha=0.25, which="both")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper left", framealpha=0.92)
    fig.text(0.5, -0.02, "Labels give the spacing g. Error bars: sd over runs (single-run points have none).",
             ha="center", fontsize=8)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight"); print("[wrote]", a.out)


if __name__ == "__main__":
    main()
