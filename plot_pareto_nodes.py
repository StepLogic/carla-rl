"""SR and SPL against map size in CARLA Town02 (user 2026-09-16: "stick to one town"), the same four node budgets for every mapping method.

The budgets are set by the Spacing sweep (15 / 8 / 6 / 2 m). Entropy + Spacing uses the same spacings, SGM keeps
edge 55 / consistency 55 and varies its metric gate to hit the node count, NoMaD drives the Spacing maps. Dense (every
trace sample) and Entropy (z=2 spikes only) cannot be sized, so they are horizontal reference lines.
Each point is the mean over the complete runs of the same 40 episodes found on disk; incomplete runs are skipped.
Run from the icra-2027 folder: python3 <worktree>/plot_pareto_nodes.py
"""
import collections, glob, json, re, sys
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9})

TOWNS = {
    "Town02": dict(runs=["results/drive_plan_c/*.jsonl", "results/drive_plan_c/repeat/*.jsonl",
                         "results/drive_plan_c/equal_points/*.jsonl", "results/nomad/nomad_*v6ncap*.jsonl",
                         "results/nomad/repeat/*.jsonl", "results/nomad/equal_points/*.jsonl"],
                   seq="results/maps/sequence", sgm_gates=["24", "10", "7", "2"],
                   ent_sgm_gates=["44", "14", "9", "2"]),
    "Town07": dict(runs=["results/town07/*.jsonl", "results/town07/repeat/*.jsonl", "results/town07/equal_points/*.jsonl",
                         "results/town07/nomad/*.jsonl", "results/town07/nomad/repeat/*.jsonl",
                         "results/town07/nomad/equal_points/*.jsonl"],
                   seq="results/town07/maps/sequence", sgm_gates=["20", "9.5", "6.5", "2"],
                   ent_sgm_gates=[]),
}
SPACINGS = ["15", "8", "6", "2"]
# method -> (colour, marker); categorical slots in fixed order, marker shape as secondary encoding
LINES = {"Spacing": ("#2a78d6", "o"), "Entropy + Spacing": ("#eb6834", "D"), "SGM": ("#4a3aa7", "s"),
         "NoMaD": ("#c98500", "v")}
# user 2026-09-19 "how about we report madent + sgm": the madent seeds pinned inside the SGM frame. Opt-in
# (--with-entropy-sgm), and it writes its own figure/table, so the paper's four-arm artefacts are untouched.
# Its hue was checked against the four above: worst CVD dE 8.6 (marker shape is the second encoding).
ENT_SGM = ("Entropy + SGM", "#c0398b", "^")
REFS = {"Dense": ("#555555", ":"), "Entropy": ("#1baf7a", "--")}


def grid(town):
    g = {}
    for s in SPACINGS:
        g[f"uniform{s}_trace"] = "Spacing"
        g[f"hybrid_uniform{s}_entropy_thin4"] = "Entropy + Spacing"
        g[f"nomad_uniform{s}_trace_v6ncap_20kmh"] = "NoMaD"
    for s in TOWNS[town]["sgm_gates"]:
        g[f"sgm_g{s}_e55_c55"] = "SGM"
    if "--with-entropy-sgm" in sys.argv:
        for s in TOWNS[town]["ent_sgm_gates"]:
            g[f"hybrid_sgm_g{s}_entropy_thin4"] = ENT_SGM[0]
    g["raw_trace"] = "Dense"
    g["entropy_nosmooth_high_thin4"] = "Entropy"
    return g


def nodes_of(seq_dir, m):
    base = re.sub(r"^nomad_(.*)_v6ncap_20kmh$", r"\1", m)
    return len(json.load(open(f"{seq_dir}/{base}_sequence.json"))["nodes"])


def load(town):
    cfg, G = TOWNS[town], grid(town)
    per_run = collections.defaultdict(list)  # map -> [(SR, SPL)], one entry per complete file (= run)
    for f in sorted({f for p in cfg["runs"] for f in glob.glob(p)}):
        if any(s in f for s in ("smoke", "debug")): continue
        by_map = collections.defaultdict(list)
        for line in open(f):
            if line.strip():
                r = json.loads(line); by_map[r["map"]].append(r)
        for m, L in by_map.items():
            if m not in G: continue
            if len(L) != 40:
                print(f"[skip] {f} {m}: {len(L)}/40 episodes", file=sys.stderr); continue
            per_run[m].append((np.mean([x["success"] for x in L]), np.mean([x["spl"] or 0 for x in L])))
    rows = []
    for m, v in per_run.items():
        a = np.array(v)
        sd = lambda c: float(a[:, c].std(ddof=1)) if len(a) > 1 else 0.0
        rows.append(dict(map=m, method=G[m], nodes=nodes_of(cfg["seq"], m), runs=len(a),
                         sr=a[:, 0].mean(), spl=a[:, 1].mean(), sr_sd=sd(0), spl_sd=sd(1)))
    return rows


def main():
    if "--with-entropy-sgm" in sys.argv:
        LINES[ENT_SGM[0]] = ENT_SGM[1:]
    linear = "--log" not in sys.argv  # user 2026-09-17 "use actual nodes": linear node axis by default
    data = {t: load(t) for t in TOWNS}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), sharex="col", constrained_layout=True)
    for j, town in enumerate(TOWNS):
        rows = data[town]
        for i, (key, label) in enumerate([("sr", "Success rate (SR)"), ("spl", "SPL")]):
            ax = axes[i, j]
            for meth, (c, ls) in REFS.items():
                R = [r for r in rows if r["method"] == meth]
                if R:
                    ax.axhline(R[0][key], color=c, ls=ls, lw=1.2, zorder=1, label=meth)
            for meth, (c, mk) in LINES.items():
                R = sorted([r for r in rows if r["method"] == meth], key=lambda r: r["nodes"])
                if R:
                    ax.plot([r["nodes"] for r in R], [r[key] for r in R], color=c, marker=mk, ms=4.5, lw=1.5,
                            label=meth, zorder=3)
            if linear:
                ax.set_xlim(0, 900)
            else:
                ax.set_xscale("log")
                ax.set_xlim(45, 1000)
                ax.set_xticks([50, 100, 200, 500, 1000], ["50", "100", "200", "500", "1000"])
                ax.minorticks_off()
            ax.set_ylim(0, 0.65)
            ax.grid(True, axis="y", color="#ececec", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 1: ax.set_xlabel("Nodes" if linear else "Nodes (log scale)")
            if j == 0: ax.set_ylabel(label)
            if i == 0: ax.set_title(town)
    h, l = axes[0, 0].get_legend_handles_labels()
    order = list(LINES) + list(REFS)
    hl = sorted(zip(h, l), key=lambda x: order.index(x[1]))
    fig.legend([x[0] for x in hl], [x[1] for x in hl], loc="outside upper center", ncol=6, frameon=False, fontsize=8,
               handletextpad=0.4, handlelength=2.4)
    stem = ("figures/pareto_nodes_sr_spl" + ("" if linear else "_log")
            + ("_entropy_sgm" if ENT_SGM[0] in LINES else ""))
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=200)

    md = ["# SR / SPL at equal node budgets (CARLA, 40 episodes per run)\n",
          "| Town | Method | Map | Nodes | Runs | SR | SPL |", "|---|---|---|---|---|---|---|"]
    for t in TOWNS:
        md += [f"| {t} | {r['method']} | {r['map']} | {r['nodes']} | {r['runs']} | {r['sr']:.3f} ± {r['sr_sd']:.3f} | "
               f"{r['spl']:.3f} ± {r['spl_sd']:.3f} |"
               for r in sorted(data[t], key=lambda r: (order.index(r["method"]), r["nodes"]))]
    open("tables/pareto_nodes_sr_spl" + ("_entropy_sgm" if ENT_SGM[0] in LINES else "") + ".md",
         "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
