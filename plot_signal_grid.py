"""SR by uncertainty signal and filler, at four matched budgets (user 2026-09-20 "ablate SR").

Left: the uniform filler. Right: the SGM filler. Bars are the three signals, count-matched at 30 seeds;
the dashed line in each panel is that filler with NO seeds, which is what seeding has to beat.
All rows driven under the DINOv2 localiser so the grid is internally comparable. Single runs except where
the table marks otherwise, so read the ordering, not the individual gaps.
Run from the icra-2027 folder.
"""
import collections, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SIG = [("entropy", "Policy entropy", "#c0398b"), ("epistemic", "Epistemic", "#2a78d6"),
       ("aleatoric", "Aleatoric", "#eb6834")]
BUD = ["~60", "~115", "~150", "~550"]
UG = {"~60": "15", "~115": "8", "~150": "6", "~550": "2"}
SG = {"~60": {"entropy": "44", "epistemic": "44", "aleatoric": "44"},
      "~115": {"entropy": "14", "epistemic": "13", "aleatoric": "13"},
      "~150": {"entropy": "9", "epistemic": "9", "aleatoric": "9"},
      "~550": {"entropy": "2", "epistemic": "2", "aleatoric": "2"}}
UNSEEDED_U = {"~60": "uniform15_trace", "~115": "uniform8_trace", "~150": "uniform6_trace", "~550": "uniform2_trace"}
UNSEEDED_S = {"~60": "sgm_g24_e55_c55", "~115": "sgm_g10_e55_c55", "~150": "sgm_g7_e55_c55", "~550": "sgm_g2_e55_c55"}
GLOBS = ["results/drive_plan_c/localiser/dinov2_bank_*.jsonl", "results/drive_plan_c/signal_grid/*.jsonl"]

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9})


def load():
    per = {}
    for f in sorted({f for g in GLOBS for f in glob.glob(g)}):
        if any(s in f for s in ("smoke", "debug")):
            continue
        by = collections.defaultdict(dict)
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if r.get("localiser", "").split(":")[0] == "dinov2_bank":
                    by[r["map"]][(r["set"], r["index"])] = r
        for m, d in by.items():
            if len(d) == 40:
                per.setdefault(m, []).append(d)
    return {m: float(np.mean([np.mean([r["success"] for r in d.values()]) for d in v])) for m, v in per.items()}


def main():
    sr = load()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), sharey=True, constrained_layout=True)
    x = np.arange(len(BUD)); w = 0.26
    for ax, filler in zip(axes, ("Uniform", "SGM")):
        for j, (key, label, colour) in enumerate(SIG):
            vals = []
            for b in BUD:
                m = f"hybrid_uniform{UG[b]}_{key}_thin4" if filler == "Uniform" else f"hybrid_sgm_g{SG[b][key]}_{key}_thin4"
                vals.append(sr.get(m, np.nan))
            ax.bar(x + (j - 1) * w, vals, w, color=colour, label=label, zorder=3)
        ref = [sr.get((UNSEEDED_U if filler == "Uniform" else UNSEEDED_S)[b], np.nan) for b in BUD]
        for k, v in enumerate(ref):
            ax.plot([x[k] - 1.6 * w, x[k] + 1.6 * w], [v, v], color="#444444", ls="--", lw=1.4, zorder=4,
                    label="No seeds (filler alone)" if k == 0 and filler == "Uniform" else None)
        ax.set_title(f"{filler} filler", fontsize=10)
        ax.set_xticks(x, [b.replace("~", "≈") + " nodes" for b in BUD])
        ax.grid(True, axis="y", color="#ececec", lw=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 0.62)
    axes[0].set_ylabel("Success rate (SR)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle("Town02, DINOv2 localiser: seeding a map with uncertainty, 30 seeds per signal", fontsize=10)
    for e in ("pdf", "png"):
        fig.savefig(f"figures/signal_grid_sr_town02.{e}", **({"dpi": 200} if e == "png" else {}))
    print("-> figures/signal_grid_sr_town02.{pdf,png}")


if __name__ == "__main__":
    main()
