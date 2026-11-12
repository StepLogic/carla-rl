"""Pareto view of the placement grid: SR and SPL against map size (user 2026-09-20).

Colour is the SEED SIGNAL, line style is the FILLER -- a composite encoding, because eight separate hues
would not survive a colour-blind check and the two factors are what the reader has to separate. Grey is
the unseeded control for each filler. Triangles at 30 nodes are the seed-only maps (no filler at all).
Town02, DINOv2 localiser throughout. Run from the icra-2027 folder.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid_rows import rows, nodes  # noqa: E402

COLOUR = {"entropy": "#c0398b", "epistemic": "#2a78d6", "aleatoric": "#eb6834", None: "#6b6b6b"}
NAME = {"entropy": "Policy entropy", "epistemic": "Epistemic", "aleatoric": "Aleatoric", None: "No seeds"}
STYLE = {"Uniform": ("-", "o"), "SGM": ("--", "s")}

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9})


def main():
    R = [r for r in rows() if r["d"]]
    series = {}
    for r in R:
        if r.get("fixed"):
            continue
        if r["ours"]:
            key = (r.get("signal"), r["filler"])
        elif r["arm"].startswith("Spacing"):
            key = (None, "Uniform")
        elif r["arm"].startswith("SGM"):
            key = (None, "SGM")
        else:
            continue
        series.setdefault(key, []).append((nodes(r["map"]), r["d"][0], r["d"][1]))
    fixed = {r["arm"]: (nodes(r["map"]), r["d"][0], r["d"][1]) for r in R if r.get("fixed")}

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True)
    for ax, (idx, ylab) in zip(axes, [(1, "Success rate (SR)"), (2, "SPL")]):
        dense = fixed.get("Dense (every sample)")
        if dense:
            ax.axhline(dense[idx], color="#555555", ls=":", lw=1.2, zorder=1)
            ax.text(600, dense[idx] + 0.008, "Dense (1155 nodes)", fontsize=7, color="#555555", ha="right")
        for (sig, filler), pts in sorted(series.items(), key=lambda kv: str(kv[0])):
            pts = sorted(pts)
            ls, mk = STYLE[filler]
            ax.plot([p[0] for p in pts], [p[idx] for p in pts], color=COLOUR[sig], ls=ls, marker=mk,
                    ms=4.5, lw=1.5, zorder=3)
        # the three seed-only maps all sit at 30 nodes and two of them score identically, so they are
        # nudged apart on x -- the marker still reads at its own 30-node budget.
        for dx, (name, key) in zip((-9, 0, 9), (("Policy entropy only", "entropy"),
                                                ("Epistemic only", "epistemic"),
                                                ("Aleatoric only", "aleatoric"))):
            v = fixed.get(name)
            if v:
                ax.plot([v[0] + dx], [v[idx]], marker="^", ms=7, color=COLOUR[key], ls="", zorder=4)
        ax.set_xlabel("Nodes"); ax.set_ylabel(ylab)
        ax.set_xlim(0, 640); ax.set_ylim(0, 0.62)
        ax.grid(True, axis="y", color="#ececec", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].annotate("Seed-only maps\n(30 nodes, no filler)", xy=(30, 0.19), xytext=(95, 0.075),
                     fontsize=7, color="#444444",
                     arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8))
    hue = [Line2D([], [], color=COLOUR[k], lw=2.2) for k in (None, "entropy", "epistemic", "aleatoric")]
    sty = [Line2D([], [], color="#444444", ls=STYLE[f][0], marker=STYLE[f][1], ms=4.5, lw=1.5)
           for f in ("Uniform", "SGM")]
    fig.legend(hue + sty, [NAME[k] for k in (None, "entropy", "epistemic", "aleatoric")]
               + ["Uniform filler", "SGM filler"],
               loc="outside lower center", ncol=6, frameon=False, fontsize=8.5, handletextpad=0.5)
    fig.suptitle("Town02 under the DINOv2 localiser: success against map size, by seed signal and filler",
                 fontsize=10)
    for e in ("pdf", "png"):
        fig.savefig(f"figures/pareto_grid_town02.{e}", **({"dpi": 200} if e == "png" else {}))
    print("-> figures/pareto_grid_town02.{pdf,png}")


if __name__ == "__main__":
    main()
