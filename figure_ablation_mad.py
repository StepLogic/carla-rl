"""Ablation figure: is the placement result an artefact of detector settings?

Panel (a) is the precision/recall trade-off as the threshold moves, with the
node count annotated -- because a setting that buys recall by placing more nodes
has proved nothing.

Panel (b) is the thing that matters: recall MINUS a uniform arm re-matched to
the same merged node count, for every configuration tried. If the advantage
depended on the settings, this would scatter.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PEAKS, UNIFORM, MUTED, INK, INK2, GRID = "#0072B2", "#E69F00", "#8a8a8a", "#000000", "#3d3d3d", "#dcdcdc"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})

d = json.load(open("results/ablation_mad.json"))
cfg, base = d["configs"], d["base_rate_precision_5m"]
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw={"width_ratios": [1, 1.25]})

ax = axes[0]
zs = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
rec = [cfg[f"z={z}"]["recall"] * 100 for z in zs]
pre = [cfg[f"z={z}"]["precision"] * 100 for z in zs]
n = [cfg[f"z={z}"]["nodes"] for z in zs]
ax.plot(zs, rec, color=PEAKS, marker="o", markersize=3.4, linewidth=1.3,
        markeredgecolor="white", markeredgewidth=0.5, label="Recall (<10 m)")
ax.plot(zs, pre, color=UNIFORM, marker="s", markersize=3.2, linewidth=1.3,
        markeredgecolor="white", markeredgewidth=0.5, label="Precision (<5 m)")
ax.axhline(base * 100, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
ax.text(3.95, base * 100 + 2.5, "Chance", fontsize=6.3, color=INK2, ha="right")
for z, r, c in zip(zs, rec, n):
    ax.annotate(f"{c}", (z, r), textcoords="offset points", xytext=(0, 5),
                ha="center", fontsize=5.8, color=INK2)
ax.set_xlabel("MAD threshold $z$")
ax.set_ylabel("Percent")
ax.set_ylim(0, 108)
ax.set_title("(a) Threshold trades recall for precision", pad=3)
ax.legend(frameon=False, loc="center right", handlelength=1.5)

ax = axes[1]
keys = [k for k in cfg if cfg[k] and np.isfinite(cfg[k]["uni_recall"])]
delta = [(cfg[k]["recall"] - cfg[k]["uni_recall"]) * 100 for k in keys]
order = np.argsort(delta)
ys = np.arange(len(keys))
ax.axvline(0, color=MUTED, linewidth=0.8)
ax.barh(ys, [delta[i] for i in order], color=PEAKS, height=0.7,
        edgecolor="white", linewidth=0.4)
ax.set_yticks(ys)
ax.set_yticklabels([keys[i] for i in order], fontsize=5.6)
ax.set_xlabel("Recall advantage over uniform at equal node count (pp)")
ax.set_title("(b) Advantage holds across every setting tried", pad=3)
ax.set_ylim(-0.7, len(keys) - 0.3)
ax.tick_params(axis="y", length=0)

for a in axes:
    a.grid(True, color=GRID, linewidth=0.45); a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_color(MUTED)
    a.tick_params(colors=INK2)
fig.tight_layout(pad=0.35, w_pad=1.0)
for ext, kw in ((".pdf", {}), (".png", {"dpi": 600})):
    fig.savefig("results/fig_ablation_mad" + ext, bbox_inches="tight", facecolor="white", **kw)
    print("[Fig] -> results/fig_ablation_mad" + ext)
print(f"[Fig] delta: min {min(delta):+.1f}  median {np.median(delta):+.1f}  max {max(delta):+.1f} pp"
      f"  over {len(delta)} configs")
