"""How low can the MAD threshold go before the advantage disappears?

Two limits, and they do not coincide:
  * the RECALL limit -- below z~2.0 every episode gets a node and recall is 100%
  * the ADVANTAGE limit -- below z~0.5 uniform ALSO reaches 100%, so the
    advantage at equal node count is exactly zero
Between them the advantage decays smoothly. The point of the figure is that the
floor is set by the BASELINE catching up, not by the detector failing.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ablate_mad as A

PEAKS, UNIFORM, MUTED, INK, INK2, GRID = "#0072B2", "#E69F00", "#8a8a8a", "#000000", "#3d3d3d", "#dcdcdc"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})

eps = json.load(open("evidence/2026-09-09_entropy_at_junctions/data/"
                     "entropy_nogoal_500cap.json"))["episodes"]
ZS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]
rows = [(z, A.evaluate(eps, z=z)) for z in ZS]
rows = [(z, r) for z, r in rows if r]
z = [a for a, _ in rows]
rec = [r["recall"] * 100 for _, r in rows]
uni = [r["uni_recall"] * 100 for _, r in rows]
dlt = [(r["recall"] - r["uni_recall"]) * 100 for _, r in rows]
nod = [r["nodes"] for _, r in rows]

fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))

ax = axes[0]
ax.axvspan(0, 0.6, color=MUTED, alpha=0.16, linewidth=0)
ax.text(0.30, 50, "Saturated:\nuniform also\nreaches 100%", fontsize=6.0,
        color=INK2, ha="center", va="center")
ax.plot(z, rec, color=PEAKS, marker="o", markersize=3.2, linewidth=1.3,
        markeredgecolor="white", markeredgewidth=0.5, label="Entropy peaks")
ax.plot(z, uni, color=UNIFORM, marker="s", markersize=3.0, linewidth=1.3,
        markeredgecolor="white", markeredgewidth=0.5, label="Uniform, count-matched")
ax.set_xlabel("MAD threshold $z$")
ax.set_ylabel("Junction recall (<10 m), %")
ax.set_ylim(0, 108)
ax.set_title("(a) The baseline catches up, not the detector", pad=3)
ax.legend(frameon=False, loc="lower right", handlelength=1.5)

ax = axes[1]
ax.axhline(0, color=MUTED, linewidth=0.8)
ax.plot(z, dlt, color=PEAKS, marker="o", markersize=3.4, linewidth=1.4,
        markeredgecolor="white", markeredgewidth=0.5)
best = int(np.argmax(dlt))
ax.plot([z[best]], [dlt[best]], "o", color=PEAKS, markersize=7,
        markeredgecolor=INK, markeredgewidth=0.9, zorder=5)
ax.annotate(f"Best: $z$={z[best]}\n{dlt[best]:+.1f} pp, {nod[best]} nodes",
            (z[best], dlt[best]), textcoords="offset points", xytext=(6, -20),
            fontsize=6.3, color=INK)
for zz, dd, nn in zip(z, dlt, nod):
    if zz in (0.5, 1.0, 3.0, 4.0):
        ax.annotate(f"{nn}", (zz, dd), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=5.8, color=INK2)
ax.set_xlabel("MAD threshold $z$")
ax.set_ylabel("Advantage over uniform (pp)")
ax.set_title("(b) Advantage vs threshold (labels = node count)", pad=3)

for a in axes:
    a.grid(True, color=GRID, linewidth=0.45); a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_color(MUTED)
    a.tick_params(colors=INK2)
fig.tight_layout(pad=0.35, w_pad=1.1)
for ext, kw in ((".pdf", {}), (".png", {"dpi": 600})):
    fig.savefig("results/fig_z_floor" + ext, bbox_inches="tight", facecolor="white", **kw)
    print("[Fig] -> results/fig_z_floor" + ext)
print(f"[Fig] best z={z[best]} at {dlt[best]:+.1f} pp with {nod[best]} nodes; "
      f"advantage reaches zero at z<=0.5")
