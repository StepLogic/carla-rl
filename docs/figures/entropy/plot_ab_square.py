import json, math, sys, numpy as np, matplotlib
matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"] = 42; matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
mode = sys.argv[1]                       # "drive" or "z25"
z = np.load("egy_trace_ent.npz")
allE = np.concatenate([z["Town02_ent"], z["Town03_ent"]])
vmin, vmax = np.percentile(allE, 1), np.percentile(allE, 99)
# two equal square panels side by side, colour bar underneath, minimal margins
W, gap, cbh, cbgap, lab = 6.0, 0.12, 0.16, 0.10, 0.28
fig = plt.figure(figsize=(2 * W + gap, W + lab + cbgap + cbh + 0.45), facecolor="white")
FW, FH = fig.get_size_inches()
axes = [fig.add_axes([0, (cbh + cbgap + 0.45) / FH, W / FW, W / FH]),
        fig.add_axes([(W + gap) / FW, (cbh + cbgap + 0.45) / FH, W / FW, W / FH])]
cax = fig.add_axes([0.0, 0.45 / FH, 1.0, cbh / FH])
for ax, T, L in zip(axes, ("Town02", "Town03"), ("A", "B")):
    m = json.load(open(f"topdown_{T}.json")); img = np.asarray(Image.open(f"topdown_{T}.png"))[..., :3]
    img = 0.35 + 0.65 * np.fliplr(img.transpose(1, 0, 2)).astype(float) / 255.0
    f = m["W"] / (2 * math.tan(math.radians(m["fov"]) / 2))
    proj = lambda P: (m["W"] / 2 + f * (P[:, 0] - m["cx"]) / m["z"], m["H"] / 2 + f * (P[:, 1] - m["cy"]) / m["z"])
    tu, tv = proj(z[f"{T}_pos"])
    cx, cy = (tu.min() + tu.max()) / 2, (tv.min() + tv.max()) / 2
    half = max(tu.max() - tu.min(), tv.max() - tv.min()) / 2 + 25
    ax.imshow(img)
    if mode == "drive":
        E = z[f"{T}_ent"]; o = np.argsort(E)
        sc = ax.scatter(tu[o], tv[o], s=3 if T == "Town02" else 1.2, c=E[o], cmap="inferno", vmin=vmin, vmax=vmax, lw=0, zorder=2)
    else:
        N = json.load(open(f"z25_{T}.json"))["nodes"]
        K = [n for n in N if n.get("kind", "key") == "key"]
        u, v = proj(np.array([n["pos"][:2] for n in K]))
        sc = ax.scatter(u, v, s=46 if T == "Town02" else 20, c=[n["entropy"] for n in K], cmap="inferno", vmin=vmin, vmax=vmax,
                        lw=0.6, edgecolors="#1a1a1a", zorder=3)
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy + half, cy - half); ax.axis("off")
    ax.text(0.0, 1.005, L, transform=ax.transAxes, fontsize=20, fontweight="bold", va="bottom", ha="left")
cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cb.set_label("Entropy", fontsize=13); cb.ax.tick_params(labelsize=11)
name = "entropy_drive_AB" if mode == "drive" else "entropy_keys_z25_AB"
fig.savefig(f"{name}.pdf", facecolor="white", bbox_inches="tight", pad_inches=0.02, dpi=250)
fig.savefig(f"{name}.png", facecolor="white", bbox_inches="tight", pad_inches=0.02, dpi=90)
print(name)
