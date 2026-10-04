"""What the localiser swap bought, by arm and map size (user 2026-09-19 "replace localization with dinov2").

Top row: success rate. Bottom row: median start localisation error, which is the quantity the swap was
meant to fix -- read them together, because the two only move together on the sparse maps.
Three lines per panel, same 40 episodes and same maps throughout; only the features node matching uses
differ. `mobilenet_bank` is the control: same encoder, node features re-captured, so any gap between it
and DINOv2 is the encoder alone.
Run from the icra-2027 folder.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
from localiser_results_table import load, sign_test, BASE, ARMS  # noqa: E402

MODES = [("mobilenet_trace", "MobileNet (shipped)", "#6b6b6b", "o", "-"),
         ("mobilenet_bank", "MobileNet + bank (control)", "#2a78d6", "s", "--"),
         ("dinov2_bank", "DINOv2", "#c0398b", "^", "-")]
SEQ = "results/maps/sequence"

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9})


def main():
    import json
    runs = {"mobilenet_trace": load(BASE, "mobilenet_trace"),
            "mobilenet_bank": load(["results/drive_plan_c/localiser/mobilenet_bank_*.jsonl"], "mobilenet_bank"),
            "dinov2_bank": load(["results/drive_plan_c/localiser/dinov2_bank_*.jsonl"], "dinov2_bank")}
    nodes_of = lambda m: len(json.load(open(f"{SEQ}/{m}_sequence.json"))["nodes"])

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.6), sharex=True, constrained_layout=True)
    for j, (arm, maps) in enumerate(ARMS):
        n = [nodes_of(m) for m in maps]
        for key, label, c, mk, ls in MODES:
            sr = [np.mean([r["success"] for r in runs[key][m].values()]) if m in runs[key] else np.nan
                  for m in maps]
            er = [np.median([r["start_node_err_m"] for r in runs[key][m].values()]) if m in runs[key] else np.nan
                  for m in maps]
            axes[0, j].plot(n, sr, color=c, marker=mk, ms=5, lw=1.6, ls=ls, label=label, zorder=3)
            axes[1, j].plot(n, er, color=c, marker=mk, ms=5, lw=1.6, ls=ls, zorder=3)
        # mark the paired sign tests that reached p < 0.1 against the shipped row
        for m, x in zip(maps, n):
            if m in runs["dinov2_bank"] and m in runs["mobilenet_trace"]:
                A, B = runs["dinov2_bank"][m], runs["mobilenet_trace"][m]
                a = {k: A[k]["success"] for k in A if k in B}
                _wa, _wb, p = sign_test(a, {k: B[k]["success"] for k in a})
                if p < 0.1:
                    y = np.mean([r["success"] for r in A.values()])
                    # the leftmost point sits on the axis edge; push its label inboard
                    off, ha = ((14, 10), "left") if x == min(n) else ((0, 10), "center")
                    axes[0, j].annotate(f"p={p:.3f}".rstrip("0"), (x, y), textcoords="offset points",
                                        xytext=off, ha=ha, fontsize=7, color="#c0398b")
        axes[0, j].set_title(arm, fontsize=10)
        for i in (0, 1):
            axes[i, j].grid(True, axis="y", color="#ececec", lw=0.6)
            axes[i, j].spines[["top", "right"]].set_visible(False)
            axes[i, j].set_xlim(0, 640)
        axes[0, j].set_ylim(0, 0.62)
        axes[1, j].set_yscale("log"); axes[1, j].set_ylim(0.5, 60)
        axes[1, j].set_yticks([1, 3, 10, 30], ["1", "3", "10", "30"])
        axes[1, j].set_xlabel("Nodes")
    axes[0, 0].set_ylabel("Success rate (SR)")
    axes[1, 0].set_ylabel("Start localisation error (m)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False, fontsize=8.5, handletextpad=0.5)
    fig.suptitle("Town02: the same maps and episodes, three sources of features for node matching", fontsize=10)
    for e in ("pdf", "png"):
        fig.savefig(f"figures/localiser_sweep_town02.{e}", **({"dpi": 200} if e == "png" else {}))
    print("-> figures/localiser_sweep_town02.{pdf,png}")


if __name__ == "__main__":
    main()
